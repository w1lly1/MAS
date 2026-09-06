import json
import os
import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from .base_agent import BaseAgent, Message
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.database.weaviate import WeaviateVectorService
from infrastructure.config.ai_agents import get_ai_agent_config
from infrastructure.config.prompts import get_prompt
from infrastructure.reports import report_manager
from utils.prompt_budgeting import prepare_generation_prompt, semantic_truncate_text, resolve_model_max_tokens
from utils import log, LogLevel


class AIDrivenSecondPassAnalysisAgent(BaseAgent):
    """二次分析代理。

    目标：在可读性增强之前，融合数据库知识对 consolidated 报告做二次修正：
    1. 纠正模型误判（严重级别、来源标签等）
    2. 补充模型漏报（基于历史知识高相似命中）
    """

    def __init__(self):
        super().__init__(
            agent_id="ai_second_pass_analysis_agent",
            name="AI驱动二次分析代理",
        )

        self.agent_config = get_ai_agent_config().get_second_pass_agent_config()
        self.enable_second_pass = self.agent_config.get("enabled", True)
        self.enable_weaviate_query = self.agent_config.get("enable_weaviate_query", True)
        # 二次阶段「非生成式」：默认关闭 LLM 纠错/补漏/总结，走硬编码检索派生
        # （与论文"二次阶段禁止自由生成式缺口发现"一致，且省显存、避免 OOM）。
        self.enable_llm_second_pass = self.agent_config.get("enable_llm_second_pass", False)
        self.fallback_to_original = self.agent_config.get("fallback_to_original_on_error", True)
        self.weaviate_top_k = int(self.agent_config.get("weaviate_top_k", 5))
        self.similarity_threshold = float(self.agent_config.get("similarity_threshold", 0.78))
        # 统一门控判别式（两通道 DNF）的三个阈值：
        #   admit = F(x) ∧ [ s(x) ≥ gate_structured_threshold
        #                    ∨ ( v(x) ≥ similarity_threshold
        #                        ∧ a(x) ≥ gate_anchor_threshold
        #                        ∧ s(x) ≥ gate_weak_structure_threshold ) ]
        self.gate_structured_threshold = float(
            self.agent_config.get("gate_structured_threshold", 0.65)
        )
        self.gate_anchor_threshold = float(
            self.agent_config.get("gate_anchor_threshold", 0.35)
        )
        self.gate_weak_structure_threshold = float(
            self.agent_config.get("gate_weak_structure_threshold", 0.2)
        )
        self.max_new_findings = int(self.agent_config.get("max_new_findings", 5))
        self.max_sqlite_patterns = int(self.agent_config.get("max_sqlite_patterns", 200))
        self.llm_max_input_chars = int(self.agent_config.get("llm_max_input_chars", 9000))
        self.gap_code_chunk_chars = int(self.agent_config.get("gap_code_chunk_chars", 1200))
        self.gap_chunk_overlap_lines = int(self.agent_config.get("gap_chunk_overlap_lines", 2))
        self.max_gap_code_chunks = int(
            self.agent_config.get("max_gap_code_chunks", max(20, self.max_new_findings * 8))
        )
        default_layer_bonus = {
            "semantic": 0.08,
            "solution": 0.05,
            "code_pattern": 0.03,
            "full": 0.01,
        }
        configured_layer_bonus = self.agent_config.get("layer_bonus") or {}
        self.layer_bonus_map: Dict[str, float] = {
            str(k).strip().lower(): float(v)
            for k, v in {**default_layer_bonus, **configured_layer_bonus}.items()
            if str(k).strip()
        }
        self.layer_bonus_require_similarity_gate = bool(
            self.agent_config.get("layer_bonus_require_similarity_gate", True)
        )
        # 错误代码克隆检测（问题1修复）：用"diff 前错误代码的连续 token 序列"
        # 替代"文件路径 + 行号"作为结构化命中的主匹配键。
        self.error_code_clone_min_tokens = int(
            self.agent_config.get("error_code_clone_min_tokens", 4)
        )

        self.used_device = "gpu"
        self.text_generator = None
        self.model_name = self.agent_config.get("model_name", "gpt2")
        self.fallback_model = self.agent_config.get("fallback_model", "distilgpt2")
        self._shared_generator_injected = False

        self.db_service = DatabaseService()
        self.vector_service = WeaviateVectorService()
        self._weaviate_connect_attempted = False
        self._llm_init_attempted = False
        self.models_loaded = False
        self._debug_log_run_id = None
        self._debug_log_path = None

    async def initialize(self):
        try:
            self.is_running = True
            if self.enable_llm_second_pass:
                await self._initialize_models()
            if self.enable_weaviate_query:
                connected = self.vector_service.connect(auto_create_schema=False)
                if connected:
                    log("second_pass_agent", LogLevel.INFO, "✅ Weaviate 已连接，二次分析检索启用")
                else:
                    log("second_pass_agent", LogLevel.WARNING, "⚠️ Weaviate 未连接，二次分析将只使用 SQLite 规则")
            log("second_pass_agent", LogLevel.INFO, "✅ 二次分析代理初始化完成")
        except Exception as e:
            log("second_pass_agent", LogLevel.WARNING, f"⚠️ 初始化异常，降级为仅透传模式: {e}")

    async def stop(self):
        await super().stop()
        try:
            self.vector_service.disconnect()
        except Exception:
            pass

    def set_shared_generator(self, generator, tokenizer=None):
        """注入共享的文本生成 pipeline（如 Qwen），避免重复加载 gpt2。"""
        if generator is None:
            return
        self.text_generator = generator
        if tokenizer is None:
            tokenizer = getattr(generator, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass
        self._shared_generator_injected = True
        self._llm_init_attempted = True
        self.models_loaded = True
        log("second_pass_agent", LogLevel.INFO, "✅ 二次分析Agent已注入共享文本生成模型")

    async def _initialize_models(self):
        """初始化二次分析LLM，失败时保留硬编码回退路径。"""
        if self._llm_init_attempted:
            return
        self._llm_init_attempted = True

        if self._shared_generator_injected and self.text_generator is not None:
            self.models_loaded = True
            log("second_pass_agent", LogLevel.INFO, "♻️ 二次分析使用已注入的共享文本生成模型")
            return

        try:
            if self.used_device not in ["cpu", "gpu"]:
                self.used_device = "cpu"

            device = -1 if self.used_device == "cpu" else 0
            if self.used_device == "cpu":
                cpu_threads = int(self.agent_config.get("cpu_threads", 4))
                torch.set_num_threads(cpu_threads)

            cache_dir = get_ai_agent_config().get_model_cache_dir()
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.abspath(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

            model_name = self.model_name
            log("second_pass_agent", LogLevel.INFO, f"🤖 初始化二次分析LLM: {model_name}")

            try:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                    )
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        trust_remote_code=False,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        low_cpu_mem_usage=True,
                    )
            except Exception as main_model_err:
                log("second_pass_agent", LogLevel.WARNING, f"⚠️ 主模型初始化失败，尝试备用模型 {self.fallback_model}: {main_model_err}")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.fallback_model,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    trust_remote_code=False,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.fallback_model,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                )

            self.text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

            if self.text_generator.tokenizer.pad_token is None:
                self.text_generator.tokenizer.pad_token = self.text_generator.tokenizer.eos_token

            self.models_loaded = True
            log("second_pass_agent", LogLevel.INFO, "✅ 二次分析LLM初始化完成")
        except Exception as e:
            log("second_pass_agent", LogLevel.WARNING, f"⚠️ 二次分析LLM初始化失败，启用硬编码回退: {e}")
            self.text_generator = None
            self.models_loaded = False

    async def handle_message(self, message: Message):
        if message.message_type != "analyze_consolidated_report_for_second_pass":
            return

        run_id = message.content.get("run_id")
        requirement_id = message.content.get("requirement_id")
        file_path = message.content.get("file_path")
        report_data = message.content.get("report_data")
        original_analysis = message.content.get("original_analysis")
        self._ensure_debug_log_path(run_id)
        if isinstance(original_analysis, dict):
            self._debug_log(
                run_id,
                "received original_analysis",
                {"keys": list(original_analysis.keys())},
            )
        else:
            self._debug_log(run_id, "received original_analysis (invalid)")

        if not isinstance(report_data, dict):
            log("second_pass_agent", LogLevel.WARNING, "⚠️ 未收到有效 report_data，回退原始链路")
            await self._forward_to_readability(
                report_data if isinstance(report_data, dict) else {},
                run_id,
                requirement_id,
                file_path,
                validation_failed=True,
            )
            return

        is_valid, validation_errors = self._validate_json_report(report_data)
        if not is_valid:
            log(
                "second_pass_agent",
                LogLevel.WARNING,
                f"⚠️ consolidated JSON 校验失败，将透传原始结果: {validation_errors}",
            )
            await self._forward_to_readability(
                report_data,
                run_id,
                requirement_id,
                file_path,
                validation_failed=True,
                validation_errors=validation_errors,
            )
            return

        # 在开始二次分析前，始终保存一份不经过二次分析的纯 LLM 输出，供对比
        try:
            base_name = str(report_data.get("sanitized_name") or f"req_{report_data.get('requirement_id', 'unknown')}")
            filename = f"consolidated_{base_name}.json"
            report_manager.generate_run_scoped_report(run_id=run_id, content=report_data, filename=filename, subdir="pureLLM/consolidated")
        except Exception:
            pass

        try:
            # 执行两轮独立的二次分析并分别保存：
            # - 第1轮 (all_only)：仅使用数据库的全量层，保存到 run_id/fullLayer/consolidated
            # - 第2轮 (all layers)：使用所有分层，保存为二次分析完整结果（保存在 second_pass_consolidated_*_r2）
            refined_round1 = await self._run_second_pass(report_data, original_analysis=original_analysis, layer_mode="all_only")
            await self._persist_second_pass_report(refined_round1, round_num=1, subdir="fullLayer/consolidated")
            await self._forward_to_readability(refined_round1, run_id, requirement_id, file_path, second_pass_round=1, weaviate_layer_mode="all_only")

            refined_round2 = await self._run_second_pass(report_data, original_analysis=original_analysis, layer_mode=None)
            await self._persist_second_pass_report(refined_round2, round_num=2, subdir="second_pass/consolidated")
            await self._forward_to_readability(refined_round2, run_id, requirement_id, file_path, second_pass_round=2, weaviate_layer_mode="all_layers")
            log(
                "second_pass_agent",
                LogLevel.INFO,
                f"✅ 二次分析完成 run_id={run_id} requirement_id={requirement_id}"
            )
        except Exception as e:
            log("second_pass_agent", LogLevel.ERROR, f"❌ 二次分析失败: {e}")
            if self.fallback_to_original:
                await self._forward_to_readability(
                    report_data,
                    run_id,
                    requirement_id,
                    file_path,
                    second_pass_error=str(e),
                )

    def _validate_json_report(self, report_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        # 要求：在二次分析阶段之前明确验证其可 JSON 序列化。
        try:
            serialized = json.dumps(report_data, ensure_ascii=False)
            reparsed = json.loads(serialized)
            if not isinstance(reparsed, dict):
                errors.append("report_data 不是 JSON object")
        except Exception as e:
            errors.append(f"report_data 非法 JSON: {e}")
            return False, errors

        required_fields = ["run_id", "requirement_id", "issues", "analysis_types"]
        for field in required_fields:
            if field not in report_data:
                errors.append(f"缺少字段: {field}")

        issues = report_data.get("issues", [])
        if not isinstance(issues, list):
            errors.append("字段 issues 必须是 list")

        return len(errors) == 0, errors

    async def _run_second_pass(
        self,
        report_data: Dict[str, Any],
        original_analysis: Optional[Dict[str, Any]] = None,
        layer_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        original_issues = report_data.get("issues", [])
        if not isinstance(original_issues, list):
            original_issues = []

        # 根据 layer_mode 决定是否限制检索到数据库的 patterns 或 weaviate layer
        sqlite_patterns = await self.db_service.get_issue_patterns(status="active")
        sqlite_patterns = sqlite_patterns[: self.max_sqlite_patterns]

        corrected_issues: List[Dict[str, Any]] = []
        confidence_adjustments: List[Dict[str, Any]] = []
        retrieval_evidence: List[Dict[str, Any]] = []
        gap_retrieval_evidence: List[Dict[str, Any]] = []
        new_findings: List[Dict[str, Any]] = []

        # 查询1：用一轮 consolidated 输出查库，验判现有问题
        for issue in original_issues:
            if not isinstance(issue, dict):
                continue
            self._debug_log(
                report_data.get("run_id"),
                "collect evidence for issue",
                {
                    "file": issue.get("file"),
                    "line": issue.get("line"),
                    "source": issue.get("source"),
                    "severity": issue.get("severity"),
                    "description": issue.get("description"),
                },
            )
            evidence = await self._collect_evidence(issue, sqlite_patterns, layer_mode=layer_mode)
            evidence["query_channel"] = "validation_from_consolidated"
            evidence["query_pass_label"] = "一轮LLM/consolidated分析命中数据库"
            retrieval_evidence.append(evidence)

        # Task 1: LLM纠错（失败则回退到硬编码纠错）
        llm_correction_ok = False
        if self.enable_llm_second_pass and self.text_generator:
            try:
                corrected_issues, confidence_adjustments = await self._llm_issue_correction(
                    original_issues,
                    retrieval_evidence,
                )
                llm_correction_ok = True
            except Exception as e:
                log("second_pass_agent", LogLevel.WARNING, f"⚠️ LLM纠错失败，回退硬编码纠错: {e}")

        if not llm_correction_ok:
            corrected_issues = []
            confidence_adjustments = []
            for idx, issue in enumerate(original_issues):
                if not isinstance(issue, dict):
                    continue
                evidence = retrieval_evidence[idx] if idx < len(retrieval_evidence) else {}
                corrected_issue, adjustment = self._apply_corrections(issue, evidence)
                corrected_issues.append(corrected_issue)
                if adjustment:
                    confidence_adjustments.append(adjustment)

        # 查询2：对原始源代码按上下文分片 → 查库 → 喂 LLM 做相似度补漏
        code_chunks = self._build_source_code_chunks(
            report_data=report_data,
            original_issues=original_issues,
            fallback_file=report_data.get("file"),
        )
        gap_retrieval_evidence = await self._collect_gap_evidence_from_code_chunks(
            code_chunks=code_chunks,
            sqlite_patterns=sqlite_patterns,
            layer_mode=layer_mode,
            run_id=report_data.get("run_id"),
        )
        self._debug_log(
            report_data.get("run_id"),
            "gap evidence collected from code chunks",
            {
                "code_chunk_count": len(code_chunks),
                "gap_evidence_count": len(gap_retrieval_evidence),
            },
        )

        # Task 2: LLM补漏（失败/产出为空则回退到硬编码补漏）
        llm_gap_ok = False
        if self.enable_llm_second_pass and self.text_generator:
            try:
                llm_gap_findings = await self._llm_gap_discovery(
                    corrected_issues,
                    gap_retrieval_evidence=gap_retrieval_evidence,
                    run_id=report_data.get("run_id"),
                    requirement_id=report_data.get("requirement_id"),
                    file_path=report_data.get("file"),
                    fallback_retrieval_evidence=retrieval_evidence,
                )
                if llm_gap_findings:
                    new_findings = llm_gap_findings
                    llm_gap_ok = True
                else:
                    log("second_pass_agent", LogLevel.WARNING, "⚠️ LLM补漏产出为空，回退硬编码补漏")
            except Exception as e:
                log("second_pass_agent", LogLevel.WARNING, f"⚠️ LLM补漏失败，回退硬编码补漏: {e}")

        if not llm_gap_ok:
            new_findings = self._derive_new_findings_from_gap_evidence(
                gap_retrieval_evidence=gap_retrieval_evidence,
                fallback_issues=corrected_issues,
                fallback_evidence=retrieval_evidence,
                run_id=report_data.get("run_id"),
                requirement_id=report_data.get("requirement_id"),
                file_path=report_data.get("file"),
            )

        merged_issues = self._dedupe_issues(corrected_issues + new_findings)
        merged_issues = self._rank_issues_by_evidence(merged_issues)
        severity_stats = self._build_severity_stats(merged_issues)

        refined = dict(report_data)
        refined["original_issues"] = list(original_issues)
        refined["original_issue_count"] = len(original_issues)
        refined["issues"] = merged_issues
        refined["issue_count"] = len(merged_issues)
        refined["severity_stats"] = severity_stats
        refined["second_pass_version"] = "1.0"
        refined["corrected_issues"] = confidence_adjustments
        refined["new_findings"] = new_findings
        refined["retrieval_evidence"] = retrieval_evidence
        refined["gap_retrieval_evidence"] = gap_retrieval_evidence
        refined["gap_code_chunks"] = [
            {
                "file": c.get("file"),
                "start_line": c.get("start_line"),
                "end_line": c.get("end_line"),
                "chunk_index": c.get("chunk_index"),
                "char_count": len(str(c.get("text") or "")),
            }
            for c in code_chunks
        ]
        second_pass_summary = {
            "original_issue_count": len(original_issues),
            "corrected_issue_count": len(confidence_adjustments),
            "new_finding_count": len(new_findings),
            "final_issue_count": len(merged_issues),
            "llm_correction_used": llm_correction_ok,
            "llm_gap_discovery_used": llm_gap_ok,
            "gap_evidence_count": len(gap_retrieval_evidence),
            "code_chunk_count": len(code_chunks),
        }

        # Task 3: LLM总结（失败使用规则总结）
        llm_summary = None
        if self.enable_llm_second_pass and self.text_generator:
            try:
                llm_summary = await self._llm_second_pass_summary(second_pass_summary)
            except Exception as e:
                log("second_pass_agent", LogLevel.WARNING, f"⚠️ LLM总结失败，使用规则总结: {e}")

        if isinstance(llm_summary, dict):
            second_pass_summary.update(llm_summary)

        refined["second_pass_summary"] = second_pass_summary
        analysis_types = refined.get("analysis_types", [])
        if isinstance(analysis_types, list) and "second_pass_analysis" not in analysis_types:
            analysis_types.append("second_pass_analysis")
            refined["analysis_types"] = analysis_types

        return refined

    async def _llm_issue_correction(
        self,
        issues: List[Dict[str, Any]],
        retrieval_evidence: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not issues:
            return [], []

        semantic_chunks = self._build_semantic_chunks(issues, retrieval_evidence)

        corrected_issues: List[Dict[str, Any]] = list(issues)
        confidence_adjustments: List[Dict[str, Any]] = []
        parsed_any_chunk = False

        for chunk in semantic_chunks:
            chunk_issues = [unit["issue"] for unit in chunk]
            chunk_evidence = [unit["evidence"] for unit in chunk]

            prompt = get_prompt(
                task_type="analysis_report",
                variant="second_pass_correction",
                issues_json=json.dumps(chunk_issues, ensure_ascii=False),
                retrieval_evidence_json=json.dumps(chunk_evidence, ensure_ascii=False),
            )
            generated = await self._run_generation_inference(
                prompt,
                max_new_tokens=384,
                temperature=0.2,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.text_generator.tokenizer.eos_token_id if self.text_generator else None,
            )
            parsed = self._parse_json_object(generated)
            if not parsed:
                continue

            parsed_any_chunk = True
            index_map: Dict[int, Dict[str, Any]] = {}
            for item in parsed.get("corrected_issues", []):
                if not isinstance(item, dict):
                    continue
                local_idx = item.get("index")
                issue_obj = item.get("issue")
                if isinstance(local_idx, int) and isinstance(issue_obj, dict):
                    if 0 <= local_idx < len(chunk):
                        global_idx = int(chunk[local_idx]["index"])
                        index_map[global_idx] = issue_obj

            for global_idx, corrected in index_map.items():
                if 0 <= global_idx < len(corrected_issues):
                    corrected_issues[global_idx] = corrected

            adjustments = parsed.get("confidence_adjustments", [])
            if isinstance(adjustments, list):
                confidence_adjustments.extend(adjustments)

        if not parsed_any_chunk:
            raise ValueError("LLM纠错输出不可解析")

        return corrected_issues, confidence_adjustments

    async def _llm_gap_discovery(
        self,
        issues: List[Dict[str, Any]],
        gap_retrieval_evidence: List[Dict[str, Any]],
        run_id: Optional[str],
        requirement_id: Optional[int],
        file_path: Optional[str],
        fallback_retrieval_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """基于源代码分片查库证据，让 LLM 做相似度匹配并补漏。"""
        evidence_for_gap = list(gap_retrieval_evidence or [])
        if not evidence_for_gap and issues and fallback_retrieval_evidence:
            # 无源码分片时回退查询1证据，保持兼容
            evidence_for_gap = []
            for idx, issue in enumerate(issues):
                if not isinstance(issue, dict):
                    continue
                ev = fallback_retrieval_evidence[idx] if idx < len(fallback_retrieval_evidence) else {}
                if not isinstance(ev, dict):
                    ev = {}
                packed = dict(ev)
                packed["code_chunk"] = {
                    "file": issue.get("file") or file_path,
                    "start_line": issue.get("line"),
                    "end_line": issue.get("line"),
                    "text": str(issue.get("description") or "")[: self.gap_code_chunk_chars],
                    "chunk_index": idx,
                }
                evidence_for_gap.append(packed)

        if not evidence_for_gap:
            return []

        normalized: List[Dict[str, Any]] = []
        parsed_any_chunk = False
        max_chars = max(800, int(self.llm_max_input_chars))

        for evidence in evidence_for_gap:
            if len(normalized) >= self.max_new_findings:
                break
            if not isinstance(evidence, dict):
                continue

            code_chunk = evidence.get("code_chunk") if isinstance(evidence.get("code_chunk"), dict) else {}
            chunk_file = str(code_chunk.get("file") or evidence.get("issue_file") or file_path or "").strip()
            related_reported = [
                issue
                for issue in issues
                if isinstance(issue, dict)
                and (
                    not chunk_file
                    or str(issue.get("file") or "").strip() == chunk_file
                )
            ]
            if not related_reported:
                related_reported = list(issues)

            # 控制喂给 LLM 的代码与证据体积
            code_payload = {
                "file": chunk_file,
                "start_line": code_chunk.get("start_line"),
                "end_line": code_chunk.get("end_line"),
                "chunk_index": code_chunk.get("chunk_index"),
                "text": str(code_chunk.get("text") or "")[:max_chars],
            }
            # 确定性过滤：门控未通过(discarded/low_confidence)的候选不喂 LLM，
            # 使门控成为硬闸门（离线回放结果 = 真实结果）。
            gate_pass = {"formal_hit", "explanatory_hit"}
            surviving_candidates = [
                c for c in (evidence.get("candidates") or [])
                if isinstance(c, dict) and c.get("gating_decision") in gate_pass
            ][: self.weaviate_top_k]
            if not surviving_candidates:
                # 该分片没有通过门控的候选，喂 LLM 无意义；跳过以提速并避免 OOM/超时。
                continue
            surviving_sids = {c.get("sqlite_id") for c in surviving_candidates}
            surviving_weaviate_hits = [
                h for h in (evidence.get("weaviate_hits") or [])
                if isinstance(h, dict) and h.get("sqlite_id") in surviving_sids
            ][: self.weaviate_top_k]
            evidence_payload = {
                "query_channel": evidence.get("query_channel"),
                "weaviate_hits": surviving_weaviate_hits,
                "evidence_hits": (evidence.get("evidence_hits") or [])[: self.weaviate_top_k],
                "candidates": [
                    {
                        "gating_decision": c.get("gating_decision"),
                        "error_type": c.get("error_type"),
                        "severity": c.get("severity"),
                        "sqlite_id": c.get("sqlite_id"),
                        "semantic_score": c.get("semantic_score"),
                        "total_score": c.get("total_score"),
                        "vector_layer": c.get("vector_layer"),
                        "matched_layers": c.get("matched_layers"),
                        "matched_layer_details": c.get("matched_layer_details"),
                        "layer_bonus": c.get("layer_bonus"),
                        "confidence_components": c.get("confidence_components"),
                        "solution": c.get("solution"),
                        "reasoning": c.get("reasoning"),
                    }
                    for c in surviving_candidates
                ],
            }

            self._debug_log(
                run_id,
                "code chunk prepared for gap LLM",
                {
                    "file": chunk_file,
                    "start_line": code_payload.get("start_line"),
                    "end_line": code_payload.get("end_line"),
                    "text_chars": len(code_payload.get("text") or ""),
                    "candidate_count": len(evidence_payload.get("candidates") or []),
                },
            )

            prompt = get_prompt(
                task_type="analysis_report",
                variant="second_pass_gap_discovery",
                issues_json=json.dumps(related_reported, ensure_ascii=False)[:max_chars],
                retrieval_evidence_json=json.dumps(evidence_payload, ensure_ascii=False)[:max_chars],
                raw_analysis_json=json.dumps(code_payload, ensure_ascii=False)[:max_chars],
                run_id=run_id or "",
                requirement_id=requirement_id or 0,
                file_path=chunk_file or file_path or "",
            )
            generated = await self._run_generation_inference(
                prompt,
                max_new_tokens=384,
                temperature=0.2,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.text_generator.tokenizer.eos_token_id if self.text_generator else None,
            )
            parsed = self._parse_json_object(generated)
            if not parsed:
                continue

            parsed_any_chunk = True
            findings = parsed.get("new_findings", [])
            if not isinstance(findings, list):
                continue

            for f in findings:
                if len(normalized) >= self.max_new_findings:
                    break
                if not isinstance(f, dict):
                    continue
                normalized.append(
                    {
                        "requirement_id": f.get("requirement_id", requirement_id),
                        "file": f.get("file", chunk_file or file_path),
                        "source": f.get("source", "db_supplemented"),
                        "severity": f.get("severity", "medium"),
                        "line": f.get("line", code_chunk.get("start_line")),
                        "description": f.get("description", "源代码分片与历史知识相似，可能漏报"),
                        "tool": f.get("tool", "second_pass_analysis"),
                        "run_id": f.get("run_id", run_id),
                        "evidence": f.get("evidence", {}),
                    }
                )

        if not parsed_any_chunk:
            raise ValueError("LLM补漏输出不可解析")

        return normalized

    async def _llm_second_pass_summary(self, summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = get_prompt(
            task_type="analysis_report",
            variant="second_pass_summary",
            original_issue_count=summary.get("original_issue_count", 0),
            corrected_issue_count=summary.get("corrected_issue_count", 0),
            new_finding_count=summary.get("new_finding_count", 0),
            final_issue_count=summary.get("final_issue_count", 0),
        )
        generated = await self._run_generation_inference(
            prompt,
            max_new_tokens=220,
            temperature=0.2,
            do_sample=True,
            return_full_text=False,
            pad_token_id=self.text_generator.tokenizer.eos_token_id if self.text_generator else None,
        )
        parsed = self._parse_json_object(generated)
        if not parsed:
            return None
        sp = parsed.get("second_pass_summary")
        if isinstance(sp, dict):
            return sp
        return None

    def _resolve_source_file_path(self, file_path: Optional[str]) -> Optional[str]:
        if not file_path:
            return None
        p = str(file_path).strip()
        if not p:
            return None
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        if os.path.isfile(p):
            return p
        return None

    def _collect_gap_source_files(
        self,
        report_data: Dict[str, Any],
        original_issues: List[Dict[str, Any]],
        fallback_file: Optional[str] = None,
    ) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def add_path(path: Optional[str]):
            resolved = self._resolve_source_file_path(path)
            if not resolved:
                return
            key = os.path.normcase(os.path.abspath(resolved))
            if key in seen:
                return
            seen.add(key)
            candidates.append(resolved)

        add_path(fallback_file)
        add_path(report_data.get("file") if isinstance(report_data, dict) else None)
        for issue in original_issues:
            if isinstance(issue, dict):
                add_path(issue.get("file"))
        return candidates

    def _split_file_into_context_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """按上下文字符预算切分源文件，相邻分片保留少量行重叠。"""
        chunk_chars = max(200, int(self.gap_code_chunk_chars))
        overlap_lines = max(0, int(self.gap_chunk_overlap_lines))
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return []

        if not lines:
            return []

        chunks: List[Dict[str, Any]] = []
        start_idx = 0
        total = len(lines)
        while start_idx < total:
            current: List[str] = []
            current_size = 0
            end_idx = start_idx
            while end_idx < total:
                line = lines[end_idx]
                line_len = len(line)
                if current and current_size + line_len > chunk_chars:
                    break
                current.append(line)
                current_size += line_len
                end_idx += 1
                if current_size >= chunk_chars:
                    break

            if not current:
                current = [lines[start_idx]]
                end_idx = start_idx + 1

            text = "".join(current).strip("\n")
            if text.strip():
                chunks.append(
                    {
                        "file": file_path,
                        "start_line": start_idx + 1,
                        "end_line": end_idx,
                        "text": text,
                        "chunk_index": len(chunks),
                    }
                )

            if end_idx >= total:
                break
            next_start = max(start_idx + 1, end_idx - overlap_lines)
            if next_start <= start_idx:
                next_start = end_idx
            start_idx = next_start

        return chunks

    def _build_source_code_chunks(
        self,
        report_data: Dict[str, Any],
        original_issues: List[Dict[str, Any]],
        fallback_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        files = self._collect_gap_source_files(report_data, original_issues, fallback_file=fallback_file)
        all_chunks: List[Dict[str, Any]] = []
        for file_path in files:
            all_chunks.extend(self._split_file_into_context_chunks(file_path))
            if len(all_chunks) >= self.max_gap_code_chunks:
                break
        return all_chunks[: self.max_gap_code_chunks]

    def _code_chunk_as_issue(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        text = str(chunk.get("text") or "")
        start_line = chunk.get("start_line")
        end_line = chunk.get("end_line")
        preview = text[:500]
        return {
            "description": f"source_code_chunk L{start_line}-{end_line}: {preview}",
            "source": "source_code_chunk",
            "severity": "medium",
            "file": chunk.get("file"),
            "line": start_line,
            "line_number": start_line,
            "chunk_start_line": start_line,
            "chunk_end_line": end_line,
            "location": f"第{start_line}-{end_line}行",
            "code_snippet": text[:2000],
            "tool": "second_pass_gap_chunk",
        }

    async def _collect_gap_evidence_from_code_chunks(
        self,
        code_chunks: List[Dict[str, Any]],
        sqlite_patterns: List[Dict[str, Any]],
        layer_mode: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询2：对源代码上下文分片独立检索 Weaviate/SQLite。"""
        gap_evidence: List[Dict[str, Any]] = []
        if not code_chunks:
            return gap_evidence

        for chunk in code_chunks[: self.max_gap_code_chunks]:
            if not isinstance(chunk, dict):
                continue
            if not str(chunk.get("text") or "").strip():
                continue
            issue_like = self._code_chunk_as_issue(chunk)
            if run_id:
                issue_like["run_id"] = run_id
            evidence = await self._collect_evidence(
                issue_like,
                sqlite_patterns,
                layer_mode=layer_mode,
            )
            evidence["code_chunk"] = {
                "file": chunk.get("file"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("text"),
            }
            evidence["query_channel"] = "gap_from_original_analysis"
            evidence["query_pass_label"] = "二轮原始源代码分片命中数据库"
            gap_evidence.append(evidence)

        return gap_evidence

    def _derive_new_findings_from_gap_evidence(
        self,
        gap_retrieval_evidence: List[Dict[str, Any]],
        fallback_issues: List[Dict[str, Any]],
        fallback_evidence: List[Dict[str, Any]],
        run_id: Optional[str],
        requirement_id: Optional[int],
        file_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        """硬编码补漏：优先用源代码分片查库证据；无则回退查询1证据。"""
        new_findings: List[Dict[str, Any]] = []

        if gap_retrieval_evidence:
            best_by_key: Dict[Any, Dict[str, Any]] = {}
            for evidence in gap_retrieval_evidence:
                code_chunk = evidence.get("code_chunk") if isinstance(evidence, dict) else None
                if isinstance(code_chunk, dict):
                    issue_like = self._code_chunk_as_issue(code_chunk)
                else:
                    issue_like = {
                        "description": (evidence or {}).get("issue_description"),
                        "line": None,
                        "file": (evidence or {}).get("issue_file") or file_path,
                    }
                candidates = self._derive_new_findings(
                    issue=issue_like,
                    evidence=evidence if isinstance(evidence, dict) else {},
                    run_id=run_id,
                    requirement_id=requirement_id,
                    file_path=issue_like.get("file") or file_path,
                )
                for candidate in candidates:
                    refined_line = self._resolve_gap_finding_line(
                        code_chunk if isinstance(code_chunk, dict) else {},
                        candidate,
                        hit_hint=candidate.get("evidence")
                        if isinstance(candidate.get("evidence"), dict)
                        else None,
                    )
                    if refined_line is not None:
                        candidate["line"] = refined_line
                    elif candidate.get("line") is None and isinstance(code_chunk, dict):
                        candidate["line"] = code_chunk.get("start_line")

                    ev = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
                    key = ev.get("sqlite_id") or candidate.get("description")
                    prev = best_by_key.get(key)
                    if prev is None:
                        best_by_key[key] = candidate
                        continue
                    prev_ev = prev.get("evidence") if isinstance(prev.get("evidence"), dict) else {}
                    prev_score = (
                        float(prev_ev.get("structured_score") or 0.0),
                        float(prev_ev.get("total_score") or 0.0),
                        abs(int(prev.get("line") or 1)),
                    )
                    new_score = (
                        float(ev.get("structured_score") or 0.0),
                        float(ev.get("total_score") or 0.0),
                        abs(int(candidate.get("line") or 1)),
                    )
                    if new_score >= prev_score:
                        best_by_key[key] = candidate

            ranked = sorted(
                best_by_key.values(),
                key=lambda c: (
                    float((c.get("evidence") or {}).get("structured_score") or 0.0),
                    float((c.get("evidence") or {}).get("total_score") or 0.0),
                    abs(int(c.get("line") or 1)),
                ),
                reverse=True,
            )
            for candidate in ranked:
                if len(new_findings) >= self.max_new_findings:
                    break
                new_findings.append(candidate)
            return new_findings

        for idx, issue in enumerate(fallback_issues):
            if len(new_findings) >= self.max_new_findings:
                break
            evidence = fallback_evidence[idx] if idx < len(fallback_evidence) else {}
            candidates = self._derive_new_findings(
                issue=issue,
                evidence=evidence,
                run_id=run_id,
                requirement_id=requirement_id,
                file_path=file_path,
            )
            for candidate in candidates:
                if len(new_findings) >= self.max_new_findings:
                    break
                new_findings.append(candidate)
        return new_findings

    async def _collect_evidence(self, issue: Dict[str, Any], sqlite_patterns: List[Dict[str, Any]], layer_mode: Optional[str] = None) -> Dict[str, Any]:
        issue_desc = str(issue.get("description") or "").strip()
        issue_source = str(issue.get("source") or "").strip()
        issue_severity = str(issue.get("severity") or "").strip().lower()
        issue_file = str(issue.get("file") or "").strip()

        evidence: Dict[str, Any] = {
            "issue_description": issue_desc,
            "issue_source": issue_source,
            "issue_severity": issue_severity,
            "issue_file": issue_file,
            "weaviate_hits": [],
            "sqlite_hits": [],
            "curated_issue_hits": [],
            "candidates": [],
            # 仅保留正式的证据命中（formal/explanatory），供可读性层展示
            "evidence_hits": [],
            "low_confidence_hits": [],
        }

        # SQLite 结构化匹配
        for pattern in sqlite_patterns:
            match_info = self._evaluate_pattern_match(pattern, issue_desc, issue_source, issue_file, issue)
            # if match_info.get("structured_score", 0.0) > 0.0:
            #     self._debug_log(
            #         issue.get("run_id"),
            #         "sqlite pattern checked",
            #         {
            #             "pattern_id": pattern.get("id"),
            #             "error_type": pattern.get("error_type"),
            #             "file_pattern": pattern.get("file_pattern"),
            #             "language": pattern.get("language"),
            #             "matched": match_info.get("matched"),
            #             "matched_fields": match_info.get("matched_fields"),
            #             "structured_score": match_info.get("structured_score"),
            #             "context_score": match_info.get("context_score"),
            #         },
            #     )
            if match_info["matched"]:
                sqlite_hit = {
                    "id": pattern.get("id"),
                    "error_type": pattern.get("error_type"),
                    "severity": pattern.get("severity"),
                    "solution": pattern.get("solution"),
                    "matched_fields": match_info["matched_fields"],
                    "structured_score": match_info["structured_score"],
                    "context_score": match_info["context_score"],
                    # 附上 DB 可读字段，供渲染使用
                    "error_description": pattern.get("error_description"),
                    "problematic_pattern": pattern.get("problematic_pattern"),
                    "file_pattern": pattern.get("file_pattern"),
                    "class_pattern": pattern.get("class_pattern"),
                    "language": pattern.get("language"),
                    "framework": pattern.get("framework"),
                }
                evidence["sqlite_hits"].append(sqlite_hit)
                candidate = self._build_candidate_from_sqlite(sqlite_hit, issue_desc, issue)
                self._gate_candidate(candidate)
                # self._debug_log(
                #     issue.get("run_id"),
                #     "sqlite candidate gated",
                #     {
                #         "sqlite_id": candidate.get("sqlite_id"),
                #         "error_type": candidate.get("error_type"),
                #         "severity": candidate.get("severity"),
                #         "structured_score": candidate.get("structured_score"),
                #         "semantic_score": candidate.get("semantic_score"),
                #         "context_score": candidate.get("context_score"),
                #         "penalty_score": candidate.get("penalty_score"),
                #         "total_score": candidate.get("total_score"),
                #         "gating_decision": candidate.get("gating_decision"),
                #     },
                # )
                evidence["candidates"].append(candidate)
                # 若通过门控进入 formal/explanatory，则把它作为正式证据记入 evidence_hits
                if candidate.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                    hit = dict(candidate)
                    # 辅助字段来自 sqlite_hit
                    hit.update({
                        "error_description": sqlite_hit.get("error_description"),
                        "problematic_pattern": sqlite_hit.get("problematic_pattern"),
                        "file_pattern": sqlite_hit.get("file_pattern"),
                        "class_pattern": sqlite_hit.get("class_pattern"),
                        "language": sqlite_hit.get("language"),
                        "framework": sqlite_hit.get("framework"),
                        "sqlite_id": sqlite_hit.get("id"),
                    })
                    # 尝试提取代码片段（基于 issue 的 file/line）
                    try:
                        line_no = int(issue.get("line")) if issue.get("line") is not None else None
                    except Exception:
                        line_no = None
                    code_snip = self._extract_code_snippet(issue_file, line_no)
                    if code_snip:
                        hit["code_snippet"] = code_snip
                    evidence["evidence_hits"].append(hit)
                elif candidate.get("gating_decision") == "low_confidence_hit":
                    evidence["low_confidence_hits"].append(dict(candidate))

                if len(evidence["sqlite_hits"]) >= self.weaviate_top_k:
                    break

        # CuratedIssue 结构化补强：本地已确认样本（含 BigVul resolved/open）做结构锚点。
        curated_issues: List[Dict[str, Any]] = []
        try:
            curated_all = await self.db_service.get_curated_issues()
            curated_issues = [
                item
                for item in (curated_all or [])
                if str(item.get("status") or "").strip().lower() in {"resolved", "open"}
            ]
        except Exception as e:
            self._debug_log(issue.get("run_id"), "curated issue fetch failed", {"error": str(e)})

        for curated in curated_issues:
            match_info = self._match_curated_issue(curated, issue, issue_file)
            if match_info.get("structured_score", 0.0) <= 0.0:
                continue

            curated_hit = {
                "id": curated.get("id"),
                "pattern_id": curated.get("pattern_id"),
                "severity": curated.get("severity"),
                "solution": curated.get("solution"),
                "project_path": curated.get("project_path"),
                "file_path": curated.get("file_path"),
                "start_line": curated.get("start_line"),
                "end_line": curated.get("end_line"),
                "problem_phenomenon": curated.get("problem_phenomenon"),
                "root_cause": curated.get("root_cause"),
                "status": curated.get("status"),
                "structured_score": match_info.get("structured_score", 0.0),
                "context_score": match_info.get("context_score", 0.0),
                "matched_fields": match_info.get("matched_fields", []),
            }
            evidence["curated_issue_hits"].append(curated_hit)
            candidate = self._build_candidate_from_curated_issue(curated_hit, issue_desc, issue_file, issue)
            self._gate_candidate(candidate)
            # self._debug_log(
            #     issue.get("run_id"),
            #     "curated candidate gated",
            #     {
            #         "curated_id": candidate.get("sqlite_id"),
            #         "structured_score": candidate.get("structured_score"),
            #         "semantic_score": candidate.get("semantic_score"),
            #         "context_score": candidate.get("context_score"),
            #         "anchor_score": candidate.get("anchor_score"),
            #         "total_score": candidate.get("total_score"),
            #         "gating_decision": candidate.get("gating_decision"),
            #     },
            # )
            evidence["candidates"].append(candidate)
            if candidate.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                hit = dict(candidate)
                hit.update({
                    "curated_issue_id": curated_hit.get("id"),
                    "pattern_id": curated_hit.get("pattern_id"),
                    "file_path": curated_hit.get("file_path"),
                    "start_line": curated_hit.get("start_line"),
                    "end_line": curated_hit.get("end_line"),
                    "problem_phenomenon": curated_hit.get("problem_phenomenon"),
                    "root_cause": curated_hit.get("root_cause"),
                })
                try:
                    line_no = int(issue.get("line")) if issue.get("line") is not None else None
                except Exception:
                    line_no = None
                code_snip = self._extract_code_snippet(issue_file, line_no)
                if code_snip:
                    hit["code_snippet"] = code_snip
                evidence["evidence_hits"].append(hit)
            elif candidate.get("gating_decision") == "low_confidence_hit":
                evidence["low_confidence_hits"].append(dict(candidate))

        # Weaviate 语义匹配
        # 根据 layer_mode 可控制仅使用 'full' 层（对应于无分层知识）
        if layer_mode == 'all_only':
            # 标记本 issue 以便后续 weaviate 搜索使用单层 'full'
            issue['_requested_layer_mode'] = 'all_only'

        if self.enable_weaviate_query and not self._weaviate_connect_attempted and not self.vector_service.is_connected():
            self._weaviate_connect_attempted = True
            connected = self.vector_service.connect(auto_create_schema=False)
            if connected:
                log("second_pass_agent", LogLevel.INFO, "✅ Weaviate 连接成功，启用语义检索")
            else:
                log("second_pass_agent", LogLevel.WARNING, "⚠️ Weaviate 连接不可用，跳过语义检索")

        if self.enable_weaviate_query and self.vector_service.is_connected() and issue_desc:
            signature = self._semantic_signature(issue)
            query_parts = [f"[{issue_source}] {issue_desc}"]
            for key in [
                "analysis_type",
                "source_category",
                "issue_type",
                "function_name",
                "location",
                "line_number",
                "recommendation",
                "severity",
                "tool",
            ]:
                value = str(issue.get(key) or "").strip()
                if value:
                    query_parts.append(f"{key}:{value}")
            if issue_file:
                basename = os.path.basename(issue_file)
                ext = os.path.splitext(basename)[1].lower().lstrip(".")
                if basename:
                    query_parts.append(f"file:{basename}")
                if ext:
                    query_parts.append(f"ext:{ext}")
            details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
            for key in [
                "operation",
                "outer_loop",
                "inner_loop",
                "recursive_call_line",
                "pattern_matched",
                "io_type",
                "estimated_complexity",
            ]:
                value = str(details.get(key) or "").strip()
                if value:
                    query_parts.append(f"{key}:{value}")
            snippet = str(issue.get("code_snippet") or "").strip()
            if snippet:
                query_parts.append(f"snippet:{snippet[:200]}")
            query_parts.append(f"sig:{signature}")
            query_text = " | ".join(query_parts)
            # 根据传入的 layer_mode，只查询特定的层
            layers_to_query = ["semantic", "code_pattern", "solution", "full"]
            # layer_mode == 'all_only' 表示仅使用默认的全量层（对应 'full'）
            if isinstance(issue.get('run_id'), str):
                pass
            # if caller requested all_only, restrict to ['full'] only
            lm = None
            try:
                lm = issue.get('_requested_layer_mode')
            except Exception:
                lm = None
            # fallback: try to read from self if set earlier
            if lm is None:
                lm = getattr(self, '_requested_layer_mode', None)
            if lm == 'all_only':
                layers_to_query = ['full']
            seen_hits: set[tuple[Optional[int], str]] = set()
            layer_candidates: List[Dict[str, Any]] = []
            for layer in layers_to_query:
                # 分层查询向量：code_pattern 层用代码片段做 code→code，其余层用语义文本
                if layer == "code_pattern" and snippet:
                    qv = self._default_embed(snippet[:2000], layer)
                else:
                    qv = self._default_embed(query_text, layer)
                results = self.vector_service.search_knowledge_items(
                    query_vector=qv,
                    limit=self.weaviate_top_k,
                    layer=layer,
                )
                for item in results:
                    item_layer = str(item.get("vector_layer") or layer).strip().lower()
                    if layer == "full" and item_layer and item_layer != "full":
                        continue
                    key = (item.get("sqlite_id"), item_layer)
                    if key in seen_hits:
                        continue
                    seen_hits.add(key)

                    distance = item.get("_additional", {}).get("distance", 2.0)
                    similarity = 1.0 - (float(distance) / 2.0)
                    weaviate_hit = {
                        "sqlite_id": item.get("sqlite_id"),
                        "vector_layer": item_layer,
                        "error_type": item.get("error_type"),
                        "severity": item.get("severity"),
                        "solution": item.get("solution"),
                        "language": item.get("language"),
                        "framework": item.get("framework"),
                        "error_description": item.get("error_description"),
                        "problematic_pattern": item.get("problematic_pattern"),
                        "file_pattern": item.get("file_pattern"),
                        "class_pattern": item.get("class_pattern"),
                        "distance": distance,
                        "similarity": similarity,
                    }
                    evidence["weaviate_hits"].append(weaviate_hit)
                    layer_candidates.append(
                        self._build_candidate_from_weaviate(weaviate_hit, issue_desc, issue_file, issue)
                    )

            for candidate in self._merge_weaviate_candidates_by_sqlite_id(layer_candidates):
                self._backfill_weaviate_candidate_solution(
                    candidate,
                    weaviate_hits=evidence["weaviate_hits"],
                    sqlite_patterns=sqlite_patterns,
                )
                self._apply_file_function_anchors(candidate, issue, issue_file)
                self._gate_candidate(candidate)
                evidence["candidates"].append(candidate)
                if candidate.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                    hit = dict(candidate)
                    hit.update({
                        "sqlite_id": candidate.get("sqlite_id"),
                        "semantic_score": candidate.get("semantic_score"),
                        "vector_layer": candidate.get("vector_layer"),
                        "matched_layers": candidate.get("matched_layers"),
                        "matched_layer_details": candidate.get("matched_layer_details"),
                        "error_description": next(
                            (
                                h.get("error_description")
                                for h in evidence["weaviate_hits"]
                                if h.get("sqlite_id") == candidate.get("sqlite_id")
                                and h.get("error_description")
                            ),
                            None,
                        ),
                        "problematic_pattern": next(
                            (
                                h.get("problematic_pattern")
                                for h in evidence["weaviate_hits"]
                                if h.get("sqlite_id") == candidate.get("sqlite_id")
                                and h.get("problematic_pattern")
                            ),
                            None,
                        ),
                    })
                    try:
                        line_no = int(issue.get("line")) if issue.get("line") is not None else None
                    except Exception:
                        line_no = None
                    code_snip = self._extract_code_snippet(issue_file, line_no)
                    if code_snip:
                        hit["code_snippet"] = code_snip
                    evidence["evidence_hits"].append(hit)
                elif candidate.get("gating_decision") == "low_confidence_hit":
                    evidence["low_confidence_hits"].append(dict(candidate))

        # 不再输出候选审计块；evidence_hits 已包含 formal/explanatory 命中，供可读性层渲染

        return evidence

    def _match_curated_issue(
        self,
        curated_issue: Dict[str, Any],
        issue: Dict[str, Any],
        issue_file: str,
    ) -> Dict[str, Any]:
        curated_file = str(curated_issue.get("file_path") or "").strip().lower()
        issue_file_l = str(issue_file or "").strip().lower()
        issue_line = None
        try:
            issue_line = int(issue.get("line") or issue.get("line_number") or 0) or None
        except Exception:
            issue_line = None

        matched_fields: List[str] = []
        structured_score = 0.0
        context_score = 0.0

        # ----------------------------------------------------------------- #
        # 主匹配键：错误代码克隆检测（问题1修复）。
        # 用"diff 前错误代码的连续 token 序列"替代"文件路径 + 行号"，
        # 行号/路径只作为元数据展示，不参与结构化打分。
        # ----------------------------------------------------------------- #
        solution_text = str(curated_issue.get("solution") or "")
        issue_code = str(issue.get("code_snippet") or "")
        error_code_hit = self._error_code_clone_matched(solution_text, issue_code)
        if error_code_hit:
            matched_fields.append("error_code_clone")
            structured_score += 0.5

        # 文件 basename 锚定：降级为辅助证据（命中错误代码后的小加分），
        # 不再作为"必须满足"的前置条件。
        issue_base = self._normalize_source_basename(issue_file_l)
        curated_base = self._normalize_source_basename(curated_file)
        if issue_base and curated_base and issue_base == curated_base:
            matched_fields.append("basename_match")
            structured_score += 0.15

        # 描述词面辅助证据
        issue_desc = str(issue.get("description") or "").lower()
        phenomenon = str(curated_issue.get("problem_phenomenon") or "").lower()
        root_cause = str(curated_issue.get("root_cause") or "").lower()
        if phenomenon and any(token in issue_desc for token in phenomenon.split()[:3] if token):
            matched_fields.append("phenomenon_in_description")
            structured_score += 0.1
        if root_cause and any(token in issue_desc for token in root_cause.split()[:3] if token):
            matched_fields.append("root_cause_in_description")
            structured_score += 0.08

        # 未命中错误代码克隆：仅靠文件名/描述词面不足以晋升，
        # 封顶 0.4（低于 0.45 门限），从而消除"行号/路径查表"式命中。
        if not error_code_hit:
            structured_score = min(structured_score, 0.4)

        matched = structured_score >= 0.45
        if matched:
            context_score = 0.15 if issue_line is not None else 0.05

        return {
            "matched": matched,
            "matched_fields": matched_fields,
            "structured_score": min(1.0, structured_score),
            "context_score": min(0.2, context_score),
            "rejection_reason": "" if matched else "curated_no_error_code_clone",
        }

    def _extract_code_snippet(self, file_path: str, line: Optional[int], context_lines: int = 2) -> Optional[Dict[str, Any]]:
        """
        从给定文件和行号提取包含上下文的代码片段。
        返回字典: {"start_line": int, "end_line": int, "snippet": str} 或 None
        """
        if not file_path or line is None:
            return None
        try:
            # 处理 Windows 路径和相对路径
            p = file_path
            if not os.path.isabs(p):
                # 相对于工程根尝试解析
                p = os.path.join(os.getcwd(), p)
            if not os.path.exists(p):
                return None
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
            idx = max(0, line - 1)
            start = max(0, idx - context_lines)
            end = min(len(all_lines), idx + context_lines + 1)
            snippet = ''.join(all_lines[start:end])
            return {"start_line": start + 1, "end_line": end, "snippet": snippet}
        except Exception:
            return None

    def _apply_corrections(
        self,
        issue: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        corrected = dict(issue)
        adjustment: Optional[Dict[str, Any]] = None

        candidates = [c for c in evidence.get("candidates", []) if c.get("gating_decision") == "formal_hit"]
        if candidates:
            best_hit = sorted(candidates, key=lambda c: c.get("total_score", 0), reverse=True)[0]
            old_severity = str(corrected.get("severity") or "low").lower()
            new_severity = str(best_hit.get("severity") or old_severity).lower()
            if new_severity and new_severity != old_severity:
                corrected["severity"] = new_severity
                corrected["source"] = f"{corrected.get('source', 'unknown')}_db_corrected"
                adjustment = {
                    "issue_description": corrected.get("description"),
                    "reason": best_hit.get("reasoning", "dual_channel_correction"),
                    "old_severity": old_severity,
                    "new_severity": new_severity,
                    "similarity": best_hit.get("semantic_score"),
                    "sqlite_id": best_hit.get("sqlite_id"),
                    "match_score": best_hit.get("total_score"),
                    "gating_decision": best_hit.get("gating_decision"),
                }
            corrected["second_pass_evidence"] = best_hit

        return corrected, adjustment

    def _derive_new_findings(
        self,
        issue: Dict[str, Any],
        evidence: Dict[str, Any],
        run_id: Optional[str],
        requirement_id: Optional[int],
        file_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        issue_desc = str(issue.get("description") or "")

        for hit in evidence.get("candidates", []):
            if hit.get("gating_decision") not in ("formal_hit", "explanatory_hit"):
                continue
            # 双保险：无文件/函数锚定的 weaviate 命中不进入最终报告
            if str(hit.get("channel") or "").lower() == "weaviate" and not self._has_promotion_anchor(hit):
                continue

            candidate_desc = str(hit.get("error_type") or "潜在已知问题模式命中")
            if candidate_desc and candidate_desc.lower() in issue_desc.lower():
                continue

            error_description = str(hit.get("error_description") or "").strip()
            class_pattern = str(hit.get("class_pattern") or "").strip()
            title_hint = ""
            if error_description:
                # Prefer CVE id if present at start of related fields
                cve_match = re.search(r"\bCVE-\d{4}-\d+\b", error_description, flags=re.IGNORECASE)
                if cve_match:
                    title_hint = cve_match.group(0)
                elif class_pattern:
                    title_hint = class_pattern
            display_name = title_hint or candidate_desc
            description = f"历史知识命中: {display_name}"
            if error_description:
                short_desc = error_description if len(error_description) <= 180 else error_description[:177] + "..."
                description = f"{description} — {short_desc}"

            severity = self._demote_unanchored_severity(hit.get("severity"), hit)
            channel = str(hit.get("channel") or "").strip().lower()
            primary_channel = (
                "curated"
                if channel == "curated_issue"
                else ("weaviate" if channel == "weaviate" else (channel or "unknown"))
            )

            # curated 命中优先用知识库行号（若落在当前 gap 分片内）
            finding_line = issue.get("line")
            try:
                curated_line = int(hit.get("start_line") or 0) or None
            except Exception:
                curated_line = None
            try:
                chunk_start_i = int(issue.get("chunk_start_line") or 0) or None
                chunk_end_i = int(issue.get("chunk_end_line") or 0) or None
            except Exception:
                chunk_start_i = None
                chunk_end_i = None
            if (
                curated_line is not None
                and chunk_start_i is not None
                and chunk_end_i is not None
                and chunk_start_i <= curated_line <= chunk_end_i
            ):
                finding_line = curated_line

            findings.append(
                {
                    "requirement_id": requirement_id,
                    "file": file_path,
                    "source": "db_supplemented",
                    "severity": severity,
                    "line": finding_line,
                    "description": description,
                    "tool": "second_pass_analysis",
                    "run_id": run_id,
                    "evidence": {
                        "channel": hit.get("channel"),
                        "primary_channel": primary_channel,
                        "sqlite_id": hit.get("sqlite_id"),
                        "semantic_score": hit.get("semantic_score"),
                        "structured_score": hit.get("structured_score"),
                        "context_score": hit.get("context_score"),
                        "anchor_score": hit.get("anchor_score"),
                        "anchor_bonus": hit.get("anchor_bonus"),
                        "layer_bonus": hit.get("layer_bonus"),
                        "penalty_score": hit.get("penalty_score"),
                        "total_score": hit.get("total_score"),
                        "confidence_components": hit.get("confidence_components"),
                        "confidence_formula": hit.get("confidence_formula"),
                        "vector_layer": hit.get("vector_layer"),
                        "matched_layers": hit.get("matched_layers"),
                        "matched_layer_details": hit.get("matched_layer_details"),
                        "matched_fields": hit.get("matched_fields"),
                        "error_description": error_description,
                        "class_pattern": class_pattern,
                        "file_pattern": hit.get("file_pattern"),
                        "start_line": hit.get("start_line"),
                        "recommended_solution": hit.get("solution"),
                        "solution": hit.get("solution"),
                        "reasoning": hit.get("reasoning"),
                        "rejection_reason": hit.get("rejection_reason"),
                    },
                }
            )

        return findings

    def _evaluate_pattern_match(
        self,
        pattern: Dict[str, Any],
        description: str,
        source: str,
        file_path: str,
        issue: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pattern_error_type = str(pattern.get("error_type") or "").strip().lower()
        pattern_desc = str(pattern.get("error_description") or "").strip().lower()
        pattern_file = str(pattern.get("file_pattern") or "").strip().lower()
        pattern_language = str(pattern.get("language") or "").strip().lower()
        pattern_problematic = str(pattern.get("problematic_pattern") or "").strip().lower()
        description_l = description.lower()
        source_l = source.lower()
        run_id = issue.get("run_id") if isinstance(issue, dict) else None
        issue_func = str((issue or {}).get("function_name") or "").strip().lower()
        issue_snippet = str((issue or {}).get("code_snippet") or "").strip().lower()
        issue_location = str((issue or {}).get("location") or "").strip().lower()

        matched_fields: List[str] = []
        structured_score = 0.0
        context_score = 0.0

        if pattern_error_type and pattern_error_type in description_l:
            matched_fields.append("error_type_in_description")
            structured_score += 0.4
        if pattern_error_type and source_l and pattern_error_type in source_l:
            matched_fields.append("error_type_in_source")
            structured_score += 0.2
        if pattern_desc and pattern_desc[:32] and pattern_desc[:32] in description_l:
            matched_fields.append("error_description_prefix")
            structured_score += 0.3
        if pattern_problematic and pattern_problematic[:24] and pattern_problematic[:24] in description_l:
            matched_fields.append("problematic_pattern_prefix")
            structured_score += 0.2
        if pattern_file and pattern_file in file_path.lower():
            matched_fields.append("file_pattern")
            structured_score += 0.2
        elif pattern_file:
            pattern_base = os.path.basename(pattern_file.replace("\\", "/"))
            issue_base = os.path.basename(file_path.replace("\\", "/")).lower()
            if pattern_base and (
                pattern_base == issue_base
                or pattern_base in file_path.lower()
                or issue_base in pattern_file
            ):
                matched_fields.append("file_pattern")
                structured_score += 0.2
        pattern_class = str(pattern.get("class_pattern") or "").strip().lower()
        code_haystack = f"{description_l}\n{issue_snippet}"
        if pattern_class and pattern_class in code_haystack:
            matched_fields.append("class_pattern_in_code")
            structured_score += 0.25
        if issue_func and issue_func in description_l:
            matched_fields.append("function_in_description")
            structured_score += 0.1
        if issue_location and issue_location in description_l:
            matched_fields.append("location_in_description")
            structured_score += 0.05
        if issue_snippet and pattern_problematic and pattern_problematic[:16] in issue_snippet:
            matched_fields.append("pattern_in_snippet")
            structured_score += 0.15
        if pattern_language and self._language_matches_file(pattern_language, file_path):
            matched_fields.append("language_match")
            context_score += 0.1

        # 描述/片段中出现文件 basename 或 CVE 描述中的路径片段
        if file_path:
            issue_base = os.path.basename(file_path.replace("\\", "/")).lower()
            if issue_base and len(issue_base) > 2 and issue_base in pattern_desc:
                if "file_basename_in_description" not in matched_fields:
                    matched_fields.append("file_basename_in_description")
                    structured_score += 0.15

        matched = structured_score >= 0.3
        # if structured_score > 0.0 and not matched:
        #     self._debug_log(
        #         run_id,
        #         "sqlite near-miss",
        #         {
        #             "pattern_id": pattern.get("id"),
        #             "error_type": pattern.get("error_type"),
        #             "file_pattern": pattern.get("file_pattern"),
        #             "language": pattern.get("language"),
        #             "structured_score": min(1.0, structured_score),
        #             "context_score": min(0.2, context_score),
        #             "matched_fields": matched_fields,
        #             "issue_file": file_path,
        #             "issue_location": (issue or {}).get("location"),
        #             "issue_function": (issue or {}).get("function_name"),
        #         },
        #     )
        return {
            "matched": matched,
            "matched_fields": matched_fields,
            "structured_score": min(1.0, structured_score),
            "context_score": min(0.2, context_score),
        }

    def _language_matches_file(self, language: str, file_path: str) -> bool:
        if not file_path:
            return False
        ext = os.path.splitext(file_path)[1].lower()
        language = language.lower()
        mapping = {
            "c": {".c", ".h"},
            "cpp": {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"},
            "c++": {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"},
            "python": {".py"},
            "java": {".java"},
        }
        return ext in mapping.get(language, set())

    def _build_candidate_from_sqlite(self, hit: Dict[str, Any], issue_desc: str, issue: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "channel": "sqlite",
            "sqlite_id": hit.get("id"),
            "run_id": (issue or {}).get("run_id"),
            "error_type": hit.get("error_type"),
            "severity": hit.get("severity"),
            "solution": hit.get("solution"),
            "structured_score": float(hit.get("structured_score", 0.0)),
            "semantic_score": 0.0,
            "context_score": float(hit.get("context_score", 0.0)),
            "anchor_score": self._calc_anchor_score(issue),
            "penalty_score": 0.0,
            "total_score": 0.0,
            "matched_fields": hit.get("matched_fields", []),
            "issue_summary": issue_desc[:160],
            "_current_code": (issue or {}).get("code_snippet") or "",
            "reasoning": "sqlite_structured_match",
            "rejection_reason": "",
            "gating_decision": "",
        }

    def _build_candidate_from_weaviate(
        self,
        hit: Dict[str, Any],
        issue_desc: str,
        file_path: str,
        issue: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context_score = 0.0
        language = str(hit.get("language") or "").strip().lower()
        if language and self._language_matches_file(language, file_path):
            context_score += 0.1
        vector_layer = str(hit.get("vector_layer") or "").strip().lower()
        reasoning = "weaviate_semantic_match"
        if vector_layer:
            reasoning = f"weaviate_{vector_layer}_match"
        return {
            "channel": "weaviate",
            "sqlite_id": hit.get("sqlite_id"),
            "run_id": (issue or {}).get("run_id"),
            "vector_layer": vector_layer,
            "matched_layers": [vector_layer] if vector_layer else [],
            "error_type": hit.get("error_type"),
            "severity": hit.get("severity"),
            "solution": hit.get("solution"),
            "error_description": hit.get("error_description"),
            "problematic_pattern": hit.get("problematic_pattern"),
            "file_pattern": hit.get("file_pattern"),
            "class_pattern": hit.get("class_pattern"),
            "structured_score": 0.0,
            "semantic_score": float(hit.get("similarity", 0.0)),
            "context_score": context_score,
            "anchor_score": self._calc_anchor_score(issue),
            "penalty_score": 0.0,
            "total_score": 0.0,
            "matched_fields": [],
            "issue_summary": issue_desc[:160],
            "_current_code": (issue or {}).get("code_snippet") or "",
            "reasoning": reasoning,
            "rejection_reason": "",
            "gating_decision": "",
        }

    def _build_candidate_from_curated_issue(
        self,
        hit: Dict[str, Any],
        issue_desc: str,
        file_path: str,
        issue: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context_score = float(hit.get("context_score", 0.0))
        matched_fields = hit.get("matched_fields", []) or []
        # error_code_clone 命中归到 code_pattern 视图名下（代码精确匹配是 code_pattern 视图的实现）
        view_layer = "code_pattern" if "error_code_clone" in matched_fields else None
        return {
            "channel": "curated_issue",
            "sqlite_id": hit.get("id"),
            "run_id": (issue or {}).get("run_id"),
            "error_type": str(hit.get("problem_phenomenon") or hit.get("root_cause") or "curated_issue")[:80],
            "severity": hit.get("severity"),
            "solution": hit.get("solution"),
            "structured_score": float(hit.get("structured_score", 0.0)),
            "semantic_score": 0.0,
            "context_score": context_score,
            "anchor_score": self._calc_anchor_score(issue),
            "penalty_score": 0.0,
            "total_score": 0.0,
            "matched_fields": matched_fields,
            "vector_layer": view_layer,
            "matched_layers": [view_layer] if view_layer else [],
            "issue_summary": issue_desc[:160],
            "_current_code": (issue or {}).get("code_snippet") or "",
            "reasoning": "curated_issue_structural_match",
            "rejection_reason": "",
            "gating_decision": "",
            "start_line": hit.get("start_line"),
            "end_line": hit.get("end_line"),
            "file_pattern": hit.get("file_path"),
            "error_description": hit.get("problem_phenomenon") or hit.get("root_cause"),
            "_analysis_file": file_path,
        }

    def _calc_anchor_score(self, issue: Optional[Dict[str, Any]]) -> float:
        if not isinstance(issue, dict):
            return 0.0
        score = 0.0
        if str(issue.get("function_name") or "").strip():
            score += 0.2
        if str(issue.get("location") or "").strip():
            score += 0.15
        if str(issue.get("line_number") or issue.get("line") or "").strip():
            score += 0.15
        if str(issue.get("code_snippet") or "").strip():
            score += 0.25
        if str(issue.get("recommendation") or "").strip():
            score += 0.1
        details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
        if details:
            if str(details.get("pattern_matched") or "").strip():
                score += 0.1
            if str(details.get("operation") or "").strip():
                score += 0.05
        return min(1.0, score)

    def _configured_layer_bonus(self, vector_layer: str) -> float:
        layer = str(vector_layer or "").strip().lower()
        if not layer:
            return 0.0
        return float(getattr(self, "layer_bonus_map", {}).get(layer, 0.0))

    def _build_matched_layer_details(self, layers: List[Any]) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in layers or []:
            layer = str(raw or "").strip().lower()
            if not layer or layer in seen:
                continue
            seen.add(layer)
            details.append(
                {
                    "layer": layer,
                    "bonus": round(self._configured_layer_bonus(layer), 4),
                }
            )
        return details

    def _apply_file_function_anchors(
        self,
        candidate: Dict[str, Any],
        issue: Optional[Dict[str, Any]],
        issue_file: str,
    ) -> None:
        """用当前文件 basename / 函数名锚定抬高 structured，压过无关 CVE 噪声。"""
        if not isinstance(candidate, dict):
            return
        matched_fields = list(candidate.get("matched_fields") or [])
        structured = float(candidate.get("structured_score") or 0.0)
        issue_desc = str((issue or {}).get("description") or "").lower()
        issue_snippet = str((issue or {}).get("code_snippet") or "").lower()
        haystack = f"{issue_desc}\n{issue_snippet}"

        error_desc = str(candidate.get("error_description") or "").lower()
        file_pattern = str(candidate.get("file_pattern") or "").strip().lower()
        class_pattern = str(candidate.get("class_pattern") or "").strip().lower()

        issue_base = self._normalize_source_basename(issue_file)
        knowledge_base = self._normalize_source_basename(file_pattern) if file_pattern else ""
        if issue_base and len(issue_base) > 2:
            same_file = bool(knowledge_base and issue_base == knowledge_base)
            if (
                same_file
                or issue_base in error_desc
                or (file_pattern and (issue_base in file_pattern or knowledge_base in str(issue_file or "").lower()))
            ):
                if "file_basename_anchor" not in matched_fields:
                    matched_fields.append("file_basename_anchor")
                    structured += 0.2

        if class_pattern and class_pattern in haystack:
            if "class_pattern_in_code" not in matched_fields:
                matched_fields.append("class_pattern_in_code")
                structured += 0.25
        elif class_pattern and class_pattern in error_desc and issue_base and issue_base in haystack:
            # gap chunk 含文件上下文且描述指向同函数
            if "class_pattern_desc_anchor" not in matched_fields:
                matched_fields.append("class_pattern_desc_anchor")
                structured += 0.15

        # 从 error_description 抽函数名，若出现在代码片中也加分
        if error_desc and "function" in error_desc:
            m = re.search(r"\b([a-z_][a-z0-9_]{3,})\s+function\b", error_desc)
            if m:
                fname = m.group(1)
                if fname in haystack and "function_name_in_code" not in matched_fields:
                    matched_fields.append("function_name_in_code")
                    structured += 0.25
                    if not class_pattern:
                        candidate["class_pattern"] = fname

        candidate["matched_fields"] = matched_fields
        candidate["structured_score"] = min(1.0, structured)
        candidate["_analysis_file"] = issue_file

    def _resolve_gap_finding_line(
        self,
        code_chunk: Dict[str, Any],
        finding: Dict[str, Any],
        hit_hint: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """优先把补漏行号定位到 chunk 内函数名/关键词，避免一律落到 chunk 首行。"""
        text = str(code_chunk.get("text") or "")
        if not text:
            return finding.get("line")
        try:
            start = int(code_chunk.get("start_line") or 1)
        except Exception:
            start = 1

        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        try:
            end = int(code_chunk.get("end_line") or start)
        except Exception:
            end = start

        # curated 知识行号落在分片内时优先采用
        for source in (evidence, hit_hint or {}, finding):
            if not isinstance(source, dict):
                continue
            try:
                curated_line = int(source.get("start_line") or 0) or None
            except Exception:
                curated_line = None
            if curated_line is not None and start <= curated_line <= end:
                return curated_line

        needles: List[str] = []
        for source in (evidence, hit_hint or {}, finding):
            if not isinstance(source, dict):
                continue
            for key in ("class_pattern",):
                value = str(source.get(key) or "").strip()
                if value:
                    needles.append(value)
            desc = str(source.get("error_description") or "").strip()
            if desc:
                m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s+function\b", desc, flags=re.IGNORECASE)
                if m:
                    needles.append(m.group(1))
                # Also try common path/file tokens from description
                path_match = re.search(r"([A-Za-z0-9_./-]+\.(?:c|h|cpp|cc|py|java))", desc)
                if path_match:
                    needles.append(os.path.basename(path_match.group(1)))
                    needles.append(self._normalize_source_basename(path_match.group(1)))

        seen = set()
        for needle in needles:
            key = needle.lower()
            if not needle or key in seen:
                continue
            seen.add(key)
            idx = text.find(needle)
            if idx < 0:
                idx = text.lower().find(key)
            if idx >= 0:
                return start + text[:idx].count("\n")

        # 若当前行已是 chunk 中部则保留；否则用 start_line（仍优于硬编码 1）
        current = finding.get("line")
        try:
            if current is not None and int(current) >= start:
                return int(current)
        except Exception:
            pass
        return start

    def _pick_nonempty_solution(self, items: List[Dict[str, Any]]) -> str:
        """从同 sqlite_id 的多层命中中挑选 solution，优先 solution/full 层。"""
        layer_priority = {
            "solution": 0,
            "full": 1,
            "code_pattern": 2,
            "semantic": 3,
        }

        def _rank(item: Dict[str, Any]) -> Tuple[int, int]:
            layer = str(item.get("vector_layer") or "").strip().lower()
            has_solution = 0 if str(item.get("solution") or "").strip() else 1
            return (has_solution, layer_priority.get(layer, 99))

        for item in sorted((i for i in items if isinstance(i, dict)), key=_rank):
            solution = str(item.get("solution") or "").strip()
            if solution:
                return solution
        return ""

    def _backfill_weaviate_candidate_solution(
        self,
        candidate: Dict[str, Any],
        weaviate_hits: Optional[List[Dict[str, Any]]] = None,
        sqlite_patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """回填 solution / error_description / class_pattern / file_pattern（同 sqlite_id）。"""
        if not isinstance(candidate, dict):
            return

        sid = candidate.get("sqlite_id")
        if sid is None:
            return

        sibling_hits = [
            hit
            for hit in (weaviate_hits or [])
            if isinstance(hit, dict) and hit.get("sqlite_id") == sid
        ]

        if not str(candidate.get("solution") or "").strip():
            solution = self._pick_nonempty_solution(sibling_hits + [candidate])
            if solution:
                candidate["solution"] = solution

        def _fill_field(field: str, sources: List[Dict[str, Any]]) -> None:
            if str(candidate.get(field) or "").strip():
                return
            for item in sources:
                value = str(item.get(field) or "").strip()
                if value:
                    candidate[field] = value
                    return

        meta_sources: List[Dict[str, Any]] = list(sibling_hits)
        for pattern in sqlite_patterns or []:
            if isinstance(pattern, dict) and pattern.get("id") == sid:
                meta_sources.append(
                    {
                        "solution": pattern.get("solution"),
                        "error_description": pattern.get("error_description"),
                        "class_pattern": pattern.get("class_pattern"),
                        "file_pattern": pattern.get("file_pattern"),
                    }
                )
                break

        if self.enable_weaviate_query and self.vector_service.is_connected():
            if not str(candidate.get("solution") or "").strip() or not str(
                candidate.get("error_description") or ""
            ).strip():
                try:
                    items = self.vector_service.get_knowledge_items(sqlite_id=int(sid), limit=8)
                except Exception:
                    items = []
                meta_sources.extend(items or [])

        for field in ("solution", "error_description", "class_pattern", "file_pattern"):
            _fill_field(field, meta_sources)

    def _merge_weaviate_candidates_by_sqlite_id(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """按 sqlite_id 融合多层命中：取最大语义分，身份优先稀疏层。"""
        if not candidates:
            return []

        groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        orphan_idx = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            sid = candidate.get("sqlite_id")
            if sid is None:
                groups[f"__orphan_{orphan_idx}"] = [candidate]
                orphan_idx += 1
            else:
                groups[sid].append(candidate)

        merged: List[Dict[str, Any]] = []
        for group in groups.values():
            if not group:
                continue
            if len(group) == 1:
                item = dict(group[0])
                layer = str(item.get("vector_layer") or "").strip().lower()
                layers = list(item.get("matched_layers") or [])
                if layer and layer not in layers:
                    layers = [layer] + [x for x in layers if x != layer]
                item["matched_layers"] = layers or ([layer] if layer else [])
                item["matched_layer_details"] = self._build_matched_layer_details(item["matched_layers"])
                solution = self._pick_nonempty_solution(group)
                if solution:
                    item["solution"] = solution
                merged.append(item)
                continue

            layers: List[str] = []
            for item in group:
                layer = str(item.get("vector_layer") or "").strip().lower()
                if layer and layer not in layers:
                    layers.append(layer)

            best_semantic = max(float(item.get("semantic_score") or 0.0) for item in group)
            best_context = max(float(item.get("context_score") or 0.0) for item in group)
            gated = [
                item
                for item in group
                if float(item.get("semantic_score") or 0.0) >= self.similarity_threshold
            ]
            pick_from = gated if gated else group

            def _identity_key(item: Dict[str, Any]) -> Tuple[float, float]:
                layer = str(item.get("vector_layer") or "").strip().lower()
                return (
                    self._configured_layer_bonus(layer),
                    float(item.get("semantic_score") or 0.0),
                )

            identity = max(pick_from, key=_identity_key)
            merged_item = dict(identity)
            merged_item["semantic_score"] = best_semantic
            merged_item["context_score"] = best_context
            merged_item["matched_layers"] = layers
            merged_item["matched_layer_details"] = self._build_matched_layer_details(layers)
            identity_layer = str(merged_item.get("vector_layer") or "").strip().lower()
            if identity_layer:
                merged_item["reasoning"] = f"weaviate_{identity_layer}_match"
            solution = self._pick_nonempty_solution(group)
            if solution:
                merged_item["solution"] = solution
            for field in ("error_description", "class_pattern", "file_pattern"):
                if str(merged_item.get(field) or "").strip():
                    continue
                for item in group:
                    value = str(item.get(field) or "").strip()
                    if value:
                        merged_item[field] = value
                        break
            merged.append(merged_item)

        return merged

    @staticmethod
    def _normalize_source_basename(path: str) -> str:
        """将普通路径或 BigVul 扁平文件名归一为可比较的 basename。

        例:
          src/kadmin/server/schpw.c -> schpw.c
          src__kadmin__server__schpw.c -> schpw.c
          arch__powerpc__kernel__traps.c -> traps.c
        """
        if not path:
            return ""
        name = os.path.basename(str(path).replace("\\", "/")).strip().lower()
        if not name:
            return ""
        if "__" in name:
            name = name.split("__")[-1]
        return name

    # 统一结构化分 s(x) 的证据字段（与通道无关，合并三套分通道累加）。
    _UNIFIED_STRUCT_FIELDS = {
        # 词法词元连续子串命中（主匹配键）
        "error_code_clone": 0.5,
        # 同文件名（basename_match 为 curated 通道等价字段）
        "file_basename_anchor": 0.2,
        "basename_match": 0.2,
        # 类名命中
        "class_pattern_in_code": 0.25,
        # 函数名命中
        "function_name_in_code": 0.25,
        "function_in_description": 0.25,
    }
    # 弱证据字段：统一计入 +0.1 一项（封顶，见 _unified_structured_score）
    _UNIFIED_WEAK_FIELDS = {
        "phenomenon_in_description",
        "root_cause_in_description",
        "error_type_in_description",
        "error_type_in_source",
        "error_description_prefix",
        "problematic_pattern_prefix",
        "location_in_description",
        "pattern_in_snippet",
        "file_basename_in_description",
        "file_pattern",
    }

    @classmethod
    def _unified_structured_score(cls, matched_fields) -> float:
        """统一证据计数公式 s(x)（与通道无关）。

        s(x) = 0.5·[词法词元连续子串命中]
             + 0.2·[同文件名]
             + 0.25·[类名命中]
             + 0.25·[函数名命中]
             + 0.1·[描述词面/现象等弱证据，封顶]

        取代原 curated/sqlite/weaviate 三套分通道累加，使门控判别式可用
        同一把尺子描述。返回值 [0, 1]。
        """
        mf = set(matched_fields or [])
        s = 0.0
        for field, weight in cls._UNIFIED_STRUCT_FIELDS.items():
            if field in mf:
                s += weight
        if mf & cls._UNIFIED_WEAK_FIELDS:
            s += 0.1
        return min(1.0, s)

    # ------------------------------------------------------------------ #
    # 错误代码克隆检测（问题1修复）
    # ------------------------------------------------------------------ #
    _CODE_TOKEN_RE = re.compile(
        r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|->|==|!=|<=|>=|\+\+|--|[+\-*/%=<>!&|^~]"
    )
    # C 语言通用 token：整段只含这些的片段视为无判别力，予以丢弃
    _CODE_GENERIC_TOKENS = {
        "if", "else", "for", "while", "return", "int", "char", "void", "size", "len",
        "sizeof", "null", "true", "false", "0", "1", "break", "continue", "goto",
        "case", "switch", "do", "struct", "unsigned", "long", "short", "static", "const",
        "err", "ret", "i", "j", "k", "buf", "data", "ptr", "tmp", "res", "len_", "count",
    }

    def _tokenize_code(self, text: str) -> List[str]:
        return self._CODE_TOKEN_RE.findall(text or "")

    def _is_contiguous_subseq(self, needle: List[str], haystack: List[str]) -> bool:
        """判断 needle 是否作为【连续、原样、有序】的子串出现在 haystack 中。"""
        n = len(needle)
        if n == 0:
            return False
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                return True
        return False

    def _extract_error_code_fragments(self, solution: str) -> List[List[str]]:
        """从 solution 抽出 'Remove incorrect logic' 错误代码，token 化为连续序列。

        只保留：token 数 >= error_code_clone_min_tokens 且含非通用 token 的片段。
        返回完整 token 序列（不过滤通用 token），供连续子串匹配使用。
        """
        if not solution:
            return []
        m = re.search(
            r"Remove incorrect logic:\s*(.+?)(?:\.\s*Ensure corrected path:|$)",
            solution,
            re.DOTALL,
        )
        if not m:
            return []
        frags, seen = [], set()
        for part in re.split(r";;", m.group(1)):
            p = part.strip()
            if p and p not in seen:
                seen.add(p)
                frags.append(p)
        out = []
        for f in frags:
            toks = self._tokenize_code(f)
            if len(toks) < self.error_code_clone_min_tokens:
                continue
            if all(t.lower() in self._CODE_GENERIC_TOKENS for t in toks):
                continue
            out.append(toks)
        return out

    def _error_code_clone_matched(self, solution: str, current_code: str) -> bool:
        """错误代码克隆检测：solution 里的错误代码连续 token 序列是否出现在当前代码。"""
        if not solution or not current_code:
            return False
        frags = self._extract_error_code_fragments(solution)
        if not frags:
            return False
        current_toks = self._tokenize_code(current_code)
        return any(self._is_contiguous_subseq(f, current_toks) for f in frags)

    def _knowledge_file_basename(self, candidate: Dict[str, Any]) -> str:
        """从 file_pattern 或 error_description 解析知识条目关联文件 basename。"""
        file_pattern = str(candidate.get("file_pattern") or "").strip().replace("\\", "/")
        if file_pattern:
            base = self._normalize_source_basename(file_pattern)
            if base:
                return base
        desc = str(candidate.get("error_description") or "")
        match = re.search(
            r"([A-Za-z0-9_./\\-]+\.(?:c|h|cpp|cc|S|py|java))",
            desc,
        )
        if match:
            return self._normalize_source_basename(match.group(1))
        return ""

    def _has_promotion_anchor(self, hit: Dict[str, Any]) -> bool:
        structured = float(hit.get("structured_score") or 0.0)
        if structured >= 0.2:
            return True
        fields = set(hit.get("matched_fields") or [])
        return bool(
            fields
            & {
                "class_pattern_in_code",
                "function_name_in_code",
                "file_basename_anchor",
                "file_pattern",
                "class_pattern_desc_anchor",
                "basename_match",
                "basename_in_curated_path",
                "curated_basename_in_issue_path",
            }
        )

    def _demote_unanchored_severity(self, severity: Any, hit: Dict[str, Any]) -> str:
        sev = str(severity or "medium").strip().lower() or "medium"
        if self._has_promotion_anchor(hit):
            return sev
        if sev in {"high", "critical", "medium"}:
            return "info"
        return sev if sev else "info"

    def _candidate_code_fixed(self, candidate: Dict[str, Any]) -> bool:
        """门控层统一检测：当前代码是否已应用修复（错误代码 token 连续子串已消失）。

        与召回判定 _error_code_clone_matched 用同一把尺子（token 化 + 连续子串），
        避免裸字符串包含因空白/换行差异把"未修复"误判成"已修复"。
        """
        solution = str(candidate.get("solution") or "")
        current_code = str(candidate.get("_current_code") or "")
        if not solution or not current_code:
            return False
        frags = self._extract_error_code_fragments(solution)
        if not frags:
            # 无法提取可判定的错误代码片段 → 不能断言"已修复"，保留
            return False
        current_toks = self._tokenize_code(current_code)
        return not any(self._is_contiguous_subseq(f, current_toks) for f in frags)

    def _gate_candidate(self, candidate: Dict[str, Any]) -> None:
        generic_terms = {"threading", "insert", "update", "delete"}
        error_type = str(candidate.get("error_type") or "").strip().lower()
        channel = str(candidate.get("channel") or "").strip().lower()
        run_id = candidate.get("run_id")
        vector_layer = str(candidate.get("vector_layer") or "").strip().lower()
        structured = float(candidate.get("structured_score", 0.0))
        semantic = float(candidate.get("semantic_score", 0.0))
        context = float(candidate.get("context_score", 0.0))
        anchor = float(candidate.get("anchor_score", 0.0))
        matched_fields = candidate.get("matched_fields") or []
        # 统一结构化分 s(x)：门控判定统一用同一把尺子（与通道无关）。
        # 原 structured_score 仍保留用于报告展示与排序，但不再参与晋升判定。
        unified_s = self._unified_structured_score(matched_fields)

        anchor_bonus = 0.0
        if "pattern_in_snippet" in matched_fields:
            anchor_bonus += 0.12
        if "file_pattern" in matched_fields:
            anchor_bonus += 0.08
        if "function_in_description" in matched_fields:
            anchor_bonus += 0.05
        if "location_in_description" in matched_fields:
            anchor_bonus += 0.03
        anchor_bonus = min(0.2, anchor_bonus)

        layer_bonus = 0.0
        if vector_layer:
            layer_bonus = float(getattr(self, "layer_bonus_map", {}).get(vector_layer, 0.0))
        require_sim_gate = bool(getattr(self, "layer_bonus_require_similarity_gate", True))
        if require_sim_gate and semantic < self.similarity_threshold:
            layer_bonus = 0.0

        penalty = 0.0
        if error_type in generic_terms and structured < 0.4:
            penalty += 0.2
        if semantic < self.similarity_threshold and structured < 0.4:
            penalty += 0.1
        if anchor < 0.2 and structured < 0.4:
            penalty += 0.05

        # 置信度各分项权重（可调整）
        if channel == "curated_issue":
            weights = {
                "structured": 0.55,
                "semantic": 0.0,
                "context": 0.15,
                "anchor": 0.20,
            }
        else:
            weights = {
                "structured": 0.50,
                "semantic": 0.35,
                "context": 0.10,
                "anchor": 0.05,
            }

        # 各分项原始贡献
        comp_structured = structured * weights.get("structured", 0.0)
        comp_semantic = semantic * weights.get("semantic", 0.0)
        comp_context = context * weights.get("context", 0.0)
        comp_anchor = anchor * weights.get("anchor", 0.0)

        total = comp_structured + comp_semantic + comp_context + comp_anchor + anchor_bonus + layer_bonus - penalty

        candidate["penalty_score"] = round(penalty, 4)
        candidate["total_score"] = round(max(0.0, total), 4)
        candidate["anchor_bonus"] = round(anchor_bonus, 4)
        candidate["layer_bonus"] = round(layer_bonus, 4)
        if not candidate.get("matched_layers") and vector_layer:
            candidate["matched_layers"] = [vector_layer]
        if not candidate.get("matched_layer_details"):
            candidate["matched_layer_details"] = self._build_matched_layer_details(
                candidate.get("matched_layers") or ([vector_layer] if vector_layer else [])
            )

        # 记录置信度分项明细与使用的公式，便于报告展示
        candidate["confidence_components"] = {
            "structured": round(structured, 4),
            "semantic": round(semantic, 4),
            "context": round(context, 4),
            "anchor": round(anchor, 4),
            "anchor_bonus": round(anchor_bonus, 4),
            "layer_bonus": round(layer_bonus, 4),
            "penalty": round(penalty, 4),
            "component_contributions": {
                "structured_contribution": round(comp_structured, 4),
                "semantic_contribution": round(comp_semantic, 4),
                "context_contribution": round(comp_context, 4),
                "anchor_contribution": round(comp_anchor, 4),
            },
        }
        if channel == "curated_issue":
            candidate["confidence_formula"] = "total = structured*0.55 + context*0.15 + anchor*0.20 + anchor_bonus + layer_bonus - penalty"
        else:
            candidate["confidence_formula"] = "total = structured*0.5 + semantic*0.35 + context*0.1 + anchor*0.05 + anchor_bonus + layer_bonus - penalty"

        # code anchor 只保留"真实命中当前代码/文件"的强锚点。
        # class_pattern_desc_anchor（类名只出现在 KB 描述里、未出现在当前代码）是弱语义锚，
        # 不再是 code anchor——否则纯语义误报会绕过跨文件/弱结构两道拦截。
        code_anchor_fields = {
            "class_pattern_in_code",
            "function_name_in_code",
            "file_basename_anchor",
            "file_pattern",
        }
        has_code_anchor = bool(set(matched_fields) & code_anchor_fields)

        # Weaviate：异文件知识不得晋升（除非当前代码片已命中函数名）
        if channel == "weaviate":
            analysis_base = self._normalize_source_basename(
                str(candidate.get("_analysis_file") or "")
            )
            knowledge_base = self._knowledge_file_basename(candidate)
            if (
                knowledge_base
                and analysis_base
                and knowledge_base != analysis_base
                and not has_code_anchor
            ):
                candidate["gating_decision"] = "discarded_hit"
                candidate["rejection_reason"] = "cross_file_mismatch"
                return
            # 无结构化/文件锚定的纯语义命中：仅保留为低置信，不进入最终 findings
            if unified_s < self.gate_weak_structure_threshold and not has_code_anchor:
                if semantic >= self.similarity_threshold:
                    candidate["gating_decision"] = "low_confidence_hit"
                    candidate["rejection_reason"] = "weak_structure_no_file_anchor"
                else:
                    candidate["gating_decision"] = "discarded_hit"
                    candidate["rejection_reason"] = "low_confidence_or_generic"
                return

        # 方向B：门控层统一"代码已修复"检测（curated / weaviate / sqlite 通道共用）
        if self._candidate_code_fixed(candidate):
            candidate["gating_decision"] = "discarded_hit"
            candidate["rejection_reason"] = "code_already_fixed"
            return

        # 两通道析取门控（统一判别式）：
        #   情况① 词法-结构通道：s(x) ≥ θ_s
        #   情况② 语义通道：v(x) ≥ τ 且 a(x) ≥ θ_a 且 s(x) ≥ θ_w
        # 情况①命中判 formal（强证据），情况②命中判 explanatory（语义确认）。
        if unified_s >= self.gate_structured_threshold:
            candidate["gating_decision"] = "formal_hit"
        elif (
            semantic >= self.similarity_threshold
            and anchor >= self.gate_anchor_threshold
            and unified_s >= self.gate_weak_structure_threshold
        ):
            candidate["gating_decision"] = "explanatory_hit"
        elif semantic >= self.similarity_threshold and unified_s < self.gate_weak_structure_threshold:
            candidate["gating_decision"] = "low_confidence_hit"
            candidate["rejection_reason"] = "weak_structure_high_semantic"
        else:
            candidate["gating_decision"] = "discarded_hit"
            candidate["rejection_reason"] = "low_confidence_or_generic"

        # 记录统一判别式信息，便于报告与复现
        candidate["unified_structured_score"] = round(unified_s, 4)
        candidate["gate_formula"] = (
            "admit = F(x) & ( s(x)>=theta_s | ( v(x)>=tau & a(x)>=theta_a & s(x)>=theta_w ) )"
        )
        candidate["gate_params"] = {
            "theta_s": self.gate_structured_threshold,
            "tau": self.similarity_threshold,
            "theta_a": self.gate_anchor_threshold,
            "theta_w": self.gate_weak_structure_threshold,
        }

        # only emit minimal gating debug info; full numeric scoring is persisted in report JSON
        # if candidate.get("gating_decision") in {"discarded_hit", "low_confidence_hit"}:
        #     self._debug_log(
        #         run_id,
        #         "gating decision",
        #         {
        #             "sqlite_id": candidate.get("sqlite_id"),
        #             "gating_decision": candidate.get("gating_decision"),
        #             "rejection_reason": candidate.get("rejection_reason"),
        #         },
        #     )

    def _ensure_debug_log_path(self, run_id: Optional[str]) -> None:
        if not run_id:
            return
        if self._debug_log_run_id == run_id and self._debug_log_path:
            return
        run_root = report_manager.resolve_run_root(str(run_id))
        run_root.mkdir(parents=True, exist_ok=True)
        self._debug_log_run_id = run_id
        self._debug_log_path = run_root / "second_pass_debug.log"

    def _debug_log(self, run_id: Optional[str], message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not message:
            return
        if run_id:
            self._ensure_debug_log_path(run_id)
        log_payload = payload or {}
        log("second_pass_agent", LogLevel.DEBUG, f"[DEBUG] {message} | {log_payload}")
        if self._debug_log_path:
            try:
                timestamp = datetime.now().isoformat()
                line = json.dumps(
                    {"ts": timestamp, "message": message, "payload": log_payload},
                    ensure_ascii=False,
                )
                with open(self._debug_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _build_severity_stats(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for item in issues:
            sev = str(item.get("severity") or "low").lower()
            stats[sev] = stats.get(sev, 0) + 1
        return stats

    def _dedupe_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for issue in issues:
            desc = str(issue.get("description") or "").strip().lower()
            source = str(issue.get("source") or "").strip().lower()
            line = str(issue.get("line") or "")
            key = (desc, source, line)
            if key in seen:
                continue
            seen.add(key)
            # 对已混入的 db 命中再做一次无锚定降级
            if source == "db_supplemented":
                ev = issue.get("evidence") if isinstance(issue.get("evidence"), dict) else {}
                if ev:
                    issue = dict(issue)
                    issue["severity"] = self._demote_unanchored_severity(issue.get("severity"), ev)
                    channel = str(ev.get("channel") or "").strip().lower()
                    if not ev.get("primary_channel"):
                        ev = dict(ev)
                        if channel == "curated_issue":
                            ev["primary_channel"] = "curated"
                        elif channel == "weaviate":
                            ev["primary_channel"] = "weaviate"
                        elif channel:
                            ev["primary_channel"] = channel
                    issue["evidence"] = ev
            deduped.append(issue)
        return deduped

    def _rank_issues_by_evidence(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 structured/total 排序，使有锚定的真命中排在前面。"""
        severity_rank = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "info": 1,
        }

        def _key(item: Dict[str, Any]) -> Tuple[float, float, int]:
            ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            if not ev:
                ev = item.get("second_pass_evidence") if isinstance(item.get("second_pass_evidence"), dict) else {}
            structured = float(ev.get("structured_score") or 0.0)
            total = float(ev.get("total_score") or item.get("total_score") or 0.0)
            sev = severity_rank.get(str(item.get("severity") or "").lower(), 0)
            return (structured, total, sev)

        return sorted(issues, key=_key, reverse=True)

    def _default_embed(self, text: str, layer=None) -> List[float]:
        """分层向量生成（code_pattern→codebert，其余→distilbert）。"""
        from infrastructure.embeddings.codebert_embedder import embed_text
        return embed_text(text, layer)

    def _truncate_for_prompt(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text[: self.llm_max_input_chars]

    def _semantic_signature(self, issue: Dict[str, Any]) -> str:
        file_path = str(issue.get("file") or "").strip().lower()
        source = str(issue.get("source") or "").strip().lower()
        severity = str(issue.get("severity") or "").strip().lower()
        tool = str(issue.get("tool") or "").strip().lower()
        requirement_id = str(issue.get("requirement_id") or "")
        issue_type = str(issue.get("issue_type") or issue.get("type") or "").strip().lower()
        function_name = str(issue.get("function_name") or "").strip().lower()
        location = str(issue.get("location") or "").strip().lower()
        line_number = str(issue.get("line_number") or issue.get("line") or "")
        code_snippet = str(issue.get("code_snippet") or "").strip().lower()
        anchor = " ".join([token for token in [function_name, location, line_number] if token])
        snippet_hash = ""
        if code_snippet:
            snippet_hash = hashlib.sha256(code_snippet.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return "|".join([
            file_path,
            source,
            severity,
            tool,
            requirement_id,
            issue_type,
            anchor,
            snippet_hash,
        ])

    def _build_semantic_chunks(
        self,
        issues: List[Dict[str, Any]],
        retrieval_evidence: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        # 语义分批：以 issue+evidence 为不可拆分单元，先按语义签名聚类，再按预算装箱。
        grouped_units: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        grouped_order: List[str] = []

        for idx, issue in enumerate(issues):
            if not isinstance(issue, dict):
                continue
            evidence = retrieval_evidence[idx] if idx < len(retrieval_evidence) else {}
            sig = self._semantic_signature(issue)
            if sig not in grouped_units:
                grouped_order.append(sig)
            grouped_units[sig].append(
                {
                    "index": idx,
                    "issue": issue,
                    "evidence": evidence if isinstance(evidence, dict) else {},
                }
            )

        chunks: List[List[Dict[str, Any]]] = []
        max_chars = max(800, int(self.llm_max_input_chars))

        for sig in grouped_order:
            units = grouped_units[sig]
            current_chunk: List[Dict[str, Any]] = []
            current_size = 0

            for unit in units:
                unit_blob = {
                    "issue": unit["issue"],
                    "evidence": unit["evidence"],
                }
                unit_size = len(json.dumps(unit_blob, ensure_ascii=False))

                # 如果单个语义单元已超预算，仍保持完整并单独成块。
                if unit_size >= max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = []
                        current_size = 0
                    chunks.append([unit])
                    continue

                if current_chunk and (current_size + unit_size > max_chars):
                    chunks.append(current_chunk)
                    current_chunk = [unit]
                    current_size = unit_size
                else:
                    current_chunk.append(unit)
                    current_size += unit_size

            if current_chunk:
                chunks.append(current_chunk)

        return chunks

    def _build_raw_units(
        self,
        original_analysis: Optional[Dict[str, Any]],
        fallback_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(original_analysis, dict):
            return []

        raw_units: List[Dict[str, Any]] = []

        def append_units(items: Any, source_type: str):
            if not isinstance(items, list):
                return
            for item in items:
                if not isinstance(item, dict):
                    continue
                desc = item.get("description") or item.get("message") or item.get("detail")
                if not desc:
                    continue
                line_no = None
                if item.get("line") is not None:
                    try:
                        line_no = int(item.get("line"))
                    except Exception:
                        line_no = None
                file_path = item.get("file") or fallback_file
                unit = {
                    "source_type": source_type,
                    "description": desc,
                    "severity": item.get("severity") or item.get("priority"),
                    "line": line_no,
                    "tool": item.get("tool"),
                    "file": file_path,
                }
                if line_no is not None and file_path:
                    code_snip = self._extract_code_snippet(file_path, line_no)
                    if code_snip:
                        unit["code_snippet"] = code_snip
                raw_units.append(unit)

        static_res = original_analysis.get("static_analysis", {})
        if isinstance(static_res, dict):
            append_units(static_res.get("quality_issues"), "static_quality")
            append_units(static_res.get("security_issues"), "static_security")
            append_units(static_res.get("type_issues"), "static_type")
            append_units(static_res.get("style_issues"), "static_style")

        ai_res = original_analysis.get("ai_analysis", {})
        if isinstance(ai_res, dict):
            final_report = ai_res.get("final_report", {}) if isinstance(ai_res.get("final_report"), dict) else {}
            recs = final_report.get("recommendations", {}) if isinstance(final_report.get("recommendations"), dict) else {}
            append_units(recs.get("immediate_fixes"), "ai_immediate_fix")
            append_units(recs.get("quality_enhancements"), "ai_quality_enhancement")

        sec_res = original_analysis.get("security_analysis", {})
        if isinstance(sec_res, dict):
            sec_ai = sec_res.get("ai_security_analysis", {}) if isinstance(sec_res.get("ai_security_analysis"), dict) else {}
            append_units(sec_ai.get("vulnerabilities_detected"), "security_ai")

        perf_res = original_analysis.get("performance_analysis", {})
        if isinstance(perf_res, dict):
            perf_ai = perf_res.get("ai_performance_analysis", {}) if isinstance(perf_res.get("ai_performance_analysis"), dict) else {}
            append_units(perf_ai.get("performance_bottlenecks"), "performance_ai")

        return raw_units

    def _build_raw_chunk_for_files(
        self,
        raw_units: List[Dict[str, Any]],
        chunk_files: Optional[set],
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        if not raw_units:
            return []

        if chunk_files:
            candidates = [
                unit
                for unit in raw_units
                if not unit.get("file") or str(unit.get("file")) in chunk_files
            ]
        else:
            candidates = list(raw_units)

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        order: List[str] = []
        for unit in candidates:
            signature = f"{unit.get('file') or ''}|{unit.get('source_type') or ''}"
            if signature not in grouped:
                order.append(signature)
            grouped[signature].append(unit)

        chunk: List[Dict[str, Any]] = []
        current_size = 0
        budget = max(800, int(max_chars))

        for signature in order:
            for unit in grouped[signature]:
                unit_size = len(json.dumps(unit, ensure_ascii=False))
                if unit_size >= budget:
                    if not chunk:
                        return [unit]
                    return chunk
                if chunk and (current_size + unit_size > budget):
                    return chunk
                chunk.append(unit)
                current_size += unit_size

        return chunk

    def _resolve_model_max_tokens(self, tokenizer, fallback: int = 1024) -> int:
        return resolve_model_max_tokens(tokenizer, fallback=fallback)

    def _truncate_text_for_model(self, tokenizer, text: str, max_tokens: int) -> str:
        return semantic_truncate_text(tokenizer, text, max_tokens)

    async def _run_generation_inference(self, prompt: str, **kwargs):
        if not self.text_generator or not prompt:
            return []

        tokenizer = getattr(self.text_generator, "tokenizer", None)
        model_max = self._resolve_model_max_tokens(tokenizer, fallback=1024)

        effective_kwargs = dict(kwargs)
        requested_new = int(effective_kwargs.get("max_new_tokens", 256) or 256)
        prompt, _, requested_new = prepare_generation_prompt(
            tokenizer,
            prompt,
            requested_new,
            fallback_model_max=model_max,
            safety_margin=64,
        )

        if "max_length" in effective_kwargs:
            effective_kwargs.pop("max_length", None)

        input_budget = model_max - requested_new
        if input_budget < 64:
            requested_new = max(32, model_max // 4)
            input_budget = max(64, model_max - requested_new)

        safe_prompt = semantic_truncate_text(tokenizer, prompt, input_budget)

        effective_kwargs["max_new_tokens"] = requested_new
        effective_kwargs["truncation"] = True

        try:
            return await asyncio.to_thread(self.text_generator, safe_prompt, **effective_kwargs)
        except Exception as e:
            log("second_pass_agent", LogLevel.WARNING, f"⚠️ 二次分析LLM推理失败: {e}")
            return []

    def _sanitize_json_like_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _parse_json_object(self, generated: Any) -> Dict[str, Any]:
        text = ""
        if isinstance(generated, list) and generated and isinstance(generated[0], dict):
            text = str(generated[0].get("generated_text", "")).strip()
        elif isinstance(generated, str):
            text = generated.strip()
        if not text:
            return {}

        cleaned = self._sanitize_json_like_text(text)
        candidates = [cleaned]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return {}

    async def _persist_second_pass_report(self, report_data: Dict[str, Any], round_num: Optional[int] = None, subdir: Optional[str] = None) -> None:
        run_id = report_data.get("run_id")
        if not run_id:
            return

        base_name = str(report_data.get("sanitized_name") or f"req_{report_data.get('requirement_id', 'unknown')}")
        round_suffix = f"_r{round_num}" if round_num is not None else ""
        filename = f"second_pass_consolidated_{base_name}{round_suffix}.json"
        target_subdir = subdir if subdir is not None else "consolidated"
        # embed a concise scoring summary into the report_data before persisting
        try:
            scoring_summary = {
                "generated_at": datetime.now().isoformat(),
                "issue_count": len(report_data.get("issues", [])),
                "scored_issues": [],
            }
            for issue in report_data.get("issues", []):
                # include evidence scoring if present
                evidence = issue.get("evidence") or {}
                if evidence:
                    # if agent already provided detailed confidence_components, reuse them
                    comps = evidence.get("confidence_components")
                    formula = evidence.get("confidence_formula")
                    total_score = evidence.get("total_score")

                    if not comps:
                        # compute breakdown from available raw scores
                        structured = float(evidence.get("structured_score") or 0.0)
                        semantic = float(evidence.get("semantic_score") or 0.0)
                        context = float(evidence.get("context_score") or 0.0)
                        anchor = float(evidence.get("anchor_score") or 0.0)
                        anchor_bonus = float(evidence.get("anchor_bonus") or 0.0)
                        layer_bonus = float(evidence.get("layer_bonus") or 0.0)
                        penalty = float(evidence.get("penalty_score") or 0.0)

                        channel = str(evidence.get("channel") or "").lower()
                        if channel == "curated_issue":
                            weights = {"structured": 0.55, "semantic": 0.0, "context": 0.15, "anchor": 0.2}
                            formula = "total = structured*0.55 + context*0.15 + anchor*0.20 + anchor_bonus + layer_bonus - penalty"
                        else:
                            weights = {"structured": 0.5, "semantic": 0.35, "context": 0.1, "anchor": 0.05}
                            formula = "total = structured*0.5 + semantic*0.35 + context*0.1 + anchor*0.05 + anchor_bonus + layer_bonus - penalty"

                        comp_structured = round(structured * weights.get("structured", 0.0), 4)
                        comp_semantic = round(semantic * weights.get("semantic", 0.0), 4)
                        comp_context = round(context * weights.get("context", 0.0), 4)
                        comp_anchor = round(anchor * weights.get("anchor", 0.0), 4)

                        comps = {
                            "structured": round(structured, 4),
                            "semantic": round(semantic, 4),
                            "context": round(context, 4),
                            "anchor": round(anchor, 4),
                            "anchor_bonus": round(anchor_bonus, 4),
                            "layer_bonus": round(layer_bonus, 4),
                            "penalty": round(penalty, 4),
                            "component_contributions": {
                                "structured_contribution": comp_structured,
                                "semantic_contribution": comp_semantic,
                                "context_contribution": comp_context,
                                "anchor_contribution": comp_anchor,
                            },
                        }

                        # 总分始终按分项重算，避免沿用残缺字段下的旧 total_score
                        total_score = round(
                            max(0.0, comp_structured + comp_semantic + comp_context + comp_anchor + anchor_bonus + layer_bonus - penalty),
                            4,
                        )

                    scoring_summary["scored_issues"].append(
                        {
                            "line": issue.get("line"),
                            "description": issue.get("description"),
                            "channel": evidence.get("channel"),
                            "total_score": total_score,
                            "confidence_components": comps,
                            "confidence_formula": formula,
                        }
                    )
        except Exception:
            scoring_summary = {"generated_at": datetime.now().isoformat(), "issue_count": 0, "scored_issues": []}

        report_data["scoring_summary"] = scoring_summary

        path = report_manager.generate_run_scoped_report(
            run_id=run_id,
            content=report_data,
            filename=filename,
            subdir=target_subdir,
        )
        log("second_pass_agent", LogLevel.INFO, f"📝 二次分析报告已写入: {path}")

    async def _forward_to_readability(
        self,
        report_data: Dict[str, Any],
        run_id: Optional[str],
        requirement_id: Optional[int],
        file_path: Optional[str],
        validation_failed: bool = False,
        validation_errors: Optional[List[str]] = None,
        second_pass_error: Optional[str] = None,
        pure_llm: bool = False,
        second_pass_round: Optional[int] = None,
        weaviate_layer_mode: Optional[str] = None,
    ):
        content = {
            "requirement_id": requirement_id,
            "run_id": run_id,
            "file_path": file_path,
            "analysis_type": "second_pass_report",
            "report_data": report_data,
            "validation_failed": validation_failed,
            "validation_errors": validation_errors or [],
            "second_pass_error": second_pass_error,
            "pure_llm": pure_llm,
            "second_pass_round": second_pass_round,
            "weaviate_layer_mode": weaviate_layer_mode,
        }

        msg = Message(
            id=f"{run_id}_{requirement_id}_readability",
            sender=self.agent_id,
            receiver="ai_readability_enhancement_agent",
            content=content,
            timestamp=datetime.now().timestamp(),
            message_type="analyze_consolidated_report",
        )
        from .agent_manager import AgentManager

        await AgentManager.get_instance().route_message(msg)

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "second_pass_agent_ready",
            "enabled": self.enable_second_pass,
            "weaviate_connected": self.vector_service.is_connected(),
        }
