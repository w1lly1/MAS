import json
import os
import asyncio
import hashlib
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
        self.enable_llm_second_pass = self.agent_config.get("enable_llm_second_pass", True)
        self.fallback_to_original = self.agent_config.get("fallback_to_original_on_error", True)
        self.weaviate_top_k = int(self.agent_config.get("weaviate_top_k", 5))
        self.similarity_threshold = float(self.agent_config.get("similarity_threshold", 0.78))
        self.max_new_findings = int(self.agent_config.get("max_new_findings", 5))
        self.max_sqlite_patterns = int(self.agent_config.get("max_sqlite_patterns", 200))
        self.llm_max_input_chars = int(self.agent_config.get("llm_max_input_chars", 9000))

        self.used_device = "gpu"
        self.text_generator = None
        self.model_name = self.agent_config.get("model_name", "gpt2")
        self.fallback_model = self.agent_config.get("fallback_model", "distilgpt2")

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

    async def _initialize_models(self):
        """初始化二次分析LLM，失败时保留硬编码回退路径。"""
        if self._llm_init_attempted:
            return
        self._llm_init_attempted = True

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

        if not self.enable_second_pass:
            await self._forward_to_readability(report_data, run_id, requirement_id, file_path)
            return

        try:
            refined = await self._run_second_pass(report_data, original_analysis=original_analysis)
            await self._persist_second_pass_report(refined)
            await self._forward_to_readability(refined, run_id, requirement_id, file_path)
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
    ) -> Dict[str, Any]:
        original_issues = report_data.get("issues", [])
        if not isinstance(original_issues, list):
            original_issues = []

        sqlite_patterns = await self.db_service.get_issue_patterns(status="active")
        sqlite_patterns = sqlite_patterns[: self.max_sqlite_patterns]

        corrected_issues: List[Dict[str, Any]] = []
        confidence_adjustments: List[Dict[str, Any]] = []
        retrieval_evidence: List[Dict[str, Any]] = []
        new_findings: List[Dict[str, Any]] = []

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
            evidence = await self._collect_evidence(issue, sqlite_patterns)
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

        # Task 2: LLM补漏（失败则回退到硬编码补漏）
        llm_gap_ok = False
        if self.enable_llm_second_pass and self.text_generator:
            try:
                new_findings = await self._llm_gap_discovery(
                    corrected_issues,
                    retrieval_evidence,
                    run_id=report_data.get("run_id"),
                    requirement_id=report_data.get("requirement_id"),
                    file_path=report_data.get("file"),
                    original_analysis=original_analysis,
                )
                llm_gap_ok = True
            except Exception as e:
                log("second_pass_agent", LogLevel.WARNING, f"⚠️ LLM补漏失败，回退硬编码补漏: {e}")

        if not llm_gap_ok:
            new_findings = []
            for idx, issue in enumerate(corrected_issues):
                evidence = retrieval_evidence[idx] if idx < len(retrieval_evidence) else {}
                candidates = self._derive_new_findings(
                    issue=issue,
                    evidence=evidence,
                    run_id=report_data.get("run_id"),
                    requirement_id=report_data.get("requirement_id"),
                    file_path=report_data.get("file"),
                )
                for candidate in candidates:
                    if len(new_findings) >= self.max_new_findings:
                        break
                    new_findings.append(candidate)

        merged_issues = self._dedupe_issues(corrected_issues + new_findings)
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
        second_pass_summary = {
            "original_issue_count": len(original_issues),
            "corrected_issue_count": len(confidence_adjustments),
            "new_finding_count": len(new_findings),
            "final_issue_count": len(merged_issues),
            "llm_correction_used": llm_correction_ok,
            "llm_gap_discovery_used": llm_gap_ok,
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
                max_new_tokens=900,
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
        retrieval_evidence: List[Dict[str, Any]],
        run_id: Optional[str],
        requirement_id: Optional[int],
        file_path: Optional[str],
        original_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not issues:
            return []

        semantic_chunks = self._build_semantic_chunks(issues, retrieval_evidence)
        normalized: List[Dict[str, Any]] = []
        parsed_any_chunk = False

        raw_units = self._build_raw_units(original_analysis, fallback_file=file_path)
        self._debug_log(
            run_id,
            "raw_units built",
            {
                "count": len(raw_units),
                "sample": raw_units[:3],
            },
        )

        for chunk in semantic_chunks:
            if len(normalized) >= self.max_new_findings:
                break

            chunk_issues = [unit["issue"] for unit in chunk]
            chunk_evidence = [unit["evidence"] for unit in chunk]
            chunk_files = {
                str(unit.get("file") or "").strip()
                for unit in chunk_issues
                if isinstance(unit, dict)
            }
            raw_chunk = self._build_raw_chunk_for_files(
                raw_units,
                chunk_files,
                max_chars=self.llm_max_input_chars,
            )
            raw_analysis_json = json.dumps(raw_chunk, ensure_ascii=False)
            self._debug_log(
                run_id,
                "raw_chunk prepared",
                {
                    "chunk_files": list(chunk_files)[:5],
                    "raw_chunk_count": len(raw_chunk),
                    "raw_chunk_chars": len(raw_analysis_json),
                },
            )

            prompt = get_prompt(
                task_type="analysis_report",
                variant="second_pass_gap_discovery",
                issues_json=json.dumps(chunk_issues, ensure_ascii=False),
                retrieval_evidence_json=json.dumps(chunk_evidence, ensure_ascii=False),
                raw_analysis_json=raw_analysis_json,
                run_id=run_id or "",
                requirement_id=requirement_id or 0,
                file_path=file_path or "",
            )
            generated = await self._run_generation_inference(
                prompt,
                max_new_tokens=900,
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
                        "file": f.get("file", file_path),
                        "source": f.get("source", "db_supplemented"),
                        "severity": f.get("severity", "medium"),
                        "line": f.get("line"),
                        "description": f.get("description", "历史知识命中，可能漏报"),
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

    async def _collect_evidence(self, issue: Dict[str, Any], sqlite_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            if match_info.get("structured_score", 0.0) > 0.0:
                self._debug_log(
                    issue.get("run_id"),
                    "sqlite pattern checked",
                    {
                        "pattern_id": pattern.get("id"),
                        "error_type": pattern.get("error_type"),
                        "file_pattern": pattern.get("file_pattern"),
                        "language": pattern.get("language"),
                        "matched": match_info.get("matched"),
                        "matched_fields": match_info.get("matched_fields"),
                        "structured_score": match_info.get("structured_score"),
                        "context_score": match_info.get("context_score"),
                    },
                )
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
                self._debug_log(
                    issue.get("run_id"),
                    "sqlite candidate gated",
                    {
                        "sqlite_id": candidate.get("sqlite_id"),
                        "error_type": candidate.get("error_type"),
                        "severity": candidate.get("severity"),
                        "structured_score": candidate.get("structured_score"),
                        "semantic_score": candidate.get("semantic_score"),
                        "context_score": candidate.get("context_score"),
                        "penalty_score": candidate.get("penalty_score"),
                        "total_score": candidate.get("total_score"),
                        "gating_decision": candidate.get("gating_decision"),
                    },
                )
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

        # CuratedIssue 结构化补强：只用本地数据库中已确认的样本做结构锚点，不直接进入 LLM prompt。
        curated_issues: List[Dict[str, Any]] = []
        try:
            curated_issues = await self.db_service.get_curated_issues(status="resolved")
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
            self._debug_log(
                issue.get("run_id"),
                "curated candidate gated",
                {
                    "curated_id": candidate.get("sqlite_id"),
                    "structured_score": candidate.get("structured_score"),
                    "semantic_score": candidate.get("semantic_score"),
                    "context_score": candidate.get("context_score"),
                    "anchor_score": candidate.get("anchor_score"),
                    "total_score": candidate.get("total_score"),
                    "gating_decision": candidate.get("gating_decision"),
                },
            )
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
            query_vector = self._default_embed(query_text)
            layers_to_query = ["semantic", "code_pattern", "solution", "full"]
            seen_hits: set[tuple[Optional[int], str]] = set()
            for layer in layers_to_query:
                results = self.vector_service.search_knowledge_items(
                    query_vector=query_vector,
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
                    "distance": distance,
                    "similarity": similarity,
                }
                evidence["weaviate_hits"].append(weaviate_hit)
                candidate = self._build_candidate_from_weaviate(weaviate_hit, issue_desc, issue_file, issue)
                self._gate_candidate(candidate)
                self._debug_log(
                    issue.get("run_id"),
                    "weaviate candidate gated",
                    {
                        "sqlite_id": candidate.get("sqlite_id"),
                        "error_type": candidate.get("error_type"),
                        "severity": candidate.get("severity"),
                        "semantic_score": candidate.get("semantic_score"),
                        "context_score": candidate.get("context_score"),
                        "penalty_score": candidate.get("penalty_score"),
                        "total_score": candidate.get("total_score"),
                        "gating_decision": candidate.get("gating_decision"),
                    },
                )
                evidence["candidates"].append(candidate)
                if candidate.get("gating_decision") in {"formal_hit", "explanatory_hit"}:
                    hit = dict(candidate)
                    hit.update({
                        "sqlite_id": weaviate_hit.get("sqlite_id"),
                        "semantic_score": weaviate_hit.get("similarity"),
                        "vector_layer": weaviate_hit.get("vector_layer"),
                        "error_description": weaviate_hit.get("error_description"),
                        "problematic_pattern": weaviate_hit.get("problematic_pattern"),
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

        issue_base = os.path.basename(issue_file_l) if issue_file_l else ""
        curated_base = os.path.basename(curated_file) if curated_file else ""
        normalized_curated = curated_file.replace("\\", "/")
        normalized_issue = issue_file_l.replace("\\", "/")

        if issue_base and curated_base and issue_base == curated_base:
            matched_fields.append("basename_match")
            structured_score += 0.28
        if issue_base and curated_file and issue_base in curated_file:
            matched_fields.append("basename_in_curated_path")
            structured_score += 0.22
        if curated_base and issue_file_l and curated_base in issue_file_l:
            matched_fields.append("curated_basename_in_issue_path")
            structured_score += 0.22
        if normalized_curated and normalized_issue:
            curated_tokens = [token for token in normalized_curated.split("/") if token]
            issue_tokens = [token for token in normalized_issue.split("/") if token]
            overlap = set(curated_tokens) & set(issue_tokens)
            if overlap:
                matched_fields.append("path_token_overlap")
                structured_score += min(0.15, 0.03 * len(overlap))

        start_line = curated_issue.get("start_line")
        end_line = curated_issue.get("end_line")
        if issue_line is not None and start_line is not None and end_line is not None:
            try:
                start_i = int(start_line)
                end_i = int(end_line)
                if start_i <= issue_line <= end_i:
                    matched_fields.append("line_in_curated_range")
                    structured_score += 0.4
                elif abs(issue_line - start_i) <= 8 or abs(issue_line - end_i) <= 8:
                    matched_fields.append("near_curated_range")
                    structured_score += 0.2
            except Exception:
                pass

        issue_desc = str(issue.get("description") or "").lower()
        phenomenon = str(curated_issue.get("problem_phenomenon") or "").lower()
        root_cause = str(curated_issue.get("root_cause") or "").lower()
        if phenomenon and any(token in issue_desc for token in phenomenon.split()[:3] if token):
            matched_fields.append("phenomenon_in_description")
            structured_score += 0.1
        if root_cause and any(token in issue_desc for token in root_cause.split()[:3] if token):
            matched_fields.append("root_cause_in_description")
            structured_score += 0.08

        matched = structured_score >= 0.45
        if matched:
            context_score = 0.15 if issue_line is not None else 0.05
        elif structured_score > 0.0:
            self._debug_log(
                issue.get("run_id"),
                "curated issue near-miss",
                {
                    "curated_id": curated_issue.get("id"),
                    "file_path": curated_issue.get("file_path"),
                    "structured_score": round(min(1.0, structured_score), 3),
                    "matched_fields": matched_fields,
                },
            )

        return {
            "matched": matched,
            "matched_fields": matched_fields,
            "structured_score": min(1.0, structured_score),
            "context_score": min(0.2, context_score),
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
            if hit.get("gating_decision") != "explanatory_hit":
                continue

            candidate_desc = str(hit.get("error_type") or "潜在已知问题模式命中")
            if candidate_desc and candidate_desc.lower() in issue_desc.lower():
                continue

            findings.append(
                {
                    "requirement_id": requirement_id,
                    "file": file_path,
                    "source": "db_supplemented",
                    "severity": (hit.get("severity") or "medium"),
                    "line": issue.get("line"),
                    "description": f"历史知识命中: {candidate_desc}",
                    "tool": "second_pass_analysis",
                    "run_id": run_id,
                    "evidence": {
                        "channel": hit.get("channel"),
                        "sqlite_id": hit.get("sqlite_id"),
                        "semantic_score": hit.get("semantic_score"),
                        "structured_score": hit.get("structured_score"),
                        "total_score": hit.get("total_score"),
                        "matched_fields": hit.get("matched_fields"),
                        "recommended_solution": hit.get("solution"),
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

        matched = structured_score >= 0.3
        if structured_score > 0.0 and not matched:
            self._debug_log(
                run_id,
                "sqlite near-miss",
                {
                    "pattern_id": pattern.get("id"),
                    "error_type": pattern.get("error_type"),
                    "file_pattern": pattern.get("file_pattern"),
                    "language": pattern.get("language"),
                    "structured_score": min(1.0, structured_score),
                    "context_score": min(0.2, context_score),
                    "matched_fields": matched_fields,
                    "issue_file": file_path,
                    "issue_location": (issue or {}).get("location"),
                    "issue_function": (issue or {}).get("function_name"),
                },
            )
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
            "error_type": hit.get("error_type"),
            "severity": hit.get("severity"),
            "solution": hit.get("solution"),
            "structured_score": 0.0,
            "semantic_score": float(hit.get("similarity", 0.0)),
            "context_score": context_score,
            "anchor_score": self._calc_anchor_score(issue),
            "penalty_score": 0.0,
            "total_score": 0.0,
            "matched_fields": [],
            "issue_summary": issue_desc[:160],
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
            "matched_fields": hit.get("matched_fields", []),
            "issue_summary": issue_desc[:160],
            "reasoning": "curated_issue_structural_match",
            "rejection_reason": "",
            "gating_decision": "",
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
        if vector_layer in {"full", "code_pattern"}:
            layer_bonus += 0.05
        elif vector_layer == "solution":
            layer_bonus += 0.03

        penalty = 0.0
        if error_type in generic_terms and structured < 0.4:
            penalty += 0.2
        if semantic < self.similarity_threshold and structured < 0.4:
            penalty += 0.1
        if anchor < 0.2 and structured < 0.4:
            penalty += 0.05

        if channel == "curated_issue":
            total = (structured * 0.55) + (context * 0.15) + (anchor * 0.2) + anchor_bonus + layer_bonus - penalty
        else:
            total = (structured * 0.5) + (semantic * 0.35) + (context * 0.1) + (anchor * 0.05) + anchor_bonus + layer_bonus - penalty
        candidate["penalty_score"] = penalty
        candidate["total_score"] = round(max(0.0, total), 4)
        candidate["anchor_bonus"] = round(anchor_bonus, 4)
        candidate["layer_bonus"] = round(layer_bonus, 4)

        if channel == "curated_issue" and structured >= 0.75 and anchor >= 0.35:
            candidate["gating_decision"] = "formal_hit"
        elif channel == "curated_issue" and structured >= 0.45 and anchor >= 0.2:
            candidate["gating_decision"] = "explanatory_hit"
        elif structured >= 0.6 or (structured >= 0.5 and anchor >= 0.2 and anchor_bonus >= 0.1):
            candidate["gating_decision"] = "formal_hit"
        elif candidate["total_score"] >= 0.55 and structured >= 0.45 and anchor >= 0.2:
            candidate["gating_decision"] = "explanatory_hit"
        elif semantic >= self.similarity_threshold and anchor >= 0.35:
            candidate["gating_decision"] = "explanatory_hit"
        elif semantic >= self.similarity_threshold and anchor < 0.3 and structured < 0.3:
            candidate["gating_decision"] = "low_confidence_hit"
            candidate["rejection_reason"] = "weak_structure_high_semantic"
        else:
            candidate["gating_decision"] = "discarded_hit"
            candidate["rejection_reason"] = "low_confidence_or_generic"

        if candidate.get("gating_decision") in {"discarded_hit", "low_confidence_hit"}:
            self._debug_log(
                run_id,
                "gating decision",
                {
                    "sqlite_id": candidate.get("sqlite_id"),
                    "channel": channel,
                    "vector_layer": vector_layer,
                    "error_type": candidate.get("error_type"),
                    "structured_score": structured,
                    "semantic_score": semantic,
                    "context_score": context,
                    "anchor_score": anchor,
                    "anchor_bonus": candidate.get("anchor_bonus"),
                    "layer_bonus": candidate.get("layer_bonus"),
                    "penalty_score": candidate.get("penalty_score"),
                    "total_score": candidate.get("total_score"),
                    "matched_fields": matched_fields,
                    "similarity_threshold": self.similarity_threshold,
                    "gating_decision": candidate.get("gating_decision"),
                    "rejection_reason": candidate.get("rejection_reason"),
                },
            )

    def _ensure_debug_log_path(self, run_id: Optional[str]) -> None:
        if not run_id:
            return
        if self._debug_log_run_id == run_id and self._debug_log_path:
            return
        run_root = report_manager.directories["analysis"] / str(run_id)
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
            deduped.append(issue)
        return deduped

    def _default_embed(self, text: str) -> List[float]:
        if text is None:
            text = ""
        total = float(sum(ord(c) for c in text))
        length = float(len(text) or 1)
        return [
            length,
            (total % 991) / 991.0,
            (total % 313) / 313.0,
        ]

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

    async def _persist_second_pass_report(self, report_data: Dict[str, Any]) -> None:
        run_id = report_data.get("run_id")
        if not run_id:
            return

        base_name = str(report_data.get("sanitized_name") or f"req_{report_data.get('requirement_id', 'unknown')}")
        filename = f"second_pass_consolidated_{base_name}.json"
        path = report_manager.generate_run_scoped_report(
            run_id=run_id,
            content=report_data,
            filename=filename,
            subdir="consolidated",
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
