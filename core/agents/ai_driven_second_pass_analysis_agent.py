import json
import os
import asyncio
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
            refined = await self._run_second_pass(report_data)
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

    async def _run_second_pass(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
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
    ) -> List[Dict[str, Any]]:
        if not issues:
            return []

        semantic_chunks = self._build_semantic_chunks(issues, retrieval_evidence)
        normalized: List[Dict[str, Any]] = []
        parsed_any_chunk = False

        for chunk in semantic_chunks:
            if len(normalized) >= self.max_new_findings:
                break

            chunk_issues = [unit["issue"] for unit in chunk]
            chunk_evidence = [unit["evidence"] for unit in chunk]

            prompt = get_prompt(
                task_type="analysis_report",
                variant="second_pass_gap_discovery",
                issues_json=json.dumps(chunk_issues, ensure_ascii=False),
                retrieval_evidence_json=json.dumps(chunk_evidence, ensure_ascii=False),
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

        evidence: Dict[str, Any] = {
            "issue_description": issue_desc,
            "issue_source": issue_source,
            "issue_severity": issue_severity,
            "weaviate_hits": [],
            "sqlite_hits": [],
        }

        # SQLite 结构化匹配
        for pattern in sqlite_patterns:
            if self._pattern_matches_issue(pattern, issue_desc, issue_source):
                evidence["sqlite_hits"].append(
                    {
                        "id": pattern.get("id"),
                        "error_type": pattern.get("error_type"),
                        "severity": pattern.get("severity"),
                        "solution": pattern.get("solution"),
                    }
                )
                if len(evidence["sqlite_hits"]) >= self.weaviate_top_k:
                    break

        # Weaviate 语义匹配
        if self.enable_weaviate_query and not self._weaviate_connect_attempted and not self.vector_service.is_connected():
            self._weaviate_connect_attempted = True
            connected = self.vector_service.connect(auto_create_schema=False)
            if connected:
                log("second_pass_agent", LogLevel.INFO, "✅ Weaviate 连接成功，启用语义检索")
            else:
                log("second_pass_agent", LogLevel.WARNING, "⚠️ Weaviate 连接不可用，跳过语义检索")

        if self.enable_weaviate_query and self.vector_service.is_connected() and issue_desc:
            query_vector = self._default_embed(f"[{issue_source}] {issue_desc}")
            results = self.vector_service.search_knowledge_items(
                query_vector=query_vector,
                limit=self.weaviate_top_k,
                layer="semantic",
            )
            for item in results:
                distance = item.get("_additional", {}).get("distance", 2.0)
                similarity = 1.0 - (float(distance) / 2.0)
                evidence["weaviate_hits"].append(
                    {
                        "sqlite_id": item.get("sqlite_id"),
                        "error_type": item.get("error_type"),
                        "severity": item.get("severity"),
                        "solution": item.get("solution"),
                        "distance": distance,
                        "similarity": similarity,
                    }
                )

        return evidence

    def _apply_corrections(
        self,
        issue: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        corrected = dict(issue)
        adjustment: Optional[Dict[str, Any]] = None

        best_hit = None
        weaviate_hits = evidence.get("weaviate_hits", [])
        if weaviate_hits:
            best_hit = sorted(weaviate_hits, key=lambda h: h.get("similarity", 0), reverse=True)[0]

        if best_hit and float(best_hit.get("similarity", 0.0)) >= self.similarity_threshold:
            old_severity = str(corrected.get("severity") or "low").lower()
            new_severity = str(best_hit.get("severity") or old_severity).lower()
            if new_severity and new_severity != old_severity:
                corrected["severity"] = new_severity
                corrected["source"] = f"{corrected.get('source', 'unknown')}_db_corrected"
                adjustment = {
                    "issue_description": corrected.get("description"),
                    "reason": "weaviate_similarity_correction",
                    "old_severity": old_severity,
                    "new_severity": new_severity,
                    "similarity": best_hit.get("similarity"),
                    "sqlite_id": best_hit.get("sqlite_id"),
                }

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

        for hit in evidence.get("weaviate_hits", []):
            similarity = float(hit.get("similarity", 0.0))
            if similarity < (self.similarity_threshold + 0.06):
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
                    "description": f"历史知识命中，可能漏报: {candidate_desc}",
                    "tool": "second_pass_analysis",
                    "run_id": run_id,
                    "evidence": {
                        "sqlite_id": hit.get("sqlite_id"),
                        "similarity": similarity,
                        "recommended_solution": hit.get("solution"),
                    },
                }
            )

        return findings

    def _pattern_matches_issue(self, pattern: Dict[str, Any], description: str, source: str) -> bool:
        pattern_error_type = str(pattern.get("error_type") or "").strip().lower()
        pattern_desc = str(pattern.get("error_description") or "").strip().lower()
        description_l = description.lower()
        source_l = source.lower()

        if pattern_error_type and pattern_error_type in description_l:
            return True
        if pattern_desc and pattern_desc[:32] and pattern_desc[:32] in description_l:
            return True
        if pattern_error_type and source_l and pattern_error_type in source_l:
            return True
        return False

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
        return "|".join([file_path, source, severity, tool, requirement_id])

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
