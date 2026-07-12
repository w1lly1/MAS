"""
AI驱动的可读性增强代理 - 将复杂的JSON分析报告转换为易读的Markdown格式摘要

这个代理的核心功能是：
1. 扫描analysis/{run_id}目录下所有JSON文件（agents和consolidated目录）
2. 分析和分类问题
3. 生成Markdown格式的易理解的中文摘要
4. 创建analysis/{run_id}/readability_enhancement目录结构
5. 保存增强后的Markdown文件

CPU友好设计：使用轻量级的文本处理，无需大型生成模型
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .base_agent import BaseAgent, Message
from infrastructure.reports import report_manager
from utils import log, LogLevel


class AIDrivenReadabilityEnhancementAgent(BaseAgent):
    """
    AI驱动的可读性增强代理
    
    职责：
    1. 扫描 analysis/{run_id}/ 目录下所有JSON文件
    2. 分析和分类问题
    3. 生成Markdown格式的易理解的中文摘要
    4. 创建analysis/{run_id}/readability_enhancement/目录
    5. 保存agents和consolidated两类增强文件
    
    特点：
    - 完全基于CPU运行，无需GPU
    - 生成Markdown格式输出
    - 自动扫描和处理run_id目录下的所有报告
    """
    
    def __init__(self):
        super().__init__(
            agent_id="ai_readability_enhancement_agent",
            name="AI驱动的可读性增强代理"
        )
        
        # 从统一配置获取
        from infrastructure.config.ai_agents import get_ai_agent_config
        self.agent_config = get_ai_agent_config().get_readability_agent_config()
        
        # 缓存配置
        self.enable_cache = self.agent_config.get("enable_report_cache", True)
        self.report_cache = {} if self.enable_cache else None
        self.reports_base_dir = Path(__file__).parent.parent.parent / "reports" / "analysis"
        
    async def initialize(self):
        """初始化代理"""
        try:
            log("readability_enhancement_agent", LogLevel.INFO, f"初始化可读性增强代理")
            self.is_running = True
            log("readability_enhancement_agent", LogLevel.INFO, "✅ 可读性增强代理初始化完成")
        except Exception as e:
            log("readability_enhancement_agent", LogLevel.ERROR, f"初始化失败: {e}")
            raise
    
    async def handle_message(self, message: Message):
        """处理接收到的消息

        现在支持接收由二次分析代理转发的消息，包含额外字段:
        - `pure_llm`: 如果为 True，则同时把源 consolidated 保存到 `pureLLM` 子目录
        - `second_pass_round`: 用于区分第一轮/第二轮结果并在可读性输出中保留独立副本
        """
        if message.message_type == "analyze_consolidated_report":
            try:
                run_id = message.content.get("run_id")
                requirement_id = message.content.get("requirement_id")
                pure_llm = message.content.get("pure_llm", False)
                second_pass_round = message.content.get("second_pass_round")

                # 如果是 pure LLM 输出，保存一份到 pureLLM 目录
                if pure_llm and run_id and isinstance(message.content.get("report_data"), dict):
                    try:
                        base_name = str(message.content["report_data"].get("sanitized_name") or f"req_{requirement_id}")
                        filename = f"consolidated_{base_name}.json"
                        report_manager.generate_run_scoped_report(run_id=run_id, content=message.content["report_data"], filename=filename, subdir="pureLLM/consolidated")
                        log("readability_enhancement_agent", LogLevel.INFO, f"💾 已保存 pureLLM 输出: {run_id}/pureLLM/consolidated/{filename}")
                    except Exception as e:
                        log("readability_enhancement_agent", LogLevel.WARNING, f"⚠️ 保存 pureLLM 输出失败: {e}")

                # 对于普通流程，继续扫描并增强 run 下的 consolidated 文件
                await self.enhance_run_reports(run_id)
                log("readability_enhancement_agent", LogLevel.INFO, f"✅ 可读性增强完成: run_id={run_id}")
            except Exception as e:
                log("readability_enhancement_agent", LogLevel.ERROR, f"❌ 处理消息失败: {e}")
    
    async def enhance_run_reports(self, run_id: str) -> bool:
        """
        扫描并增强指定run_id下的所有报告
        
        Args:
            run_id: 运行ID
        
        Returns:
            是否成功处理
        """
        try:
            run_dir = self.reports_base_dir / run_id
            
            if not run_dir.exists():
                log("readability_enhancement_agent", LogLevel.WARNING, f"⚠️  run_id目录不存在: {run_dir}")
                return False
            
            log("readability_enhancement_agent", LogLevel.INFO, f"🔍 扫描报告目录: {run_dir}")
            
            # 创建输出目录结构：readability_enhancement 下包含三个子目录用于对比
            enhancement_dir = run_dir / "readability_enhancement"
            consolidated_out = enhancement_dir / "consolidated"
            pure_out = enhancement_dir / "pureLLM"
            full_out = enhancement_dir / "fullLayer"

            consolidated_out.mkdir(parents=True, exist_ok=True)
            pure_out.mkdir(parents=True, exist_ok=True)
            full_out.mkdir(parents=True, exist_ok=True)

            log("readability_enhancement_agent", LogLevel.INFO, f"📁 已创建输出目录: {enhancement_dir}")

            # 处理不同来源目录下的 JSON 文件：真实二次分析(second_pass)、pureLLM、fullLayer
            # 真实对照路径优先使用 second_pass/consolidated（含双查询证据），否则回退到原始 consolidated
            second_pass_source_dir = run_dir / "second_pass" / "consolidated"
            consolidated_source_dir = run_dir / "consolidated"
            real_source_dir = second_pass_source_dir if second_pass_source_dir.exists() else consolidated_source_dir
            if real_source_dir.exists():
                for json_file in real_source_dir.glob("*.json"):
                    await self._enhance_single_report(json_file, consolidated_out, "consolidated")

            pure_source_dir = run_dir / "pureLLM" / "consolidated"
            if pure_source_dir.exists():
                for json_file in pure_source_dir.glob("*.json"):
                    await self._enhance_single_report(json_file, pure_out, "pureLLM")

            full_source_dir = run_dir / "fullLayer" / "consolidated"
            if full_source_dir.exists():
                for json_file in full_source_dir.glob("*.json"):
                    await self._enhance_single_report(json_file, full_out, "fullLayer")
            
            log("readability_enhancement_agent", LogLevel.INFO, f"✅ run_id {run_id} 的所有报告已完成可读性增强")
            return True
            
        except Exception as e:
            log("readability_enhancement_agent", LogLevel.ERROR, f"❌ 增强报告失败: {e}")
            return False
    
    async def _enhance_single_report(
        self,
        json_file: Path,
        output_dir: Path,
        category: str
    ) -> bool:
        """
        增强单个JSON报告
        
        Args:
            json_file: 源JSON文件路径
            output_dir: 输出目录
            category: 文件类别 ("agents" 或 "consolidated")
        
        Returns:
            是否成功处理
        """
        try:
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            log("readability_enhancement_agent", LogLevel.INFO, f"📄 处理文件: {json_file.name}")
            
            # 如果报告的 evidence 缺少 confidence_components，尝试从可用分数字段计算并注入，
            # 以保证后续渲染能展示分项与公式
            try:
                scored_issues = report_data.get("scoring_summary", {}).get("scored_issues", [])
            except Exception:
                scored_issues = []

            for issue in report_data.get("issues", []):
                ev = issue.get("second_pass_evidence") or issue.get("evidence")
                if not isinstance(ev, dict):
                    continue

                # 若 evidence 缺分项，尝试从检索候选中按 sqlite_id 回填（兼容旧报告）
                self._hydrate_evidence_scores_from_retrieval(report_data, ev)

                # 优先使用已存在的 confidence_components，并确保 total 与分项一致
                if ev.get("confidence_components"):
                    ev["total_score"] = self._total_score_from_components(
                        ev.get("confidence_components") or {},
                        channel=ev.get("channel"),
                    )
                    continue

                # 1) 如果 scoring_summary 中存在匹配项，优先使用分项；总分按分项重算以保持一致
                injected = False
                try:
                    for s in scored_issues:
                        if s.get("line") == issue.get("line") and (
                            not issue.get("description")
                            or s.get("description") == issue.get("description")
                        ):
                            comps = s.get("confidence_components")
                            formula = s.get("confidence_formula")
                            if comps:
                                ev["confidence_components"] = comps
                                if formula:
                                    ev["confidence_formula"] = formula
                                # 用分项重算总分，避免沿用拷贝不完整字段后的陈旧 total_score
                                ev["total_score"] = self._total_score_from_components(
                                    comps,
                                    channel=ev.get("channel") or s.get("channel"),
                                )
                                injected = True
                                break
                except Exception:
                    injected = False

                if injected:
                    continue

                # 2) 否则根据现有原始分数字段计算
                try:
                    structured = float(ev.get("structured_score") or 0.0)
                    semantic = float(ev.get("semantic_score") or 0.0)
                    context = float(ev.get("context_score") or 0.0)
                    anchor = float(ev.get("anchor_score") or 0.0)
                    anchor_bonus = float(ev.get("anchor_bonus") or 0.0)
                    layer_bonus = float(ev.get("layer_bonus") or 0.0)
                    penalty = float(ev.get("penalty_score") or 0.0)

                    channel = str(ev.get("channel") or "").lower()
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

                    ev["confidence_components"] = comps
                    ev["confidence_formula"] = formula
                    ev["total_score"] = round(
                        max(0.0, comp_structured + comp_semantic + comp_context + comp_anchor + anchor_bonus + layer_bonus - penalty),
                        4,
                    )
                except Exception:
                    # 不影响主流程
                    pass

            # 额外：生成 vectorDebug.json，并回写 MD 可追溯的 hit_index
            try:
                vector_debug_payload = self._build_vector_debug_payload(report_data)
                self._attach_vector_debug_refs(report_data, vector_debug_payload)
                if vector_debug_payload.get("hits"):
                    dbg_path = output_dir / "vectorDebug.json"
                    with open(dbg_path, 'w', encoding='utf-8') as df:
                        json.dump(vector_debug_payload, df, ensure_ascii=False, indent=2)
                    summary = vector_debug_payload.get("summary") or {}
                    log(
                        "readability_enhancement_agent",
                        LogLevel.INFO,
                        (
                            f"💾 已写入向量调试文件: {dbg_path} "
                            f"(validation={summary.get('validation_from_consolidated_count', 0)}, "
                            f"gap={summary.get('gap_from_original_analysis_count', 0)}, "
                            f"effective={summary.get('produces_valid_output_total', 0)})"
                        ),
                    )
            except Exception as e:
                log("readability_enhancement_agent", LogLevel.WARNING, f"⚠️ 写入 vectorDebug 失败: {e}")

            # 生成Markdown摘要
            markdown_content = self._generate_markdown_summary(report_data, category)
            
            # 生成输出文件名
            base_name = json_file.stem  # 去掉.json
            output_file = output_dir / f"{base_name}.md"
            
            # 保存Markdown文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            log("readability_enhancement_agent", LogLevel.INFO, f"✅ 已保存: {output_file}")
            return True
            
        except Exception as e:
            log("readability_enhancement_agent", LogLevel.ERROR, f"❌ 处理文件 {json_file.name} 失败: {e}")
            return False

    def _build_vector_debug_payload(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建双查询命中调试载荷，区分一轮验判与二轮原始输入查漏。"""
        channel_meta = {
            "validation_from_consolidated": {
                "query_pass": "validation_from_consolidated",
                "query_pass_label": "一轮LLM/consolidated分析命中数据库",
                "source_field": "retrieval_evidence",
            },
            "gap_from_original_analysis": {
                "query_pass": "gap_from_original_analysis",
                "query_pass_label": "二轮原始源代码分片命中数据库",
                "source_field": "gap_retrieval_evidence",
            },
        }

        # 最终报告中实际引用的 sqlite_id → 输出问题摘要
        output_refs = self._collect_report_output_sqlite_refs(report_data)

        hits: List[Dict[str, Any]] = []
        validation_count = 0
        gap_count = 0
        validation_matched = 0
        gap_matched = 0
        validation_effective = 0
        gap_effective = 0

        def append_hits(items: Any, default_channel: str):
            nonlocal validation_count, gap_count, validation_matched, gap_matched
            nonlocal validation_effective, gap_effective
            if not isinstance(items, list):
                return
            for item in items:
                if not isinstance(item, dict):
                    continue
                channel = str(item.get("query_channel") or default_channel).strip() or default_channel
                if channel not in channel_meta:
                    channel = default_channel
                label = item.get("query_pass_label") or channel_meta[channel]["query_pass_label"]
                weaviate_hits = item.get("weaviate_hits") or []
                evidence_hits = item.get("evidence_hits") or []
                code_chunk = item.get("code_chunk") if isinstance(item.get("code_chunk"), dict) else {}
                evidence_by_sqlite = {
                    eh.get("sqlite_id"): eh
                    for eh in evidence_hits
                    if isinstance(eh, dict) and eh.get("sqlite_id") is not None
                }

                for hit in weaviate_hits:
                    if not isinstance(hit, dict):
                        continue
                    sqlite_id = hit.get("sqlite_id")
                    matched = sqlite_id in evidence_by_sqlite
                    gated = evidence_by_sqlite.get(sqlite_id) if matched else {}
                    linked_outputs = list(output_refs.get(sqlite_id) or [])
                    produces_valid_output = bool(linked_outputs)
                    hit_index = len(hits)
                    debug_item = {
                        "hit_index": hit_index,
                        "query_pass": channel,
                        "query_pass_label": label,
                        "issue_description": item.get("issue_description"),
                        "issue_file": item.get("issue_file"),
                        "issue_line": item.get("issue_file"),
                        "sqlite_id": sqlite_id,
                        "vector_layer": hit.get("vector_layer"),
                        "distance": hit.get("distance"),
                        "similarity": hit.get("similarity"),
                        "error_type": hit.get("error_type"),
                        "severity": hit.get("severity"),
                        # 是否进入门控后的正式证据池
                        "matched_as_evidence": matched,
                        "gating_decision": gated.get("gating_decision") if matched else None,
                        # 是否真正出现在最终二次分析输出问题中
                        "produces_valid_output": produces_valid_output,
                        "output_issue_count": len(linked_outputs),
                        "output_issues": [
                            {**dict(item), "hit_index": hit_index}
                            for item in linked_outputs[:5]
                        ],
                    }
                    if channel == "gap_from_original_analysis":
                        debug_item["code_chunk_start_line"] = code_chunk.get("start_line")
                        debug_item["code_chunk_end_line"] = code_chunk.get("end_line")
                        debug_item["code_chunk_index"] = code_chunk.get("chunk_index")
                        text = str(code_chunk.get("text") or "")
                        debug_item["code_chunk_preview"] = text[:200]
                    hits.append(debug_item)
                    if channel == "gap_from_original_analysis":
                        gap_count += 1
                        if matched:
                            gap_matched += 1
                        if produces_valid_output:
                            gap_effective += 1
                    else:
                        validation_count += 1
                        if matched:
                            validation_matched += 1
                        if produces_valid_output:
                            validation_effective += 1

        append_hits(report_data.get("retrieval_evidence") or [], "validation_from_consolidated")
        append_hits(report_data.get("gap_retrieval_evidence") or [], "gap_from_original_analysis")

        return {
            "schema_version": 3,
            "description": (
                "双查询数据库命中调试。"
                "validation_from_consolidated=一轮LLM/consolidated验判；"
                "gap_from_original_analysis=二轮原始源代码分片查漏。"
                "matched_as_evidence=进入正式证据池；"
                "produces_valid_output=该节点对应 sqlite_id 出现在最终报告问题中。"
            ),
            "summary": {
                "validation_from_consolidated_count": validation_count,
                "gap_from_original_analysis_count": gap_count,
                "matched_as_evidence_validation": validation_matched,
                "matched_as_evidence_gap": gap_matched,
                "produces_valid_output_validation": validation_effective,
                "produces_valid_output_gap": gap_effective,
                "produces_valid_output_total": validation_effective + gap_effective,
                "total_hit_count": len(hits),
            },
            "effective_hit_indexes": [h["hit_index"] for h in hits if h.get("produces_valid_output")],
            "hits": hits,
        }

    def _collect_report_output_sqlite_refs(
        self,
        report_data: Dict[str, Any],
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """收集最终报告问题中引用的 sqlite_id，用于判定 vectorDebug 节点是否产生有效输出。"""
        refs: Dict[Any, List[Dict[str, Any]]] = {}

        def add_ref(sqlite_id: Any, issue: Dict[str, Any], origin: str):
            if sqlite_id is None:
                return
            bucket = refs.setdefault(sqlite_id, [])
            summary = {
                "origin": origin,
                "source": issue.get("source"),
                "severity": issue.get("severity"),
                "line": issue.get("line"),
                "description": str(issue.get("description") or "")[:160],
            }
            # 去重：同 origin+line+description 不重复记
            key = (summary["origin"], summary["line"], summary["description"])
            if any(
                (b.get("origin"), b.get("line"), b.get("description")) == key
                for b in bucket
            ):
                return
            bucket.append(summary)

        for issue in report_data.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            ev = issue.get("second_pass_evidence") or issue.get("evidence")
            if isinstance(ev, dict):
                add_ref(ev.get("sqlite_id"), issue, "issues")

        for finding in report_data.get("new_findings") or []:
            if not isinstance(finding, dict):
                continue
            ev = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
            add_ref(ev.get("sqlite_id"), finding, "new_findings")

        return refs

    def _hydrate_evidence_scores_from_retrieval(
        self,
        report_data: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> None:
        """从 retrieval/gap 候选回填完整打分字段，修复旧报告只保留 total/semantic 的问题。"""
        if evidence.get("confidence_components"):
            return
        sqlite_id = evidence.get("sqlite_id")
        if sqlite_id is None:
            return

        best = None
        for bucket_name in ("gap_retrieval_evidence", "retrieval_evidence"):
            for item in report_data.get(bucket_name) or []:
                if not isinstance(item, dict):
                    continue
                for cand in item.get("candidates") or []:
                    if not isinstance(cand, dict):
                        continue
                    if cand.get("sqlite_id") != sqlite_id:
                        continue
                    if best is None or float(cand.get("total_score") or 0) >= float(best.get("total_score") or 0):
                        best = cand

        if not best:
            return

        for key in (
            "semantic_score",
            "structured_score",
            "context_score",
            "anchor_score",
            "anchor_bonus",
            "layer_bonus",
            "penalty_score",
            "confidence_components",
            "confidence_formula",
            "vector_layer",
            "matched_layers",
            "matched_layer_details",
            "matched_fields",
            "error_description",
            "class_pattern",
            "file_pattern",
            "reasoning",
        ):
            if evidence.get(key) in (None, "", [], {}) and best.get(key) is not None:
                evidence[key] = best.get(key)
        if best.get("total_score") is not None:
            evidence["total_score"] = best.get("total_score")
        if not evidence.get("channel") and best.get("channel"):
            evidence["channel"] = best.get("channel")

    def _total_score_from_components(
        self,
        comps: Dict[str, Any],
        channel: Any = None,
    ) -> float:
        """按分项原始分与权重重算总分，保证与公式展示一致。"""
        weights = self._confidence_formula_weights(channel)
        structured = float(comps.get("structured") or 0.0)
        semantic = float(comps.get("semantic") or 0.0)
        context = float(comps.get("context") or 0.0)
        anchor = float(comps.get("anchor") or 0.0)
        anchor_bonus = float(comps.get("anchor_bonus") or 0.0)
        layer_bonus = float(comps.get("layer_bonus") or 0.0)
        penalty = float(comps.get("penalty") or 0.0)
        total = (
            structured * weights.get("structured", 0.0)
            + semantic * weights.get("semantic", 0.0)
            + context * weights.get("context", 0.0)
            + anchor * weights.get("anchor", 0.0)
            + anchor_bonus
            + layer_bonus
            - penalty
        )
        return round(max(0.0, total), 4)

    def _parse_history_hit_error_type(self, description: Any) -> Optional[str]:
        text = str(description or "").strip()
        prefix = "历史知识命中:"
        if prefix not in text:
            return None
        return text.split(prefix, 1)[1].strip().split()[0].strip(" ,;，；") or None

    def _attach_vector_debug_refs(
        self,
        report_data: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> None:
        """把 vectorDebug hit_index 挂到 issue.evidence，供 MD 追溯。"""
        hits = payload.get("hits") if isinstance(payload, dict) else None
        if not isinstance(hits, list) or not hits:
            return

        effective_hits = [
            h for h in hits
            if isinstance(h, dict) and h.get("produces_valid_output") and h.get("sqlite_id") is not None
        ]

        def pick_refs(issue: Dict[str, Any], evidence: Dict[str, Any]) -> None:
            sqlite_id = evidence.get("sqlite_id")
            if sqlite_id is None:
                return
            error_type = self._parse_history_hit_error_type(issue.get("description"))
            candidates = [h for h in effective_hits if h.get("sqlite_id") == sqlite_id]
            if error_type:
                typed = [
                    h for h in candidates
                    if str(h.get("error_type") or "").strip().lower() == error_type.lower()
                ]
                if typed:
                    candidates = typed

            def sort_key(h: Dict[str, Any]):
                gap_first = 0 if h.get("query_pass") == "gap_from_original_analysis" else 1
                return (gap_first, int(h.get("hit_index") or 0))

            candidates = sorted(candidates, key=sort_key)
            if not candidates:
                return

            indexes = [int(h["hit_index"]) for h in candidates if h.get("hit_index") is not None][:10]
            primary = candidates[0]
            evidence["vector_debug_hit_index"] = primary.get("hit_index")
            evidence["vector_debug_hit_indexes"] = indexes
            evidence["vector_debug_query_pass"] = primary.get("query_pass")

        for issue in report_data.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            ev = issue.get("second_pass_evidence") or issue.get("evidence")
            if isinstance(ev, dict):
                pick_refs(issue, ev)

        for finding in report_data.get("new_findings") or []:
            if not isinstance(finding, dict):
                continue
            ev = finding.get("evidence")
            if isinstance(ev, dict):
                pick_refs(finding, ev)

    def _confidence_formula_weights(self, channel: Any) -> Dict[str, float]:
        if str(channel or "").strip().lower() == "curated_issue":
            return {
                "structured": 0.55,
                "semantic": 0.0,
                "context": 0.15,
                "anchor": 0.2,
            }
        return {
            "structured": 0.5,
            "semantic": 0.35,
            "context": 0.1,
            "anchor": 0.05,
        }

    def _format_confidence_formula_lines(
        self,
        comps: Dict[str, Any],
        channel: Any,
        indent: str = "      ",
    ) -> List[str]:
        """把分项得分内嵌进公式行。"""
        weights = self._confidence_formula_weights(channel)
        structured = comps.get("structured", 0.0)
        semantic = comps.get("semantic", 0.0)
        context = comps.get("context", 0.0)
        anchor = comps.get("anchor", 0.0)
        anchor_bonus = comps.get("anchor_bonus", 0.0)
        layer_bonus = comps.get("layer_bonus", 0.0)
        penalty = comps.get("penalty", 0.0)

        lines = [
            f"{indent}total",
            f"{indent}= structured({structured})*{weights['structured']}",
        ]
        if weights.get("semantic", 0.0) > 0:
            lines.append(f"{indent}+ semantic({semantic})*{weights['semantic']}")
        lines.append(f"{indent}+ context({context})*{weights['context']}")
        lines.append(f"{indent}+ anchor({anchor})*{weights['anchor']}")
        lines.append(f"{indent}+ anchor_bonus({anchor_bonus})")
        lines.append(f"{indent}+ layer_bonus({layer_bonus})")
        lines.append(f"{indent}- penalty({penalty})")
        return lines

    def _render_evidence_item(self, issue: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        description = issue.get("description", "No description")
        line_num = issue.get("line")
        line_info = f"第 {line_num} 行" if line_num else "未定位行号"
        channel = evidence.get("channel", "unknown")
        primary_channel = evidence.get("primary_channel") or (
            "curated"
            if str(channel).lower() == "curated_issue"
            else ("weaviate" if str(channel).lower() == "weaviate" else channel)
        )
        score = evidence.get("total_score", "")
        matched_fields = ", ".join(evidence.get("matched_fields", []) or [])
        reasoning = evidence.get("reasoning") or ""
        rejection = evidence.get("rejection_reason") or ""
        solution = evidence.get("solution") or evidence.get("recommended_solution") or ""

        parts = [
            f"- **问题**: {description}",
            f"  - 位置: {line_info}",
            f"  - 命中通道: {channel}",
            f"  - 主通道(primary_channel): {primary_channel}",
        ]
        if primary_channel == "curated":
            parts.append("  - 分层说明: curated 通道不受 fullLayer/多层影响；r1≈r2 属预期")
        if score != "":
            parts.append(f"  - 命中评分: {score}")
            comps = evidence.get("confidence_components") or {}
            if comps:
                parts.append("  - 置信度计算公式:")
                parts.extend(self._format_confidence_formula_lines(comps, channel, indent="      "))

        hit_index = evidence.get("vector_debug_hit_index")
        query_pass = evidence.get("vector_debug_query_pass")
        sqlite_id = evidence.get("sqlite_id")
        if hit_index is not None:
            pass_text = query_pass or "unknown"
            sid_text = f", sqlite_id={sqlite_id}" if sqlite_id is not None else ""
            parts.append(f"  - vectorDebug节点: #{hit_index} ({pass_text}{sid_text})")
            related = evidence.get("vector_debug_hit_indexes") or []
            if isinstance(related, list) and related:
                related_text = ", ".join(f"#{idx}" for idx in related)
                parts.append(f"  - 相关节点: {related_text}")

        if matched_fields:
            parts.append(f"  - 结构化命中字段: {matched_fields}")
        matched_layer_details = evidence.get("matched_layer_details") or []
        if isinstance(matched_layer_details, list) and matched_layer_details:
            layer_text = ", ".join(
                f"{d.get('layer')}(+{d.get('bonus')})"
                for d in matched_layer_details
                if isinstance(d, dict) and d.get("layer") is not None
            )
            if layer_text:
                parts.append(f"  - 命中分层: {layer_text}")
        elif evidence.get("matched_layers"):
            layers = evidence.get("matched_layers") or []
            if isinstance(layers, list) and layers:
                parts.append(f"  - 命中分层: {', '.join(str(x) for x in layers)}")
        elif primary_channel == "curated":
            parts.append("  - 命中分层: N/A（curated）")
        error_description = str(evidence.get("error_description") or "").strip()
        if error_description:
            short_desc = (
                error_description
                if len(error_description) <= 220
                else error_description[:217] + "..."
            )
            parts.append(f"  - 知识摘要: {short_desc}")
        class_pattern = str(evidence.get("class_pattern") or "").strip()
        if class_pattern:
            parts.append(f"  - 锚定函数: {class_pattern}")
        if reasoning:
            parts.append(f"  - 命中原因: {reasoning}")
        if solution:
            parts.append(f"  - 建议动作: {solution}")
        if rejection:
            parts.append(f"  - 未采纳原因: {rejection}")

        return "\n".join(parts) + "\n"

    def _render_candidate_item(self, index: int, candidate: Dict[str, Any]) -> str:
        error_type = candidate.get("error_type") or "unknown"
        channel = candidate.get("channel", "unknown")
        score = candidate.get("total_score", "")
        decision = candidate.get("gating_decision", "")
        rejection = candidate.get("rejection_reason", "")
        matched_fields = ", ".join(candidate.get("matched_fields", []) or [])
        summary = candidate.get("issue_summary", "")

        parts = [
            f"{index}. {error_type}",
            f"  - 通道: {channel}",
        ]
        if score != "":
            parts.append(f"  - 评分: {score}")
            comps = candidate.get("confidence_components") or {}
            if comps:
                parts.append("  - 置信度计算公式:")
                parts.extend(self._format_confidence_formula_lines(comps, channel, indent="      "))
        if matched_fields:
            parts.append(f"  - 命中字段: {matched_fields}")
        if decision:
            parts.append(f"  - 门控结果: {decision}")
        if rejection:
            parts.append(f"  - 未采纳原因: {rejection}")
        if summary:
            parts.append(f"  - 当前问题摘要: {summary}")

        return "\n".join(parts) + "\n"
    
    def _group_issues_by_severity(self, issues: List[Dict]) -> Dict[str, List[Dict]]:
        """按严重程度分组问题"""
        grouped = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }
        
        for issue in issues:
            severity = issue.get("severity", "low")
            if severity in grouped:
                grouped[severity].append(issue)
        
        return grouped
    
    def _group_issues_by_source(self, issues: List[Dict]) -> Dict[str, List[Dict]]:
        """按来源分组问题"""
        grouped = {}
        
        for issue in issues:
            source = issue.get("source", "unknown")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(issue)
        
        return grouped
    
    def _generate_markdown_summary(self, report_data: Dict[str, Any], category: str) -> str:
        """
        生成Markdown格式的报告摘要
        
        Args:
            report_data: 报告数据
            category: 文件类别
        
        Returns:
            Markdown格式的文本
        """
        lines = []
        
        # 标题
        file_name = report_data.get("file", report_data.get("readable_file", "Unknown File"))
        lines.append(f"# 代码分析报告 - {file_name}\n")
        
        # 基本信息
        lines.append("## 📋 基本信息\n")
        
        requirement_id = report_data.get("requirement_id")
        run_id = report_data.get("run_id")
        
        if requirement_id:
            lines.append(f"- **需求ID**: {requirement_id}")
        if run_id:
            lines.append(f"- **运行ID**: {run_id}")
        
        status = report_data.get("status", "unknown")
        lines.append(f"- **状态**: {status}")
        
        analysis_types = report_data.get("analysis_types", [])
        if analysis_types:
            types_str = ", ".join(analysis_types)
            lines.append(f"- **分析类型**: {types_str}")
        
        lines.append("")

        # 统计信息
        issue_count = report_data.get("issue_count", 0)
        severity_stats = report_data.get("severity_stats", {})
        
        lines.append("## 📊 问题统计\n")
        lines.append(f"**总问题数**: {issue_count}\n")
        
        if severity_stats:
            lines.append("### 严重程度分布\n")
            for severity in ["critical", "high", "medium", "low", "info"]:
                count = severity_stats.get(severity, 0)
                if count > 0:
                    severity_cn = self._translate_severity(severity)
                    lines.append(f"- {severity_cn}: {count}")
            lines.append("")
        
        # 问题详情
        issues = report_data.get("issues", [])
        if issues:
            lines.append("## 🔍 问题详情\n")
            
            # 按严重程度分组
            grouped_issues = self._group_issues_by_severity(issues)
            
            for severity in ["critical", "high", "medium", "low", "info"]:
                severity_issues = grouped_issues.get(severity, [])
                if not severity_issues:
                    continue
                
                severity_cn = self._translate_severity(severity)
                lines.append(f"### {severity_cn}问题 ({len(severity_issues)}个)\n")
                
                # 按source分类
                by_source = self._group_issues_by_source(severity_issues)
                
                for source in sorted(by_source.keys()):
                    source_issues = by_source[source]
                    source_cn = self._translate_source(source)
                    
                    lines.append(f"#### {source_cn}\n")
                    
                    for idx, issue in enumerate(source_issues[:5], 1):  # 每个来源最多显示5个
                        description = issue.get("description", "No description")
                        line_num = issue.get("line")

                        if line_num:
                            lines.append(f"{idx}. **第 {line_num} 行**: {description}")
                        else:
                            lines.append(f"{idx}. {description}")

                        # 如果二次分析提供了证据（strong hit），在问题下直接展示证据摘要与代码片段
                        ev = issue.get("second_pass_evidence") or issue.get("evidence")
                        if isinstance(ev, dict):
                            lines.append("")
                            lines.append(self._render_evidence_item(issue, ev))

                    if len(source_issues) > 5:
                        lines.append(f"   ... 还有 {len(source_issues) - 5} 个问题\n")
                    else:
                        lines.append("")

        # 改进建议
        lines.append("## 💡 改进建议\n")
        
        if issue_count > 0:
            critical_count = severity_stats.get("critical", 0)
            high_count = severity_stats.get("high", 0)
            medium_count = severity_stats.get("medium", 0)
            low_count = severity_stats.get("low", 0)
            info_count = severity_stats.get("info", 0)
            
            if critical_count > 0:
                lines.append(f"### 🚨 立即处理\n")
                lines.append(f"- 检测到 {critical_count} 个严重问题，需要优先修复")
                lines.append(f"- 建议立即进行影响分析和修复规划\n")
            
            if high_count > 0:
                lines.append(f"### 🔴 高优先级\n")
                lines.append(f"- 检测到 {high_count} 个高级问题")
                lines.append(f"- 建议在本轮迭代中完成修复\n")
            
            if medium_count > 0:
                lines.append(f"### 🟡 中优先级\n")
                lines.append(f"- 检测到 {medium_count} 个中等问题")
                lines.append(f"- 建议在下一个周期内逐步改进\n")
            
            if low_count > 0:
                lines.append(f"### 🟢 低优先级\n")
                lines.append(f"- 检测到 {low_count} 个低级问题")
                lines.append(f"- 建议在代码维护中持续改进\n")
            if info_count > 0:
                lines.append(f"### 🔵 提示/观察\n")
                lines.append(f"- 检测到 {info_count} 个提示类问题")
                lines.append(f"- 建议作为优化线索持续跟踪\n")
        else:
            lines.append("✅ 未检测到问题，代码质量良好！\n")
        
        # 工作量估计
        estimated_effort = self._estimate_effort(issue_count)
        lines.append("## ⏱️ 工作量估计\n")
        lines.append(f"**预计修复工作量**: {estimated_effort}\n")
        
        # 分析详情（仅consolidated类型）
        if category == "consolidated" and "analysis_types" in report_data:
            lines.append("## 📈 分析详情\n")
            
            for analysis_type in analysis_types:
                type_cn = self._translate_analysis_type(analysis_type)
                count = self._count_issues_by_type(issues, analysis_type)
                if count > 0:
                    lines.append(f"- **{type_cn}**: {count} 个问题")
            
            lines.append("")
        
        # 页脚
        lines.append("---\n")
        lines.append(f"*本报告由AI可读性增强代理自动生成 | 生成时间: {self._get_current_time()}*\n")
        
        return "\n".join(lines)

    def _count_issues_by_type(self, issues: List[Dict], analysis_type: str) -> int:
        """计算某种分析类型的问题数量"""
        # 根据issue的source字段推断分析类型
        type_mapping = {
            "security_analysis": ["security_vulnerability", "security_risk"],
            "performance_analysis": ["performance_bottleneck"],
            "static_analysis": ["style", "quality"],
            "ai_analysis": []
        }
        
        sources = type_mapping.get(analysis_type, [])
        count = 0
        
        for issue in issues:
            source = issue.get("source", "")
            if source in sources:
                count += 1
        
        return count
    
    def _translate_severity(self, severity: str) -> str:
        """翻译严重程度"""
        mapping = {
            "critical": "🚨 严重",
            "high": "🔴 高",
            "medium": "🟡 中",
            "low": "🟢 低",
            "info": "🔵 提示"
        }
        return mapping.get(severity, severity)
    
    def _translate_source(self, source: str) -> str:
        """翻译问题来源"""
        mapping = {
            "style": "代码风格",
            "quality": "代码质量",
            "security": "安全问题",
            "security_vulnerability": "安全漏洞",
            "security_risk": "安全风险",
            "performance": "性能问题",
            "performance_bottleneck": "性能瓶颈",
            "complexity": "复杂度问题"
        }
        return mapping.get(source, source)
    
    def _translate_analysis_type(self, analysis_type: str) -> str:
        """翻译分析类型"""
        mapping = {
            "security_analysis": "安全分析",
            "performance_analysis": "性能分析",
            "static_analysis": "静态分析",
            "ai_analysis": "AI分析"
        }
        return mapping.get(analysis_type, analysis_type)
    
    def _estimate_effort(self, issue_count: int) -> str:
        """估计修复工作量"""
        if issue_count == 0:
            return "无"
        elif issue_count < 10:
            return "低 (~0.5-1天)"
        elif issue_count < 30:
            return "中 (~2-3天)"
        elif issue_count < 60:
            return "高 (~1周)"
        else:
            return "非常高 (>1周)"
    
    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务的实现（BaseAgent抽象方法）
        
        对于可读性增强代理，任务通过消息处理而不是直接任务执行
        此方法提供备用的同步任务接口
        """
        try:
            if isinstance(task_data, dict) and "run_id" in task_data:
                run_id = task_data.get("run_id")
                result = await self.enhance_run_reports(run_id)
                return {
                    "status": "success" if result else "failed",
                    "run_id": run_id,
                    "message": f"可读性增强完成" if result else "可读性增强失败"
                }
            else:
                return {
                    "status": "error",
                    "message": "任务数据格式错误，需要包含run_id"
                }
        except Exception as e:
            log("readability_enhancement_agent", LogLevel.ERROR, f"任务执行失败: {e}")
            return {
                "status": "error",
                "message": f"任务执行失败: {str(e)}"
            }
