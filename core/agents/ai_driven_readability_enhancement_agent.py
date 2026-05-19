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
        """处理接收到的消息"""
        if message.message_type == "analyze_consolidated_report":
            # 处理来自SummaryAgent的转发消息
            try:
                run_id = message.content.get("run_id")
                requirement_id = message.content.get("requirement_id")
                
                # 扫描该run_id下的所有报告并进行可读性增强
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
            
            # 创建输出目录结构：仅保留汇总层的可读性增强结果
            enhancement_dir = run_dir / "readability_enhancement"
            consolidated_dir = enhancement_dir / "consolidated"
            
            consolidated_dir.mkdir(parents=True, exist_ok=True)
            
            log("readability_enhancement_agent", LogLevel.INFO, f"📁 已创建输出目录: {enhancement_dir}")
            
            # 处理consolidated目录下的JSON文件
            consolidated_source_dir = run_dir / "consolidated"
            if consolidated_source_dir.exists():
                for json_file in consolidated_source_dir.glob("*.json"):
                    await self._enhance_single_report(json_file, consolidated_dir, "consolidated")
            
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
            for severity in ["critical", "high", "medium", "low"]:
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
            
            for severity in ["critical", "high", "medium", "low"]:
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

    def _render_evidence_item(self, issue: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        description = issue.get("description", "No description")
        line_num = issue.get("line")
        line_info = f"第 {line_num} 行" if line_num else "未定位行号"
        channel = evidence.get("channel", "unknown")
        score = evidence.get("total_score", "")
        matched_fields = ", ".join(evidence.get("matched_fields", []) or [])
        reasoning = evidence.get("reasoning") or ""
        rejection = evidence.get("rejection_reason") or ""
        solution = evidence.get("solution") or evidence.get("recommended_solution") or ""

        parts = [
            f"- **问题**: {description}",
            f"  - 位置: {line_info}",
            f"  - 命中通道: {channel}",
        ]
        if score != "":
            parts.append(f"  - 命中评分: {score}")
        if matched_fields:
            parts.append(f"  - 结构化命中字段: {matched_fields}")
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
            "low": []
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
            "low": "🟢 低"
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
