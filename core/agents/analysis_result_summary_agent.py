import pandas as pd
from datetime import datetime
from typing import Dict, Any
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.reports import report_manager

class SummaryAgent(BaseAgent):
    def __init__(self):
        super().__init__("summary_agent", "结果汇总输出智能体")
        self.db_service = DatabaseService()
        self.analysis_results = {}
        self.run_meta = {}  # run_id -> {'expected': set(ids), 'completed': set(ids), 'issues': []}
    
    async def handle_message(self, message: Message):
        if message.message_type == 'run_init':
            run_id = message.content.get('run_id')
            req_ids = set(message.content.get('requirement_ids', []))
            if run_id:
                self.run_meta[run_id] = {'expected': req_ids, 'completed': set(), 'issues': [], 'target_directory': message.content.get('target_directory')}
            return
        """处理分析结果"""
        if message.message_type == "analysis_result":
            run_id = message.content.get('run_id')
            requirement_id = message.content.get("requirement_id")
            analysis_type = message.content.get("analysis_type") or message.content.get("agent_type")
            result = message.content.get("result") or message.content.get("results")
            file_path = message.content.get("file_path")
            if requirement_id not in self.analysis_results:
                self.analysis_results[requirement_id] = {"files": set(), "types": set(), "data": {}, "file_path": file_path, "run_id": run_id}
            record = self.analysis_results[requirement_id]
            if run_id:
                record['run_id'] = run_id
            if file_path:
                record["file_path"] = file_path
            record["types"].add(analysis_type)
            record["data"][analysis_type] = result
            record["files"].add(file_path)
            await self._try_generate_consolidated_report(requirement_id)

    async def _try_generate_consolidated_report(self, requirement_id: int):
        record = self.analysis_results.get(requirement_id)
        if not record:
            return
        collected = record["types"]
        run_id = None
        for res in record['data'].values():
            if isinstance(res, dict):
                candidate = res.get('run_id') or res.get('analysis_run_id')
                if candidate:
                    run_id = candidate
                    break
        # 放宽条件: 只要收到 static_analysis 就生成初步报告, 若后续再补充AI结果可再生成升级版(此处简单跳过二次)
        if "static_analysis" in collected:
            await self._generate_consolidated_report(requirement_id, record)
            if run_id and run_id in self.run_meta:
                static_data = record['data']
                issues_local = []
                static_res = static_data.get("static_analysis", {})
                for q in static_res.get("quality_issues", [])[:50]:
                    issues_local.append({'source': 'quality', 'severity': q.get('severity','low')})
                for s in static_res.get("security_issues", [])[:50]:
                    issues_local.append({'source': 'security', 'severity': s.get('severity','low')})
                for t in static_res.get("type_issues", [])[:50]:
                    issues_local.append({'source': 'type', 'severity': t.get('severity','low')})
                for st in static_res.get("style_issues", [])[:30]:
                    issues_local.append({'source': 'style', 'severity': st.get('severity','low')})
                self.run_meta[run_id]['issues'].extend(issues_local)
                self.run_meta[run_id]['completed'].add(requirement_id)
                await self._maybe_finalize_run(run_id)
            if requirement_id in self.analysis_results:
                del self.analysis_results[requirement_id]

    async def _generate_consolidated_report(self, requirement_id: int, record):
        data = record["data"]
        file_path = record.get("file_path")
        run_id = record.get('run_id')
        issues = []
        def add_issue(src, description, severity="low", line=None, tool=None):
            issues.append({
                "requirement_id": requirement_id,
                "file": file_path,
                "source": src,
                "severity": severity,
                "line": line,
                "description": description,
                "tool": tool,
                "run_id": run_id
            })
        # static issues
        static_res = data.get("static_analysis", {})
        for q in static_res.get("quality_issues", [])[:50]:
            add_issue("quality", q.get("message"), q.get("severity"), q.get("line"), q.get("tool"))
        for s in static_res.get("security_issues", [])[:50]:
            add_issue("security", s.get("message"), s.get("severity"), s.get("line"), s.get("tool"))
        for t in static_res.get("type_issues", [])[:50]:
            add_issue("type", t.get("message"), t.get("severity"), t.get("line"), t.get("tool"))
        for st in static_res.get("style_issues", [])[:30]:
            add_issue("style", st.get("message"), st.get("severity"), st.get("line"), st.get("tool"))
        # ai quality
        ai_res = data.get("ai_analysis", {})
        final_report = ai_res.get("final_report", {})
        for fix in final_report.get("recommendations", {}).get("immediate_fixes", [])[:20]:
            add_issue("ai_quality_fix", fix.get("description"), fix.get("severity", "high"))
        for enh in final_report.get("recommendations", {}).get("quality_enhancements", [])[:20]:
            add_issue("ai_quality_enhancement", enh.get("description"), enh.get("priority", "medium"))
        # security
        sec_res = data.get("security_analysis", {}).get("ai_security_analysis", {})
        for vuln in sec_res.get("vulnerabilities_detected", [])[:30]:
            add_issue("security_ai", vuln.get("description"), vuln.get("severity"))
        # performance
        perf_res = data.get("performance_analysis", {}).get("ai_performance_analysis", {})
        for bn in perf_res.get("performance_bottlenecks", [])[:30]:
            add_issue("performance_bottleneck", bn.get("description"), bn.get("severity"))
        # 汇总统计
        severity_stats = {}
        for it in issues:
            sev = it.get("severity", "low")
            severity_stats[sev] = severity_stats.get(sev, 0) + 1
        report_payload = {
            "status": "completed",
            "requirement_id": requirement_id,
            "file": file_path,
            "run_id": run_id,
            "issue_count": len(issues),
            "severity_stats": severity_stats,
            "issues": issues,
            "analysis_types": list(record.get("types", []))
        }
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"consolidated_req_{requirement_id}_{run_id or 'no_run'}_{ts}.json"
        report_path = report_manager.generate_analysis_report(report_payload, filename=filename)
        print(f"✅ 综合分析完成(ID={requirement_id}) -> {report_path} 发现问题: {len(issues)} 高危: {severity_stats.get('critical',0)+severity_stats.get('high',0)}")
        return
    
    async def _maybe_finalize_run(self, run_id: str):
        meta = self.run_meta.get(run_id)
        if not meta:
            return
        if meta['expected'] and meta['expected'] == meta['completed']:
            # aggregate severity stats
            severity_stats = {}
            for it in meta['issues']:
                sev = it.get('severity','low')
                severity_stats[sev] = severity_stats.get(sev,0)+1
            report_payload = {
                'status': 'run_completed',
                'run_id': run_id,
                'requirements_processed': len(meta['completed']),
                'issue_count': len(meta['issues']),
                'severity_stats': severity_stats,
                'target_directory': meta.get('target_directory')
            }
            from datetime import datetime
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'run_summary_{ts}_{run_id}.json'
            path = report_manager.generate_analysis_report(report_payload, filename=filename)
            print(f"✅ 运行级综合报告生成: {path} 总问题: {len(meta['issues'])}")
            del self.run_meta[run_id]
    
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行汇总任务"""
        return {"status": "summary_agent_ready"}
    
    def aggregate_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy sync aggregation interface used in tests.
        Expects a dict with keys like code_quality/security/performance each containing a score and issue lists.
        Returns unified summary with overall_score, summary_by_category, priority_issues, recommendations."""
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        summary_by_category = {}
        collected_scores = []
        priority_issues = []
        for category, data in analysis_results.items():
            score = data.get("score")
            if isinstance(score, (int, float)):
                collected_scores.append(score)
            # unify issues field names
            issues = []
            if 'issues' in data and isinstance(data['issues'], list):
                for it in data['issues']:
                    issues.append({
                        "type": it.get("type", category),
                        "severity": it.get("severity", "low"),
                        "count": it.get("count", 1),
                        "category": category
                    })
            if 'vulnerabilities' in data:
                for v in data['vulnerabilities']:
                    issues.append({
                        "type": v.get("type", "vulnerability"),
                        "severity": v.get("severity", "medium"),
                        "count": v.get("count", 1),
                        "category": category
                    })
            if 'bottlenecks' in data:
                for b in data['bottlenecks']:
                    issues.append({
                        "type": b.get("type", "bottleneck"),
                        "severity": b.get("severity", "medium"),
                        "count": b.get("count", 1),
                        "category": category
                    })
            summary_by_category[category] = {
                "score": score,
                "issue_count": sum(i.get("count", 1) for i in issues),
                "issues": issues
            }
            priority_issues.extend(issues)
        overall_score = round(sum(collected_scores)/len(collected_scores), 2) if collected_scores else 0
        priority_issues = self.prioritize_issues(priority_issues)
        recommendations = self._generate_basic_recommendations(summary_by_category)
        return {
            "overall_score": overall_score,
            "summary_by_category": summary_by_category,
            "priority_issues": priority_issues,
            "recommendations": recommendations
        }

    def prioritize_issues(self, issues):
        """Sort issues by severity critical>high>medium>low>info preserving relative order otherwise."""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        return sorted(issues, key=lambda x: severity_order.get(x.get("severity", "low"), 5))

    def generate_executive_summary(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Produce high-level executive summary from aggregated numeric data."""
        overall = aggregated_data.get('overall_score', 0)
        total_issues = aggregated_data.get('total_issues') or sum(
            v.get('issues', v.get('issue_count', 0)) if isinstance(v, dict) else 0
            for v in (aggregated_data.get('categories') or {}).values()
        )
        critical = aggregated_data.get('critical_issues', 0)
        high = aggregated_data.get('high_issues', 0)
        headline = f"Overall score {overall} with {total_issues} issues (critical:{critical} high:{high})"
        key_metrics = {
            "overall_score": overall,
            "issues_total": total_issues,
            "critical": critical,
            "high": high
        }
        main_concerns = []
        if critical > 0:
            main_concerns.append("存在关键高危问题")
        if overall < 70:
            main_concerns.append("整体质量需提升")
        success_areas = []
        if overall >= 80:
            success_areas.append("总体评分良好")
        return {
            "headline": headline,
            "key_metrics": key_metrics,
            "main_concerns": main_concerns,
            "success_areas": success_areas
        }

    def calculate_trends(self, historical_data):
        """Simple trend analysis comparing first and last entries."""
        if not historical_data:
            return {"score_trend": "stable", "issues_trend": "stable", "improvement_rate": 0}
        start = historical_data[0]
        end = historical_data[-1]
        score_trend = "improving" if end.get('overall_score',0) > start.get('overall_score',0) else ("declining" if end.get('overall_score',0) < start.get('overall_score',0) else "stable")
        issues_trend = "decreasing" if end.get('issues',0) < start.get('issues',0) else ("increasing" if end.get('issues',0) > start.get('issues',0) else "stable")
        improvement_rate = 0
        if start.get('overall_score'):
            improvement_rate = round((end.get('overall_score',0)-start.get('overall_score',0))/start.get('overall_score')*100,2)
        return {"score_trend": score_trend, "issues_trend": issues_trend, "improvement_rate": improvement_rate}

    def analyze_correlations(self, category_data: Dict[str, Any]):
        """Naive correlation insight placeholders between categories based on simple heuristics."""
        correlations = []
        if 'code_quality' in category_data and 'performance' in category_data:
            cq = category_data['code_quality']
            perf = category_data['performance']
            if cq.get('complexity_issues',0) > 0 and perf.get('memory_usage') == 'high':
                correlations.append({
                    "categories": ["code_quality", "performance"],
                    "relationship": "High complexity may contribute to memory usage",
                    "impact": "medium"
                })
        if 'code_quality' in category_data and 'security' in category_data:
            sec = category_data['security']
            if sec.get('vulnerabilities',0) > 0 and category_data['code_quality'].get('maintainability_score', 100) < 80:
                correlations.append({
                    "categories": ["code_quality", "security"],
                    "relationship": "Lower maintainability can hide security issues",
                    "impact": "high"
                })
        return correlations

    def generate_action_plan(self, prioritized_issues):
        """Split prioritized issues into immediate/short/long term buckets."""
        immediate = [i for i in prioritized_issues if i.get('severity') in ('critical','high')][:5]
        short = [i for i in prioritized_issues if i.get('severity') == 'medium'][:5]
        long_term = [i for i in prioritized_issues if i.get('severity') in ('low','info')][:5]
        timeline = {
            "immediate": "0-2 days",
            "short_term": "1-2 weeks",
            "long_term": "1-2 months"
        }
        return {
            "immediate_actions": immediate,
            "short_term_goals": short,
            "long_term_improvements": long_term,
            "timeline": timeline
        }

    def create_comprehensive_report(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        project = comprehensive_results.get('project_info', {})
        analysis = comprehensive_results.get('analysis_results', {})
        aggregated = {}
        for k,v in analysis.items():
            aggregated[k] = {
                "score": v.get('score'),
                "issues": v.get('issues') or v.get('vulnerabilities') or v.get('bottlenecks'),
            }
        overall_scores = [v.get('score') for v in analysis.values() if isinstance(v.get('score'), (int,float))]
        overall_score = round(sum(overall_scores)/len(overall_scores),2) if overall_scores else 0
        exec_summary = self.generate_executive_summary({
            'overall_score': overall_score,
            'total_issues': sum((v.get('issues') or v.get('vulnerabilities') or v.get('bottlenecks') or 0) if isinstance(v, dict) else 0 for v in analysis.values()),
            'critical_issues': 0,
            'high_issues': 0,
            'categories': aggregated
        })
        recommendations = ["修复高风险问题", "优化性能瓶颈", "提升代码可维护性"]
        next_steps = ["制定整改时间表", "跟踪指标变化", "二次审查"]
        return {
            "project_overview": project,
            "executive_summary": exec_summary,
            "detailed_analysis": analysis,
            "recommendations": recommendations,
            "next_steps": next_steps
        }

    def export_report(self, summary_data: Dict[str, Any], format: str = 'json') -> str:
        import json
        format = format.lower()
        if format == 'json':
            return json.dumps(summary_data, ensure_ascii=False, indent=2)
        if format == 'markdown':
            lines = ["# Summary Report"]
            for k,v in summary_data.items():
                if isinstance(v, (dict,list)):
                    lines.append(f"## {k}")
                    lines.append(f"`{str(v)[:500]}`")
                else:
                    lines.append(f"- **{k}**: {v}")
            return "\n".join(lines)
        if format == 'html':
            import html
            body = ''.join(f"<h2>{k}</h2><pre>{html.escape(str(v)[:1000])}</pre>" for k,v in summary_data.items())
            return f"<html><body><h1>Summary Report</h1>{body}</body></html>"
        return str(summary_data)

    def _generate_basic_recommendations(self, summary_by_category: Dict[str, Any]):
        recs = []
        for cat, val in summary_by_category.items():
            score = val.get('score') or 0
            if score < 75:
                recs.append(f"改进 {cat} 以提升得分")
        if not recs:
            recs.append("维持当前质量并持续监控")
        return recs