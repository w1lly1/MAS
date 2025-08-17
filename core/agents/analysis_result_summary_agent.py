import pandas as pd
from datetime import datetime
from typing import Dict, Any
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService

class SummaryAgent(BaseAgent):
    def __init__(self):
        super().__init__("summary_agent", "结果汇总输出智能体")
        self.db_service = DatabaseService()
        self.analysis_results = {}
        
    async def handle_message(self, message: Message):
        """处理分析结果"""
        if message.message_type == "analysis_result":
            requirement_id = message.content.get("requirement_id")
            analysis_type = message.content.get("analysis_type")
            result = message.content.get("result")
            
            # 收集分析结果
            if requirement_id not in self.analysis_results:
                self.analysis_results[requirement_id] = {}
                
            self.analysis_results[requirement_id][analysis_type] = result
            
            # 检查是否收齐所有分析结果
            await self._check_and_generate_report(requirement_id)
            
    async def _check_and_generate_report(self, requirement_id: int):
        """检查是否可以生成报告"""
        results = self.analysis_results.get(requirement_id, {})
        
        # 检查是否有静态分析和AI分析结果
        if "static_analysis" in results and "ai_analysis" in results:
            await self._generate_excel_report(requirement_id, results)
            
    async def _generate_excel_report(self, requirement_id: int, results: Dict[str, Any]):
        """生成Excel报告"""
        try:
            filename = f"code_analysis_report_{requirement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 汇总页
                await self._create_summary_sheet(writer, requirement_id, results)
                
                # 静态分析详情页
                if "static_analysis" in results:
                    await self._create_static_analysis_sheet(writer, results["static_analysis"])
                    
                # AI分析详情页
                if "ai_analysis" in results:
                    await self._create_ai_analysis_sheet(writer, results["ai_analysis"])
                    
            print(f"✅ 分析报告已生成: {filename}")
            
            # 清理已处理的结果
            if requirement_id in self.analysis_results:
                del self.analysis_results[requirement_id]
                
        except Exception as e:
            print(f"Failed to generate Excel report: {e}")
            
    async def _create_summary_sheet(self, writer, requirement_id: int, results: Dict[str, Any]):
        """创建汇总页"""
        summary_data = {
            "项目": ["分析ID", "分析时间", "总体评分", "关键问题数", "安全问题数", "建议数"],
            "结果": [
                requirement_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                results.get("ai_analysis", {}).get("code_quality_score", "N/A"),
                results.get("static_analysis", {}).get("summary", {}).get("critical_issues", 0),
                results.get("static_analysis", {}).get("summary", {}).get("security_issues", 0),
                len(results.get("ai_analysis", {}).get("recommendations", []))
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='汇总', index=False)
        
    async def _create_static_analysis_sheet(self, writer, static_results: Dict[str, Any]):
        """创建静态分析详情页"""
        issues_data = []
        
        # 处理Pylint结果
        if "tools_results" in static_results and "pylint" in static_results["tools_results"]:
            pylint_data = static_results["tools_results"]["pylint"]
            if pylint_data.get("status") == "success":
                for issue in pylint_data.get("output", []):
                    issues_data.append({
                        "工具": "Pylint",
                        "文件": issue.get("path", ""),
                        "行号": issue.get("line", ""),
                        "类型": issue.get("type", ""),
                        "消息": issue.get("message", ""),
                        "严重程度": issue.get("message-id", "")
                    })
                    
        # 处理Bandit结果
        if "tools_results" in static_results and "bandit" in static_results["tools_results"]:
            bandit_data = static_results["tools_results"]["bandit"]
            if bandit_data.get("status") == "success" and "results" in bandit_data.get("output", {}):
                for issue in bandit_data["output"]["results"]:
                    issues_data.append({
                        "工具": "Bandit",
                        "文件": issue.get("filename", ""),
                        "行号": issue.get("line_number", ""),
                        "类型": "Security",
                        "消息": issue.get("issue_text", ""),
                        "严重程度": issue.get("issue_severity", "")
                    })
                    
        if issues_data:
            df_static = pd.DataFrame(issues_data)
            df_static.to_excel(writer, sheet_name='静态分析详情', index=False)
        else:
            df_empty = pd.DataFrame({"消息": ["未发现静态分析问题"]})
            df_empty.to_excel(writer, sheet_name='静态分析详情', index=False)
            
    async def _create_ai_analysis_sheet(self, writer, ai_results: Dict[str, Any]):
        """创建AI分析详情页"""
        ai_data = []
        
        # AI发现的问题
        for issue in ai_results.get("potential_issues", []):
            ai_data.append({
                "类型": issue.get("type", ""),
                "描述": issue.get("description", ""),
                "严重程度": issue.get("severity", ""),
                "来源": "AI分析"
            })
            
        # AI建议
        for i, recommendation in enumerate(ai_results.get("recommendations", []), 1):
            ai_data.append({
                "类型": "建议",
                "描述": recommendation,
                "严重程度": "建议",
                "来源": f"AI建议{i}"
            })
            
        if ai_data:
            df_ai = pd.DataFrame(ai_data)
            df_ai.to_excel(writer, sheet_name='AI分析详情', index=False)
        else:
            df_empty = pd.DataFrame({"消息": ["AI分析未发现问题"]})
            df_empty.to_excel(writer, sheet_name='AI分析详情', index=False)
            
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行汇总任务"""
        return {"status": "summary_agent_ready"}