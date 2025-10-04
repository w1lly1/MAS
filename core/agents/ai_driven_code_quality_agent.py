import torch
import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, Any, List
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.config.prompts import get_prompt

class AIDrivenCodeQualityAgent(BaseAgent):
    """AI-driven code quality analysis agent - utilizing AI model capabilities"""
    
    def __init__(self):
        super().__init__("ai_code_quality_agent", "AI Code Quality Agent")  # rename for legacy test expectation
        # legacy alias
        self.legacy_display_name = "AI驱动代码质量分析智能体"
        self.db_service = DatabaseService()
        self.model_config = HUGGINGFACE_CONFIG["models"]["code_quality"]
        
        # AI模型组件
        self.code_understanding_model = None
        self.text_generation_model = None
        self.classification_model = None
        
        # 移除硬编码的prompt,改用配置文件
        # self.quality_analysis_prompt 和其他prompt现在从prompts.py获取

    async def _initialize_models(self):
        """Initialize AI model - optimized for CPU environment"""
        try:
            model_name = self.model_config["name"]
            cache_dir = HUGGINGFACE_CONFIG["cache_dir"]
            
            # 强制使用CPU,避免GPU相关错误
            device = -1  # CPU only
            torch.set_num_threads(4)  # 限制CPU线程数
            
            print(f"🤖 正在加载代码理解模型 (CPU模式): {model_name}")
            print(f"💾 缓存目录: {cache_dir}")
            
            # 1. 优先加载轻量级模型
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    local_files_only=False,  # 允许下载
                    trust_remote_code=False
                )
                
                # 确保tokenizer有pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print("✅ Tokenizer加载成功")
                
                # 使用更轻量的pipeline而不是直接加载模型
                self.classification_model = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device=device,
                    cache_dir=cache_dir,
                    model_kwargs={
                        "torch_dtype": torch.float32,  # 使用float32减少内存
                        "low_cpu_mem_usage": True
                    },
                    tokenizer_kwargs={
                        "padding": True,
                        "truncation": True,
                        "max_length": 512  # 设置tokenizer最大长度
                    }
                )
                print("✅ 分类模型加载成功")
                
            except Exception as model_error:
                print(f"⚠️ 主模型加载失败,尝试备用模型: {model_error}")
                # 降级到DistilBERT (更轻量)
                fallback_model = "distilbert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model, 
                    cache_dir=cache_dir
                )
                
                # 确保fallback tokenizer有pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.classification_model = pipeline(
                    "text-classification",
                    model=fallback_model,
                    device=device,
                    cache_dir=cache_dir,
                    tokenizer_kwargs={
                        "padding": True,
                        "truncation": True,
                        "max_length": 512  # 设置tokenizer最大长度
                    }
                )
                print(f"✅ 备用模型加载成功: {fallback_model}")
            
            # 2. 使用CPU友好的文本生成 (可选,性能要求高时可禁用)
            try:
                # 使用更小的模型用于文本生成
                self.text_generation_model = pipeline(
                    "text-generation",
                    model="gpt2",  # 改为更轻量的GPT-2
                    device=device,
                    cache_dir=cache_dir,
                    model_kwargs={"low_cpu_mem_usage": True},
                    tokenizer_kwargs={
                        "padding": True,
                        "truncation": True,
                        "max_length": 512  # 设置tokenizer最大长度
                    }
                )
                
                # 确保pipeline的tokenizer有pad_token
                if self.text_generation_model.tokenizer.pad_token is None:
                    self.text_generation_model.tokenizer.pad_token = self.text_generation_model.tokenizer.eos_token
                
                print("✅ 文本生成模型加载成功")
            except Exception as gen_error:
                print(f"⚠️ 文本生成模型加载失败,将使用模板生成: {gen_error}")
                self.text_generation_model = None
            
            # 3. 设置代码理解模型引用
            self.code_understanding_model = self.classification_model
            
            print(f"✅ AI模型初始化完成 (CPU模式)")
            
        except Exception as e:
            print(f"❌ AI模型初始化失败: {e}")
            print("🔄 切换到无AI模式,使用基础分析")
            self.code_understanding_model = None
            self.classification_model = None
            self.text_generation_model = None
            
    async def handle_message(self, message: Message):
        """Process code quality analysis request"""
        if message.message_type == "quality_analysis_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            
            print(f"🤖 AI代码质量分析开始 - 需求ID: {requirement_id}")
            
            if not self.code_understanding_model:
                await self._initialize_models()
            
            # 先请求静态扫描结果
            await self.send_message(
                receiver="static_scan_agent",
                content={
                    "requirement_id": requirement_id,
                    "code_content": code_content,
                    "code_directory": code_directory
                },
                message_type="static_scan_request"
            )
            
            print(f"📋 已请求静态扫描结果 - 需求ID: {requirement_id}")
            
        elif message.message_type == "static_scan_complete":
            # 接收静态扫描结果并进行AI综合分析
            requirement_id = message.content.get("requirement_id")
            static_scan_results = message.content.get("static_scan_results", {})
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            file_path = message.content.get("file_path")
            run_id = message.content.get('run_id')
            
            print(f"📊 收到静态扫描结果,开始AI综合分析 - 需求ID: {requirement_id}")
            
            # 执行AI驱动的综合分析
            result = await self._ai_comprehensive_analysis(
                code_content, code_directory, static_scan_results
            )
            
            # 发送最终结果
            await self.send_message(
                receiver="user_comm_agent",
                content={
                    "requirement_id": requirement_id,
                    "agent_type": "ai_code_quality",
                    "results": result,
                    "analysis_complete": True,
                    "file_path": file_path,
                    "run_id": run_id
                },
                message_type="analysis_result"
            )
            await self.send_message(
                receiver="summary_agent",
                content={
                    "requirement_id": requirement_id,
                    "analysis_type": "ai_analysis",
                    "result": result,
                    "file_path": file_path,
                    "run_id": run_id
                },
                message_type="analysis_result"
            )
            
            print(f"✅ AI综合代码质量分析完成 - 需求ID: {requirement_id}")

    async def _ai_comprehensive_analysis(self, code_content: str, code_directory: str, 
                                        static_scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """AI-driven comprehensive code quality analysis (combined with static scan results)"""
        
        try:
            print("🧠 AI正在综合分析代码质量和静态扫描结果...")
            
            # 1. 执行原有的AI分析
            ai_analysis = await self._ai_driven_quality_analysis(code_content, code_directory)
            
            # 2. 解析和理解静态扫描结果
            static_analysis_insights = await self._analyze_static_scan_results(static_scan_results)
            
            # 3. AI综合评估:结合静态分析和AI理解
            comprehensive_assessment = await self._ai_comprehensive_assessment(
                ai_analysis, static_scan_results, static_analysis_insights
            )
            
            # 4. AI生成整合建议
            integrated_suggestions = await self._generate_integrated_suggestions(
                ai_analysis, static_scan_results, comprehensive_assessment
            )
            
            # 5. 生成最终质量报告
            final_report = await self._generate_final_quality_report(
                ai_analysis, static_scan_results, comprehensive_assessment, integrated_suggestions
            )
            
            print("✅ AI综合分析完成,生成最终质量报告")
            
            return {
                "analysis_type": "comprehensive_ai_static_analysis",
                "ai_analysis": ai_analysis,
                "static_scan_results": static_scan_results,
                "static_analysis_insights": static_analysis_insights,
                "comprehensive_assessment": comprehensive_assessment,
                "integrated_suggestions": integrated_suggestions,
                "final_report": final_report,
                "analysis_timestamp": self._get_current_time(),
                "analysis_status": "completed"
            }
            
        except Exception as e:
            print(f"❌ AI综合分析过程中出错: {e}")
            return {
                "analysis_type": "comprehensive_analysis_error",
                "error_message": str(e),
                "analysis_status": "failed"
            }

    async def _analyze_static_scan_results(self, static_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze static scan results and extract key insights"""
        insights = {
            "critical_issues_summary": [],
            "pattern_analysis": {},
            "quality_trends": {},
            "improvement_priority": [],
            "tool_effectiveness": {}
        }
        
        try:
            # 分析质量问题
            quality_issues = static_results.get("quality_issues", [])
            security_issues = static_results.get("security_issues", [])
            type_issues = static_results.get("type_issues", [])
            
            # 提取严重问题
            critical_issues = [
                issue for issue in quality_issues + security_issues + type_issues 
                if issue.get("severity") in ["critical", "high"]
            ]
            
            insights["critical_issues_summary"] = [
                {
                    "type": issue.get("type"),
                    "message": issue.get("message"),
                    "tool": issue.get("tool"),
                    "severity": issue.get("severity")
                }
                for issue in critical_issues[:10]  # 前10个严重问题
            ]
            
            # 分析复杂度数据
            complexity_analysis = static_results.get("complexity_analysis", {})
            insights["complexity_insights"] = {
                "maintainability_index": complexity_analysis.get("maintainability_index", 0),
                "average_complexity": complexity_analysis.get("average_complexity", 0),
                "complexity_distribution": complexity_analysis.get("cyclomatic_complexity", {})
            }
            
            # 工具效果评估
            tools_used = static_results.get("tools_used", [])
            summary = static_results.get("summary", {})
            
            insights["tool_effectiveness"] = {
                "tools_used": tools_used,
                "issues_found": summary.get("total_issues", 0),
                "quality_score": summary.get("quality_score", 0),
                "coverage": len(tools_used) / 5.0 * 100  # 假设最多5个工具
            }
            
        except Exception as e:
            insights["analysis_error"] = str(e)
            
        return insights

    async def _ai_comprehensive_assessment(self, ai_analysis: Dict[str, Any], 
                                         static_results: Dict[str, Any],
                                         static_insights: Dict[str, Any]) -> Dict[str, Any]:
        """AI-driven comprehensive evaluation"""
        
        assessment = {
            "overall_quality_score": 0.0,
            "confidence_level": 0.0,
            "assessment_dimensions": {},
            "consistency_analysis": {},
            "risk_assessment": {}
        }
        
        try:
            # 提取AI分析结果
            ai_quality = ai_analysis.get("quality_classification", {}).get("confidence", 0.5)
            ai_complexity = ai_analysis.get("code_embeddings_summary", {}).get("semantic_complexity", 0.5)
            
            # 提取静态分析结果
            static_quality = static_results.get("summary", {}).get("quality_score", 5.0) / 10.0
            static_issues = static_results.get("summary", {}).get("total_issues", 0)
            
            # 综合评分计算
            # AI理解权重40%,静态分析权重60%
            overall_score = (ai_quality * 0.4 + static_quality * 0.6) * 10.0
            assessment["overall_quality_score"] = round(overall_score, 2)
            
            # 一致性分析
            score_difference = abs(ai_quality * 10 - static_quality * 10)
            assessment["consistency_analysis"] = {
                "ai_static_alignment": "high" if score_difference < 2.0 else "medium" if score_difference < 4.0 else "low",
                "score_difference": round(score_difference, 2),
                "assessment_reliability": "high" if score_difference < 2.0 else "medium"
            }
            
            # 风险评估
            critical_issues = len([
                issue for issue in static_insights.get("critical_issues_summary", [])
                if issue.get("severity") == "critical"
            ])
            
            assessment["risk_assessment"] = {
                "critical_risk_level": "high" if critical_issues > 3 else "medium" if critical_issues > 0 else "low",
                "security_risk": "high" if any("security" in issue.get("type", "") 
                                             for issue in static_insights.get("critical_issues_summary", [])) else "low",
                "maintainability_risk": "high" if static_insights.get("complexity_insights", {}).get("maintainability_index", 50) < 30 else "low"
            }
            
            # 置信度评估
            tool_coverage = static_insights.get("tool_effectiveness", {}).get("coverage", 0)
            ai_confidence = ai_analysis.get("ai_confidence", 0.8)
            
            assessment["confidence_level"] = round((tool_coverage / 100.0 * 0.3 + ai_confidence * 0.7), 2)
            
        except Exception as e:
            assessment["assessment_error"] = str(e)
            
        return assessment

    async def _generate_integrated_suggestions(self, ai_analysis: Dict[str, Any],
                                             static_results: Dict[str, Any],
                                             assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated improvement suggestions"""
        
        suggestions = {
            "immediate_actions": [],
            "quality_improvements": [],
            "ai_enhanced_recommendations": [],
            "tool_based_fixes": [],
            "strategic_improvements": []
        }
        
        try:
            # 基于静态分析的即时修复建议
            static_issues = static_results.get("quality_issues", []) + static_results.get("security_issues", [])
            critical_static_issues = [issue for issue in static_issues if issue.get("severity") in ["critical", "high"]]
            
            for issue in critical_static_issues[:5]:
                suggestions["immediate_actions"].append({
                    "type": "static_fix",
                    "description": f"修复 {issue.get('tool')} 检测到的问题: {issue.get('message')}",
                    "line": issue.get("line", 0),
                    "severity": issue.get("severity"),
                    "tool": issue.get("tool")
                })
            
            # 基于AI分析的质量改进建议
            ai_suggestions = ai_analysis.get("improvement_suggestions", [])
            for suggestion in ai_suggestions[:3]:
                suggestions["ai_enhanced_recommendations"].append({
                    "type": "ai_insight",
                    "description": suggestion.get("description", ""),
                    "priority": suggestion.get("priority", "medium"),
                    "category": suggestion.get("category", "general")
                })
            
            # 战略性改进建议
            overall_score = assessment.get("overall_quality_score", 5.0)
            if overall_score < 6.0:
                suggestions["strategic_improvements"].append({
                    "type": "architecture_review",
                    "description": "代码质量分数偏低,建议进行架构审查和重构规划",
                    "priority": "high"
                })
            
            risk_level = assessment.get("risk_assessment", {}).get("critical_risk_level", "low")
            if risk_level == "high":
                suggestions["strategic_improvements"].append({
                    "type": "risk_mitigation",
                    "description": "存在高风险问题,建议立即制定风险缓解计划",
                    "priority": "critical"
                })
                
        except Exception as e:
            suggestions["generation_error"] = str(e)
            
        return suggestions

    async def _generate_final_quality_report(self, ai_analysis: Dict[str, Any],
                                           static_results: Dict[str, Any],
                                           assessment: Dict[str, Any],
                                           suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final quality report"""
        
        report = {
            "executive_summary": {},
            "detailed_findings": {},
            "recommendations": {},
            "next_steps": [],
            "quality_metrics": {}
        }
        
        try:
            # 执行摘要
            overall_score = assessment.get("overall_quality_score", 5.0)
            total_issues = static_results.get("summary", {}).get("total_issues", 0)
            
            report["executive_summary"] = {
                "overall_quality_score": overall_score,
                "quality_grade": self._score_to_grade(overall_score),
                "total_issues_found": total_issues,
                "critical_issues": len(suggestions.get("immediate_actions", [])),
                "analysis_confidence": assessment.get("confidence_level", 0.8),
                "primary_concerns": self._extract_primary_concerns(static_results, assessment)
            }
            
            # 详细发现
            report["detailed_findings"] = {
                "ai_insights": {
                    "semantic_understanding": ai_analysis.get("code_embeddings_summary", {}),
                    "ai_quality_assessment": ai_analysis.get("quality_classification", {}),
                    "ai_generated_analysis": ai_analysis.get("detailed_analysis", {})
                },
                "static_analysis_results": {
                    "tool_coverage": static_results.get("tools_used", []),
                    "issue_breakdown": static_results.get("summary", {}),
                    "complexity_metrics": static_results.get("complexity_analysis", {})
                },
                "consistency_check": assessment.get("consistency_analysis", {})
            }
            
            # 建议汇总
            report["recommendations"] = {
                "immediate_fixes": suggestions.get("immediate_actions", []),
                "quality_enhancements": suggestions.get("ai_enhanced_recommendations", []),
                "strategic_improvements": suggestions.get("strategic_improvements", [])
            }
            
            # 下一步行动
            report["next_steps"] = self._generate_next_steps(assessment, suggestions)
            
        except Exception as e:
            report["report_generation_error"] = str(e)
            
        return report

    def _extract_primary_concerns(self, static_results: Dict[str, Any], 
                                 assessment: Dict[str, Any]) -> List[str]:
        """Extract key concerns"""
        concerns = []
        
        # 安全问题
        security_issues = static_results.get("security_issues", [])
        if security_issues:
            concerns.append(f"发现 {len(security_issues)} 个安全问题")
        
        # 复杂度问题
        complexity = static_results.get("complexity_analysis", {})
        maintainability = complexity.get("maintainability_index", 50)
        if maintainability < 30:
            concerns.append("代码可维护性指数偏低")
        
        # 风险评估
        risk_level = assessment.get("risk_assessment", {}).get("critical_risk_level", "low")
        if risk_level == "high":
            concerns.append("存在高风险代码质量问题")
        
        return concerns[:3]  # 最多3个主要关注点

    def _generate_next_steps(self, assessment: Dict[str, Any], 
                           suggestions: Dict[str, Any]) -> List[str]:
        """Generate next action recommendations"""
        steps = []
        
        # 基于即时行动建议
        immediate_actions = suggestions.get("immediate_actions", [])
        if immediate_actions:
            steps.append(f"立即修复 {len(immediate_actions)} 个高优先级问题")
        
        # 基于质量分数
        overall_score = assessment.get("overall_quality_score", 5.0)
        if overall_score < 7.0:
            steps.append("制定代码质量提升计划")
        
        # 基于工具覆盖率
        confidence = assessment.get("confidence_level", 0.8)
        if confidence < 0.7:
            steps.append("考虑增加更多静态分析工具以提高检测覆盖率")
        
        return steps

    def _score_to_grade(self, score: float) -> str:
        """Convert score to grade"""
        if score >= 9.0:
            return "A"
        elif score >= 8.0:
            return "B"
        elif score >= 7.0:
            return "C"
        elif score >= 6.0:
            return "D"
        else:
            return "F"

    async def _ai_driven_quality_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """AI-driven code quality analysis"""
        
        try:
            print("🧠 AI正在理解代码结构和语义...")
            
            # 1. 读取所有代码文件
            all_code_content = await self._read_code_files(code_directory) if code_directory else code_content
            
            # 2. AI语义理解 - 使用CodeBERT理解代码
            code_embeddings = await self._get_code_embeddings(all_code_content)
            
            # 3. AI质量分类 - 使用分类模型
            quality_classification = await self._classify_code_quality(all_code_content)
            
            # 4. AI生成分析报告 - 使用prompt工程
            analysis_report = await self._generate_quality_report(all_code_content)
            
            # 5. AI生成改进建议
            improvement_suggestions = await self._generate_improvement_suggestions(all_code_content)
            
            # 6. AI重构建议
            refactoring_suggestions = await self._generate_refactoring_suggestions(all_code_content)
            
            print("✅ AI分析完成,生成综合质量报告")
            
            return {
                "ai_analysis_type": "comprehensive_quality_analysis",
                "model_used": self.model_config["name"],
                "code_embeddings_summary": code_embeddings,
                "quality_classification": quality_classification,
                "detailed_analysis": analysis_report,
                "improvement_suggestions": improvement_suggestions,
                "refactoring_suggestions": refactoring_suggestions,
                "ai_confidence": 0.85,  # AI模型置信度
                "analysis_timestamp": self._get_current_time(),
                "analysis_status": "completed"
            }
            
        except Exception as e:
            print(f"❌ AI分析过程中出错: {e}")
            return {
                "ai_analysis_type": "error",
                "error_message": str(e),
                "analysis_status": "failed"
            }

    async def _get_code_embeddings(self, code_content: str) -> Dict[str, Any]:
        """Use AI model to get code embedding representation - CPU optimized version"""
        try:
            if not self.classification_model:
                return {"error": "AI模型未加载", "fallback": True}
            
            # 分块处理大代码文件,减少内存使用
            chunks = self._split_code_into_chunks(code_content, max_length=256)  # 减小块大小
            embeddings_summary = []
            
            print(f"📊 处理 {len(chunks)} 个代码块...")
            
            for i, chunk in enumerate(chunks[:3]):  # 进一步限制处理块数
                try:
                    # 使用pipeline而不是直接调用模型,减少内存占用
                    result = self.classification_model(
                        chunk[:200],  # 限制输入长度
                        truncation=True  # 明确启用截断
                    )
                    
                    if result and len(result) > 0:
                        score = result[0].get('score', 0.5)
                        embeddings_summary.append({
                            "chunk_index": i,
                            "semantic_score": float(score),
                            "chunk_length": len(chunk),
                            "model_confidence": float(score)
                        })
                    
                    # 添加延迟,避免CPU过载
                    await asyncio.sleep(0.1)
                    
                except Exception as chunk_error:
                    print(f"⚠️ 处理块 {i} 时出错: {chunk_error}")
                    continue
            
            if embeddings_summary:
                avg_score = sum(item["semantic_score"] for item in embeddings_summary) / len(embeddings_summary)
                return {
                    "total_chunks": len(chunks),
                    "processed_chunks": len(embeddings_summary),
                    "embedding_summary": embeddings_summary,
                    "semantic_complexity": avg_score,
                    "processing_mode": "cpu_optimized"
                }
            else:
                return {
                    "error": "无法处理任何代码块",
                    "fallback": True,
                    "processing_mode": "cpu_optimized"
                }
            
        except Exception as e:
            print(f"⚠️ 嵌入生成失败,使用简化分析: {e}")
            return {
                "error": f"嵌入生成失败: {e}",
                "fallback_analysis": {
                    "code_length": len(code_content),
                    "estimated_complexity": "medium" if len(code_content) > 1000 else "low"
                },
                "processing_mode": "fallback"
            }

    async def _classify_code_quality(self, code_content: str) -> Dict[str, Any]:
        """Use AI classification model to evaluate code quality"""
        try:
            # 准备分类用的prompt
            classification_prompt = f"""
            Analyze the following code for quality assessment:
            
            Code:
            {code_content[:1000]}  # 限制长度
            
            Quality aspects to evaluate:
            - Readability
            - Maintainability 
            - Performance
            - Security
            """
            
            # 使用分类模型
            if self.classification_model:
                # 限制输入长度并启用截断
                truncated_prompt = classification_prompt[:512]  # 限制到512个字符
                result = self.classification_model(
                    truncated_prompt,
                    truncation=True,
                    max_length=512
                )
                
                return {
                    "predicted_quality": result[0]["label"] if result else "UNKNOWN",
                    "confidence": result[0]["score"] if result else 0.0,
                    "model_prediction": result
                }
            else:
                return {"error": "分类模型未初始化"}
                
        except Exception as e:
            return {"error": f"质量分类失败: {e}"}

    async def _generate_quality_report(self, code_content: str) -> Dict[str, Any]:
        """Use AI to generate detailed quality analysis report"""
        try:
            # 构造专业的分析prompt - 使用配置文件
            prompt = get_prompt(
                task_type="code_analysis",
                model_name=self.model_config["name"],
                code_content=code_content[:2000],
                language="python"  # 可以根据实际情况动态确定
            )
            
            if self.text_generation_model:
                # 生成分析报告
                response = self.text_generation_model(
                    prompt,
                    max_new_tokens=256,  # 使用max_new_tokens而不是max_length
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    truncation=True,  # 明确启用截断
                    pad_token_id=self.text_generation_model.tokenizer.eos_token_id  # 设置pad_token
                )
                
                generated_text = response[0]["generated_text"] if response else "无法生成报告"
                
                # 解析AI生成的报告
                analysis_result = self._parse_ai_analysis(generated_text)
                
                return {
                    "ai_generated_report": generated_text,
                    "structured_analysis": analysis_result,
                    "generation_successful": True
                }
            else:
                # 降级到基础分析
                return self._fallback_quality_analysis(code_content)
                
        except Exception as e:
            return {"error": f"报告生成失败: {e}"}

    async def _generate_improvement_suggestions(self, code_content: str) -> List[Dict[str, Any]]:
        """AI-generated improvement suggestions"""
        try:
            improvement_prompt = f"""
            作为代码审查专家,为以下代码提供具体的改进建议:
            
            {code_content[:1500]}
            
            请提供:
            1. 优先级高的改进点
            2. 具体的修改建议
            3. 改进后的预期效果
            """
            
            if self.text_generation_model:
                response = self.text_generation_model(
                    improvement_prompt,
                    max_new_tokens=200,  # 使用max_new_tokens
                    temperature=0.6,
                    truncation=True,  # 明确启用截断
                    pad_token_id=self.text_generation_model.tokenizer.eos_token_id
                )
                
                suggestions_text = response[0]["generated_text"] if response else ""
                
                # 解析建议为结构化数据
                suggestions = self._parse_suggestions(suggestions_text)
                
                return suggestions
            else:
                return self._fallback_improvement_suggestions(code_content)
                
        except Exception as e:
            return [{"error": f"建议生成失败: {e}"}]

    async def _generate_refactoring_suggestions(self, code_content: str) -> Dict[str, Any]:
        """AI-generated refactoring suggestions"""
        try:
            # 使用配置文件中的重构prompt
            refactoring_prompt = get_prompt(
                task_type="refactoring",
                model_name=self.model_config["name"], 
                code_content=code_content[:1500],
                language="python"
            )
            
            if self.text_generation_model:
                response = self.text_generation_model(
                    refactoring_prompt,
                    max_new_tokens=256,  # 使用max_new_tokens
                    temperature=0.5,
                    truncation=True,  # 明确启用截断
                    pad_token_id=self.text_generation_model.tokenizer.eos_token_id
                )
                
                refactoring_text = response[0]["generated_text"] if response else ""
                
                return {
                    "ai_refactoring_plan": refactoring_text,
                    "refactoring_priority": "medium",
                    "estimated_effort": "2-4 hours",
                    "expected_improvements": ["可读性提升", "维护性增强", "性能优化"]
                }
            else:
                return self._fallback_refactoring_suggestions(code_content)
                
        except Exception as e:
            return {"error": f"重构建议生成失败: {e}"}

    def _split_code_into_chunks(self, code_content: str, max_length: int = 256) -> List[str]:
        """Split code into smaller chunks to fit CPU memory constraints"""
        # 这里的max_length是我们自己控制的代码块大小,而不是transformer模型的参数
        # 所以不需要担心警告
        chunk_size = max_length  # 为清晰起见重命名变量
        
        if len(code_content) <= chunk_size:
            return [code_content]
        
        chunks = []
        lines = code_content.split('\n')
        current_chunk = ""
        
        for line in lines:
            # 如果单行就超过最大长度,直接截断
            if len(line) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(line[:chunk_size])
                continue
            
            # 检查添加这一行是否会超过限制
            if len(current_chunk) + len(line) + 1 <= chunk_size:
                current_chunk += line + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 限制最大块数量以节省内存
        return chunks[:10]

    def _parse_ai_analysis(self, generated_text: str) -> Dict[str, Any]:
        """Parse AI-generated analysis text into structured data"""
        # 简单的解析逻辑,实际应用中可以更复杂
        lines = generated_text.split('\n')
        
        analysis = {
            "issues_found": [],
            "quality_score": 7.0,  # 默认分数
            "recommendations": []
        }
        
        for line in lines:
            if "问题" in line or "issue" in line.lower():
                analysis["issues_found"].append(line.strip())
            elif "建议" in line or "recommend" in line.lower():
                analysis["recommendations"].append(line.strip())
            elif "分数" in line or "score" in line.lower():
                # 尝试提取分数
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if score_match:
                    analysis["quality_score"] = float(score_match.group(1))
        
        return analysis

    def _parse_suggestions(self, suggestions_text: str) -> List[Dict[str, Any]]:
        """Parse suggestion text into structured data"""
        suggestions = []
        lines = suggestions_text.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                suggestions.append({
                    "suggestion_id": i + 1,
                    "description": line.strip(),
                    "priority": "medium",
                    "category": "general"
                })
        
        return suggestions[:5]  # 限制数量

    async def _read_code_files(self, code_directory: str) -> str:
        """Read code files from directory"""
        import os
        
        code_content = ""
        supported_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rs']
        
        try:
            for root, dirs, files in os.walk(code_directory):
                for file in files[:10]:  # 限制文件数量
                    if any(file.endswith(ext) for ext in supported_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                code_content += f"\n\n# File: {file}\n{content}\n"
                        except Exception as e:
                            continue
                            
                if len(code_content) > 10000:  # 限制总长度
                    break
                    
        except Exception as e:
            print(f"读取代码文件时出错: {e}")
            
        return code_content

    def _fallback_quality_analysis(self, code_content: str) -> Dict[str, Any]:
        """Fallback analysis when AI model is not available"""
        return {
            "fallback_analysis": True,
            "basic_metrics": {
                "lines_of_code": len(code_content.split('\n')),
                "estimated_complexity": "medium",
                "has_comments": "TODO" in code_content or "FIXME" in code_content
            },
            "basic_recommendations": [
                "Recommend using AI model for more detailed analysis",
                "Check code comments and documentation",
                "Consider adding unit tests"
            ]
        }

    def _fallback_improvement_suggestions(self, code_content: str) -> List[Dict[str, Any]]:
        """Fallback improvement suggestions"""
        return [
            {
                "suggestion_id": 1,
                "description": "Recommend using AI model for more accurate analysis",
                "priority": "high",
                "category": "system"
            }
        ]

    def _fallback_refactoring_suggestions(self, code_content: str) -> Dict[str, Any]:
        """Fallback refactoring suggestions"""
        return {
            "fallback_refactoring": True,
            "basic_suggestions": [
                "Check function length and complexity",
                "Extract duplicate code",
                "Improve naming conventions"
            ]
        }

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI-driven quality analysis task"""
        return await self._ai_driven_quality_analysis(
            task_data.get("code_content", ""),
            task_data.get("code_directory", "")
        )

    def _get_current_time(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    # Compatibility layer removed (Option A). All interactions must use async message workflow.
    # If external code still calls former sync methods, raise explicit error to guide migration.
    def __getattr__(self, item):
        removed = {
            'load_model', 'analyze_code_quality', 'analyze_file',
            'generate_recommendations', 'calculate_quality_score'
        }
        if item in removed:
            raise AttributeError(
                f"'{item}' has been removed. Use async message-based requests: send 'quality_analysis_request' and listen for 'analysis_result'."
            )
        raise AttributeError(item)
