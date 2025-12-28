import os
import torch
import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, Any, List
from .base_agent import BaseAgent, Message
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.config.ai_agents import get_ai_agent_config
from infrastructure.config.prompts import get_prompt
from infrastructure.reports import report_manager
from utils import log, LogLevel

class AIDrivenCodeQualityAgent(BaseAgent):
    """AI-driven code quality analysis agent - utilizing AI model capabilities"""
    
    def __init__(self):
        super().__init__("ai_code_quality_agent", "AI Code Quality Agent")  # rename for legacy test expectation
        self.used_device = "gpu"
        self.used_device_map = None  # 添加设备映射参数
        self.db_service = DatabaseService()
        # 从统一配置获取
        self.agent_config = get_ai_agent_config().get_code_quality_agent_config()
        self.model_config = HUGGINGFACE_CONFIG["models"]["code_quality"]
        
        # AI模型组件
        self.code_understanding_model = None
        self.text_generation_model = None
        self.classification_model = None

    async def _initialize_models(self):
        """Initialize AI model - supports both CPU and GPU based on used_device"""
        try:
            # 验证 used_device 参数
            if self.used_device not in ["cpu", "gpu"]:
                log("ai_code_quality_agent", LogLevel.WARNING, f"⚠️ 无效的设备参数: {self.used_device}，回退到CPU")
                self.used_device = "cpu"
            
            # 优先使用agent专属配置，回退到HUGGINGFACE_CONFIG
            model_name = self.agent_config.get("model_name", self.model_config["name"])
            cache_dir = get_ai_agent_config().get_model_cache_dir()
            # 确保缓存目录是绝对路径（与user_comm_agent保持一致）
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.abspath(cache_dir)
            log("ai_code_quality_agent", LogLevel.INFO, f"💾 缓存目录: {cache_dir}")

            device = -1 if self.used_device == "cpu" else 0
            device_mode = "CPU" if self.used_device == "cpu" else "GPU"
            
            # 仅在CPU模式下设置线程数
            if self.used_device == "cpu":
                cpu_threads = self.agent_config.get("cpu_threads", 4)
                torch.set_num_threads(cpu_threads)
            
            log("ai_code_quality_agent", LogLevel.INFO, f"🤖 [ai_code_quality_agent] 正在加载代码理解模型 ({device_mode}模式): {model_name}")
            try:
                # 确保缓存目录存在
                os.makedirs(cache_dir, exist_ok=True)

                local_files_only = False
                # 检查模型文件是否已存在
                model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
                # 检查快照目录是否存在且不为空
                snapshots_path = os.path.join(model_path, "snapshots")
                model_files_exist = (
                    os.path.exists(model_path) and 
                    os.path.exists(snapshots_path) and 
                    os.listdir(snapshots_path)
                )

                if model_files_exist:
                    local_files_only = True
                    log("ai_code_quality_agent", LogLevel.INFO, "🔍 检测到本地缓存模型文件，将使用本地文件加载")
                else:
                    log("ai_code_quality_agent", LogLevel.INFO, "🌐 未检测到本地缓存模型，将从网络下载")
                
                log("ai_code_quality_agent", LogLevel.INFO, "🔧 使用microsoft codebert-base配置加载tokenizer...")
                if local_files_only and model_files_exist:
                    snapshot_dirs = os.listdir(snapshots_path)
                    if snapshot_dirs:
                        model_local_path = os.path.join(snapshots_path, snapshot_dirs[0])
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_local_path,
                            cache_dir=cache_dir,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    else:
                        raise Exception("[ai_code_quality_agent] 未找到有效的模型快照目录")
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        local_files_only=local_files_only
                    )
                log("ai_code_quality_agent", LogLevel.INFO, "✅ Tokenizer加载成功")

                # 配置tokenizer
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                log("ai_code_quality_agent", LogLevel.INFO, "🔧 已设置pad_token")

                log("ai_code_quality_agent", LogLevel.INFO, "🔧 正在创建对话生成pipeline...")
                model_kwargs = {
                    "torch_dtype": getattr(torch, self.agent_config.get("torch_dtype", "float32")),
                    "low_cpu_mem_usage": self.agent_config.get("low_cpu_mem_usage", True),
                    "cache_dir": cache_dir,
                }
                if local_files_only:
                    model_kwargs["local_files_only"] = True

                if local_files_only and model_files_exist:
                    # 直接使用本地模型路径而不是模型标识符
                    snapshot_dirs = os.listdir(snapshots_path)
                    if snapshot_dirs:
                        model_local_path = os.path.join(snapshots_path, snapshot_dirs[0])
                        self.conversation_model = pipeline(
                            "text-classification",
                            model=model_local_path,
                            tokenizer=self.tokenizer,
                            device=self.used_device,
                            trust_remote_code=True,
                            model_kwargs=model_kwargs
                        )
                    else:
                        raise Exception("未找到有效的模型快照目录")
                else:
                    # 在线模式或本地文件不完整时使用模型名称
                    self.conversation_model = pipeline(
                        "text-generation",
                        model=model_name,
                        tokenizer=self.tokenizer,
                        device=self.used_device,
                        trust_remote_code=True,
                        model_kwargs=model_kwargs
                    )
                log("ai_code_quality_agent", LogLevel.INFO, "✅ Pipeline创建成功")
            except Exception as model_error:
                log("ai_code_quality_agent", LogLevel.WARNING, f"⚠️ [ai_code_quality_agent] 主模型加载失败,尝试备用模型: {model_error}")
                
                if not os.path.exists(cache_dir):
                    log("ai_code_quality_agent", LogLevel.WARNING, f"❌ 缓存目录不存在: {cache_dir}")

                fallback_model = self.agent_config.get("fallback_model", "distilbert-base-uncased")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model,
                    cache_dir=cache_dir
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.classification_model = pipeline(
                    "text-classification",
                    model=fallback_model,
                    device=device
                )
                log("ai_code_quality_agent", LogLevel.INFO, f"✅ [ai_code_quality_agent] 备用模型加载成功: {fallback_model}")

            try:
                # 为 text-generation 也优先使用本地缓存（如果存在）
                tg_model_name = self.agent_config.get("text_generator_model", "gpt2")
                tg_local_files_only = False
                tg_model_path = os.path.join(cache_dir, f"models--{tg_model_name.replace('/', '--')}")
                tg_snapshots_path = os.path.join(tg_model_path, "snapshots")
                tg_model_files_exist = os.path.exists(tg_model_path) and os.path.exists(tg_snapshots_path) and bool(os.listdir(tg_snapshots_path))
                if tg_model_files_exist:
                    tg_local_files_only = True
                    log("ai_code_quality_agent", LogLevel.INFO, "🔍 检测到本地缓存模型文件，将使用本地文件加载")
                else:
                    log("ai_code_quality_agent", LogLevel.INFO, "🌐 未检测到本地缓存模型，将从网络下载")
                
                tg_model_kwargs = {"low_cpu_mem_usage": True, "cache_dir": cache_dir}
                if tg_model_files_exist:
                    tg_model_kwargs["local_files_only"] = True

                log("ai_code_quality_agent", LogLevel.INFO, f"🤖 [ai_code_quality_agent] 正在加载文本生成模型 ({'本地' if tg_model_files_exist else '网络'}) : {tg_model_name}")
                if tg_local_files_only and tg_model_files_exist:
                    tg_snapshot_dirs = os.listdir(tg_snapshots_path)
                    tg_model_local_path = os.path.join(tg_snapshots_path, tg_snapshot_dirs[0])
                    self.text_generation_model = pipeline(
                        "text-generation",
                        model=tg_model_local_path,
                        device=device,
                        model_kwargs=tg_model_kwargs
                    )
                else:
                    log("ai_code_quality_agent", LogLevel.INFO, f"🤖 [ai_code_quality_agent] 正在加载文本生成模型 (网络) : {tg_model_name}")
                    self.text_generation_model = pipeline(
                        "text-generation",
                        model=tg_model_name,
                        device=device,
                        model_kwargs=tg_model_kwargs
                    )
                if self.text_generation_model.tokenizer.pad_token is None:
                    self.text_generation_model.tokenizer.pad_token = self.text_generation_model.tokenizer.eos_token
                log("ai_code_quality_agent", LogLevel.INFO, "✅ [ai_code_quality_agent] 文本生成模型加载成功")
            except Exception as gen_error:
                log("ai_code_quality_agent", LogLevel.WARNING, f"⚠️ [ai_code_quality_agent] 文本生成模型加载失败,将使用模板生成: {gen_error}")
                self.text_generation_model = None
            self.code_understanding_model = self.classification_model
            log("ai_code_quality_agent", LogLevel.INFO, f"✅ [ai_code_quality_agent] AI模型初始化完成 ({device_mode}模式)")

        except Exception as e:
            log("ai_code_quality_agent", LogLevel.ERROR, f"❌ [ai_code_quality_agent] AI模型初始化失败: {e}")
            log("ai_code_quality_agent", LogLevel.INFO, "🔄 [ai_code_quality_agent] 切换到无AI模式,使用基础分析")
            self.code_understanding_model = None
            self.classification_model = None
            self.text_generation_model = None
            
    async def handle_message(self, message: Message):
        """Process code quality analysis request"""
        if message.message_type == "quality_analysis_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            file_path = message.content.get("file_path")
            run_id = message.content.get("run_id")
            if not self.code_understanding_model:
                await self._initialize_models()
            # 不再这里触发静态扫描，避免与集成器的初始派发造成重复 (出现 run_id=None 的第二次扫描)
            # 质量代理只等待 static_scan_complete 消息再做综合分析
            return
        elif message.message_type == "static_scan_complete":
            # 接收静态扫描结果并进行AI综合分析 (运行已结束后仍可能到达)
            requirement_id = message.content.get("requirement_id")
            static_scan_results = message.content.get("static_scan_results", {})
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            file_path = message.content.get("file_path")
            run_id = message.content.get('run_id')
            # 检测运行是否已闭合，不打印任何正常信息
            run_closed = False
            if run_id:
                from pathlib import Path as _P
                run_dir = _P(__file__).parent.parent.parent / 'reports' / 'analysis' / run_id
                if (run_dir / 'run_summary.json').exists():
                    run_closed = True
            result = await self._ai_comprehensive_analysis(
                code_content, code_directory, static_scan_results,
                silent=True  # 总是静默正常输出
            )
            if run_id:
                try:
                    agent_payload = {
                        "requirement_id": requirement_id,
                        "file_path": file_path,
                        "run_id": run_id,
                        "code_quality_result": result,
                        "generated_at": self._get_current_time()
                    }
                    report_manager.generate_run_scoped_report(run_id, agent_payload, f"quality_req_{requirement_id}.json", subdir="agents/code_quality")
                except Exception as e:
                    log("ai_code_quality_agent", LogLevel.WARNING, f"⚠️ 代码质量Agent单独报告生成失败 requirement={requirement_id} run_id={run_id}: {e}")
            await self.dispatch_message(
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
            await self.dispatch_message(
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
            # 移除完成提示打印

    async def _ai_comprehensive_analysis(self, code_content: str, code_directory: str,
                                        static_scan_results: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        """综合代码质量分析：只在错误时打印"""
        try:
            ai_analysis = await self._ai_driven_quality_analysis(code_content, code_directory, silent=True)
            static_analysis_insights = await self._analyze_static_scan_results(static_scan_results)
            comprehensive_assessment = await self._ai_comprehensive_assessment(
                ai_analysis, static_scan_results, static_analysis_insights
            )
            integrated_suggestions = await self._generate_integrated_suggestions(
                ai_analysis, static_scan_results, comprehensive_assessment
            )
            final_report = await self._generate_final_quality_report(
                ai_analysis, static_scan_results, comprehensive_assessment, integrated_suggestions
            )
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
            log("ai_code_quality_agent", LogLevel.ERROR, f"❌ AI综合分析过程中出错: {e}")
            return {
                "analysis_type": "comprehensive_analysis_error",
                "error_message": str(e),
                "analysis_status": "failed"
            }

    async def _ai_driven_quality_analysis(self, code_content: str, code_directory: str, silent: bool = False) -> Dict[str, Any]:
        """底层质量分析：仅错误打印"""
        try:
            all_code_content = await self._read_code_files(code_directory) if code_directory else code_content
            code_embeddings = await self._get_code_embeddings(all_code_content)
            quality_classification = await self._classify_code_quality(all_code_content)
            analysis_report = await self._generate_quality_report(all_code_content)
            improvement_suggestions = await self._generate_improvement_suggestions(all_code_content)
            refactoring_suggestions = await self._generate_refactoring_suggestions(all_code_content)
            return {
                "ai_analysis_type": "comprehensive_quality_analysis",
                "model_used": self.model_config["name"],
                "code_embeddings_summary": code_embeddings,
                "quality_classification": quality_classification,
                "detailed_analysis": analysis_report,
                "improvement_suggestions": improvement_suggestions,
                "refactoring_suggestions": refactoring_suggestions,
                "ai_confidence": 0.85,
                "analysis_timestamp": self._get_current_time(),
                "analysis_status": "completed"
            }
        except Exception as e:
            log("ai_code_quality_agent", LogLevel.ERROR, f"❌ AI分析过程中出错: {e}")
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
            chunks = self._split_code_into_chunks(code_content, max_length=256)
            embeddings_summary = []
            for i, chunk in enumerate(chunks[:3]):
                try:
                    result = self.classification_model(
                        chunk[:200],
                        truncation=True
                    )
                    if result and len(result) > 0:
                        score = result[0].get('score', 0.5)
                        embeddings_summary.append({
                            "chunk_index": i,
                            "semantic_score": float(score),
                            "chunk_length": len(chunk),
                            "model_confidence": float(score)
                        })
                    await asyncio.sleep(0.05)
                except Exception as chunk_error:
                    log("ai_code_quality_agent", LogLevel.WARNING, f"⚠️ 处理块 {i} 时出错: {chunk_error}")
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
            log("ai_code_quality_agent", LogLevel.WARNING, f"⚠️ 嵌入生成失败,使用简化分析: {e}")
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
            classification_prompt = f"""
            Analyze the following code for quality assessment:
            Code:
            {code_content[:1000]}
            Quality aspects to evaluate:
            - Readability
            - Maintainability 
            - Performance
            - Security
            """
            if self.classification_model:
                truncated_prompt = classification_prompt[:512]
                result = self.classification_model(
                    truncated_prompt,
                    truncation=True
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

    def _safe_generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str | None:
        """安全封装 text-generation，基于 token 截断避免超长导致 index out of range。
        返回生成文本或 None (失败)。"""
        if not self.text_generation_model:
            return None
        try:
            tokenizer = self.text_generation_model.tokenizer
            model = self.text_generation_model.model
            max_ctx = getattr(model.config, 'n_positions', 1024)
            # 编码不加生成提示，避免重复特殊token
            input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            reserve = max_new_tokens
            if len(input_ids) + reserve > max_ctx:
                # 截断到可用长度
                keep = max_ctx - reserve
                input_ids = input_ids[:keep]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            # 调用 pipeline
            out = self.text_generation_model(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            if out and len(out) > 0:
                return out[0].get('generated_text', '')
            return None
        except Exception as e:
            import traceback
            log("ai_code_quality_agent", LogLevel.ERROR, f"❌ 文本生成失败: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return None

    async def _generate_quality_report(self, code_content: str) -> Dict[str, Any]:
        """Use AI to generate detailed quality analysis report (安全生成)"""
        try:
            prompt = get_prompt(
                task_type="code_analysis",
                model_name=self.model_config["name"],
                code_content=code_content[:2000],
                language="python"
            )
            if self.text_generation_model:
                generated_text = self._safe_generate(prompt, max_new_tokens=128, temperature=0.7)
                if generated_text is None or not generated_text.strip():
                    return self._fallback_quality_analysis(code_content) | {"generation_error": "text_generation_failed"}
                analysis_result = self._parse_ai_analysis(generated_text)
                return {
                    "ai_generated_report": generated_text,
                    "structured_analysis": analysis_result,
                    "generation_successful": True
                }
            else:
                return self._fallback_quality_analysis(code_content)
        except Exception as e:
            return {"error": f"报告生成失败: {e}"}

    async def _generate_improvement_suggestions(self, code_content: str) -> List[Dict[str, Any]]:
        """AI-generated improvement suggestions (安全生成)"""
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
                generated_text = self._safe_generate(improvement_prompt, max_new_tokens=96, temperature=0.65)
                if generated_text is None:
                    return self._fallback_improvement_suggestions(code_content) + [{"error": "suggestion_generation_failed"}]
                suggestions = self._parse_suggestions(generated_text)
                return suggestions if suggestions else self._fallback_improvement_suggestions(code_content)
            else:
                return self._fallback_improvement_suggestions(code_content)
        except Exception as e:
            return [{"error": f"建议生成失败: {e}"}]

    async def _generate_refactoring_suggestions(self, code_content: str) -> Dict[str, Any]:
        """AI-generated refactoring suggestions (安全生成)"""
        try:
            refactoring_prompt = get_prompt(
                task_type="refactoring",
                model_name=self.model_config["name"],
                code_content=code_content[:1500],
                language="python"
            )
            if self.text_generation_model:
                generated_text = self._safe_generate(refactoring_prompt, max_new_tokens=128, temperature=0.55)
                if generated_text is None:
                    return self._fallback_refactoring_suggestions(code_content) | {"generation_error": "refactoring_generation_failed"}
                return {
                    "ai_refactoring_plan": generated_text,
                    "refactoring_priority": "medium",
                    "estimated_effort": "2-4 hours",
                    "expected_improvements": ["可读性提升", "维护性增强", "性能优化"]
                }
            else:
                return self._fallback_refactoring_suggestions(code_content)
        except Exception as e:
            return {"error": f"重构建议生成失败: {e}"}

    async def _analyze_static_scan_results(self, static_scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """从静态扫描结果中提炼结构化洞察。"""
        try:
            if not isinstance(static_scan_results, dict):
                return {"error": "static_scan_results 非字典", "received_type": str(type(static_scan_results))}
            summary = static_scan_results.get("summary", {}) or {}
            severity = summary.get("severity_breakdown", {}) or {}
            total_issues = summary.get("total_issues", 0)
            grade = summary.get("quality_grade")
            tools = static_scan_results.get("tools_used", [])
            language = static_scan_results.get("language")
            recs = summary.get("recommendations", [])
            insights: List[str] = []
            if total_issues == 0:
                insights.append("未发现静态问题，代码基础健康。")
            else:
                hi = severity.get("high", 0) + severity.get("critical", 0)
                if hi:
                    insights.append(f"存在 {hi} 个高/严重级别问题，需优先处理。")
                md = severity.get("medium", 0)
                if md:
                    insights.append(f"有 {md} 个中等级问题，可排期处理。")
                lo = severity.get("low", 0)
                if lo > 15:
                    insights.append("低等级样式/约定问题较多，考虑引入自动格式化。")
            if summary.get("has_security_issues"):
                insights.append("检测到安全相关静态问题，需结合安全分析结果确认风险。")
            if summary.get("has_type_issues"):
                insights.append("存在类型检查问题，建议补全类型注解。")
            if grade:
                insights.append(f"静态质量等级: {grade}")
            return {
                "static_summary": {
                    "total_issues": total_issues,
                    "severity_breakdown": severity,
                    "quality_grade": grade,
                    "language": language,
                    "tools_used": tools
                },
                "insights": insights,
                "raw_recommendations": recs
            }
        except Exception as e:
            return {"error": f"静态扫描结果分析失败: {e}"}

    async def _ai_comprehensive_assessment(self, ai_analysis: Dict[str, Any], static_scan_results: Dict[str, Any], static_analysis_insights: Dict[str, Any]) -> Dict[str, Any]:
        """结合 AI 与静态扫描信息生成综合评估。"""
        try:
            ai_quality = ai_analysis.get("quality_classification", {})
            static_summary = static_analysis_insights.get("static_summary", {})
            severity = static_summary.get("severity_breakdown", {})
            total = static_summary.get("total_issues", 0)
            grade = static_summary.get("quality_grade")
            risk = "low"
            if severity.get("critical", 0) > 0:
                risk = "critical"
            elif severity.get("high", 0) > 0:
                risk = "high"
            elif severity.get("medium", 0) > 5:
                risk = "medium"
            return {
                "risk_level": risk,
                "estimated_quality_grade": grade or "UNKNOWN",
                "total_issues": total,
                "ai_predicted_quality": ai_quality.get("predicted_quality"),
                "ai_confidence": ai_quality.get("confidence"),
                "key_static_insights": static_analysis_insights.get("insights", [])
            }
        except Exception as e:
            return {"error": f"综合评估失败: {e}"}

    async def _generate_integrated_suggestions(self, ai_analysis: Dict[str, Any], static_scan_results: Dict[str, Any], comprehensive_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """整合多来源建议。"""
        try:
            suggestions: List[Dict[str, Any]] = []
            for s in ai_analysis.get("improvement_suggestions", [])[:5]:
                if isinstance(s, dict):
                    suggestions.append({
                        "source": "ai_improvement",
                        "description": s.get("description"),
                        "priority": s.get("priority", "medium")
                    })
                elif isinstance(s, str):
                    suggestions.append({"source": "ai_improvement", "description": s, "priority": "medium"})
            for r in static_scan_results.get("summary", {}).get("recommendations", [])[:5]:
                suggestions.append({"source": "static_scan", "description": r, "priority": "medium"})
            risk = comprehensive_assessment.get("risk_level")
            if risk in {"high", "critical"}:
                suggestions.append({"source": "risk_assessment", "description": "优先修复高风险及潜在安全漏洞。", "priority": "high"})
            if not suggestions:
                suggestions.append({"source": "general", "description": "整体质量良好，无需立即动作。", "priority": "low"})
            return suggestions
        except Exception as e:
            return [{"error": f"整合建议生成失败: {e}"}]

    async def _generate_final_quality_report(self, ai_analysis: Dict[str, Any], static_scan_results: Dict[str, Any], comprehensive_assessment: Dict[str, Any], integrated_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成最终聚合报告。"""
        try:
            return {
                "overview": {
                    "risk_level": comprehensive_assessment.get("risk_level"),
                    "estimated_quality_grade": comprehensive_assessment.get("estimated_quality_grade"),
                    "total_static_issues": comprehensive_assessment.get("total_issues"),
                    "ai_quality_prediction": comprehensive_assessment.get("ai_predicted_quality"),
                    "ai_confidence": comprehensive_assessment.get("ai_confidence")
                },
                "integrated_suggestions": integrated_suggestions,
                "key_static_insights": comprehensive_assessment.get("key_static_insights", []),
                "ai_refactoring_plan": ai_analysis.get("refactoring_suggestions"),
                "generated_timestamp": self._get_current_time()
            }
        except Exception as e:
            return {"error": f"最终报告生成失败: {e}"}

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
            log("ai_code_quality_agent", LogLevel.WARNING, f"读取代码文件时出错: {e}")
            
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
    
    # DEBUG log:
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