import os
import json
import re
import torch
import asyncio
import hashlib
from collections import OrderedDict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, Message
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.config.ai_agents import get_ai_agent_config
from infrastructure.config.prompts import get_prompt
from infrastructure.reports import report_manager
from utils import log, LogLevel

class AIDrivenSecurityAgent(BaseAgent):
    """AI驱动的安全分析智能体 - 基于prompt工程和模型推理"""
    
    def __init__(self):
        super().__init__("ai_security_agent", "AI Security Analysis Agent")
        self.db_service = DatabaseService()
        self.used_device = "gpu"
        self.used_device_map = None  # 添加设备映射参数
        # 从统一配置获取
        self.agent_config = get_ai_agent_config().get_security_agent_config()
        self.model_config = HUGGINGFACE_CONFIG["models"]["security"]
        # 移除本地硬编码prompt，统一使用 prompts.get_prompt
        self.security_model = None
        self.vulnerability_classifier = None
        self.threat_analyzer = None
        self.text_generator = None
        self._inference_cache: OrderedDict[str, Any] = OrderedDict()
        self._inference_cache_max_entries = int(self.agent_config.get("inference_cache_max_entries", 256))
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "classification_hits": 0,
            "classification_misses": 0,
            "generation_hits": 0,
            "generation_misses": 0,
        }
        self._run_cache_stats = dict(self._cache_stats)
        
    async def _initialize_models(self):
        """初始化AI模型 - 支持 CPU/GPU 动态选择"""
        try:
            # 验证 used_device 参数
            if self.used_device not in ["cpu", "gpu"]:
                log("ai_security_agent", LogLevel.INFO, f"⚠️ 无效的设备参数: {self.used_device}，回退到CPU")
                self.used_device = "cpu"
            
            device_mode = "CPU" if self.used_device == "cpu" else "GPU"
            log("ai_security_agent", LogLevel.INFO, f"🔧 初始化安全分析AI模型 ({device_mode}模式)...")
            
            # 优先使用agent专属配置，回退到HUGGINGFACE_CONFIG
            model_name = self.agent_config.get("model_name", "microsoft/codebert-base")
            cache_dir = HUGGINGFACE_CONFIG.get("cache_dir", "./model_cache/")
            device = -1 if self.used_device == "cpu" else 0
            
            # 仅在CPU模式下设置线程数
            if self.used_device == "cpu":
                cpu_threads = self.agent_config.get("cpu_threads", 4)
                torch.set_num_threads(cpu_threads)
            
            log("ai_security_agent", LogLevel.INFO, f"🤖 正在加载安全分析模型 ({device_mode}模式): {model_name}")
            log("ai_security_agent", LogLevel.INFO, f"💾 缓存目录: {cache_dir}")
            
            try:
                # 先尝试从本地缓存加载，如果失败则允许联网下载并缓存
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        torch_dtype=getattr(torch, self.agent_config.get("torch_dtype", "float32")),
                        low_cpu_mem_usage=self.agent_config.get("low_cpu_mem_usage", True)
                    )
                    log("ai_security_agent", LogLevel.INFO, f"✅ {model_name} 安全模型(本地缓存)初始化成功")
                except Exception as local_err:
                    log("ai_security_agent", LogLevel.INFO, f"⚠️ 本地缓存未就绪，尝试联网下载: {local_err}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        trust_remote_code=False
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        torch_dtype=getattr(torch, self.agent_config.get("torch_dtype", "float32")),
                        low_cpu_mem_usage=self.agent_config.get("low_cpu_mem_usage", True)
                    )
                    log("ai_security_agent", LogLevel.INFO, f"✅ {model_name} 安全模型(联网下载并缓存)初始化成功")

                self.security_model = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                self.vulnerability_classifier = self.security_model
            except Exception as model_error:
                log("ai_security_agent", LogLevel.INFO, f"⚠️ 主模型加载失败,尝试备用模型: {model_error}")
                fallback_model = self.agent_config.get("fallback_model", "distilbert-base-uncased")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        fallback_model,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        fallback_model,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        low_cpu_mem_usage=True
                    )
                    log("ai_security_agent", LogLevel.INFO, f"✅ 备用模型(本地缓存)加载成功: {fallback_model}")
                except Exception as fb_local_err:
                    log("ai_security_agent", LogLevel.INFO, f"⚠️ 备用模型本地缓存未就绪，尝试联网下载: {fb_local_err}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        fallback_model,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        trust_remote_code=False
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        fallback_model,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        low_cpu_mem_usage=True
                    )
                    log("ai_security_agent", LogLevel.INFO, f"✅ 备用模型(联网下载并缓存)加载成功: {fallback_model}")

                self.security_model = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                self.vulnerability_classifier = self.security_model

            try:
                # 为 text-generation 也优先使用本地缓存，如果失败则联网下载
                text_gen_model = self.agent_config.get("text_generator_model", "gpt2")
                try:
                    tokenizer_gen = AutoTokenizer.from_pretrained(
                        text_gen_model,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                    model_gen = AutoModelForCausalLM.from_pretrained(
                        text_gen_model,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        low_cpu_mem_usage=True
                    )
                    log("ai_security_agent", LogLevel.INFO, f"✅ {text_gen_model} 文本生成模型(本地缓存)加载成功")
                except Exception as tg_local_err:
                    log("ai_security_agent", LogLevel.INFO, f"⚠️ 文本生成模型本地缓存未就绪，尝试联网下载: {tg_local_err}")
                    tokenizer_gen = AutoTokenizer.from_pretrained(
                        text_gen_model,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        trust_remote_code=False
                    )
                    model_gen = AutoModelForCausalLM.from_pretrained(
                        text_gen_model,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        low_cpu_mem_usage=True
                    )
                    log("ai_security_agent", LogLevel.INFO, f"✅ {text_gen_model} 文本生成模型(联网下载并缓存)加载成功")

                self.text_generator = pipeline(
                    "text-generation",
                    model=model_gen,
                    tokenizer=tokenizer_gen,
                    device=device
                )
                if self.text_generator.tokenizer.pad_token is None:
                    self.text_generator.tokenizer.pad_token = self.text_generator.tokenizer.eos_token
                # 采用文本生成模型作为威胁建模生成器
                self.threat_analyzer = self.text_generator
            except Exception as gen_error:
                log("ai_security_agent", LogLevel.INFO, f"⚠️ 文本生成模型加载失败: {gen_error}")
                self.text_generator = None
                self.threat_analyzer = None
                
            self.models_loaded = True
            log("ai_security_agent", LogLevel.INFO, f"✅ 安全分析AI模型初始化完成 ({device_mode}模式)")
        except Exception as e:
            log("ai_security_agent", LogLevel.INFO, f"❌ 安全分析AI模型初始化失败: {e}")
            self.models_loaded = False
            self.security_model = None
            self.vulnerability_classifier = None
            self.text_generator = None
            self.threat_analyzer = None

    async def handle_message(self, message: Message):
        """处理安全分析请求"""
        if message.message_type == "security_analysis_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            file_path = message.content.get("file_path")
            run_id = message.content.get('run_id')
            
            log("ai_security_agent", LogLevel.INFO, f"🔒 AI安全分析开始 - 需求ID: {requirement_id}")
            
            if not self.security_model:
                await self._initialize_models()
            
            # 执行AI驱动的安全分析
            result = await self._ai_driven_security_analysis(code_content, code_directory)
            if run_id:
                try:
                    agent_payload = {
                        "requirement_id": requirement_id,
                        "file_path": file_path,
                        "run_id": run_id,
                        "security_result": result,
                        "generated_at": self._get_current_time()
                    }
                    report_manager.generate_run_scoped_report(run_id, agent_payload, f"security_req_{requirement_id}.json", subdir="agents/security")
                except Exception as e:
                    log("ai_security_agent", LogLevel.INFO, f"⚠️ 安全Agent单独报告生成失败 requirement={requirement_id} run_id={run_id}: {e}")
            # 发送结果
            await self.dispatch_message(
                receiver="user_comm_agent",
                content={
                    "requirement_id": requirement_id,
                    "agent_type": "ai_security",
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
                    "analysis_type": "security_analysis",
                    "result": result,
                    "file_path": file_path,
                    "run_id": run_id
                },
                message_type="analysis_result"
            )
            
            log("ai_security_agent", LogLevel.INFO, f"✅ AI安全分析完成 - 需求ID: {requirement_id}")

    async def _ai_driven_security_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """AI驱动的全面安全分析"""

        log("ai_security_agent", LogLevel.INFO, "🔍 AI正在进行深度安全分析...")
        self._reset_run_cache_stats()
        failed_steps: List[Dict[str, str]] = []

        try:
            code_context = await self._analyze_code_context(code_directory, code_content)
        except Exception as e:
            failed_steps.append({"step": "code_context", "error": str(e)})
            code_context = {
                "application_type": "unknown",
                "framework_detected": [],
                "database_usage": False,
                "network_operations": False,
                "authentication_present": False,
                "encryption_usage": False,
                "data_sensitivity": "medium",
                "fallback": True,
                "analysis_error": str(e),
            }

        try:
            vulnerabilities = await self._ai_vulnerability_detection(code_content, code_context)
        except Exception as e:
            failed_steps.append({"step": "vulnerability_detection", "error": str(e)})
            vulnerabilities = [{
                "vulnerability_id": "SEC_VULN_FALLBACK",
                "type": "analysis_error",
                "description": f"漏洞检测失败，已回退: {e}",
                "severity": "info",
                "ai_confidence": 0.0,
                "fallback": True,
            }]

        try:
            threat_model = await self._ai_threat_modeling(code_content, code_context)
        except Exception as e:
            failed_steps.append({"step": "threat_modeling", "error": str(e)})
            threat_model = self._fallback_threat_model(code_context) | {"fallback": True, "analysis_error": str(e)}

        try:
            security_rating = await self._ai_security_rating(vulnerabilities, threat_model)
        except Exception as e:
            failed_steps.append({"step": "security_rating", "error": str(e)})
            security_rating = {
                "security_score": 5.0,
                "rating_level": "Fair",
                "fallback": True,
                "error": str(e),
            }

        try:
            remediation_plan = await self._ai_remediation_planning(vulnerabilities)
        except Exception as e:
            failed_steps.append({"step": "remediation_planning", "error": str(e)})
            remediation_plan = {
                "immediate_actions": [],
                "short_term_fixes": [],
                "long_term_improvements": [],
                "estimated_effort": "unknown",
                "fallback": True,
                "error": str(e),
            }

        try:
            hardening_recommendations = await self._ai_security_hardening(code_content, code_context)
        except Exception as e:
            failed_steps.append({"step": "security_hardening", "error": str(e)})
            hardening_recommendations = [{
                "category": "fallback",
                "recommendation": f"安全加固建议生成失败，已回退: {e}",
                "priority": "info",
                "implementation": "请人工审查高风险输入点与认证流程",
                "fallback": True,
            }]

        analysis_status = "partial_success" if failed_steps else "completed"
        ai_confidence = 0.68 if failed_steps else 0.9

        log("ai_security_agent", LogLevel.INFO, "🛡️  AI安全分析完成,生成安全报告")
        raw_result = {
            "ai_security_analysis": {
                "overall_security_rating": security_rating,
                "vulnerabilities_detected": vulnerabilities,
                "threat_model": threat_model,
                "remediation_plan": remediation_plan,
                "hardening_recommendations": hardening_recommendations,
                "code_context": code_context,
                "failed_steps": failed_steps,
                "ai_confidence": ai_confidence,
                "inference_efficiency": self._get_inference_efficiency_stats(),
                "model_used": self.model_config["name"],
                "analysis_timestamp": self._get_current_time(),
            },
            "analysis_status": analysis_status,
        }
        return self._validate_security_analysis_result(raw_result)

    def _validate_security_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """统一安全分析结果结构并执行跨类别去重。"""
        if not isinstance(result, dict):
            result = {}
        analysis = result.get("ai_security_analysis", {}) if isinstance(result.get("ai_security_analysis", {}), dict) else {}

        vulnerabilities = analysis.get("vulnerabilities_detected", [])
        vulnerabilities = vulnerabilities if isinstance(vulnerabilities, list) else []
        vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)

        remediation_plan = analysis.get("remediation_plan", {})
        remediation_plan = remediation_plan if isinstance(remediation_plan, dict) else {}
        remediation_plan = self._normalize_remediation_plan(remediation_plan)

        hardening_recommendations = analysis.get("hardening_recommendations", [])
        hardening_recommendations = hardening_recommendations if isinstance(hardening_recommendations, list) else []

        remediation_plan, hardening_recommendations = self._deduplicate_remediation_and_hardening(
            remediation_plan,
            hardening_recommendations,
        )

        analysis["vulnerabilities_detected"] = vulnerabilities
        analysis["remediation_plan"] = remediation_plan
        analysis["hardening_recommendations"] = hardening_recommendations[:10]
        analysis["failed_steps"] = analysis.get("failed_steps", []) if isinstance(analysis.get("failed_steps", []), list) else []

        if not isinstance(analysis.get("overall_security_rating", {}), dict):
            analysis["overall_security_rating"] = {
                "security_score": 5.0,
                "rating_level": "Fair",
                "fallback": True,
            }

        result["ai_security_analysis"] = analysis
        if result.get("analysis_status") not in {"completed", "partial_success", "failed"}:
            result["analysis_status"] = "partial_success"
        return result

    def _normalize_remediation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "immediate_actions": [],
            "short_term_fixes": [],
            "long_term_improvements": [],
            "estimated_effort": str(plan.get("estimated_effort", "unknown")),
        }
        for key in ["immediate_actions", "short_term_fixes", "long_term_improvements"]:
            items = plan.get(key, [])
            if not isinstance(items, list):
                items = []
            for item in items:
                if isinstance(item, dict):
                    normalized[key].append({
                        "priority": self._normalize_priority_level(item.get("priority", "medium")),
                        "vulnerability_id": str(item.get("vulnerability_id", "")).strip(),
                        "type": str(item.get("type", "unknown")).strip(),
                        "fix": str(item.get("fix", "")).strip(),
                    })
                elif isinstance(item, str) and item.strip():
                    normalized[key].append({
                        "priority": "medium",
                        "vulnerability_id": "",
                        "type": "unknown",
                        "fix": item.strip(),
                    })
            normalized[key] = normalized[key][:10]
        return normalized

    def _deduplicate_remediation_and_hardening(
        self,
        remediation_plan: Dict[str, Any],
        hardening_recommendations: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        remediation_texts = set()
        for key in ["immediate_actions", "short_term_fixes", "long_term_improvements"]:
            deduped: List[Dict[str, Any]] = []
            for item in remediation_plan.get(key, []):
                text = re.sub(r"\s+", " ", str(item.get("fix", "")).strip().lower())
                if not text or text in remediation_texts:
                    continue
                remediation_texts.add(text)
                deduped.append(item)
            remediation_plan[key] = deduped[:10]

        hardening_dedup: List[Dict[str, Any]] = []
        seen_h = set()
        for item in hardening_recommendations:
            if not isinstance(item, dict):
                continue
            rec = str(item.get("recommendation", "")).strip()
            key = re.sub(r"\s+", " ", f"{item.get('category','')}|{rec}".lower())
            if not key or key in seen_h:
                continue
            # 若加固建议与修复建议语义重复，则跳过
            rec_norm = re.sub(r"\s+", " ", rec.lower())
            if rec_norm in remediation_texts:
                continue
            seen_h.add(key)
            hardening_dedup.append({
                "category": str(item.get("category", "configuration")).strip(),
                "recommendation": rec,
                "priority": self._normalize_priority_level(item.get("priority", "medium")),
                "implementation": str(item.get("implementation", "")).strip(),
                "source": str(item.get("source", "unknown")).strip() or "unknown",
            })

        return remediation_plan, hardening_dedup

    async def _analyze_code_context(self, code_directory: str, code_content: str = "") -> Dict[str, Any]:
        """分析代码环境和上下文（LLM优先，规则兜底）。"""
        context = {
            "application_type": "unknown",
            "framework_detected": [],
            "database_usage": False,
            "network_operations": False,
            "authentication_present": False,
            "encryption_usage": False,
            "data_sensitivity": "medium"
        }
        
        code_files = await self._read_security_relevant_files(code_directory) if code_directory else []
        merged_snippet = "\n".join((code_files[:3] + [code_content[:1200]])).strip()[:3200]

        # 1) LLM语义识别
        if self.threat_analyzer and merged_snippet:
            try:
                context_prompt = get_prompt(
                    task_type="security",
                    variant="context_analysis",
                    code_snippet=merged_snippet,
                    file_context=(code_directory or "")[:300],
                )
                generated = await self._run_generation_inference(
                    context_prompt,
                    max_new_tokens=220,
                    temperature=0.2,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.threat_analyzer.tokenizer.eos_token_id if self.threat_analyzer else None,
                )
                parsed = self._parse_context_analysis_result(generated)
                if parsed:
                    context.update(parsed)
                    context["source"] = "llm_context_analysis"
            except Exception as e:
                context["context_llm_error"] = str(e)

        # 2) 规则补充，确保关键字段可靠
        for file_content in code_files[:5]:
            lowered = file_content.lower()
            if "flask" in lowered or "django" in lowered or "fastapi" in lowered:
                context["application_type"] = "web_application"
                if "Python Web Framework" not in context["framework_detected"]:
                    context["framework_detected"].append("Python Web Framework")

            if any(k in lowered for k in ["password", "auth", "jwt", "oauth"]):
                context["authentication_present"] = True

            if any(k in lowered for k in ["sql", "database", "sqlite", "mysql", "postgres"]):
                context["database_usage"] = True

            if any(k in lowered for k in ["requests", "http", "socket", "urllib", "aiohttp"]):
                context["network_operations"] = True

            if any(k in lowered for k in ["encrypt", "hash", "bcrypt", "sha", "cipher"]):
                context["encryption_usage"] = True

        return context

    async def _ai_vulnerability_detection(self, code_content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI驱动的漏洞检测"""
        vulnerabilities = []
        
        try:
            # 将代码分块进行分析
            code_chunks = self._split_code_for_analysis(code_content)
            
            for i, chunk in enumerate(code_chunks[:3]):  # 限制分析块数
                security_prompt = get_prompt(
                    task_type="security",
                    variant="vulnerability_detection",
                    code_snippet=chunk
                )
                
                # 使用AI模型进行漏洞分类
                if self.vulnerability_classifier:
                    classification_result = await self._run_classification_inference(
                        f"Security analysis: {chunk[:500]}"
                    )
                    
                    # 解析AI分析结果
                    vuln_data = await self._parse_vulnerability_result(
                        classification_result, chunk, i
                    )
                    
                    if vuln_data:
                        vulnerabilities.append(vuln_data)
                
                # 使用威胁分析器生成详细分析
                if self.threat_analyzer and len(vulnerabilities) < 3:
                    threat_analysis = await self._run_generation_inference(
                        security_prompt,
                        max_new_tokens=220,
                        temperature=0.25,
                        do_sample=True,
                        return_full_text=False,
                        pad_token_id=self.threat_analyzer.tokenizer.eos_token_id if self.threat_analyzer else None,
                    )
                    
                    detailed_vuln = await self._extract_vulnerability_details(
                        threat_analysis, chunk, i
                    )
                    
                    if detailed_vuln:
                        vulnerabilities.append(detailed_vuln)
            
            # AI风险评估和优先级排序
            vulnerabilities = await self._ai_risk_assessment(vulnerabilities)

            # 去重并稳定输出
            vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)
            
        except Exception as e:
            vulnerabilities.append({
                "vulnerability_id": "AI_ERROR_001",
                "type": "analysis_error",
                "description": f"AI分析过程出错: {e}",
                "severity": "info"
            })
        
        return vulnerabilities

    async def _ai_threat_modeling(self, code_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI驱动的威胁建模"""
        try:
            threat_prompt = get_prompt(
                task_type="security",
                variant="threat_modeling",
                system_components=str(context.get("framework_detected", [])),
                data_flow="User Input -> Application -> Database -> Response",
                code_content=code_content[:1000]
            )
            if self.threat_analyzer:
                try:
                    threat_analysis = await self._run_generation_inference(
                        threat_prompt,
                        max_new_tokens=320,
                        temperature=0.3,
                        do_sample=True,
                        return_full_text=False,
                        pad_token_id=self.threat_analyzer.tokenizer.eos_token_id,
                    )
                    threat_model = await self._parse_threat_model(threat_analysis)
                    return threat_model
                except Exception as gen_err:
                    log("ai_security_agent", LogLevel.INFO, f"⚠️ 威胁建模生成失败,降级使用fallback: {gen_err}")
                    return self._fallback_threat_model(context)
            else:
                log("ai_security_agent", LogLevel.INFO, "⚠️ 威胁建模生成器未初始化,使用fallback简化模型")
                return self._fallback_threat_model(context)
        except Exception as e:
            log("ai_security_agent", LogLevel.INFO, f"⚠️ 威胁建模prompt构造或处理异常: {e}")
            return {"error": f"威胁建模失败: {e}"}

    async def _ai_security_rating(self, vulnerabilities: List[Dict[str, Any]], 
                                 threat_model: Dict[str, Any]) -> Dict[str, Any]:
        """AI驱动的安全评级"""
        try:
            rule_track = self._calculate_rule_based_security_score(vulnerabilities, threat_model)
            llm_track = await self._calculate_llm_based_security_score(vulnerabilities, threat_model)
            final_score, fusion_info = self._fuse_security_scores(rule_track, llm_track)

            rating_explanation = await self._generate_rating_explanation(
                final_score, vulnerabilities, threat_model, llm_track
            )

            return {
                "security_score": final_score,
                "rating_level": self._score_to_rating(final_score),
                "rule_based_score": rule_track.get("score"),
                "llm_based_score": llm_track.get("score") if isinstance(llm_track, dict) else None,
                "llm_confidence": llm_track.get("confidence") if isinstance(llm_track, dict) else 0.0,
                "fusion": fusion_info,
                "explanation": rating_explanation,
                "factors_considered": [
                    "漏洞数量和严重程度",
                    "威胁模型分析",
                    "安全控制措施",
                    "代码质量指标"
                ]
            }
            
        except Exception as e:
            return {"error": f"安全评级失败: {e}"}

    async def _ai_remediation_planning(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI生成修复计划"""
        try:
            remediation_plan = {
                "immediate_actions": [],
                "short_term_fixes": [],
                "long_term_improvements": [],
                "estimated_effort": "unknown"
            }

            for vuln in vulnerabilities:
                severity = vuln.get("severity", "low")
                fix_suggestion = await self._generate_fix_suggestion(vuln)

                fix_item = {
                    "priority": self._normalize_priority_level(severity),
                    "vulnerability_id": vuln.get("vulnerability_id", ""),
                    "type": vuln.get("type", "unknown"),
                    "fix": fix_suggestion,
                }

                if severity in ["critical", "high"]:
                    remediation_plan["immediate_actions"].append(fix_item)
                elif severity == "medium":
                    remediation_plan["short_term_fixes"].append(fix_item)
                else:
                    remediation_plan["long_term_improvements"].append(fix_item)

            remediation_plan["estimated_effort"] = await self._estimate_remediation_effort(vulnerabilities)
            return remediation_plan
            
        except Exception as e:
            return {"error": f"修复计划生成失败: {e}"}

    async def _ai_security_hardening(self, code_content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI生成安全加固建议"""
        hardening_recommendations = []

        try:
            custom_recommendations = await self._generate_custom_hardening(code_content, context)
            hardening_recommendations.extend(custom_recommendations)

            # 规则兜底
            if not hardening_recommendations:
                if context.get("application_type") == "web_application":
                    hardening_recommendations.append({
                        "category": "input_validation",
                        "recommendation": "实施严格输入校验与输出编码",
                        "priority": "high",
                        "implementation": "使用参数化查询和输入验证库",
                        "source": "rule_based_fallback",
                    })

                if context.get("database_usage"):
                    hardening_recommendations.append({
                        "category": "database",
                        "recommendation": "数据库账号最小权限并启用审计日志",
                        "priority": "high",
                        "implementation": "拆分读写账号并限制DDL权限",
                        "source": "rule_based_fallback",
                    })

        except Exception as e:
            hardening_recommendations.append({
                "category": "错误",
                "recommendation": f"安全加固建议生成失败: {e}",
                "priority": "info",
                "source": "fallback",
            })

        return hardening_recommendations

    # 辅助方法
    async def _parse_vulnerability_result(self, classification_result: List[Dict], 
                                        code_chunk: str, chunk_index: int) -> Dict[str, Any]:
        """解析漏洞分析结果"""
        if not classification_result:
            return None
            
        result = classification_result[0]
        confidence = float(result.get("score", 0.0) or 0.0)
        label = str(result.get("label", "UNKNOWN")).upper()
        raw_text = f"{label} {code_chunk[:160]}".lower()

        severity = "low"
        if "critical" in raw_text:
            severity = "critical"
        elif "high" in raw_text:
            severity = "high"
        elif "medium" in raw_text:
            severity = "medium"
        elif confidence > 0.88:
            severity = "medium"
        
        # 只有当置信度较高时才报告漏洞
        if confidence > 0.7:
            return {
                "vulnerability_id": f"AI_VULN_{chunk_index:03d}",
                "type": "ai_detected_issue",
                "description": f"AI检测到潜在安全问题 label={label} (置信度: {confidence:.2f})",
                "severity": severity,
                "location": f"代码块 {chunk_index + 1}",
                "code_snippet": code_chunk[:200],
                "ai_confidence": round(confidence, 3),
                "source": "classification",
            }
        
        return None

    def _split_code_for_analysis(self, code_content: str, chunk_size: int = 800) -> List[str]:
        """将代码分割成适合安全分析的块"""
        # 按函数或类分割会更好,这里简化处理
        lines = code_content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _score_to_rating(self, score: float) -> str:
        """将数字分数转换为等级评价"""
        if score >= 8.5:
            return "Excellent"
        elif score >= 7.0:
            return "Good"
        elif score >= 5.0:
            return "Fair"
        elif score >= 3.0:
            return "Poor"
        else:
            return "Critical"

    def _normalize_priority_level(self, raw_priority: Any) -> str:
        value = str(raw_priority or "medium").strip().lower()
        allowed = {"critical", "high", "medium", "low", "info"}
        return value if value in allowed else "medium"

    def _stable_serialize(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(value)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()

    def _build_inference_cache_key(self, inference_type: str, payload: str, kwargs: Dict[str, Any]) -> str:
        kw = self._stable_serialize(kwargs or {})
        raw = f"{inference_type}|{self._hash_text(payload)}|{kw}"
        return self._hash_text(raw)

    def _cache_get(self, key: str, cache_type: str) -> Any:
        if key in self._inference_cache:
            self._inference_cache.move_to_end(key)
            self._cache_stats["hits"] += 1
            self._run_cache_stats["hits"] += 1
            if cache_type == "classification":
                self._cache_stats["classification_hits"] += 1
                self._run_cache_stats["classification_hits"] += 1
            elif cache_type == "generation":
                self._cache_stats["generation_hits"] += 1
                self._run_cache_stats["generation_hits"] += 1
            return self._inference_cache[key]

        self._cache_stats["misses"] += 1
        self._run_cache_stats["misses"] += 1
        if cache_type == "classification":
            self._cache_stats["classification_misses"] += 1
            self._run_cache_stats["classification_misses"] += 1
        elif cache_type == "generation":
            self._cache_stats["generation_misses"] += 1
            self._run_cache_stats["generation_misses"] += 1
        return None

    def _cache_set(self, key: str, value: Any):
        self._inference_cache[key] = value
        self._inference_cache.move_to_end(key)
        while len(self._inference_cache) > self._inference_cache_max_entries:
            self._inference_cache.popitem(last=False)

    def _reset_run_cache_stats(self):
        self._run_cache_stats = {
            "hits": 0,
            "misses": 0,
            "classification_hits": 0,
            "classification_misses": 0,
            "generation_hits": 0,
            "generation_misses": 0,
        }

    def _get_inference_efficiency_stats(self) -> Dict[str, Any]:
        hits = int(self._run_cache_stats.get("hits", 0))
        misses = int(self._run_cache_stats.get("misses", 0))
        total = hits + misses
        return {
            "cache_enabled": True,
            "run_hits": hits,
            "run_misses": misses,
            "run_hit_rate": round((hits / total), 4) if total else 0.0,
            "run_classification_hits": int(self._run_cache_stats.get("classification_hits", 0)),
            "run_generation_hits": int(self._run_cache_stats.get("generation_hits", 0)),
            "cache_entries": len(self._inference_cache),
            "cache_max_entries": self._inference_cache_max_entries,
        }

    def _resolve_model_max_tokens(self, tokenizer: Any, fallback: int = 1024) -> int:
        if tokenizer is None:
            return fallback
        try:
            value = int(getattr(tokenizer, "model_max_length", fallback) or fallback)
            # HuggingFace may use very large sentinel values when limit is unknown.
            if value <= 0 or value > 100000:
                return fallback
            return value
        except Exception:
            return fallback

    def _truncate_text_for_model(self, tokenizer: Any, text: str, max_tokens: int) -> str:
        if not tokenizer or not text:
            return text
        try:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=True,
            )
            input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            if isinstance(input_ids, list) and input_ids:
                return tokenizer.decode(input_ids, skip_special_tokens=True)
        except Exception:
            pass
        return text

    async def _run_classification_inference(self, text: str, **kwargs):
        """在线程中执行同步分类推理，避免阻塞事件循环。"""
        if not self.vulnerability_classifier or not text:
            return []
        tokenizer = getattr(self.vulnerability_classifier, "tokenizer", None)
        model_max = self._resolve_model_max_tokens(tokenizer, fallback=512)
        safe_text = self._truncate_text_for_model(tokenizer, text, model_max)

        effective_kwargs = dict(kwargs)
        effective_kwargs["truncation"] = True
        effective_kwargs["max_length"] = model_max

        cache_key = self._build_inference_cache_key("classification", safe_text, effective_kwargs)
        cached = self._cache_get(cache_key, "classification")
        if cached is not None:
            return cached
        try:
            result = await asyncio.to_thread(self.vulnerability_classifier, safe_text, **effective_kwargs)
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            log("ai_security_agent", LogLevel.WARNING, f"⚠️ 分类推理失败: {e}")
            return []

    async def _run_generation_inference(self, prompt: str, **kwargs):
        """在线程中执行同步生成推理，避免阻塞事件循环。"""
        if not self.threat_analyzer or not prompt:
            return []

        tokenizer = getattr(self.threat_analyzer, "tokenizer", None)
        model_max = self._resolve_model_max_tokens(tokenizer, fallback=1024)

        effective_kwargs = dict(kwargs)
        requested_new = int(effective_kwargs.get("max_new_tokens", 128) or 128)
        requested_new = max(16, min(requested_new, max(16, model_max - 32)))

        # Prevent max_length semantics from conflicting with long prompts.
        if "max_length" in effective_kwargs:
            try:
                max_length_total = int(effective_kwargs.get("max_length") or 0)
            except Exception:
                max_length_total = 0
            if max_length_total > 0 and "max_new_tokens" not in kwargs:
                requested_new = max(16, min(requested_new, max_length_total // 2 if max_length_total >= 32 else 16))
            effective_kwargs.pop("max_length", None)

        input_budget = model_max - requested_new
        if input_budget < 32:
            requested_new = max(16, model_max // 4)
            input_budget = max(32, model_max - requested_new)

        safe_prompt = self._truncate_text_for_model(tokenizer, prompt, input_budget)
        effective_kwargs["max_new_tokens"] = requested_new
        effective_kwargs["truncation"] = True

        cache_key = self._build_inference_cache_key("generation", safe_prompt, effective_kwargs)
        cached = self._cache_get(cache_key, "generation")
        if cached is not None:
            return cached
        try:
            result = await asyncio.to_thread(self.threat_analyzer, safe_prompt, **effective_kwargs)
            self._cache_set(cache_key, result)
            return result
        except Exception as e:
            log("ai_security_agent", LogLevel.WARNING, f"⚠️ 生成推理失败: {e}")
            return []

    def _sanitize_json_like_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return cleaned.strip()

    def _parse_context_analysis_result(self, generated: Any) -> Dict[str, Any]:
        if isinstance(generated, list) and generated and isinstance(generated[0], dict):
            text = str(generated[0].get("generated_text", "")).strip()
        elif isinstance(generated, str):
            text = generated.strip()
        else:
            return {}

        cleaned = self._sanitize_json_like_text(text)
        candidates = [cleaned]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start:end + 1])

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            return {
                "application_type": str(obj.get("application_type", "unknown")).strip() or "unknown",
                "framework_detected": obj.get("framework_detected", []) if isinstance(obj.get("framework_detected", []), list) else [],
                "database_usage": bool(obj.get("database_usage", False)),
                "network_operations": bool(obj.get("network_operations", False)),
                "authentication_present": bool(obj.get("authentication_present", False)),
                "encryption_usage": bool(obj.get("encryption_usage", False)),
                "data_sensitivity": str(obj.get("data_sensitivity", "medium")).strip().lower() or "medium",
                "reasoning": str(obj.get("reasoning", "")).strip(),
            }
        return {}

    def _deduplicate_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        priority_rank = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}
        for item in vulnerabilities:
            if not isinstance(item, dict):
                continue
            key = re.sub(r"\s+", " ", f"{item.get('type','')}|{item.get('location','')}|{item.get('description','')}".strip().lower())
            if key not in seen:
                seen[key] = item
                continue
            old = seen[key]
            old_rank = priority_rank.get(str(old.get("severity", "low")).lower(), 2)
            new_rank = priority_rank.get(str(item.get("severity", "low")).lower(), 2)
            if new_rank > old_rank:
                seen[key] = item
        return sorted(seen.values(), key=lambda x: priority_rank.get(str(x.get("severity", "low")).lower(), 2), reverse=True)[:10]

    def _calculate_rule_based_security_score(self, vulnerabilities: List[Dict[str, Any]], threat_model: Dict[str, Any]) -> Dict[str, Any]:
        base_score = 9.5
        severity_penalty = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.4, "info": 0.1}
        penalty = 0.0
        for vuln in vulnerabilities:
            sev = str(vuln.get("severity", "low")).lower()
            penalty += severity_penalty.get(sev, 0.4)
        threat_risk = float(threat_model.get("overall_risk_score", 5.0) or 5.0)
        score = max(0.0, min(10.0, base_score - penalty - ((threat_risk - 5.0) * 0.35)))
        return {"score": round(score, 3), "penalty": round(penalty, 3), "threat_risk": round(threat_risk, 3)}

    async def _calculate_llm_based_security_score(self, vulnerabilities: List[Dict[str, Any]], threat_model: Dict[str, Any]) -> Dict[str, Any]:
        if not self.threat_analyzer:
            return {"available": False, "score": None, "confidence": 0.0, "reason": "generation_model_unavailable"}

        try:
            prompt = get_prompt(
                task_type="security",
                variant="security_rating",
                vulnerability_summary=json.dumps(vulnerabilities[:6], ensure_ascii=False),
                threat_summary=json.dumps(threat_model, ensure_ascii=False),
            )
            generated = await self._run_generation_inference(
                prompt,
                max_new_tokens=220,
                temperature=0.2,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.threat_analyzer.tokenizer.eos_token_id,
            )
            parsed = self._parse_llm_security_rating(generated)
            return parsed if parsed else {"available": False, "score": None, "confidence": 0.0, "reason": "rating_parse_failed"}
        except Exception as e:
            return {"available": False, "score": None, "confidence": 0.0, "reason": f"llm_rating_failed: {e}"}

    def _parse_llm_security_rating(self, generated: Any) -> Dict[str, Any]:
        if isinstance(generated, list) and generated and isinstance(generated[0], dict):
            text = str(generated[0].get("generated_text", "")).strip()
        elif isinstance(generated, str):
            text = generated.strip()
        else:
            return {}
        cleaned = self._sanitize_json_like_text(text)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        candidates = [cleaned]
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start:end + 1])
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            score = obj.get("llm_security_score")
            confidence = obj.get("confidence", 0.0)
            if not isinstance(score, (int, float)):
                continue
            return {
                "available": True,
                "score": round(max(0.0, min(10.0, float(score))), 3),
                "confidence": round(max(0.0, min(1.0, float(confidence or 0.0))), 3),
                "primary_risks": obj.get("primary_risks", []),
                "explanation": str(obj.get("explanation", "")).strip(),
            }
        return {}

    def _parse_fix_suggestion(self, generated: Any) -> str:
        if isinstance(generated, list) and generated and isinstance(generated[0], dict):
            text = str(generated[0].get("generated_text", "")).strip()
        elif isinstance(generated, str):
            text = generated.strip()
        else:
            return ""

        cleaned = self._sanitize_json_like_text(text)
        candidates = [cleaned]
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            candidates.append(cleaned[s:e + 1])

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            steps = obj.get("fix_steps", [])
            if isinstance(steps, list) and steps:
                step_text = "；".join(str(s).strip() for s in steps if str(s).strip())
            else:
                step_text = ""
            suggestion = obj.get("code_level_recommendation") or obj.get("verification") or ""
            combined = "；".join(filter(None, [step_text, str(suggestion).strip()]))
            if combined.strip():
                return combined.strip()

        return ""

    def _parse_hardening_recommendations(self, generated: Any) -> List[Dict[str, Any]]:
        if isinstance(generated, list) and generated and isinstance(generated[0], dict):
            text = str(generated[0].get("generated_text", "")).strip()
        elif isinstance(generated, str):
            text = generated.strip()
        else:
            return []

        cleaned = self._sanitize_json_like_text(text)
        candidates = [cleaned]
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            candidates.append(cleaned[s:e + 1])

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            items = obj.get("hardening_recommendations", [])
            if not isinstance(items, list):
                continue

            normalized: List[Dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized.append({
                    "category": str(item.get("category", "configuration")).strip(),
                    "priority": self._normalize_priority_level(item.get("priority", "medium")),
                    "recommendation": str(item.get("recommendation", "")).strip(),
                    "implementation": str(item.get("implementation", "")).strip(),
                    "source": "llm_hardening",
                })
            if normalized:
                return normalized
        return []

    def _fuse_security_scores(self, rule_track: Dict[str, Any], llm_track: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        rule_score = float(rule_track.get("score", 5.0) or 5.0)
        if not isinstance(llm_track, dict) or not llm_track.get("available") or llm_track.get("score") is None:
            return rule_score, {"strategy": "rule_only_fallback", "rule_weight": 1.0, "llm_weight": 0.0}
        llm_score = float(llm_track.get("score", rule_score))
        llm_conf = float(llm_track.get("confidence", 0.0) or 0.0)
        llm_weight = 0.65 if llm_conf >= 0.8 else 0.55 if llm_conf >= 0.55 else 0.4
        rule_weight = 1.0 - llm_weight
        score = max(0.0, min(10.0, llm_weight * llm_score + rule_weight * rule_score))
        return round(score, 3), {"strategy": "hybrid_fusion", "rule_weight": round(rule_weight, 3), "llm_weight": round(llm_weight, 3)}

    async def _estimate_remediation_effort(self, vulnerabilities: List[Dict[str, Any]]) -> str:
        critical = sum(1 for v in vulnerabilities if str(v.get("severity", "")).lower() == "critical")
        high = sum(1 for v in vulnerabilities if str(v.get("severity", "")).lower() == "high")
        total = len(vulnerabilities)
        if critical >= 2 or total >= 8:
            return "1-2 weeks"
        if critical >= 1 or high >= 2 or total >= 5:
            return "3-5 days"
        if total >= 1:
            return "1-2 days"
        return "0.5-1 day"

    async def _read_security_relevant_files(self, code_directory: str) -> List[str]:
        """读取安全相关的代码文件，增强Python/C/C++支持"""
        import os
        
        security_files = []
        # 扩展支持的文件类型，重点加强Python和C/C++
        supported_extensions = [
            '.py',           # Python
            '.cpp', '.cxx', '.cc',  # C++
            '.c', '.h', '.hpp',     # C and headers
            '.js', '.java', '.php'
        ]
        
        security_keywords = ["auth", "login", "password", "security", "crypto", "hash"]
        
        try:
            for root, dirs, files in os.walk(code_directory):
                for file in files[:15]:  # 增加文件数量限制
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    # 检查文件扩展名和支持的安全关键词
                    if (file_ext in supported_extensions or 
                        any(keyword in file.lower() for keyword in security_keywords)):
                        
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # 添加文件信息便于分析
                                file_header = f"// File: {file} ({file_ext})\n"
                                file_header += f"// Path: {file_path}\n"
                                
                                security_files.append(file_header + content)
                                
                        except UnicodeDecodeError:
                            # 尝试GBK编码
                            try:
                                with open(file_path, 'r', encoding='gbk') as f:
                                    content = f.read()
                                    file_header = f"// File: {file} ({file_ext})\n"
                                    file_header += f"// Path: {file_path}\n"
                                    security_files.append(file_header + content)
                            except Exception:
                                continue
                        except Exception as e:
                            log("ai_security_agent", LogLevel.INFO, f"读取文件失败 {file_path}: {e}")
                            continue
                            
                if len(security_files) >= 10:  # 增加文件数量限制
                    break
                    
        except Exception as e:
            log("ai_security_agent", LogLevel.INFO, f"读取安全相关文件时出错: {e}")
        
        return security_files

    def _get_current_time(self) -> str:
        """获取当前时间戳 (补充缺失的方法以避免运行时报错)"""
        import datetime
        return datetime.datetime.now().isoformat()

    # --- Newly added helper / AI synthesis methods to avoid missing attribute errors ---
    async def _ai_risk_assessment(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为检测到的漏洞执行简易风险评估与排序。
        - 计算 risk_score (0-10)
        - 规范 severity 字段 (critical/high/medium/low/info)
        - 根据评分进行排序
        """
        assessed: List[Dict[str, Any]] = []
        for v in vulnerabilities:
            sev = v.get("severity") or "info"
            # 基础严重度权重
            base = {
                "critical": 9.0,
                "high": 7.5,
                "medium": 5.5,
                "low": 3.0,
                "info": 1.0
            }.get(sev, 2.0)
            # 利用置信度提升
            confidence = float(v.get("ai_confidence", 0.5))
            risk_score = min(10.0, base + confidence * 1.5)
            v["risk_score"] = round(risk_score, 2)
            # 若 severity 缺失, 按风险评分推导
            if sev not in ["critical", "high", "medium", "low", "info"]:
                if risk_score >= 8.5:
                    v["severity"] = "high"
                elif risk_score >= 6.5:
                    v["severity"] = "medium"
                elif risk_score >= 4.0:
                    v["severity"] = "low"
                else:
                    v["severity"] = "info"
            assessed.append(v)
        # 按风险排序
        assessed.sort(key=lambda x: x.get("risk_score", 0.0), reverse=True)
        return assessed

    async def _generate_rating_explanation(
        self,
        final_score: float,
        vulnerabilities: List[Dict[str, Any]],
        threat_model: Dict[str, Any],
        llm_track: Dict[str, Any] = None,
    ) -> str:
        """生成安全评分解释文本。"""
        high_count = sum(1 for v in vulnerabilities if v.get("severity") in {"critical", "high"})
        medium_count = sum(1 for v in vulnerabilities if v.get("severity") == "medium")
        low_count = sum(1 for v in vulnerabilities if v.get("severity") == "low")
        rating_level = self._score_to_rating(final_score)
        stride_summary = threat_model.get("stride_summary") or threat_model.get("summary") or "(无详细威胁模型)"
        base_text = (
            f"总体安全评分 {final_score:.2f} ({rating_level}). "
            f"高/严重漏洞: {high_count}, 中等: {medium_count}, 低: {low_count}. "
            f"威胁建模摘要: {stride_summary}. "
            "评分基于发现漏洞的数量与严重度、威胁类别覆盖及代码上下文中的安全控制迹象。"
        )

        if isinstance(llm_track, dict):
            llm_explanation = str(llm_track.get("explanation", "")).strip()
            if llm_explanation:
                return llm_explanation

        if self.threat_analyzer:
            try:
                prompt = get_prompt(
                    task_type="security",
                    variant="security_rating",
                    vulnerability_summary=json.dumps(vulnerabilities[:6], ensure_ascii=False),
                    threat_summary=json.dumps(threat_model, ensure_ascii=False),
                )
                generated = await self._run_generation_inference(
                    prompt,
                    max_new_tokens=120,
                    temperature=0.2,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.threat_analyzer.tokenizer.eos_token_id,
                )
                parsed = self._parse_llm_security_rating(generated)
                if parsed and parsed.get("explanation"):
                    return parsed["explanation"]
            except Exception:
                pass

        return base_text

    async def _generate_fix_suggestion(self, vuln: Dict[str, Any]) -> str:
        """根据漏洞条目生成修复建议（LLM优先，规则兜底）。"""
        if self.threat_analyzer:
            try:
                prompt = get_prompt(
                    task_type="security",
                    variant="remediation_fix",
                    vulnerability_item=json.dumps(vuln, ensure_ascii=False),
                )
                generated = await self._run_generation_inference(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.25,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.threat_analyzer.tokenizer.eos_token_id,
                )
                llm_fix = self._parse_fix_suggestion(generated)
                if llm_fix:
                    return llm_fix
            except Exception:
                pass

        vtype = (vuln.get("type") or "issue").lower()
        desc = (vuln.get("description") or "").lower()
        if "injection" in vtype or "sql" in desc:
            return "使用参数化查询并严格校验/转义所有外部输入。"
        if "xss" in vtype or "script" in desc:
            return "对输出进行HTML转义并使用内容安全策略(CSP)。"
        if "auth" in desc or "login" in desc:
            return "实施强密码策略并增加多因素认证，限制失败尝试。"
        if "crypto" in desc or "encrypt" in desc or "hash" in desc:
            return "使用经验证的库(如 hashlib/cryptography)并应用盐值+迭代。"
        if "config" in desc:
            return "检查默认配置并最小化权限，移除未使用端点或调试标志。"
        # 置信度高且无匹配规则 -> 通用建议
        if vuln.get("ai_confidence", 0) > 0.8:
            return "审查此高置信度条目，添加输入验证与访问控制审查。"
        return "进行代码审查并添加输入验证、错误处理与最小权限策略。"

    async def _generate_custom_hardening(self, code_content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于代码和上下文生成定制加固建议。"""
        if self.threat_analyzer:
            try:
                prompt = get_prompt(
                    task_type="security",
                    variant="hardening",
                    code_snippet=code_content[:2500],
                    context_summary=json.dumps(context, ensure_ascii=False),
                )
                generated = await self._run_generation_inference(
                    prompt,
                    max_new_tokens=260,
                    temperature=0.25,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.threat_analyzer.tokenizer.eos_token_id,
                )
                parsed = self._parse_hardening_recommendations(generated)
                if parsed:
                    return parsed[:8]
            except Exception:
                pass

        recs: List[Dict[str, Any]] = []
        lowered = code_content.lower()
        def add(category, recommendation, priority, implementation):
            recs.append({
                "category": category,
                "recommendation": recommendation,
                "priority": priority,
                "implementation": implementation,
                "source": "rule_based_fallback",
            })
        if "exec(" in lowered or "eval(" in lowered:
            add("危险调用", "避免使用 eval/exec, 改为显式逻辑或安全解析库", "high", "移除或替换 eval/exec")
        if "subprocess" in lowered:
            add("命令执行", "使用安全参数列表并避免 shell=True", "medium", "subprocess.run([...], shell=False)")
        if "password" in lowered and "hash" not in lowered:
            add("凭据处理", "确保对密码进行哈希存储 (bcrypt/argon2)", "high", "集成 passlib 或 argon2 库")
        if "http://" in lowered:
            add("传输安全", "升级到 HTTPS 以防止中间人攻击", "medium", "替换所有 http:// 链接为 https://")
        if "debug" in lowered:
            add("调试配置", "生产环境关闭调试模式与详细错误输出", "low", "设置 DEBUG=False 并使用统一错误处理")
        # 去重与有限长度
        return recs[:8]

    # --- Threat modeling helper methods (fix for missing _fallback_threat_model) ---
    def _fallback_threat_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """在未加载生成模型时提供简易 STRIDE 威胁模型。"""
        stride_categories = [
            ("Spoofing", "可能缺少强身份认证" if not context.get("authentication_present") else "身份认证迹象存在"),
            ("Tampering", "未发现完整性校验逻辑"),
            ("Repudiation", "缺少审计/日志机制迹象"),
            ("Information Disclosure", "潜在中等风险; 未发现加密调用" if not context.get("encryption_usage") else "存在加密迹象"),
            ("Denial of Service", "资源控制逻辑未显式检测"),
            ("Elevation of Privilege", "权限边界未明确")
        ]
        analyzed = []
        total_risk = 0.0
        for name, desc in stride_categories:
            # 简单风险打分: 根据上下文缺失情况
            base = 5.0
            if "迹象存在" in desc:
                base -= 2.0
            analyzed.append({
                "category": name,
                "summary": desc,
                "risk_score": base
            })
            total_risk += base
        overall = round(total_risk / len(analyzed), 2)
        return {
            "stride_analysis": analyzed,
            "overall_risk_score": overall,
            "stride_summary": f"六类平均风险评分 {overall}",
            "model": "fallback"
        }

    async def _parse_threat_model(self, threat_analysis: Any) -> Dict[str, Any]:
        """解析模型生成的威胁建模文本为结构化数据。"""
        # 统一为文本
        if isinstance(threat_analysis, list):
            # transformers text-generation 常为 list[{'generated_text': str}]
            text = threat_analysis[0].get("generated_text", "") if threat_analysis else ""
        elif isinstance(threat_analysis, dict):
            text = threat_analysis.get("generated_text", str(threat_analysis))
        else:
            text = str(threat_analysis)
        lowered = text.lower()
        def extract_section(keyword: str) -> str:
            # 粗糙截取: 从关键字到下一个换行或 160 字符
            idx = lowered.find(keyword.lower())
            if idx == -1:
                return "未提及"
            snippet = text[idx: idx + 180]
            return snippet.split('\n')[0][:160]
        categories = ["Spoofing", "Tampering", "Repudiation", "Information Disclosure", "Denial of Service", "Elevation of Privilege"]
        stride_analysis = []
        total = 0.0
        for cat in categories:
            detail = extract_section(cat)
            # 简单评分: 出现则 4-6 之间随机, 未出现则 5 默认。这里用规则: 长度>20 => 5.5 else 4.5
            score = 5.5 if len(detail) > 20 and detail != "未提及" else 4.5
            total += score
            stride_analysis.append({
                "category": cat,
                "summary": detail,
                "risk_score": score
            })
        overall = round(total / len(stride_analysis), 2)
        return {
            "stride_analysis": stride_analysis,
            "overall_risk_score": overall,
            "stride_summary": f"生成文本解析平均风险 {overall}",
            "raw_text_length": len(text),
            "model": "generated"
        }

    async def _extract_vulnerability_details(self, threat_analysis: Any, code_chunk: str, chunk_index: int) -> Dict[str, Any]:
        """从生成的威胁分析中提取潜在漏洞详情（LLM文本解析 + 规则兜底）。"""
        if not threat_analysis:
            return None
        # 使用解析后的文本长度与关键词作为置信度估计
        if isinstance(threat_analysis, list):
            text = threat_analysis[0].get("generated_text", "")
        elif isinstance(threat_analysis, dict):
            text = threat_analysis.get("generated_text", str(threat_analysis))
        else:
            text = str(threat_analysis)
        cleaned = self._sanitize_json_like_text(text)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(cleaned[start:end + 1])
                if isinstance(obj, dict):
                    vuln_type = obj.get("type") or obj.get("vulnerability_type") or "generated_threat_indicator"
                    severity = self._normalize_priority_level(obj.get("severity", "low"))
                    description = obj.get("description") or obj.get("issue") or "生成威胁文本指出潜在风险"
                    return {
                        "vulnerability_id": f"AI_GEN_{chunk_index:03d}",
                        "type": str(vuln_type),
                        "description": str(description),
                        "severity": severity,
                        "location": str(obj.get("location", f"代码块 {chunk_index + 1}")),
                        "code_snippet": code_chunk[:160],
                        "ai_confidence": 0.72,
                        "source": "generation_json",
                    }
            except Exception:
                pass

        keywords = ["inject", "xss", "csrf", "overflow", "leak", "auth bypass", "deserialization"]
        found = [k for k in keywords if k in text.lower()]
        if not found:
            return None
        return {
            "vulnerability_id": f"AI_GEN_{chunk_index:03d}",
            "type": "generated_threat_indicator",
            "description": f"生成威胁文本中提及关键词: {', '.join(found)}",
            "severity": "medium" if len(found) > 1 else "low",
            "location": f"代码块 {chunk_index + 1}",
            "code_snippet": code_chunk[:160],
            "ai_confidence": min(0.95, 0.6 + 0.1 * len(found)),
            "source": "generation_keyword",
        }
    # ...existing code...
    # --- Backward compatibility layer for legacy synchronous tests ---
    def __getattr__(self, item):
        removed = {
            'scan_vulnerabilities', 'detect_sql_injection', 'detect_xss',
            'detect_insecure_deserialization', 'detect_hardcoded_secrets',
            'calculate_security_score', 'generate_security_report', 'analyze_file'
        }
        if item in removed:
            raise AttributeError(
                f"'{item}' removed. Use async workflow: send 'security_analysis_request' and await 'analysis_result'."
            )
        raise AttributeError(item)
    
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a security analysis task (required by BaseAgent)."""
        return await self._ai_driven_security_analysis(
            task_data.get("code_content", ""),
            task_data.get("code_directory", "")
        )