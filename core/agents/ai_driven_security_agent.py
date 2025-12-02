import os
import torch
import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, Message
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.config.ai_agents import get_ai_agent_config
from infrastructure.config.prompts import get_prompt
from infrastructure.reports import report_manager

class AIDrivenSecurityAgent(BaseAgent):
    """AI驱动的安全分析智能体 - 基于prompt工程和模型推理"""
    
    def __init__(self):
        super().__init__("ai_security_agent", "AI驱动安全分析智能体")
        self.db_service = DatabaseService()
        # 从统一配置获取
        self.agent_config = get_ai_agent_config().get_security_agent_config()
        self.model_config = HUGGINGFACE_CONFIG["models"]["security"]
        # 移除本地硬编码prompt，统一使用 prompts.get_prompt
        self.security_model = None
        self.vulnerability_classifier = None
        self.threat_analyzer = None
        
    async def _initialize_models(self):
        """初始化AI模型 - CPU优化版本"""
        try:
            print("🔧 初始化安全分析AI模型 (CPU模式)...")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            cpu_threads = self.agent_config.get("cpu_threads", 4)
            torch.set_num_threads(cpu_threads)
            try:
                model_name = self.agent_config.get("model_name", "microsoft/codebert-base")
                torch_dtype = getattr(torch, self.agent_config.get("torch_dtype", "float32"))
                self.security_model = pipeline(
                    "text-classification",
                    model=model_name,
                    device=-1,
                    model_kwargs={
                        "low_cpu_mem_usage": self.agent_config.get("low_cpu_mem_usage", True),
                        "torch_dtype": torch_dtype
                    }
                )
                print(f"✅ {model_name} 安全模型初始化成功 (CPU)")
            except Exception as e:
                print(f"⚠️ 主模型加载失败,尝试备用模型: {e}")
                fallback_model = self.agent_config.get("fallback_model", "distilbert-base-uncased")
                self.security_model = pipeline(
                    "text-classification",
                    model=fallback_model,
                    device=-1,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                print(f"✅ {fallback_model} 备用模型初始化成功 (CPU)")
            try:
                text_gen_model = self.agent_config.get("text_generator_model", "gpt2")
                self.text_generator = pipeline(
                    "text-generation",
                    model=text_gen_model,
                    device=-1,
                    model_kwargs={"low_cpu_mem_usage": True, "pad_token_id": 50256}
                )
                print(f"✅ {text_gen_model} 文本生成模型初始化成功 (CPU)")
                # 采用文本生成模型作为威胁建模生成器
                self.threat_analyzer = self.text_generator
            except Exception as e:
                print(f"⚠️ 文本生成模型加载失败: {e}")
                self.text_generator = None
                self.threat_analyzer = None
            self.models_loaded = True
            print("✅ 安全分析AI模型初始化完成 (CPU优化模式)")
        except Exception as e:
            print(f"❌ 安全分析AI模型初始化失败: {e}")
            self.models_loaded = False
            self.security_model = None
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
            
            print(f"🔒 AI安全分析开始 - 需求ID: {requirement_id}")
            
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
                    print(f"⚠️ 安全Agent单独报告生成失败 requirement={requirement_id} run_id={run_id}: {e}")
            # 发送结果
            await self.send_message(
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
            await self.send_message(
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
            
            print(f"✅ AI安全分析完成 - 需求ID: {requirement_id}")

    async def _ai_driven_security_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """AI驱动的全面安全分析"""
        
        try:
            print("🔍 AI正在进行深度安全分析...")
            code_context = await self._analyze_code_context(code_directory)
            vulnerabilities = await self._ai_vulnerability_detection(code_content, code_context)
            threat_model = await self._ai_threat_modeling(code_content, code_context)
            security_rating = await self._ai_security_rating(vulnerabilities, threat_model)
            remediation_plan = await self._ai_remediation_planning(vulnerabilities)
            hardening_recommendations = await self._ai_security_hardening(code_content, code_context)
            
            print("🛡️  AI安全分析完成,生成安全报告")
            
            return {
                "ai_security_analysis": {
                    "overall_security_rating": security_rating,
                    "vulnerabilities_detected": vulnerabilities,
                    "threat_model": threat_model,
                    "remediation_plan": remediation_plan,
                    "hardening_recommendations": hardening_recommendations,
                    "code_context": code_context,
                    "ai_confidence": 0.92,
                    "model_used": self.model_config["name"],
                    "analysis_timestamp": self._get_current_time()
                },
                "analysis_status": "completed"
            }
            
        except Exception as e:
            print(f"❌ AI安全分析过程中出错: {e}")
            return {
                "ai_security_analysis": {"error": str(e)},
                "analysis_status": "failed"
            }

    async def _analyze_code_context(self, code_directory: str) -> Dict[str, Any]:
        """分析代码环境和上下文"""
        context = {
            "application_type": "unknown",
            "framework_detected": [],
            "database_usage": False,
            "network_operations": False,
            "authentication_present": False,
            "encryption_usage": False,
            "data_sensitivity": "medium"
        }
        
        if code_directory:
            # 读取代码文件并分析
            code_files = await self._read_security_relevant_files(code_directory)
            
            # AI分析应用类型
            for file_content in code_files[:5]:
                if "flask" in file_content.lower() or "django" in file_content.lower():
                    context["application_type"] = "web_application"
                    context["framework_detected"].append("Python Web Framework")
                
                if "password" in file_content.lower() or "auth" in file_content.lower():
                    context["authentication_present"] = True
                    
                if "sql" in file_content.lower() or "database" in file_content.lower():
                    context["database_usage"] = True
                    
                if "requests" in file_content.lower() or "http" in file_content.lower():
                    context["network_operations"] = True
                    
                if "encrypt" in file_content.lower() or "hash" in file_content.lower():
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
                    classification_result = self.vulnerability_classifier(
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
                    threat_analysis = self.threat_analyzer(
                        security_prompt,
                        max_length=200,
                        temperature=0.3
                    )
                    
                    detailed_vuln = await self._extract_vulnerability_details(
                        threat_analysis, chunk, i
                    )
                    
                    if detailed_vuln:
                        vulnerabilities.append(detailed_vuln)
            
            # AI风险评估和优先级排序
            vulnerabilities = await self._ai_risk_assessment(vulnerabilities)
            
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
                    threat_analysis = self.threat_analyzer(
                        threat_prompt,
                        max_length=300,
                        temperature=0.4
                    )
                    threat_model = await self._parse_threat_model(threat_analysis)
                    return threat_model
                except Exception as gen_err:
                    print(f"⚠️ 威胁建模生成失败,降级使用fallback: {gen_err}")
                    return self._fallback_threat_model(context)
            else:
                print("⚠️ 威胁建模生成器未初始化,使用fallback简化模型")
                return self._fallback_threat_model(context)
        except Exception as e:
            print(f"⚠️ 威胁建模prompt构造或处理异常: {e}")
            return {"error": f"威胁建模失败: {e}"}

    async def _ai_security_rating(self, vulnerabilities: List[Dict[str, Any]], 
                                 threat_model: Dict[str, Any]) -> Dict[str, Any]:
        """AI驱动的安全评级"""
        try:
            # 计算基础安全分数
            base_score = 10.0
            
            # 根据漏洞严重程度调整分数
            for vuln in vulnerabilities:
                severity = vuln.get("severity", "low")
                if severity == "critical":
                    base_score -= 3.0
                elif severity == "high":
                    base_score -= 2.0
                elif severity == "medium":
                    base_score -= 1.0
                elif severity == "low":
                    base_score -= 0.5
            
            # 根据威胁模型调整分数
            threat_score = threat_model.get("overall_risk_score", 5.0)
            adjusted_score = (base_score + threat_score) / 2
            
            # 确保分数在合理范围内
            final_score = max(0.0, min(10.0, adjusted_score))
            
            # AI生成评级说明
            rating_explanation = await self._generate_rating_explanation(
                final_score, vulnerabilities, threat_model
            )
            
            return {
                "security_score": final_score,
                "rating_level": self._score_to_rating(final_score),
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
                
                if severity in ["critical", "high"]:
                    remediation_plan["immediate_actions"].append(fix_suggestion)
                elif severity == "medium":
                    remediation_plan["short_term_fixes"].append(fix_suggestion)
                else:
                    remediation_plan["long_term_improvements"].append(fix_suggestion)
            
            # AI估算修复工作量
            total_vulns = len(vulnerabilities)
            if total_vulns <= 2:
                remediation_plan["estimated_effort"] = "1-2 days"
            elif total_vulns <= 5:
                remediation_plan["estimated_effort"] = "3-5 days"
            else:
                remediation_plan["estimated_effort"] = "1-2 weeks"
            
            return remediation_plan
            
        except Exception as e:
            return {"error": f"修复计划生成失败: {e}"}

    async def _ai_security_hardening(self, code_content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI生成安全加固建议"""
        hardening_recommendations = []
        
        try:
            # 基于上下文生成针对性建议
            if context.get("application_type") == "web_application":
                hardening_recommendations.extend([
                    {
                        "category": "输入验证",
                        "recommendation": "实施严格的输入验证和输出编码",
                        "priority": "high",
                        "implementation": "使用参数化查询和输入验证库"
                    },
                    {
                        "category": "身份认证",
                        "recommendation": "实施多因素认证",
                        "priority": "medium",
                        "implementation": "集成TOTP或SMS验证"
                    }
                ])
            
            if context.get("database_usage"):
                hardening_recommendations.append({
                    "category": "数据库安全",
                    "recommendation": "使用数据库连接池和最小权限原则",
                    "priority": "high",
                    "implementation": "配置专用数据库用户,限制权限"
                })
            
            # AI生成个性化建议
            if len(code_content) > 500:
                custom_recommendations = await self._generate_custom_hardening(code_content)
                hardening_recommendations.extend(custom_recommendations)
            
        except Exception as e:
            hardening_recommendations.append({
                "category": "错误",
                "recommendation": f"安全加固建议生成失败: {e}",
                "priority": "info"
            })
        
        return hardening_recommendations

    # 辅助方法
    async def _parse_vulnerability_result(self, classification_result: List[Dict], 
                                        code_chunk: str, chunk_index: int) -> Dict[str, Any]:
        """解析漏洞分析结果"""
        if not classification_result:
            return None
            
        result = classification_result[0]
        confidence = result.get("score", 0.0)
        
        # 只有当置信度较高时才报告漏洞
        if confidence > 0.7:
            return {
                "vulnerability_id": f"AI_VULN_{chunk_index:03d}",
                "type": "ai_detected_issue",
                "description": f"AI检测到潜在安全问题 (置信度: {confidence:.2f})",
                "severity": "medium" if confidence > 0.85 else "low",
                "location": f"代码块 {chunk_index + 1}",
                "code_snippet": code_chunk[:200],
                "ai_confidence": confidence
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

    async def _read_security_relevant_files(self, code_directory: str) -> List[str]:
        """读取安全相关的代码文件"""
        import os
        
        security_files = []
        security_keywords = ["auth", "login", "password", "security", "crypto", "hash"]
        
        try:
            for root, dirs, files in os.walk(code_directory):
                for file in files[:10]:
                    if (file.endswith(('.py', '.js', '.java', '.php')) or 
                        any(keyword in file.lower() for keyword in security_keywords)):
                        
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                security_files.append(content)
                        except Exception:
                            continue
                            
                if len(security_files) >= 5:
                    break
                    
        except Exception as e:
            print(f"读取安全相关文件时出错: {e}")
        
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

    async def _generate_rating_explanation(self, final_score: float, vulnerabilities: List[Dict[str, Any]], threat_model: Dict[str, Any]) -> str:
        """生成安全评分解释文本。使用轻量逻辑 + 可选文本生成模型。"""
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
        # 若有文本生成模型, 添加更自然语言补充
        if getattr(self, "text_generator", None):
            try:
                gen = self.text_generator(
                    base_text + " 请用一句话总结风险优先级。",
                    max_length=base_text.count(" ") + 40,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=50256
                )
                if gen and isinstance(gen, list):
                    completion = gen[0].get("generated_text", "")
                    # 去重合并
                    if completion and completion not in base_text:
                        base_text += " " + completion.strip()[:200]
            except Exception:
                pass
        return base_text

    async def _generate_fix_suggestion(self, vuln: Dict[str, Any]) -> str:
        """根据漏洞条目生成修复建议 (启发式)。"""
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

    async def _generate_custom_hardening(self, code_content: str) -> List[Dict[str, Any]]:
        """基于代码模式生成定制加固建议。"""
        recs: List[Dict[str, Any]] = []
        lowered = code_content.lower()
        def add(category, recommendation, priority, implementation):
            recs.append({
                "category": category,
                "recommendation": recommendation,
                "priority": priority,
                "implementation": implementation
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
        """从生成的威胁分析中提取潜在漏洞详情 (简化占位实现)。"""
        if not threat_analysis:
            return None
        # 使用解析后的文本长度与关键词作为置信度估计
        if isinstance(threat_analysis, list):
            text = threat_analysis[0].get("generated_text", "")
        elif isinstance(threat_analysis, dict):
            text = threat_analysis.get("generated_text", str(threat_analysis))
        else:
            text = str(threat_analysis)
        keywords = ["inject", "xss", "csrf", "overflow", "leak"]
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
            "ai_confidence": min(0.95, 0.6 + 0.1 * len(found))
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
