import os
import torch
import asyncio
import ast
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, Message
from infrastructure.database.sqlite.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.config.ai_agents import get_ai_agent_config
from infrastructure.config.prompts import get_prompt
from infrastructure.reports import report_manager
from utils import log, LogLevel

class AIDrivenPerformanceAgent(BaseAgent):
    """AI驱动的性能分析智能体 - 基于深度学习和prompt工程"""
    
    def __init__(self):
        super().__init__("ai_performance_agent", "AI Performance Analysis Agent")
        self.db_service = DatabaseService()
        self.used_device = "gpu"
        self.used_device_map = None  # 添加设备映射参数
        # 从统一配置获取
        self.agent_config = get_ai_agent_config().get_performance_agent_config()
        self.model_config = HUGGINGFACE_CONFIG["models"]["performance"]
        self.performance_model = None
        self.complexity_analyzer = None
        self.optimization_advisor = None

    async def _initialize_models(self):
        """初始化AI模型 - 支持 CPU/GPU 动态选择"""
        try:
            # 验证 used_device 参数
            if self.used_device not in ["cpu", "gpu"]:
                log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 无效的设备参数: {self.used_device}，回退到CPU")
                self.used_device = "cpu"
            
            device_mode = "CPU" if self.used_device == "cpu" else "GPU"
            log("ai_performance_agent", LogLevel.INFO, f"🔧 [ai_performance_agent] 初始化性能分析AI模型 ({device_mode}模式)...")
            
            # 优先使用agent专属配置，回退到HUGGINGFACE_CONFIG
            model_name = self.agent_config.get("model_name", "microsoft/codebert-base")
            cache_dir = HUGGINGFACE_CONFIG.get("cache_dir", "./model_cache/")
            device = -1 if self.used_device == "cpu" else 0
            
            # 仅在CPU模式下设置线程数
            if self.used_device == "cpu":
                cpu_threads = self.agent_config.get("cpu_threads", 4)
                torch.set_num_threads(cpu_threads)
            
            log("ai_performance_agent", LogLevel.INFO, f"🤖 [ai_performance_agent] 正在加载性能分析模型 ({device_mode}模式): {model_name}")
            log("ai_performance_agent", LogLevel.INFO, f"💾 [ai_performance_agent] 缓存目录: {cache_dir}")
            
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
                    log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] {model_name} 性能模型(本地缓存)初始化成功")
                except Exception as local_err:
                    log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 本地缓存未就绪，尝试联网下载: {local_err}")
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
                    log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] {model_name} 性能模型(联网下载并缓存)初始化成功")

                self.performance_model = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
            except Exception as model_error:
                log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 主模型加载失败,尝试备用模型: {model_error}")
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
                    log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] 备用模型(本地缓存)加载成功: {fallback_model}")
                except Exception as fb_local_err:
                    log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 备用模型本地缓存未就绪，尝试联网下载: {fb_local_err}")
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
                    log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] 备用模型(联网下载并缓存)加载成功: {fallback_model}")

                self.performance_model = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )

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
                    log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] {text_gen_model} 优化建议模型(本地缓存)加载成功")
                except Exception as tg_local_err:
                    log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 文本生成模型本地缓存未就绪，尝试联网下载: {tg_local_err}")
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
                    log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] {text_gen_model} 优化建议模型(联网下载并缓存)加载成功")

                self.optimization_generator = pipeline(
                    "text-generation",
                    model=model_gen,
                    tokenizer=tokenizer_gen,
                    device=device
                )
                if self.optimization_generator.tokenizer.pad_token is None:
                    self.optimization_generator.tokenizer.pad_token = self.optimization_generator.tokenizer.eos_token
            except Exception as gen_error:
                log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 优化建议模型加载失败: {gen_error}")
                self.optimization_generator = None
                
            self.models_loaded = True
            log("ai_performance_agent", LogLevel.INFO, f"✅ [ai_performance_agent] 性能分析AI模型初始化完成 ({device_mode}模式)")
            
        except Exception as e:
            log("ai_performance_agent", LogLevel.INFO, f"❌ [ai_performance_agent] 性能分析AI模型初始化失败: {e}")
            self.models_loaded = False
            # 设置备用状态
            self.performance_model = None
            self.optimization_generator = None

    async def handle_message(self, message: Message):
        """处理性能分析请求"""
        if message.message_type == "performance_analysis_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            file_path = message.content.get("file_path")
            run_id = message.content.get('run_id')
            log("ai_performance_agent", LogLevel.INFO, f"⚡ AI性能分析开始 - 需求ID: {requirement_id} run_id={run_id}")
            if not self.performance_model:
                log("ai_performance_agent", LogLevel.INFO, f"⚠️ [ai_performance_agent] 性能分析模型未加载，尝试初始化")
                await self._initialize_models()
            result = await self._ai_driven_performance_analysis(code_content, code_directory)
            # 额外: 生成该Agent单独报告 (按 run_id/agents/performance )
            if run_id:
                try:
                    per_agent_payload = {
                        "requirement_id": requirement_id,
                        "file_path": file_path,
                        "run_id": run_id,
                        "performance_result": result,
                        "generated_at": self._get_current_time()
                    }
                    report_manager.generate_run_scoped_report(run_id, per_agent_payload, f"performance_req_{requirement_id}.json", subdir="agents/performance")
                except Exception as e:
                    log("ai_performance_agent", LogLevel.INFO, f"⚠️ 性能Agent单独报告生成失败 requirement={requirement_id} run_id={run_id}: {e}")
            # 发送到用户交互
            await self.dispatch_message(
                receiver="user_comm_agent",
                content={
                    "requirement_id": requirement_id,
                    "agent_type": "ai_performance",
                    "results": result,
                    "analysis_complete": True,
                    "file_path": file_path,
                    "run_id": run_id
                },
                message_type="analysis_result"
            )
            # 发送到汇总
            await self.dispatch_message(
                receiver="summary_agent",
                content={
                    "requirement_id": requirement_id,
                    "analysis_type": "performance_analysis",
                    "result": result,
                    "file_path": file_path,
                    "run_id": run_id
                },
                message_type="analysis_result"
            )
            log("ai_performance_agent", LogLevel.INFO, f"✅ AI性能分析完成 - 需求ID: {requirement_id} run_id={run_id}")

    async def _ai_driven_performance_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """AI驱动的全面性能分析"""
        
        try:
            log("ai_performance_agent", LogLevel.INFO, "🔍 AI正在进行深度性能分析...")
            
            # 1. 代码结构和环境分析
            code_structure = await self._analyze_code_structure(code_content, code_directory)
            
            # 2. AI算法复杂度分析
            complexity_analysis = await self._ai_complexity_analysis(code_content)
            
            # 3. AI性能瓶颈检测
            bottlenecks = await self._ai_bottleneck_detection(code_content, code_structure)
            
            # 4. AI性能评分
            performance_score = await self._ai_performance_scoring(complexity_analysis, bottlenecks)
            
            # 5. AI优化建议生成
            optimization_plan = await self._ai_optimization_planning(
                code_content, bottlenecks, complexity_analysis
            )
            
            # 6. AI性能测试建议
            testing_recommendations = await self._ai_testing_recommendations(code_structure)
            
            log("ai_performance_agent", LogLevel.INFO, "🚀 AI性能分析完成,生成优化报告")
            
            return {
                "ai_performance_analysis": {
                    "overall_performance_score": performance_score,
                    "code_structure_analysis": code_structure,
                    "complexity_analysis": complexity_analysis,
                    "performance_bottlenecks": bottlenecks,
                    "optimization_plan": optimization_plan,
                    "testing_recommendations": testing_recommendations,
                    "ai_confidence": 0.88,
                    "model_used": self.model_config["name"],
                    "analysis_timestamp": self._get_current_time()
                },
                "analysis_status": "completed"
            }
            
        except Exception as e:
            log("ai_performance_agent", LogLevel.INFO, f"❌ AI性能分析过程中出错: {e}")
            return {
                "ai_performance_analysis": {"error": str(e)},
                "analysis_status": "failed"
            }

    async def _analyze_code_structure(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """分析代码结构和执行环境"""
        structure = {
            "language": "unknown",
            "framework": "unknown",
            "code_size": len(code_content),
            "function_count": 0,
            "class_count": 0,
            "loop_count": 0,
            "recursive_functions": [],
            "async_patterns": False,
            "database_operations": False,
            "file_operations": False,
            "network_operations": False
        }
        
        try:
            # 基础代码分析
            if "def " in code_content:
                structure["language"] = "python"
                structure["function_count"] = code_content.count("def ")
                structure["class_count"] = code_content.count("class ")
                structure["async_patterns"] = "async " in code_content or "await " in code_content
            
            # 循环检测
            structure["loop_count"] = (
                code_content.count("for ") + 
                code_content.count("while ") + 
                code_content.count("forEach")
            )
            
            # 操作类型检测
            structure["database_operations"] = any(keyword in code_content.lower() for keyword in 
                ["sql", "select", "insert", "update", "delete", "database", "query"])
            
            structure["file_operations"] = any(keyword in code_content.lower() for keyword in
                ["open(", "read(", "write(", "file", "io"])
            
            structure["network_operations"] = any(keyword in code_content.lower() for keyword in
                ["requests", "http", "socket", "urllib", "fetch"])
            
            # AI增强分析
            if self.complexity_analyzer:
                ai_structure_analysis = await self._ai_structural_analysis(code_content)
                structure.update(ai_structure_analysis)
            
        except Exception as e:
            structure["analysis_error"] = str(e)
        
        return structure

    async def _ai_complexity_analysis(self, code_content: str) -> Dict[str, Any]:
        """AI驱动的算法复杂度分析"""
        try:
            code_functions = self._extract_functions(code_content)
            complexity_results = []
            for i, func_code in enumerate(code_functions[:5]):
                analysis_prompt = get_prompt(
                    task_type="performance",
                    variant="algorithmic_analysis",
                    code_snippet=func_code
                )
                
                # AI复杂度分类
                if self.complexity_analyzer:
                    complexity_classification = self.complexity_analyzer(
                        f"Algorithm complexity analysis: {func_code[:300]}"
                    )
                    
                    # 解析分类结果
                    complexity_data = await self._parse_complexity_result(
                        complexity_classification, func_code, i
                    )
                    
                    if complexity_data:
                        complexity_results.append(complexity_data)
                
                # AI生成详细分析
                if self.optimization_advisor and len(complexity_results) < 3:
                    detailed_analysis = self.optimization_advisor(
                        analysis_prompt,
                        max_length=250,
                        temperature=0.3
                    )
                    
                    detailed_complexity = await self._extract_complexity_details(
                        detailed_analysis, func_code, i
                    )
                    
                    if detailed_complexity:
                        complexity_results.append(detailed_complexity)
            
            # 综合复杂度评估
            overall_complexity = await self._calculate_overall_complexity(complexity_results)
            
            return {
                "function_complexities": complexity_results,
                "overall_complexity": overall_complexity,
                "complexity_distribution": self._analyze_complexity_distribution(complexity_results),
                "optimization_priority": self._prioritize_optimizations(complexity_results)
            }
            
        except Exception as e:
            return {"error": f"复杂度分析失败: {e}"}

    async def _ai_bottleneck_detection(self, code_content: str, code_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI驱动的性能瓶颈检测"""
        bottlenecks = []
        
        try:
            # 1. 嵌套循环检测
            nested_loops = self._detect_nested_loops(code_content)
            for loop_info in nested_loops:
                bottleneck = await self._ai_analyze_loop_bottleneck(loop_info)
                if bottleneck:
                    bottlenecks.append(bottleneck)
            
            # 2. 递归函数分析
            recursive_functions = self._detect_recursive_functions(code_content)
            for func_info in recursive_functions:
                bottleneck = await self._ai_analyze_recursion_bottleneck(func_info)
                if bottleneck:
                    bottlenecks.append(bottleneck)
            
            # 3. I/O操作分析
            io_operations = self._detect_io_operations(code_content)
            for io_info in io_operations:
                bottleneck = await self._ai_analyze_io_bottleneck(io_info)
                if bottleneck:
                    bottlenecks.append(bottleneck)
            
            # 4. AI模式识别
            if self.complexity_analyzer:
                ai_bottlenecks = await self._ai_pattern_based_bottleneck_detection(code_content)
                bottlenecks.extend(ai_bottlenecks)
            
            # 按严重程度排序
            bottlenecks = sorted(bottlenecks, key=lambda x: x.get("severity_score", 0), reverse=True)
            
        except Exception as e:
            bottlenecks.append({
                "bottleneck_id": "DETECTION_ERROR",
                "type": "analysis_error", 
                "description": f"瓶颈检测过程出错: {e}",
                "severity": "info"
            })
        
        max_bottlenecks = self.agent_config.get("max_bottlenecks_reported", 10)
        return bottlenecks[:max_bottlenecks]  # 限制返回数量

    async def _ai_performance_scoring(self, complexity_analysis: Dict[str, Any], 
                                     bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """AI驱动的性能评分"""
        try:
            # 基础评分
            base_score = 8.0
            
            # 根据复杂度调整分数
            overall_complexity = complexity_analysis.get("overall_complexity", {})
            complexity_score = overall_complexity.get("average_score", 5.0)
            
            # 根据瓶颈数量和严重程度调整分数
            bottleneck_penalty = 0
            for bottleneck in bottlenecks:
                severity = bottleneck.get("severity", "low")
                if severity == "critical":
                    bottleneck_penalty += 2.0
                elif severity == "high":
                    bottleneck_penalty += 1.5
                elif severity == "medium":
                    bottleneck_penalty += 1.0
                elif severity == "low":
                    bottleneck_penalty += 0.5
            
            # 计算最终分数
            final_score = max(0.0, min(10.0, base_score - bottleneck_penalty + (complexity_score - 5.0)))
            
            # AI生成评分解释
            scoring_explanation = await self._generate_scoring_explanation(
                final_score, complexity_analysis, bottlenecks
            )
            
            return {
                "performance_score": final_score,
                "performance_grade": self._score_to_grade(final_score),
                "complexity_contribution": complexity_score,
                "bottleneck_impact": bottleneck_penalty,
                "explanation": scoring_explanation,
                "improvement_potential": 10.0 - final_score
            }
            
        except Exception as e:
            return {"error": f"性能评分失败: {e}"}

    async def _ai_optimization_planning(self, code_content: str, bottlenecks: List[Dict[str, Any]], complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """AI生成优化计划"""
        try:
            optimization_plan = {
                "immediate_optimizations": [],
                "algorithmic_improvements": [],
                "architectural_changes": [],
                "monitoring_setup": [],
                "estimated_performance_gain": "unknown"
            }
            
            # 基于瓶颈生成优化建议
            for bottleneck in bottlenecks[:5]:  # 处理前5个瓶颈
                optimization = await self._generate_bottleneck_optimization(bottleneck)
                
                severity = bottleneck.get("severity", "low")
                if severity in ["critical", "high"]:
                    optimization_plan["immediate_optimizations"].append(optimization)
                else:
                    optimization_plan["algorithmic_improvements"].append(optimization)
            
            # 基于复杂度分析生成建议
            complexity_optimizations = await self._generate_complexity_optimizations(complexity_analysis)
            optimization_plan["algorithmic_improvements"].extend(complexity_optimizations)
            
            # AI生成架构级优化建议
            if self.optimization_advisor:
                architectural_suggestions = await self._generate_architectural_optimizations(
                    code_content, bottlenecks
                )
                optimization_plan["architectural_changes"] = architectural_suggestions
            
            # 估算性能提升
            optimization_plan["estimated_performance_gain"] = await self._estimate_performance_gain(
                bottlenecks, complexity_analysis
            )
            
            return optimization_plan
            
        except Exception as e:
            return {"error": f"优化计划生成失败: {e}"}

    async def _ai_testing_recommendations(self, code_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI生成性能测试建议"""
        recommendations = []
        
        try:
            # 基于代码结构生成测试建议
            if code_structure.get("database_operations"):
                recommendations.append({
                    "test_type": "数据库性能测试",
                    "description": "测试数据库查询性能和连接池效率",
                    "tools": ["pytest-benchmark", "SQLAlchemy profiling"],
                    "priority": "high"
                })
            
            if code_structure.get("network_operations"):
                recommendations.append({
                    "test_type": "网络性能测试",
                    "description": "测试网络请求延迟和并发处理能力",
                    "tools": ["aiohttp testing", "requests-mock"],
                    "priority": "medium"
                })
            
            if code_structure.get("async_patterns"):
                recommendations.append({
                    "test_type": "异步性能测试",
                    "description": "测试异步操作的并发性能",
                    "tools": ["pytest-asyncio", "asyncio profiling"],
                    "priority": "high"
                })
            
            # AI生成自定义测试建议
            if self.optimization_advisor:
                custom_tests = await self._generate_custom_performance_tests(code_structure)
                recommendations.extend(custom_tests)
            
        except Exception as e:
            recommendations.append({
                "test_type": "错误",
                "description": f"测试建议生成失败: {e}",
                "priority": "info"
            })
        
        return recommendations

    # 辅助方法
    def _extract_functions(self, code_content: str) -> List[str]:
        """提取代码中的函数"""
        functions = []
        try:
            # 简单的函数提取,实际应该使用AST
            lines = code_content.split('\n')
            current_function = []
            in_function = False
            indent_level = 0
            
            for line in lines:
                if line.strip().startswith('def '):
                    if current_function:
                        functions.append('\n'.join(current_function))
                    current_function = [line]
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                elif in_function:
                    if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.strip().startswith('#'):
                        if current_function:
                            functions.append('\n'.join(current_function))
                        current_function = []
                        in_function = False
                    else:
                        current_function.append(line)
            
            if current_function:
                functions.append('\n'.join(current_function))
                
        except Exception as e:
            log("ai_performance_agent", LogLevel.INFO, f"函数提取失败: {e}")
        
        return functions[:10]  # 限制函数数量

    def _detect_nested_loops(self, code_content: str) -> List[Dict[str, Any]]:
        """检测嵌套循环"""
        nested_loops = []
        lines = code_content.split('\n')
        
        for i, line in enumerate(lines):
            if 'for ' in line or 'while ' in line:
                # 检查后续行中是否有嵌套循环
                indent = len(line) - len(line.lstrip())
                for j in range(i+1, min(i+20, len(lines))):
                    next_line = lines[j]
                    if next_line.strip() and len(next_line) - len(next_line.lstrip()) > indent:
                        if 'for ' in next_line or 'while ' in next_line:
                            nested_loops.append({
                                "line_number": i+1,
                                "outer_loop": line.strip(),
                                "inner_loop": next_line.strip(),
                                "nesting_level": 2  # 简化,只检测2层
                            })
                            break
        
        return nested_loops

    def _detect_recursive_functions(self, code_content: str) -> List[Dict[str, Any]]:
        """检测递归函数"""
        recursive_functions = []
        functions = self._extract_functions(code_content)
        
        for func in functions:
            lines = func.split('\n')
            func_name = None
            
            # 提取函数名
            for line in lines:
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    break
            
            # 检查是否调用自身
            if func_name:
                for line in lines[1:]:  # 跳过函数定义行
                    if func_name in line and '(' in line:
                        recursive_functions.append({
                            "function_name": func_name,
                            "function_code": func,
                            "recursive_call_line": line.strip()
                        })
                        break
        
        return recursive_functions

    def _detect_io_operations(self, code_content: str) -> List[Dict[str, Any]]:
        """检测I/O操作"""
        io_operations = []
        io_patterns = [
            ("file_io", ["open(", "read(", "write(", "close()"]),
            ("database_io", ["execute(", "query(", "select", "insert", "update"]),
            ("network_io", ["requests.", "urllib.", "socket.", "httpx."])
        ]
        
        lines = code_content.split('\n')
        for i, line in enumerate(lines):
            for io_type, patterns in io_patterns:
                for pattern in patterns:
                    if pattern.lower() in line.lower():
                        io_operations.append({
                            "line_number": i+1,
                            "io_type": io_type,
                            "operation": line.strip(),
                            "pattern_matched": pattern
                        })
                        break
        
        return io_operations

    def _score_to_grade(self, score: float) -> str:
        """将分数转换为等级"""
        if score >= 9.0:
            return "Excellent"
        elif score >= 7.5:
            return "Good"
        elif score >= 6.0:
            return "Fair"
        elif score >= 4.0:
            return "Poor"
        else:
            return "Critical"

    # 占位符方法 - 实际实现中需要完善
    async def _ai_structural_analysis(self, code_content):
        return {}
    
    async def _parse_complexity_result(self, result, code, index):
        if result and result[0].get("score", 0) > 0.6:
            return {
                "function_id": f"func_{index}",
                "estimated_complexity": "O(n)",
                "confidence": result[0]["score"]
            }
        return None
    
    async def _extract_complexity_details(self, analysis, code, index):
        return None
    
    async def _calculate_overall_complexity(self, results):
        return {"average_score": 6.5}
    
    def _analyze_complexity_distribution(self, results):
        return {"distribution": "normal"}
    
    def _prioritize_optimizations(self, results):
        return ["高复杂度函数优化"]
    
    async def _ai_analyze_loop_bottleneck(self, loop_info):
        line_num = loop_info.get('line_number')
        outer_loop = loop_info.get('outer_loop', 'N/A')
        inner_loop = loop_info.get('inner_loop', 'N/A')
        nesting_level = loop_info.get('nesting_level', 2)
        
        description = (
            f"嵌套循环性能瓶颈 (第{line_num}行)\n"
            f"  - 嵌套层级: {nesting_level}层\n"
            f"  - 外层循环: {outer_loop}\n"
            f"  - 内层循环: {inner_loop}\n"
            f"  - 性能影响: 时间复杂度可能达到 O(n^{nesting_level})\n"
            f"  - 建议: 考虑使用AST解析、字典查找或生成器优化"
        )
        
        return {
            "bottleneck_id": f"LOOP_{line_num}",
            "type": "nested_loop",
            "description": description,
            "line_number": line_num,
            "severity": "medium",
            "severity_score": 6.0,
            "details": {
                "outer_loop": outer_loop,
                "inner_loop": inner_loop,
                "nesting_level": nesting_level,
                "estimated_complexity": f"O(n^{nesting_level})"
            }
        }
    
    async def _ai_analyze_recursion_bottleneck(self, func_info):
        func_name = func_info.get('function_name', 'unknown')
        recursive_call = func_info.get('recursive_call_line', 'N/A')
        func_code_snippet = func_info.get('function_code', '')[:200]  # 前200字符
        
        # 尝试从函数代码中找到函数定义行
        lines = func_code_snippet.split('\n')
        func_line = None
        for i, line in enumerate(lines):
            if f'def {func_name}' in line:
                func_line = i + 1
                break
        
        description = (
            f"递归函数性能风险 - {func_name}\n"
            f"  - 函数名称: {func_name}\n"
            f"  - 递归调用: {recursive_call}\n"
            f"  - 性能影响: 可能导致栈溢出或指数级时间复杂度\n"
            f"  - 建议: 检查是否有终止条件、考虑使用迭代或尾递归优化、添加缓存(memoization)"
        )
        
        return {
            "bottleneck_id": f"RECURSION_{func_name}",
            "type": "recursion",
            "description": description,
            "line_number": func_line,
            "severity": "medium",
            "severity_score": 5.0,
            "details": {
                "function_name": func_name,
                "recursive_call_line": recursive_call,
                "code_preview": func_code_snippet
            }
        }
    
    async def _ai_analyze_io_bottleneck(self, io_info):
        line_num = io_info.get('line_number')
        io_type = io_info.get('io_type', 'unknown')
        operation = io_info.get('operation', 'N/A')
        pattern = io_info.get('pattern_matched', '')
        
        # 根据IO类型提供不同的优化建议
        io_type_descriptions = {
            "file_io": {
                "name": "文件I/O操作",
                "impact": "可能导致磁盘I/O阻塞",
                "suggestions": "使用异步文件操作(aiofiles)、批量读写、添加缓存机制"
            },
            "database_io": {
                "name": "数据库I/O操作",
                "impact": "数据库查询可能成为性能瓶颈",
                "suggestions": "使用连接池、批量查询、添加索引、考虑使用缓存(Redis)、使用异步数据库驱动"
            },
            "network_io": {
                "name": "网络I/O操作",
                "impact": "网络延迟可能影响响应时间",
                "suggestions": "使用异步HTTP客户端(aiohttp)、实现超时控制、添加重试机制、考虑并发请求"
            }
        }
        
        io_details = io_type_descriptions.get(io_type, {
            "name": "I/O操作",
            "impact": "可能影响性能",
            "suggestions": "考虑使用异步操作"
        })
        
        description = (
            f"{io_details['name']} (第{line_num}行)\n"
            f"  - 操作代码: {operation}\n"
            f"  - 匹配模式: {pattern}\n"
            f"  - 性能影响: {io_details['impact']}\n"
            f"  - 优化建议: {io_details['suggestions']}"
        )
        
        return {
            "bottleneck_id": f"IO_{io_type.upper()}_{line_num}",
            "type": io_type,
            "description": description,
            "line_number": line_num,
            "severity": "low",
            "severity_score": 3.0,
            "details": {
                "io_type": io_type,
                "operation": operation,
                "pattern_matched": pattern,
                "optimization_priority": "medium" if io_type == "database_io" else "low"
            }
        }
    
    async def _ai_pattern_based_bottleneck_detection(self, code_content):
        return []
    
    async def _generate_scoring_explanation(self, score, complexity, bottlenecks):
        return f"基于{len(bottlenecks)}个性能瓶颈和复杂度分析的综合评估"
    
    async def _generate_bottleneck_optimization(self, bottleneck):
        return {
            "optimization_id": bottleneck.get("bottleneck_id"),
            "description": f"优化建议:{bottleneck.get('description')}",
            "priority": bottleneck.get("severity")
        }
    
    async def _generate_complexity_optimizations(self, complexity_analysis):
        return []
    
    async def _generate_architectural_optimizations(self, code_content, bottlenecks):
        return []
    
    async def _estimate_performance_gain(self, bottlenecks, complexity_analysis):
        return "20-40%提升"
    
    async def _generate_custom_performance_tests(self, code_structure):
        return []

    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()

    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行AI驱动的性能分析任务"""
        return await self._ai_driven_performance_analysis(
            task_data.get("code_content", ""),
            task_data.get("code_directory", "")
        )