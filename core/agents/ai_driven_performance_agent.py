import os
import json
import re
import torch
import asyncio
import ast
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, Message
from utils.prompt_budgeting import prepare_generation_prompt, semantic_truncate_text
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
        self.optimization_generator = None

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

        log("ai_performance_agent", LogLevel.INFO, "🔍 AI正在进行深度性能分析...")

        failed_steps: List[Dict[str, str]] = []

        # 1. 代码结构和环境分析
        try:
            code_structure = await self._analyze_code_structure(code_content, code_directory)
        except Exception as e:
            failed_steps.append({"step": "code_structure", "error": str(e)})
            code_structure = {
                "language": "unknown",
                "analysis_error": str(e),
                "fallback": True,
            }

        # 2. AI算法复杂度分析
        try:
            complexity_analysis = await self._ai_complexity_analysis(code_content)
        except Exception as e:
            failed_steps.append({"step": "complexity_analysis", "error": str(e)})
            complexity_analysis = {
                "error": str(e),
                "fallback": True,
                "overall_complexity": {"average_score": 5.0},
                "function_complexities": [],
            }

        # 3. AI性能瓶颈检测
        try:
            bottlenecks = await self._ai_bottleneck_detection(code_content, code_structure)
        except Exception as e:
            failed_steps.append({"step": "bottleneck_detection", "error": str(e)})
            bottlenecks = [{
                "bottleneck_id": "DETECTION_FALLBACK",
                "type": "analysis_error",
                "description": f"性能瓶颈检测失败，已回退: {e}",
                "severity": "info",
                "severity_score": 0.0,
            }]

        # 4. AI性能评分
        try:
            performance_score = await self._ai_performance_scoring(complexity_analysis, bottlenecks)
        except Exception as e:
            failed_steps.append({"step": "performance_scoring", "error": str(e)})
            performance_score = {
                "error": str(e),
                "performance_score": 5.0,
                "performance_grade": "Fair",
                "fallback": True,
            }

        # 5. AI优化建议生成
        try:
            optimization_plan = await self._ai_optimization_planning(
                code_content, bottlenecks, complexity_analysis
            )
        except Exception as e:
            failed_steps.append({"step": "optimization_planning", "error": str(e)})
            optimization_plan = {
                "error": str(e),
                "fallback": True,
                "immediate_optimizations": [],
                "algorithmic_improvements": [],
                "architectural_changes": [],
                "monitoring_setup": [],
                "estimated_performance_gain": "unknown",
            }

        # 6. AI性能测试建议
        try:
            testing_recommendations = await self._ai_testing_recommendations(code_structure)
        except Exception as e:
            failed_steps.append({"step": "testing_recommendations", "error": str(e)})
            testing_recommendations = [{
                "test_type": "fallback",
                "description": f"测试建议生成失败，已回退: {e}",
                "priority": "info",
            }]

        if failed_steps:
            log("ai_performance_agent", LogLevel.WARNING, f"⚠️ AI性能分析部分步骤失败: {failed_steps}")
            analysis_status = "partial_success"
            confidence = 0.65
        else:
            analysis_status = "completed"
            confidence = 0.88

        log("ai_performance_agent", LogLevel.INFO, "🚀 AI性能分析完成,生成优化报告")

        return {
            "ai_performance_analysis": {
                "overall_performance_score": performance_score,
                "code_structure_analysis": code_structure,
                "complexity_analysis": complexity_analysis,
                "performance_bottlenecks": bottlenecks,
                "optimization_plan": optimization_plan,
                "testing_recommendations": testing_recommendations,
                "failed_steps": failed_steps,
                "ai_confidence": confidence,
                "model_used": self.model_config["name"],
                "analysis_timestamp": self._get_current_time()
            },
            "analysis_status": analysis_status
        }

    async def _analyze_code_structure(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """分析代码结构和执行环境，增强Python/C++支持"""
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
            "network_operations": False,
            "memory_operations": False,  # C/C++特有的内存操作
            "pointer_usage": False,      # C/C++指针使用
            "template_usage": False,     # C++模板使用
            "preprocessor_directives": False  # C/C++预处理器指令
        }
        
        try:
            # 智能语言检测
            detected_language = self._detect_code_language(code_content)
            structure["language"] = detected_language
            
            # 基于语言的特定分析
            if detected_language == "python":
                self._analyze_python_structure(code_content, structure)
            elif detected_language in ["cpp", "c"]:
                self._analyze_cpp_structure(code_content, structure)
            
            # 通用操作类型检测
            structure["database_operations"] = any(keyword in code_content.lower() for keyword in 
                ["sql", "select", "insert", "update", "delete", "database", "query"])
            
            structure["file_operations"] = any(keyword in code_content.lower() for keyword in
                ["open(", "read(", "write(", "file", "io"])
            
            structure["network_operations"] = any(keyword in code_content.lower() for keyword in
                ["requests", "http", "socket", "urllib", "fetch"])
            
            # AI增强分析
            if self.performance_model:
                ai_structure_analysis = await self._ai_structural_analysis(code_content)
                structure.update(ai_structure_analysis)
            
        except Exception as e:
            structure["analysis_error"] = str(e)
        
        return structure

    def _detect_code_language(self, code_content: str) -> str:
        """智能检测代码语言"""
        content_lower = code_content.lower()
        
        # Python特征
        python_indicators = ['def ', 'import ', 'class ', 'self.', '__init__', 'lambda:', 'yield ']
        python_score = sum(1 for indicator in python_indicators if indicator in code_content)
        
        # C++特征
        cpp_indicators = ['#include', 'namespace', 'std::', 'cout', 'cin', 'template<', 'using namespace']
        cpp_score = sum(1 for indicator in cpp_indicators if indicator in content_lower)
        
        # C特征
        c_indicators = ['#include', 'printf', 'scanf', 'malloc', 'free', 'struct ', 'typedef']
        c_score = sum(1 for indicator in c_indicators if indicator in content_lower)
        
        # 返回得分最高的语言
        scores = {'python': python_score, 'cpp': cpp_score, 'c': c_score}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'

    def _analyze_python_structure(self, code_content: str, structure: Dict[str, Any]):
        """分析Python代码结构"""
        structure["function_count"] = code_content.count("def ")
        structure["class_count"] = code_content.count("class ")
        structure["async_patterns"] = "async " in code_content or "await " in code_content
        
        # Python特有的循环语法
        structure["loop_count"] = (
            code_content.count("for ") + 
            code_content.count("while ") + 
            code_content.count("forEach")
        )
        
        # 检测递归调用
        self._detect_recursive_calls(code_content, structure)
        
        # Python特有的操作
        structure["list_comprehensions"] = code_content.count("[") + code_content.count("]")
        structure["generator_expressions"] = code_content.count("(") + code_content.count(")")
        structure["decorator_usage"] = code_content.count("@")
        structure["context_managers"] = code_content.count("with ")

    def _analyze_cpp_structure(self, code_content: str, structure: Dict[str, Any]):
        """分析C/C++代码结构"""
        # 基本统计
        structure["function_count"] = code_content.count("{")  # 简化的函数计数
        structure["class_count"] = code_content.count("class ") + code_content.count("struct ")
        
        # C/C++特有的循环语法
        structure["loop_count"] = (
            code_content.count("for ") + 
            code_content.count("while ") + 
            code_content.count("do ")
        )
        
        # 检测递归调用
        self._detect_recursive_calls(code_content, structure)
        
        # C/C++特有的特征
        structure["memory_operations"] = (
            "malloc" in code_content or "free" in code_content or 
            "new" in code_content or "delete" in code_content or
            "alloc" in code_content
        )
        
        structure["pointer_usage"] = "*" in code_content and "&" in code_content
        structure["template_usage"] = "template<" in code_content.lower()
        structure["preprocessor_directives"] = "#" in code_content
        
        # C++特有特征
        if "std::" in code_content.lower():
            structure["stl_usage"] = True
            structure["smart_pointers"] = (
                "std::unique_ptr" in code_content or 
                "std::shared_ptr" in code_content or
                "std::weak_ptr" in code_content
            )

    def _detect_recursive_calls(self, code_content: str, structure: Dict[str, Any]):
        """检测递归调用"""
        recursive_functions = []
        
        # Python递归检测
        if structure["language"] == "python":
            lines = code_content.split('\n')
            current_function = None
            
            for i, line in enumerate(lines):
                # 检测函数定义
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    current_function = func_name
                
                # 如果在函数内部，检查是否有自我调用
                elif current_function and current_function in line and '(' in line:
                    # 排除定义行本身
                    if not line.strip().startswith('def '):
                        recursive_functions.append({
                            "function_name": current_function,
                            "line_number": i + 1,
                            "call_expression": line.strip()
                        })
        
        # C/C++递归检测
        elif structure["language"] in ["cpp", "c"]:
            lines = code_content.split('\n')
            function_definitions = []
            
            # 找到所有函数定义
            for i, line in enumerate(lines):
                if ('void ' in line or 'int ' in line or 'bool ' in line or 
                    'char ' in line or 'double ' in line or 'float ' in line) and '(' in line and '{' in line:
                    # 简化的函数名提取
                    func_part = line.split('(')[0].strip()
                    if ' ' in func_part:
                        func_name = func_part.split()[-1]
                        function_definitions.append((func_name, i + 1))
            
            # 检查函数内部是否有自我调用
            for func_name, def_line in function_definitions:
                # 查找该函数的结束位置（简化版）
                end_line = len(lines)
                for j in range(def_line, min(def_line + 50, len(lines))):
                    if lines[j].strip() == '}' and lines[j-1].strip() == '}':
                        end_line = j
                        break
                
                # 在函数体内查找自我调用
                for k in range(def_line, end_line):
                    if func_name in lines[k] and '(' in lines[k] and ';' in lines[k]:
                        recursive_functions.append({
                            "function_name": func_name,
                            "line_number": k + 1,
                            "call_expression": lines[k].strip()
                        })
        
        structure["recursive_functions"] = recursive_functions
        structure["recursive_count"] = len(recursive_functions)

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
                if self.performance_model:
                    complexity_classification = await self._run_classification_inference(
                        f"Algorithm complexity analysis: {func_code[:300]}"
                    )
                    
                    # 解析分类结果
                    complexity_data = await self._parse_complexity_result(
                        complexity_classification, func_code, i
                    )
                    
                    if complexity_data:
                        complexity_results.append(complexity_data)
                
                # AI生成详细分析
                if self.optimization_generator and len(complexity_results) < 3:
                    detailed_analysis = await self._run_generation_inference(
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
            if self.performance_model:
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
            rule_track = self._calculate_rule_based_score(complexity_analysis, bottlenecks)
            llm_track = await self._calculate_llm_based_score(complexity_analysis, bottlenecks)

            final_score, fusion_info = self._fuse_performance_scores(rule_track, llm_track)

            scoring_explanation = await self._generate_scoring_explanation(
                final_score, complexity_analysis, bottlenecks
            )

            return {
                "performance_score": final_score,
                "performance_grade": self._score_to_grade(final_score),
                "complexity_contribution": rule_track.get("complexity_score", 5.0),
                "bottleneck_impact": rule_track.get("bottleneck_penalty", 0.0),
                "rule_based_score": rule_track.get("score"),
                "llm_based_score": llm_track.get("score") if isinstance(llm_track, dict) else None,
                "llm_confidence": llm_track.get("confidence") if isinstance(llm_track, dict) else 0.0,
                "fusion": fusion_info,
                "explanation": scoring_explanation,
                "improvement_potential": 10.0 - final_score
            }
            
        except Exception as e:
            return {"error": f"性能评分失败: {e}"}

    def _calculate_rule_based_score(
        self,
        complexity_analysis: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """规则评分轨道：用于在 LLM 不稳定时提供可解释的基线。"""
        base_score = 8.0
        overall_complexity = complexity_analysis.get("overall_complexity", {}) if isinstance(complexity_analysis, dict) else {}
        complexity_score = overall_complexity.get("average_score", 5.0)
        if not isinstance(complexity_score, (int, float)):
            complexity_score = 5.0

        severity_penalty_map = {
            "critical": 2.0,
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5,
            "info": 0.2,
        }

        bottleneck_penalty = 0.0
        severity_breakdown: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        for item in bottlenecks:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "low")).lower()
            if severity not in severity_penalty_map:
                severity = "low"
            severity_breakdown[severity] += 1
            bottleneck_penalty += severity_penalty_map[severity]

        score = max(0.0, min(10.0, base_score - bottleneck_penalty + (complexity_score - 5.0)))
        return {
            "score": round(score, 3),
            "complexity_score": round(float(complexity_score), 3),
            "bottleneck_penalty": round(float(bottleneck_penalty), 3),
            "severity_breakdown": severity_breakdown,
        }

    async def _calculate_llm_based_score(
        self,
        complexity_analysis: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """LLM评分轨道：让模型根据复杂度和瓶颈语义给出倾向分。"""
        if not self.performance_model:
            return {
                "available": False,
                "score": None,
                "confidence": 0.0,
                "reason": "classification_model_unavailable",
            }

        overall = complexity_analysis.get("overall_complexity", {}) if isinstance(complexity_analysis, dict) else {}
        avg_complexity = overall.get("average_score", "unknown")
        top_bottlenecks = []
        for item in bottlenecks[:5]:
            if not isinstance(item, dict):
                continue
            top_bottlenecks.append({
                "type": item.get("type", "unknown"),
                "severity": item.get("severity", "low"),
                "severity_score": item.get("severity_score", 0),
            })

        summary_text = (
            "Performance risk scoring input. "
            f"average_complexity={avg_complexity}; "
            f"bottleneck_count={len(bottlenecks)}; "
            f"top_bottlenecks={json.dumps(top_bottlenecks, ensure_ascii=False)}"
        )

        try:
            raw = await self._run_classification_inference(summary_text)
            if not raw or not isinstance(raw, list):
                return {
                    "available": False,
                    "score": None,
                    "confidence": 0.0,
                    "reason": "empty_classification_result",
                }

            first = raw[0] if isinstance(raw[0], dict) else {}
            label = str(first.get("label", "")).upper()
            confidence = float(first.get("score", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, confidence))

            # 不同分类头标签风格差异较大，这里用宽松映射做统一归一。
            positive_tokens = ["POS", "LABEL_1", "GOOD", "LOW_RISK", "SAFE"]
            negative_tokens = ["NEG", "LABEL_0", "BAD", "HIGH_RISK", "UNSAFE"]
            neutral_tokens = ["NEU", "LABEL_2", "MID", "MEDIUM"]

            if any(token in label for token in negative_tokens):
                score = 5.5 - 4.5 * confidence
            elif any(token in label for token in positive_tokens):
                score = 5.5 + 4.0 * confidence
            elif any(token in label for token in neutral_tokens):
                score = 5.5
            else:
                # 未知标签按风险中性偏保守处理。
                score = 5.0 + 1.0 * (confidence - 0.5)

            score = max(0.0, min(10.0, score))
            return {
                "available": True,
                "score": round(float(score), 3),
                "confidence": round(float(confidence), 3),
                "label": label,
                "reason": "classification_based_score",
            }
        except Exception as e:
            return {
                "available": False,
                "score": None,
                "confidence": 0.0,
                "reason": f"llm_scoring_failed: {e}",
            }

    def _fuse_performance_scores(
        self,
        rule_track: Dict[str, Any],
        llm_track: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """融合双轨评分：优先使用高置信 LLM，同时保留规则基线。"""
        rule_score = float(rule_track.get("score", 5.0) or 5.0)

        if not isinstance(llm_track, dict) or not llm_track.get("available") or llm_track.get("score") is None:
            final_score = max(0.0, min(10.0, rule_score))
            return final_score, {
                "strategy": "rule_only_fallback",
                "rule_weight": 1.0,
                "llm_weight": 0.0,
            }

        llm_score = float(llm_track.get("score", rule_score))
        llm_conf = float(llm_track.get("confidence", 0.0) or 0.0)

        if llm_conf >= 0.8:
            llm_weight = 0.65
        elif llm_conf >= 0.55:
            llm_weight = 0.55
        else:
            llm_weight = 0.4

        rule_weight = 1.0 - llm_weight
        final_score = max(0.0, min(10.0, llm_weight * llm_score + rule_weight * rule_score))
        return round(final_score, 3), {
            "strategy": "hybrid_fusion",
            "rule_weight": round(rule_weight, 3),
            "llm_weight": round(llm_weight, 3),
        }

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
                normalized_item = self._normalize_plan_item(
                    optimization,
                    default_type="performance",
                    default_priority=bottleneck.get("severity", "medium") if isinstance(bottleneck, dict) else "medium",
                )
                
                severity = bottleneck.get("severity", "low")
                if severity in ["critical", "high"]:
                    optimization_plan["immediate_optimizations"].append(normalized_item)
                else:
                    optimization_plan["algorithmic_improvements"].append(normalized_item)
            
            # 基于复杂度分析生成建议
            complexity_optimizations = await self._generate_complexity_optimizations(complexity_analysis)
            for item in complexity_optimizations:
                optimization_plan["algorithmic_improvements"].append(
                    self._normalize_plan_item(
                        item,
                        default_type="algorithmic_improvement",
                        default_priority="medium",
                    )
                )
            
            # AI生成架构级优化建议（内部已包含 LLM 不可用时的回退）
            architectural_suggestions = await self._generate_architectural_optimizations(
                code_content, bottlenecks
            )
            optimization_plan["architectural_changes"] = self._extract_architectural_suggestion_items(
                architectural_suggestions
            )

            optimization_plan = self._deduplicate_optimization_plan(optimization_plan)
            
            # 估算性能提升
            optimization_plan["estimated_performance_gain"] = await self._estimate_performance_gain(
                bottlenecks, complexity_analysis
            )

            return self._validate_optimization_plan_schema(optimization_plan)
            
        except Exception as e:
            return {"error": f"优化计划生成失败: {e}"}

    def _validate_optimization_plan_schema(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """优化计划最终校验与归一，防止下游消费出现结构漂移。"""
        if not isinstance(plan, dict):
            plan = {}

        validated = {
            "immediate_optimizations": [],
            "algorithmic_improvements": [],
            "architectural_changes": [],
            "monitoring_setup": [],
            "estimated_performance_gain": str(plan.get("estimated_performance_gain", "unknown")),
        }

        for category in ["immediate_optimizations", "algorithmic_improvements", "architectural_changes"]:
            items = plan.get(category, [])
            if not isinstance(items, list):
                items = []
            normalized_items = [self._normalize_plan_item(item) for item in items]
            validated[category] = normalized_items[:5]

        monitoring_items = plan.get("monitoring_setup", [])
        if isinstance(monitoring_items, list):
            validated["monitoring_setup"] = [
                str(item).strip() if not isinstance(item, dict) else str(item.get("description", "")).strip()
                for item in monitoring_items[:5]
            ]

        return validated

    def _extract_architectural_suggestion_items(self, architectural_suggestions: Any) -> List[Dict[str, Any]]:
        """兼容架构建议的 dict/list 输出，并统一为建议列表。"""
        if isinstance(architectural_suggestions, dict):
            items = architectural_suggestions.get("optimization_suggestions", [])
        elif isinstance(architectural_suggestions, list):
            items = architectural_suggestions
        else:
            items = []

        normalized: List[Dict[str, Any]] = []
        for item in items:
            normalized.append(
                self._normalize_plan_item(
                    item,
                    default_type="architectural_change",
                    default_priority="medium",
                )
            )
        return normalized[:5]

    def _normalize_plan_item(
        self,
        item: Any,
        default_type: str = "general",
        default_priority: str = "medium",
    ) -> Dict[str, Any]:
        """把各来源优化建议转换成统一字段结构。"""
        if isinstance(item, str):
            description = item.strip()
            return {
                "type": default_type,
                "priority": self._normalize_priority_level(default_priority),
                "description": description,
                "expected_effect": "",
                "recommendation": description,
            }

        if not isinstance(item, dict):
            return {
                "type": default_type,
                "priority": self._normalize_priority_level(default_priority),
                "description": "",
                "expected_effect": "",
                "recommendation": "",
            }

        description = str(item.get("description") or item.get("suggestion") or "").strip()
        recommendation = str(item.get("recommendation") or item.get("solution") or description).strip()

        return {
            "suggestion_id": item.get("suggestion_id") or item.get("optimization_id"),
            "type": str(item.get("type", default_type)).strip().lower() or default_type,
            "priority": self._normalize_priority_level(item.get("priority", default_priority)),
            "description": description,
            "expected_effect": str(item.get("expected_effect", item.get("benefit", ""))).strip(),
            "location": str(item.get("location", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
            "recommendation": recommendation,
        }

    def _deduplicate_optimization_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """跨类别去重并做优先级冲突消解。"""
        categories = [
            "immediate_optimizations",
            "algorithmic_improvements",
            "architectural_changes",
        ]
        priority_rank = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}

        seen: Dict[str, Dict[str, Any]] = {}

        for category in categories:
            items = plan.get(category, [])
            if not isinstance(items, list):
                continue

            for raw_item in items:
                item = self._normalize_plan_item(raw_item)
                key_base = f"{item.get('type', 'general')}|{item.get('description', '')}|{item.get('recommendation', '')}"
                key = re.sub(r"\s+", " ", key_base.strip().lower())
                if not key:
                    continue

                if key not in seen:
                    seen[key] = {**item, "_category": category}
                    continue

                current = seen[key]
                old_rank = priority_rank.get(current.get("priority", "medium"), 3)
                new_rank = priority_rank.get(item.get("priority", "medium"), 3)

                if new_rank > old_rank:
                    merged = {**current, **item}
                else:
                    merged = {**item, **current}

                # 合并缺失的补充字段
                for field in ["expected_effect", "location", "reason"]:
                    if not merged.get(field):
                        merged[field] = item.get(field) or current.get(field) or ""

                merged["_category"] = current.get("_category") if old_rank >= new_rank else category
                seen[key] = merged

        rebuilt = {
            "immediate_optimizations": [],
            "algorithmic_improvements": [],
            "architectural_changes": [],
            "monitoring_setup": plan.get("monitoring_setup", []) if isinstance(plan.get("monitoring_setup", []), list) else [],
            "estimated_performance_gain": plan.get("estimated_performance_gain", "unknown"),
        }

        for item in seen.values():
            category = item.pop("_category", "algorithmic_improvements")
            if category not in rebuilt:
                category = "algorithmic_improvements"
            rebuilt[category].append(item)

        for category in categories:
            rebuilt[category] = sorted(
                rebuilt[category],
                key=lambda x: priority_rank.get(x.get("priority", "medium"), 3),
                reverse=True,
            )[:5]

        return rebuilt

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
            if self.optimization_generator:
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

    def _resolve_model_max_tokens(self, tokenizer: Any, fallback: int = 1024) -> int:
        if tokenizer is None:
            return fallback
        try:
            value = int(getattr(tokenizer, "model_max_length", fallback) or fallback)
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
        if not self.performance_model or not text:
            return []
        tokenizer = getattr(self.performance_model, "tokenizer", None)
        model_max = self._resolve_model_max_tokens(tokenizer, fallback=512)
        safe_text = semantic_truncate_text(tokenizer, text, model_max)

        effective_kwargs = dict(kwargs)
        effective_kwargs["truncation"] = True
        effective_kwargs["max_length"] = model_max
        try:
            return await asyncio.to_thread(self.performance_model, safe_text, **effective_kwargs)
        except Exception as e:
            log("ai_performance_agent", LogLevel.WARNING, f"⚠️ 分类推理失败: {e}")
            return []

    async def _run_generation_inference(self, prompt: str, **kwargs):
        """在线程中执行同步生成推理，避免阻塞事件循环。"""
        if not self.optimization_generator or not prompt:
            return []
        tokenizer = getattr(self.optimization_generator, "tokenizer", None)
        model_max = self._resolve_model_max_tokens(tokenizer, fallback=1024)

        effective_kwargs = dict(kwargs)
        requested_new = int(effective_kwargs.get("max_new_tokens", 128) or 128)
        prompt, _, requested_new = prepare_generation_prompt(
            tokenizer,
            prompt,
            requested_new,
            fallback_model_max=model_max,
            safety_margin=32,
        )

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

        effective_kwargs["max_new_tokens"] = requested_new
        effective_kwargs["truncation"] = True
        safe_prompt = semantic_truncate_text(tokenizer, prompt, input_budget)

        try:
            return await asyncio.to_thread(self.optimization_generator, safe_prompt, **effective_kwargs)
        except Exception as e:
            error_text = str(e)
            log("ai_performance_agent", LogLevel.WARNING, f"⚠️ 文本生成推理失败: {error_text}")

            # 对 embedding/position 越界错误做一次保守重试，避免频繁失败日志。
            if "index out of range" in error_text.lower():
                try:
                    retry_kwargs = dict(effective_kwargs)
                    retry_kwargs["max_new_tokens"] = min(64, max(16, requested_new // 2))
                    retry_prompt = self._truncate_text_for_model(tokenizer, prompt, max(96, input_budget // 2))
                    retry_result = await asyncio.to_thread(self.optimization_generator, retry_prompt, **retry_kwargs)
                    log("ai_performance_agent", LogLevel.INFO, "✅ 文本生成推理重试成功(降级参数)")
                    return retry_result
                except Exception as retry_err:
                    log("ai_performance_agent", LogLevel.WARNING, f"⚠️ 文本生成重试失败: {retry_err}")
            return []

    # 占位符方法 - 实际实现中需要完善
    async def _ai_structural_analysis(self, code_content):
        """使用轻量 LLM 信号增强结构分析结果。"""
        if not isinstance(code_content, str) or not code_content.strip():
            return {}

        result: Dict[str, Any] = {}
        snippet = code_content[:1500]

        # 1) 分类模型给出结构风险倾向
        if self.performance_model:
            try:
                classify_text = (
                    "Structural performance risk analysis for code snippet: "
                    f"{snippet}"
                )
                cls = await self._run_classification_inference(classify_text)
                if isinstance(cls, list) and cls and isinstance(cls[0], dict):
                    label = str(cls[0].get("label", "")).upper()
                    score = float(cls[0].get("score", 0.0) or 0.0)
                    result["ai_structure_risk_label"] = label
                    result["ai_structure_risk_confidence"] = round(max(0.0, min(1.0, score)), 3)
            except Exception as e:
                log("ai_performance_agent", LogLevel.WARNING, f"⚠️ 结构分类分析失败: {e}")

        # 2) 生成模型补充结构建议并提取可落地信号
        generated_text = ""
        if self.optimization_generator:
            try:
                prompt = get_prompt(
                    task_type="performance",
                    variant="algorithmic_analysis",
                    code_snippet=snippet,
                )
                generated = await self._run_generation_inference(
                    prompt,
                    max_new_tokens=180,
                    temperature=0.25,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.optimization_generator.tokenizer.eos_token_id,
                )
                if isinstance(generated, list) and generated and isinstance(generated[0], dict):
                    generated_text = str(generated[0].get("generated_text", "")).strip()
                elif isinstance(generated, str):
                    generated_text = generated.strip()
            except Exception as e:
                log("ai_performance_agent", LogLevel.WARNING, f"⚠️ 结构生成分析失败: {e}")

        if generated_text:
            result.update(self._extract_ai_structural_signals(generated_text))
            result["ai_structural_insight"] = generated_text[:400]

        return result

    def _extract_ai_structural_signals(self, analysis_text: str) -> Dict[str, Any]:
        """从 LLM 文本中提取结构化信号，避免仅依赖自由文本。"""
        if not isinstance(analysis_text, str) or not analysis_text.strip():
            return {}

        lowered = analysis_text.lower()
        signals: Dict[str, Any] = {}

        # 这些字段与现有结构字典兼容，属于增强信号，不覆盖原有检测结果。
        if any(k in lowered for k in ["async", "await", "concurrent", "parallel", "线程", "并发"]):
            signals["ai_async_signal"] = True
        if any(k in lowered for k in ["sql", "database", "query", "orm", "索引", "连接池"]):
            signals["ai_database_signal"] = True
        if any(k in lowered for k in ["cache", "memo", "缓存"]):
            signals["ai_caching_opportunity"] = True
        if any(k in lowered for k in ["i/o", "io", "disk", "network", "socket", "磁盘", "网络"]):
            signals["ai_io_hotspot_signal"] = True
        if any(k in lowered for k in ["refactor", "split", "modular", "重构", "拆分", "模块化"]):
            signals["ai_refactor_needed"] = True

        return signals
    
    async def _parse_complexity_result(self, result, code, index):
        if result and result[0].get("score", 0) > 0.6:
            return {
                "function_id": f"func_{index}",
                "estimated_complexity": "O(n)",
                "confidence": result[0]["score"]
            }
        return None
    
    async def _extract_complexity_details(self, analysis, code, index):
        text = ""
        if isinstance(analysis, list) and analysis:
            first = analysis[0]
            if isinstance(first, dict):
                text = str(first.get("generated_text", "")).strip()
        elif isinstance(analysis, str):
            text = analysis.strip()

        if not text:
            return None

        upper_text = text.upper()
        complexity_candidates = ["O(1)", "O(LOG N)", "O(N)", "O(N LOG N)", "O(N^2)", "O(N^3)", "O(2^N)"]
        estimated = "O(n)"
        for candidate in complexity_candidates:
            if candidate in upper_text:
                estimated = candidate.replace("LOG N", "log n").replace("N", "n")
                break

        hot_words = [
            "nested loop", "recursion", "database", "query", "io", "network", "cache", "sort", "scan"
        ]
        concerns = [w for w in hot_words if w in text.lower()]

        return {
            "function_id": f"func_{index}_detail",
            "estimated_complexity": estimated,
            "confidence": 0.68,
            "analysis_source": "llm_generation",
            "summary": text[:220],
            "potential_concerns": concerns,
            "code_preview": (code or "")[:160],
        }
    
    async def _calculate_overall_complexity(self, results):
        if not isinstance(results, list) or not results:
            return {"average_score": 5.0, "max_score": 5.0, "risk_level": "medium", "sample_size": 0}

        score_map = {
            "O(1)": 1.5,
            "O(LOG N)": 2.5,
            "O(N)": 4.5,
            "O(N LOG N)": 5.5,
            "O(N^2)": 7.5,
            "O(N^3)": 8.8,
            "O(2^N)": 9.6,
            "O(N!)": 10.0,
        }

        numeric_scores: List[float] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            complexity = str(item.get("estimated_complexity", "O(N)")).upper().replace(" ", "")
            normalized = complexity.replace("O(LOGN)", "O(LOG N)").replace("O(NLOGN)", "O(N LOG N)")
            score = score_map.get(normalized, 5.0)
            confidence = item.get("confidence", 0.6)
            if not isinstance(confidence, (int, float)):
                confidence = 0.6
            weighted = max(0.0, min(1.0, float(confidence))) * score + (1 - max(0.0, min(1.0, float(confidence)))) * 5.0
            numeric_scores.append(weighted)

        if not numeric_scores:
            return {"average_score": 5.0, "max_score": 5.0, "risk_level": "medium", "sample_size": 0}

        avg_score = sum(numeric_scores) / len(numeric_scores)
        max_score = max(numeric_scores)
        if avg_score >= 8.0:
            risk_level = "critical"
        elif avg_score >= 6.8:
            risk_level = "high"
        elif avg_score >= 5.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "average_score": round(avg_score, 3),
            "max_score": round(max_score, 3),
            "risk_level": risk_level,
            "sample_size": len(numeric_scores),
        }
    
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
        try:
            if not isinstance(complexity_analysis, dict):
                return self._fallback_complexity_optimizations({})

            overall = complexity_analysis.get("overall_complexity", {}) or {}
            func_items = complexity_analysis.get("function_complexities", []) or []

            focus_points: List[str] = []
            avg_score = overall.get("average_score")
            if isinstance(avg_score, (int, float)):
                if avg_score >= 7.5:
                    focus_points.append(f"整体复杂度评分偏高(average_score={avg_score})，优先考虑算法降阶。")
                elif avg_score >= 6.0:
                    focus_points.append(f"整体复杂度中等偏高(average_score={avg_score})，建议逐步优化热点函数。")

            for item in func_items[:8]:
                if not isinstance(item, dict):
                    continue
                est = str(item.get("estimated_complexity", "")).upper()
                func_id = item.get("function_id") or item.get("function_name") or "unknown"
                if any(level in est for level in ["O(N^2)", "O(N³)", "O(N^3)", "O(2^N)", "O(N!)"]):
                    focus_points.append(f"函数 {func_id} 复杂度为 {est}，建议重写核心循环或引入更优数据结构。")

            if not focus_points:
                focus_points.append("未发现明确高复杂度函数，请提供针对热路径的可观测指标与基准数据。")

            performance_issues = "\n".join(f"- {p}" for p in focus_points)
            code_snapshot = json.dumps(
                {
                    "overall_complexity": overall,
                    "function_complexities": func_items[:5],
                    "complexity_distribution": complexity_analysis.get("complexity_distribution", {}),
                },
                ensure_ascii=False,
                indent=2,
            )

            if self.optimization_generator:
                prompt = get_prompt(
                    task_type="performance",
                    variant="optimization",
                    current_code=code_snapshot[:3000],
                    performance_issues=performance_issues,
                )
                generated = await self._run_generation_inference(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.25,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.optimization_generator.tokenizer.eos_token_id,
                )
                generated_text = ""
                if isinstance(generated, list) and generated:
                    generated_text = str(generated[0].get("generated_text", "")).strip()
                elif isinstance(generated, str):
                    generated_text = generated.strip()

                if generated_text:
                    parsed = self._parse_architectural_optimization_response(generated_text)
                    llm_items = parsed.get("optimization_suggestions", []) if isinstance(parsed, dict) else []
                    if llm_items:
                        return llm_items[:5]

            return self._fallback_complexity_optimizations(complexity_analysis)
        except Exception as e:
            log("ai_performance_agent", LogLevel.WARNING, f"⚠️ 复杂度优化建议LLM生成失败，使用回退策略: {e}")
            return self._fallback_complexity_optimizations(complexity_analysis)

    def _fallback_complexity_optimizations(self, complexity_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """复杂度优化建议的规则化回退实现。"""
        suggestions: List[Dict[str, Any]] = []
        overall = complexity_analysis.get("overall_complexity", {}) if isinstance(complexity_analysis, dict) else {}
        avg_score = overall.get("average_score")

        if isinstance(avg_score, (int, float)) and avg_score >= 7.5:
            suggestions.append({
                "suggestion_id": 1,
                "type": "algorithmic_improvement",
                "priority": "high",
                "description": "整体复杂度偏高，优先对热点路径做算法降阶（如 O(n^2) -> O(n log n)）。",
                "expected_effect": "降低大输入规模下的执行时间",
                "recommendation": "引入哈希索引、双指针或分治策略并补充基准测试。",
            })
        else:
            suggestions.append({
                "suggestion_id": 1,
                "type": "monitoring",
                "priority": "medium",
                "description": "当前复杂度数据有限，建议先建立函数级性能基线。",
                "expected_effect": "定位真实热点，减少盲目优化",
                "recommendation": "为关键函数增加 profiling 与 benchmark。",
            })

        suggestions.append({
            "suggestion_id": 2,
            "type": "structural_adjustment",
            "priority": "medium",
            "description": "将复杂函数拆分为更小职责单元，降低圈复杂度与维护成本。",
            "expected_effect": "提升可维护性并减少性能回归风险",
            "recommendation": "提取热点逻辑为独立函数并缓存重复计算结果。",
        })

        return suggestions[:5]
    
    async def _generate_architectural_optimizations(self, code_content, bottlenecks):
        try:
            if not self.optimization_generator:
                return self._fallback_architectural_optimizations(code_content, bottlenecks)

            performance_issues = "\n".join(
                f"- {item.get('description', '')}"
                for item in bottlenecks[:5]
                if isinstance(item, dict)
            ) or "无明显瓶颈"

            prompt = get_prompt(
                task_type="performance",
                variant="optimization",
                current_code=code_content[:3000],
                performance_issues=performance_issues
            )
            generated_text = await self._run_generation_inference(
                prompt,
                max_new_tokens=320,
                temperature=0.3,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.optimization_generator.tokenizer.eos_token_id,
            )
            if not generated_text:
                return self._fallback_architectural_optimizations(code_content, bottlenecks)
            if isinstance(generated_text, list) and generated_text:
                generated_text = generated_text[0].get("generated_text", "")

            parsed = self._parse_architectural_optimization_response(str(generated_text))
            if not parsed:
                return self._fallback_architectural_optimizations(code_content, bottlenecks)

            return {
                "source": "llm_architectural_optimization",
                "raw_response": str(generated_text).strip(),
                "optimization_suggestions": parsed.get("optimization_suggestions", []),
                "overall_assessment": parsed.get("overall_assessment", ""),
                "estimated_effort": parsed.get("estimated_effort", ""),
                "priority": parsed.get("priority", "medium"),
            }
        except Exception as e:
            return self._fallback_architectural_optimizations(code_content, bottlenecks) | {"error": f"架构级优化建议生成失败: {e}"}

    def _parse_architectural_optimization_response(self, generated_text: str) -> Dict[str, Any]:
        """将 LLM 输出解析为结构化优化建议。"""
        if not generated_text:
            return {}

        cleaned = self._sanitize_json_like_text(generated_text)

        candidates: List[str] = []
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cleaned.find(opener)
            end = cleaned.rfind(closer)
            if start != -1 and end != -1 and end > start:
                candidates.append(cleaned[start : end + 1])

        candidates.insert(0, cleaned)

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue

            if isinstance(data, list):
                return {
                    "optimization_suggestions": self._normalize_optimization_items(data),
                    "overall_assessment": "",
                    "estimated_effort": "",
                    "priority": "medium",
                }

            if isinstance(data, dict):
                suggestions = data.get("optimization_suggestions") or data.get("suggestions") or data.get("items")
                if isinstance(suggestions, list):
                    normalized_items = self._normalize_optimization_items(suggestions)
                elif isinstance(suggestions, dict):
                    normalized_items = self._normalize_optimization_items([suggestions])
                else:
                    normalized_items = []

                return {
                    "optimization_suggestions": normalized_items,
                    "overall_assessment": str(data.get("overall_assessment", "")).strip(),
                    "estimated_effort": str(data.get("estimated_effort", "")).strip(),
                    "priority": self._normalize_priority_level(data.get("priority", "medium")),
                }

        return {}

    def _sanitize_json_like_text(self, text: str) -> str:
        """清理 LLM 返回中的常见 JSON 噪声。"""
        if not isinstance(text, str):
            return ""

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return cleaned.strip()

    def _normalize_optimization_items(self, items: List[Any]) -> List[Dict[str, Any]]:
        """把模型输出的建议项统一成稳定结构。"""
        normalized: List[Dict[str, Any]] = []
        allowed_types = {
            "algorithmic_improvement",
            "architectural_change",
            "structural_adjustment",
            "monitoring",
            "general",
            "performance",
        }

        for index, item in enumerate(items):
            if isinstance(item, str):
                normalized_item = {
                    "suggestion_id": index + 1,
                    "type": "general",
                    "priority": "medium",
                    "description": item.strip(),
                    "expected_effect": "",
                    "recommendation": item.strip(),
                }
                normalized.append(normalized_item)
                continue

            if not isinstance(item, dict):
                continue

            description = (
                item.get("description")
                or item.get("recommendation")
                or item.get("solution")
                or item.get("suggestion")
                or ""
            )
            raw_type = str(item.get("type", "general")).strip().lower()
            normalized_type = raw_type if raw_type in allowed_types else "general"

            suggestion_id = item.get("suggestion_id", index + 1)
            if not isinstance(suggestion_id, int):
                suggestion_id = index + 1

            recommendation = (
                item.get("recommendation")
                or item.get("solution")
                or description
                or "请根据瓶颈类型补充可执行的重构步骤。"
            )

            normalized.append({
                "suggestion_id": suggestion_id,
                "type": normalized_type,
                "priority": self._normalize_priority_level(item.get("priority", "medium")),
                "description": str(description).strip(),
                "expected_effect": str(item.get("expected_effect", item.get("benefit", ""))).strip(),
                "location": str(item.get("location", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "recommendation": str(recommendation).strip(),
            })

        return normalized[:5]

    def _normalize_priority_level(self, raw_priority: Any) -> str:
        """将优先级归一到允许枚举。"""
        value = str(raw_priority or "medium").strip().lower()
        allowed = {"critical", "high", "medium", "low", "info"}
        return value if value in allowed else "medium"

    def _fallback_architectural_optimizations(self, code_content, bottlenecks):
        """LLM 不可用或解析失败时的硬编码兜底。"""
        suggestions = []
        for index, bottleneck in enumerate(bottlenecks[:5]):
            if not isinstance(bottleneck, dict):
                continue
            suggestions.append({
                "suggestion_id": index + 1,
                "type": bottleneck.get("type", "performance"),
                "priority": bottleneck.get("severity", "medium"),
                "description": bottleneck.get("description", "优化性能瓶颈"),
                "expected_effect": "降低执行时间或资源消耗",
                "recommendation": "根据瓶颈类型重构相关逻辑并补充基准测试",
            })

        if not suggestions:
            suggestions.append({
                "suggestion_id": 1,
                "type": "general",
                "priority": "low",
                "description": "未检测到明显性能瓶颈，建议补充性能基准测试并观察热点路径。",
                "expected_effect": "建立性能基线，便于后续回归检测",
                "recommendation": "为核心路径增加性能测试和监控指标",
            })

        return {
            "source": "fallback_architectural_optimization",
            "optimization_suggestions": suggestions,
            "overall_assessment": "LLM 输出不可解析，已使用规则化兜底建议。",
            "estimated_effort": "1-2 hours",
            "priority": "medium",
        }
    
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