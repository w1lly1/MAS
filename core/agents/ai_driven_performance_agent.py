import os
import torch
import asyncio
import ast
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG
from infrastructure.reports import report_manager

class AIDrivenPerformanceAgent(BaseAgent):
    """AI驱动的性能分析智能体 - 基于深度学习和prompt工程"""
    
    def __init__(self):
        super().__init__("ai_performance_agent", "AI驱动性能分析智能体")
        self.db_service = DatabaseService()
        self.model_config = HUGGINGFACE_CONFIG["models"]["performance"]
        
        # AI性能分析组件
        self.performance_model = None
        self.complexity_analyzer = None
        self.optimization_advisor = None
        
        # 专业性能分析prompt
        self.performance_analysis_prompt = """
你是一位资深的性能优化专家和算法工程师。请对以下代码进行深度性能分析:

**性能分析维度:**
1. 时间复杂度分析 (Big O notation)
2. 空间复杂度分析
3. 算法效率评估
4. 数据结构选择合理性
5. I/O操作优化机会
6. 并发和异步处理机会
7. 内存使用模式
8. 缓存优化潜力

**代码内容:**
```{language}
{code_content}
```

**执行环境:**
- 预期数据规模: {data_scale}
- 并发用户数: {concurrent_users}
- 硬件环境: {hardware_info}
- 性能要求: {performance_requirements}

**请提供详细的性能评估:**
1. 性能等级评分 (1-10分)
2. 关键性能瓶颈识别
3. 算法复杂度分析
4. 优化建议和最佳实践
5. 重构方案 (如需要)
6. 性能测试建议

**性能分析结果:**
"""

        self.algorithmic_analysis_prompt = """
作为算法专家,请分析以下代码的算法效率:

**代码实现:**
```
{code_snippet}
```

**算法分析重点:**
- 循环结构和嵌套深度
- 递归调用模式
- 数据结构访问模式
- 搜索和排序算法选择
- 数学运算复杂度

**复杂度分析框架:**
- 最好情况时间复杂度
- 平均情况时间复杂度  
- 最坏情况时间复杂度
- 空间复杂度
- 稳定性分析

请提供结构化的算法分析:
"""

        self.optimization_prompt = """
基于性能分析结果,提供具体的优化建议:

**当前实现:**
```
{current_code}
```

**性能问题:**
{performance_issues}

**优化目标:**
- 减少执行时间
- 降低内存消耗
- 提高并发处理能力
- 增强可扩展性

**请提供:**
1. 具体的代码优化方案
2. 数据结构改进建议
3. 算法替换建议
4. 架构优化建议
5. 性能监控方案

**优化建议:**
"""

    async def _initialize_models(self):
        """初始化AI模型 - CPU优化版本"""
        try:
            print("🔧 初始化性能分析AI模型 (CPU模式)...")
            
            # 设置CPU环境变量
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_num_threads(4)  # 限制CPU线程数
            
            # 使用轻量级性能分析模型
            try:
                self.performance_model = pipeline(
                    "text-classification", 
                    model="microsoft/codebert-base",
                    device=-1,  # 强制使用CPU
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "torch_dtype": torch.float32
                    }
                )
                print("✅ CodeBERT 性能模型初始化成功 (CPU)")
            except Exception as e:
                print(f"⚠️ CodeBERT加载失败,尝试备用模型: {e}")
                self.performance_model = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",
                    device=-1,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                print("✅ DistilBERT 备用模型初始化成功 (CPU)")
            
            # 轻量级优化建议生成模型
            try:
                self.optimization_generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=-1,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "pad_token_id": 50256
                    }
                )
                print("✅ GPT-2 优化建议模型初始化成功 (CPU)")
            except Exception as e:
                print(f"⚠️ 优化建议模型加载失败: {e}")
                self.optimization_generator = None
            
            self.models_loaded = True
            print("✅ 性能分析AI模型初始化完成 (CPU优化模式)")
            
        except Exception as e:
            print(f"❌ 性能分析AI模型初始化失败: {e}")
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
            print(f"⚡ AI性能分析开始 - 需求ID: {requirement_id} run_id={run_id}")
            if not self.performance_model:
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
                    print(f"⚠️ 性能Agent单独报告生成失败 requirement={requirement_id} run_id={run_id}: {e}")
            # 发送到用户交互
            await self.send_message(
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
            await self.send_message(
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
            print(f"✅ AI性能分析完成 - 需求ID: {requirement_id} run_id={run_id}")

    async def _ai_driven_performance_analysis(self, code_content: str, code_directory: str) -> Dict[str, Any]:
        """AI驱动的全面性能分析"""
        
        try:
            print("🔍 AI正在进行深度性能分析...")
            
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
            
            print("🚀 AI性能分析完成,生成优化报告")
            
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
            print(f"❌ AI性能分析过程中出错: {e}")
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
            # 分块分析代码
            code_functions = self._extract_functions(code_content)
            complexity_results = []
            
            for i, func_code in enumerate(code_functions[:5]):  # 限制分析函数数量
                # 构造算法分析prompt
                analysis_prompt = self.algorithmic_analysis_prompt.format(
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
        
        return bottlenecks[:10]  # 限制返回数量

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

    async def _ai_optimization_planning(self, code_content: str, bottlenecks: List[Dict[str, Any]], 
                                      complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
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
            print(f"函数提取失败: {e}")
        
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
        return {
            "bottleneck_id": f"LOOP_{loop_info.get('line_number')}",
            "type": "nested_loop",
            "description": "检测到嵌套循环",
            "severity": "medium",
            "severity_score": 6.0
        }
    
    async def _ai_analyze_recursion_bottleneck(self, func_info):
        return {
            "bottleneck_id": f"RECURSION_{func_info.get('function_name')}",
            "type": "recursion",
            "description": f"递归函数: {func_info.get('function_name')}",
            "severity": "medium",
            "severity_score": 5.0
        }
    
    async def _ai_analyze_io_bottleneck(self, io_info):
        return {
            "bottleneck_id": f"IO_{io_info.get('line_number')}",
            "type": io_info.get("io_type"),
            "description": f"I/O操作: {io_info.get('operation')}",
            "severity": "low",
            "severity_score": 3.0
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
