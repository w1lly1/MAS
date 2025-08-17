import torch
from transformers import pipeline, AutoTokenizer
from typing import Dict, Any
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG

class PerformanceAgent(BaseAgent):
    """性能分析专用智能体 - 使用DistilBERT模型"""
    
    def __init__(self):
        super().__init__("performance_agent", "性能分析智能体")
        self.db_service = DatabaseService()
        self.model_config = HUGGINGFACE_CONFIG["models"]["performance"]
        self.pipeline = None
        self.performance_patterns = [
            "for.*in.*range", "while.*True", "nested.*loop",
            "O(n²)", "O(n³)", "recursive", "time.sleep"
        ]
        
    async def _initialize_model(self):
        """初始化性能分析模型"""
        try:
            model_name = self.model_config["name"]
            cache_dir = HUGGINGFACE_CONFIG["cache_dir"]
            
            self.pipeline = pipeline(
                "text-classification",
                model=model_name,
                cache_dir=cache_dir,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"✅ 性能分析模型加载成功: {model_name}")
            
        except Exception as e:
            print(f"❌ 性能分析模型加载失败: {e}")
            self.pipeline = None
            
    async def handle_message(self, message: Message):
        """处理性能分析请求"""
        if message.message_type == "performance_analysis_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            
            result = await self.execute_task({
                "requirement_id": requirement_id,
                "code_content": code_content,
                "code_directory": code_directory
            })
            
            await self.send_message(
                receiver="summary_agent",
                content={
                    "requirement_id": requirement_id,
                    "analysis_type": "performance",
                    "result": result
                },
                message_type="analysis_result"
            )
            
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行性能分析"""
        if not self.pipeline:
            await self._initialize_model()
            
        code_content = task_data.get("code_content", "")
        code_directory = task_data.get("code_directory", "")
        
        if not code_content and code_directory:
            code_content = await self._extract_code_from_directory(code_directory)
            
        performance_result = await self._analyze_performance(code_content)
        
        # 保存结果
        requirement_id = task_data.get("requirement_id")
        if requirement_id:
            await self.db_service.save_analysis_result(
                requirement_id=requirement_id,
                agent_type="performance",
                result_data=performance_result,
                status="completed"
            )
        
        return performance_result
        
    async def _extract_code_from_directory(self, directory: str) -> str:
        """从目录提取代码内容"""
        import os
        code_content = ""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.py', '.js', '.java', '.cpp', '.c')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code_content += f"# File: {file_path}\n"
                                code_content += f.read() + "\n\n"
                        except Exception:
                            continue
        except Exception as e:
            print(f"读取目录失败: {e}")
        return code_content[:4000]
        
    async def _analyze_performance(self, code_content: str) -> Dict[str, Any]:
        """执行性能分析"""
        bottlenecks = []
        performance_score = 100
        
        # 基于模式的性能检查
        lines = code_content.split('\n')
        nested_loop_depth = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip().lower()
            
            # 检测嵌套循环
            if 'for ' in line_stripped or 'while ' in line_stripped:
                indent_level = len(line) - len(line.lstrip())
                if indent_level > 4:  # 嵌套循环
                    bottlenecks.append({
                        "type": "nested_loop",
                        "line": i + 1,
                        "description": "检测到嵌套循环，可能影响性能",
                        "severity": "medium"
                    })
                    performance_score -= 10
                    
            # 检测递归
            if 'def ' in line_stripped:
                func_name = line_stripped.split('def ')[1].split('(')[0]
                if func_name in code_content.lower():
                    bottlenecks.append({
                        "type": "recursion",
                        "line": i + 1,
                        "description": f"函数 {func_name} 可能使用递归",
                        "severity": "low"
                    })
                    performance_score -= 5
                    
        # 使用AI模型分析（如果可用）
        if self.pipeline and code_content.strip():
            try:
                if len(code_content) > 512:
                    code_content = code_content[:512]
                    
                result = self.pipeline(code_content)
                model_confidence = result[0]['score'] if result else 0.5
                
                if model_confidence < 0.8:
                    bottlenecks.append({
                        "type": "ai_detected",
                        "description": "AI模型检测到潜在性能问题",
                        "severity": "low",
                        "confidence": model_confidence
                    })
                    performance_score -= 5
                    
            except Exception as e:
                print(f"AI性能分析失败: {e}")
                
        performance_score = max(0, performance_score)
        
        return {
            "performance_score": performance_score,
            "bottleneck_count": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "optimizations": self._generate_optimization_suggestions(bottlenecks),
            "model_used": self.model_config["name"],
            "analysis_status": "completed"
        }
        
    def _generate_optimization_suggestions(self, bottlenecks) -> list:
        """生成优化建议"""
        suggestions = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "nested_loop":
                suggestions.append("考虑使用更高效的算法减少循环嵌套")
            elif bottleneck["type"] == "recursion":
                suggestions.append("考虑使用迭代替代递归")
                
        if not suggestions:
            suggestions.append("代码性能良好，未发现明显瓶颈")
            
        return suggestions