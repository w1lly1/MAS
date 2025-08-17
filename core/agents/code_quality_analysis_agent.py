import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG

class CodeQualityAgent(BaseAgent):
    """代码质量分析专用智能体 - 使用CodeBERT模型"""
    
    def __init__(self):
        super().__init__("code_quality_agent", "代码质量分析智能体")
        self.db_service = DatabaseService()
        self.model_config = HUGGINGFACE_CONFIG["models"]["code_quality"]
        self.pipeline = None
        self.tokenizer = None
        
    async def _initialize_model(self):
        """初始化CodeBERT模型"""
        try:
            model_name = self.model_config["name"]
            # 使用本地缓存目录
            cache_dir = HUGGINGFACE_CONFIG["cache_dir"]
            
            # 初始化tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            
            # 使用sequence classification pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=self.tokenizer,
                cache_dir=cache_dir,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"✅ CodeBERT模型加载成功: {model_name}")
            
        except Exception as e:
            print(f"❌ 代码质量模型加载失败: {e}")
            self.pipeline = None
            
    async def handle_message(self, message: Message):
        """处理代码质量分析请求"""
        if message.message_type == "quality_analysis_request":
            requirement_id = message.content.get("requirement_id")
            code_content = message.content.get("code_content", "")
            code_directory = message.content.get("code_directory", "")
            
            result = await self.execute_task({
                "requirement_id": requirement_id,
                "code_content": code_content,
                "code_directory": code_directory
            })
            
            # 发送结果给汇总智能体
            await self.send_message(
                receiver="summary_agent",
                content={
                    "requirement_id": requirement_id,
                    "analysis_type": "code_quality",
                    "result": result
                },
                message_type="analysis_result"
            )
            
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行代码质量分析"""
        if not self.pipeline:
            await self._initialize_model()
            
        if not self.pipeline:
            return {
                "error": "Model not available",
                "quality_score": 0,
                "model_used": self.model_config["name"]
            }
            
        code_content = task_data.get("code_content", "")
        code_directory = task_data.get("code_directory", "")
        
        # 如果没有代码内容，尝试从目录读取
        if not code_content and code_directory:
            code_content = await self._extract_code_from_directory(code_directory)
            
        # 使用CodeBERT分析代码质量
        quality_result = await self._analyze_code_quality(code_content)
        
        # 保存结果
        requirement_id = task_data.get("requirement_id")
        if requirement_id:
            await self.db_service.save_analysis_result(
                requirement_id=requirement_id,
                agent_type="code_quality",
                result_data=quality_result,
                status="completed"
            )
        
        return quality_result
        
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
        return code_content[:4000]  # 限制长度避免模型输入过长
        
    async def _analyze_code_quality(self, code_content: str) -> Dict[str, Any]:
        """使用CodeBERT分析代码质量"""
        if not code_content.strip():
            return {
                "quality_score": 0,
                "issues": ["代码内容为空"],
                "recommendations": [],
                "model_used": self.model_config["name"]
            }
            
        try:
            # 截断代码内容以适应模型输入限制
            max_length = 512
            if len(code_content) > max_length:
                code_content = code_content[:max_length]
                
            # 使用pipeline分析
            # 注意：CodeBERT原本用于代码理解，这里我们模拟质量评分
            result = self.pipeline(code_content)
            
            # 基于模型输出计算质量分数
            confidence = result[0]['score'] if result else 0.5
            quality_score = int(confidence * 100)
            
            # 基于启发式规则生成问题和建议
            issues = []
            recommendations = []
            
            # 简单的代码质量检查
            if len(code_content.split('\n')) > 100:
                issues.append("函数/文件过长，建议拆分")
                recommendations.append("将长函数拆分为多个小函数")
                
            if 'TODO' in code_content or 'FIXME' in code_content:
                issues.append("存在未完成的代码标记")
                recommendations.append("完成所有TODO和FIXME标记的代码")
                
            if code_content.count('def ') > 10:
                issues.append("函数数量较多，考虑模块化")
                recommendations.append("考虑将相关函数组织到类或模块中")
                
            return {
                "quality_score": quality_score,
                "issues": issues,
                "recommendations": recommendations,
                "model_confidence": confidence,
                "model_used": self.model_config["name"],
                "analysis_status": "completed"
            }
            
        except Exception as e:
            return {
                "quality_score": 50,  # 默认分数
                "issues": [f"模型分析失败: {str(e)}"],
                "recommendations": ["请检查代码格式"],
                "model_used": self.model_config["name"],
                "analysis_status": "error"
            }