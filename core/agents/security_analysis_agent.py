import torch
from transformers import pipeline, AutoTokenizer
from typing import Dict, Any
from .base_agent import BaseAgent, Message
from infrastructure.database.service import DatabaseService
from infrastructure.config.settings import HUGGINGFACE_CONFIG

class SecurityAgent(BaseAgent):
    """安全分析专用智能体 - 使用CodeBERTa模型"""
    
    def __init__(self):
        super().__init__("security_agent", "安全分析智能体")
        self.db_service = DatabaseService()
        self.model_config = HUGGINGFACE_CONFIG["models"]["security"]
        self.pipeline = None
        self.security_patterns = [
            "eval(", "exec(", "os.system", "subprocess.call",
            "input(", "raw_input(", "open(", "__import__",
            "sql", "SELECT", "INSERT", "UPDATE", "DELETE"
        ]
        
    async def _initialize_model(self):
        """初始化安全分析模型"""
        try:
            model_name = self.model_config["name"]
            cache_dir = HUGGINGFACE_CONFIG["cache_dir"]
            
            self.pipeline = pipeline(
                "text-classification",
                model=model_name,
                cache_dir=cache_dir,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"✅ 安全分析模型加载成功: {model_name}")
            
        except Exception as e:
            print(f"❌ 安全分析模型加载失败: {e}")
            self.pipeline = None
            
    async def handle_message(self, message: Message):
        """处理安全分析请求"""
        if message.message_type == "security_analysis_request":
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
                    "analysis_type": "security",
                    "result": result
                },
                message_type="analysis_result"
            )
            
    async def _execute_task_impl(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行安全分析"""
        if not self.pipeline:
            await self._initialize_model()
            
        code_content = task_data.get("code_content", "")
        code_directory = task_data.get("code_directory", "")
        
        if not code_content and code_directory:
            code_content = await self._extract_code_from_directory(code_directory)
            
        # 执行安全分析
        security_result = await self._analyze_security(code_content)
        
        # 保存结果
        requirement_id = task_data.get("requirement_id")
        if requirement_id:
            await self.db_service.save_analysis_result(
                requirement_id=requirement_id,
                agent_type="security",
                result_data=security_result,
                status="completed"
            )
        
        return security_result
        
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
        
    async def _analyze_security(self, code_content: str) -> Dict[str, Any]:
        """执行安全分析"""
        vulnerabilities = []
        security_score = 100
        
        # 基于模式的安全检查
        for pattern in self.security_patterns:
            if pattern.lower() in code_content.lower():
                vulnerabilities.append({
                    "type": "dangerous_function",
                    "pattern": pattern,
                    "description": f"检测到潜在危险函数: {pattern}",
                    "severity": "high" if pattern in ["eval(", "exec(", "os.system"] else "medium"
                })
                security_score -= 15
                
        # 使用AI模型进行额外分析（如果可用）
        if self.pipeline and code_content.strip():
            try:
                # 截断代码内容
                if len(code_content) > 512:
                    code_content = code_content[:512]
                    
                result = self.pipeline(code_content)
                model_confidence = result[0]['score'] if result else 0.5
                
                # 基于模型置信度调整安全评分
                if model_confidence < 0.7:
                    vulnerabilities.append({
                        "type": "ai_detected",
                        "description": "AI模型检测到潜在安全问题",
                        "severity": "medium",
                        "confidence": model_confidence
                    })
                    security_score -= 10
                    
            except Exception as e:
                print(f"AI安全分析失败: {e}")
                
        # 确保分数不为负
        security_score = max(0, security_score)
        
        return {
            "security_score": security_score,
            "vulnerability_count": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "recommendations": self._generate_security_recommendations(vulnerabilities),
            "model_used": self.model_config["name"],
            "analysis_status": "completed"
        }
        
    def _generate_security_recommendations(self, vulnerabilities) -> list:
        """生成安全建议"""
        recommendations = []
        
        if vulnerabilities:
            recommendations.append("审查所有标记的潜在安全漏洞")
            recommendations.append("使用安全的替代函数")
            recommendations.append("实施输入验证和输出编码")
            recommendations.append("定期进行安全代码审查")
        else:
            recommendations.append("当前代码未发现明显安全问题")
            recommendations.append("建议持续关注安全最佳实践")
            
        return recommendations