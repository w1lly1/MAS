"""
MAS 多智能体系统

智能体模块包含:
- BaseAgent: 基础智能体抽象类
- AgentManager: 智能体管理器
- UserCommunicationAgent: 用户沟通智能体
- StaticCodeScanAgent: 静态代码扫描智能体  
- CodeQualityAgent: 代码质量分析智能体 (CodeBERT)
- SecurityAgent: 安全分析智能体 (CodeBERTa)
- PerformanceAgent: 性能分析智能体 (DistilBERT)
- SummaryAgent: 结果汇总输出智能体
"""

from .base_agent import BaseAgent, Message, AgentStatus
from .agent_manager import AgentManager
from .user_communicate_agent import UserCommunicationAgent
from .static_code_scan_agent import StaticCodeScanAgent
from .code_quality_analysis_agent import CodeQualityAgent
from .security_analysis_agent import SecurityAgent
from .performance_analysis_agent import PerformanceAgent
from .analysis_result_summary_agent import SummaryAgent

__all__ = [
    # 基础类
    "BaseAgent",
    "Message", 
    "AgentStatus",
    
    # 管理器
    "AgentManager",
    
    # 智能体
    "UserCommunicationAgent",
    "StaticCodeScanAgent", 
    "CodeQualityAgent",
    "SecurityAgent",
    "PerformanceAgent",
    "SummaryAgent"
]

# 智能体类型映射
AGENT_TYPES = {
    "user_communication": UserCommunicationAgent,
    "static_code_scan": StaticCodeScanAgent,
    "code_quality": CodeQualityAgent,
    "security": SecurityAgent,
    "performance": PerformanceAgent,
    "summary": SummaryAgent
}

def create_agent(agent_type: str) -> BaseAgent:
    """
    工厂方法创建智能体实例
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unsupported agent type: {agent_type}")
        
    return AGENT_TYPES[agent_type]()