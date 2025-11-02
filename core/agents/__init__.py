"""
MAS 多智能体系统 - 智能体模块初始化
包含所有分析智能体的定义和导入
"""

from .base_agent import BaseAgent, Message
from .agent_manager import AgentManager
from .ai_driven_user_communication_agent import AIDrivenUserCommunicationAgent
from .analysis_result_summary_agent import SummaryAgent

# AI驱动的智能体
from .ai_driven_code_quality_agent import AIDrivenCodeQualityAgent
from .ai_driven_security_agent import AIDrivenSecurityAgent
from .ai_driven_performance_agent import AIDrivenPerformanceAgent

# 传统静态扫描智能体
from .static_scan_agent import StaticCodeScanAgent

# AI驱动的可读性增强代理
from .ai_driven_readability_enhancement_agent import AIDrivenReadabilityEnhancementAgent

__all__ = [
    # 基础类
    "BaseAgent",
    "Message", 
    
    # 管理器
    "AgentManager",
    
    # 智能体
    "AIDrivenUserCommunicationAgent",
    "SummaryAgent",
    "AIDrivenCodeQualityAgent",
    "AIDrivenSecurityAgent",
    "AIDrivenPerformanceAgent",
    "StaticCodeScanAgent",
    "AIDrivenReadabilityEnhancementAgent"
]

# 智能体类型映射
AGENT_TYPES = {
    "ai_user_communication": AIDrivenUserCommunicationAgent,
    "summary": SummaryAgent,
    "ai_code_quality": AIDrivenCodeQualityAgent,
    "ai_security": AIDrivenSecurityAgent,
    "ai_performance": AIDrivenPerformanceAgent,
    "static_scan": StaticCodeScanAgent,
    "ai_readability_enhancement": AIDrivenReadabilityEnhancementAgent
}

def create_agent(agent_type: str) -> BaseAgent:
    """
    工厂方法创建智能体实例
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unsupported agent type: {agent_type}")
        
    return AGENT_TYPES[agent_type]()