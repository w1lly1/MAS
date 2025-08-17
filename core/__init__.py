"""
MAS 核心业务层

包含智能体系统和业务逻辑
"""

from .agents_integration import AgentIntegration, get_agent_integration_system

__all__ = ["AgentIntegration", "get_agent_integration_system"]