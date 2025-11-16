"""
AI智能体系统配置管理
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

class AgentMode(Enum):
    """智能体运行模式"""
    AI_DRIVEN = "ai_driven"

class AIAgentConfig:
    """AI智能体配置管理器"""
    
    def __init__(self):
        # 使用相对路径，配置文件在同一目录下
        self.config_file = Path(__file__).parent / "ai_agent_config.json"
        self.default_config = {
            "agent_mode": AgentMode.AI_DRIVEN.value,
            "ai_model_settings": {
                "enable_code_quality_ai": True,
                "enable_security_ai": True,
                "enable_performance_ai": True,
                "enable_static_scan_ai": True,
                "enable_user_communication_ai": True,
                "model_timeout_seconds": 30,
                "max_code_length": 10000,
                "ai_confidence_threshold": 0.7
            },
            "user_communication_settings": {
                "hybrid_mode": True,
                "ai_intent_analysis": True,
                "ai_response_generation": True,
                "natural_language_understanding": True,
                "context_memory": True,
                "fallback_to_traditional": True,
                "ai_confidence_threshold": 0.8
            },
            "performance_settings": {
                "parallel_ai_processing": True,
                "cache_ai_results": True,
                "max_concurrent_ai_tasks": 3
            },
            "prompt_engineering": {
                "use_dynamic_prompts": True,
                "context_aware_prompts": True,
                "multi_language_prompts": True,
                "custom_prompt_templates": {
                    "user_intent_analysis": "分析用户意图并提取关键信息: {user_message}",
                    "requirement_extraction": "从用户消息中提取代码分析需求: {user_message}",
                    "response_generation": "生成专业友好的回应: {context}"
                }
            }
        }
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置以确保所有键都存在
                    return self._merge_configs(self.default_config, config)
            else:
                # 创建默认配置文件
                self._save_config(self.default_config)
                return self.default_config
        except Exception as e:
            print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
            return self.default_config
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置，用户配置优先"""
        merged = default.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 配置文件保存失败: {e}")
    
    def get_agent_mode(self) -> AgentMode:
        """获取当前智能体运行模式"""
        mode_str = self.config.get("agent_mode", AgentMode.AI_DRIVEN.value)
        try:
            return AgentMode(mode_str)
        except ValueError:
            return AgentMode.AI_DRIVEN
    
    def set_agent_mode(self, mode: AgentMode):
        """设置智能体运行模式"""
        self.config["agent_mode"] = mode.value
        self._save_config(self.config)
    
    def is_ai_enabled_for_agent(self, agent_type: str) -> bool:
        """检查指定智能体是否启用AI功能"""
        ai_settings = self.config.get("ai_model_settings", {})
        setting_key = f"enable_{agent_type}_ai"
        return ai_settings.get(setting_key, True)
    
    def get_ai_model_timeout(self) -> int:
        """获取AI模型超时时间"""
        return self.config.get("ai_model_settings", {}).get("model_timeout_seconds", 30)
    
    def get_max_code_length(self) -> int:
        """获取最大代码长度限制"""
        return self.config.get("ai_model_settings", {}).get("max_code_length", 10000)
    
    def get_ai_confidence_threshold(self) -> float:
        """获取AI置信度阈值"""
        return self.config.get("ai_model_settings", {}).get("ai_confidence_threshold", 0.7)
    
    def get_max_concurrent_ai_tasks(self) -> int:
        """获取最大并发AI任务数"""
        return self.config.get("performance_settings", {}).get("max_concurrent_ai_tasks", 3)
    
    def is_parallel_processing_enabled(self) -> bool:
        """是否启用并行AI处理"""
        return self.config.get("performance_settings", {}).get("parallel_ai_processing", True)
    
    def is_ai_caching_enabled(self) -> bool:
        """是否启用AI结果缓存"""
        return self.config.get("performance_settings", {}).get("cache_ai_results", True)
    
    def get_custom_prompt_template(self, agent_type: str) -> Optional[str]:
        """获取自定义prompt模板"""
        templates = self.config.get("prompt_engineering", {}).get("custom_prompt_templates", {})
        return templates.get(agent_type)
    
    def set_custom_prompt_template(self, agent_type: str, template: str):
        """设置自定义prompt模板"""
        if "prompt_engineering" not in self.config:
            self.config["prompt_engineering"] = {}
        if "custom_prompt_templates" not in self.config["prompt_engineering"]:
            self.config["prompt_engineering"]["custom_prompt_templates"] = {}
        
        self.config["prompt_engineering"]["custom_prompt_templates"][agent_type] = template
        self._save_config(self.config)
    
    def get_agent_selection_strategy(self) -> Dict[str, str]:
        """获取智能体选择策略"""
        # 始终使用AI驱动模式
        return {
            "code_quality": "ai_code_quality",
            "security": "ai_security", 
            "performance": "ai_performance",
            "static_scan": "static_scan"
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        self.config = self._merge_configs(self.config, updates)
        self._save_config(self.config)
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.config = self.default_config.copy()
        self._save_config(self.config)
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查必要的配置项
            required_keys = ["agent_mode", "ai_model_settings"]
            for key in required_keys:
                if key not in self.config:
                    return False
            
            # 验证agent_mode
            mode_str = self.config["agent_mode"]
            AgentMode(mode_str)  # 如果无效会抛出异常
            
            return True
        except Exception:
            return False
    
    def should_use_traditional_fallback(self) -> bool:
        """是否应该使用传统模式作为降级备用"""
        # AI驱动模式下，始终尝试AI优先，但可以降级到传统模式
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "agent_mode": self.get_agent_mode().value,
            "ai_enabled_agents": {
                agent_type: self.is_ai_enabled_for_agent(agent_type)
                for agent_type in ["code_quality", "security", "performance", "static_scan"]
            },
            "performance_settings": {
                "parallel_processing": self.is_parallel_processing_enabled(),
                "max_concurrent_tasks": self.get_max_concurrent_ai_tasks(),
                "caching_enabled": self.is_ai_caching_enabled()
            },
            "fallback_enabled": self.should_use_traditional_fallback(),
            "config_valid": self.validate_config()
        }
    
    def is_user_communication_ai_enabled(self) -> bool:
        """检查是否启用用户交流AI"""
        return self.config.get("ai_model_settings", {}).get("enable_user_communication_ai", True)
    
    def is_hybrid_communication_mode(self) -> bool:
        """检查是否启用混合交流模式"""
        return self.config.get("user_communication_settings", {}).get("hybrid_mode", True)
    
    def get_user_communication_ai_confidence_threshold(self) -> float:
        """获取用户交流AI置信度阈值"""
        return self.config.get("user_communication_settings", {}).get("ai_confidence_threshold", 0.8)
    
    def is_natural_language_understanding_enabled(self) -> bool:
        """检查是否启用自然语言理解"""
        return self.config.get("user_communication_settings", {}).get("natural_language_understanding", True)
    
    def is_context_memory_enabled(self) -> bool:
        """检查是否启用上下文记忆"""
        return self.config.get("user_communication_settings", {}).get("context_memory", True)
    
    def should_fallback_to_traditional_communication(self) -> bool:
        """检查是否应该降级到传统交流模式"""
        return self.config.get("user_communication_settings", {}).get("fallback_to_traditional", True)
    
    def get_user_communication_prompt_templates(self) -> Dict[str, str]:
        """获取用户交流相关的提示词模板"""
        all_templates = self.config.get("prompt_engineering", {}).get("custom_prompt_templates", {})
        return {
            "user_intent_analysis": all_templates.get("user_intent_analysis", "分析用户意图并提取关键信息: {user_message}"),
            "requirement_extraction": all_templates.get("requirement_extraction", "从用户消息中提取代码分析需求: {user_message}"),
            "response_generation": all_templates.get("response_generation", "生成专业友好的回应: {context}")
        }
        
    def set_user_communication_ai_enabled(self, enabled: bool):
        """设置用户交流AI启用状态"""
        if "ai_model_settings" not in self.config:
            self.config["ai_model_settings"] = {}
        self.config["ai_model_settings"]["enable_user_communication_ai"] = enabled
        self._save_config(self.config)
    
    def set_hybrid_communication_mode(self, enabled: bool):
        """设置混合交流模式"""
        if "user_communication_settings" not in self.config:
            self.config["user_communication_settings"] = {}
        self.config["user_communication_settings"]["hybrid_mode"] = enabled
        self._save_config(self.config)
    
    # === 新增：各智能体专属配置访问方法 ===
    
    def get_code_quality_agent_config(self) -> Dict[str, Any]:
        """获取代码质量智能体配置"""
        return self.config.get("code_quality_agent", {})
    
    def get_security_agent_config(self) -> Dict[str, Any]:
        """获取安全分析智能体配置"""
        return self.config.get("security_agent", {})
    
    def get_performance_agent_config(self) -> Dict[str, Any]:
        """获取性能分析智能体配置"""
        return self.config.get("performance_agent", {})
    
    def get_user_communication_agent_config(self) -> Dict[str, Any]:
        """获取用户交互智能体配置"""
        return self.config.get("user_communication_agent", {})
    
    def get_static_scan_agent_config(self) -> Dict[str, Any]:
        """获取静态扫描智能体配置"""
        return self.config.get("static_scan_agent", {})
    
    def get_readability_agent_config(self) -> Dict[str, Any]:
        """获取可读性增强智能体配置"""
        return self.config.get("readability_enhancement_agent", {})
    
    def get_model_cache_dir(self) -> str:
        """获取模型缓存目录"""
        return self.config.get("model_configuration", {}).get("cache_dir", "./model_cache/")

# 全局配置实例
_ai_agent_config = None

def get_ai_agent_config() -> AIAgentConfig:
    """获取AI智能体配置实例（单例模式）"""
    global _ai_agent_config
    if _ai_agent_config is None:
        _ai_agent_config = AIAgentConfig()
    return _ai_agent_config

def print_config_status():
    """打印当前配置状态"""
    config = get_ai_agent_config()
    summary = config.get_config_summary()
    
    print("\n🤖 AI智能体系统配置状态:")
    print(f"运行模式: {summary['agent_mode']} (AI驱动)")
    print(f"配置有效: {'✅' if summary['config_valid'] else '❌'}")
    
    print("\nAI功能启用状态:")
    for agent, enabled in summary['ai_enabled_agents'].items():
        status = "✅" if enabled else "❌"
        print(f"  {agent}: {status}")
    
    print(f"\n性能设置:")
    print(f"  并行处理: {'✅' if summary['performance_settings']['parallel_processing'] else '❌'}")
    print(f"  最大并发任务: {summary['performance_settings']['max_concurrent_tasks']}")
    print(f"  结果缓存: {'✅' if summary['performance_settings']['caching_enabled'] else '❌'}")
