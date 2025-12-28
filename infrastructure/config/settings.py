import os
from pathlib import Path
from .prompts import get_prompt, list_supported_tasks, PROMPT_MAPPING

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 报告配置
REPORTS_CONFIG = {
    "base_dir": "/var/fpwork/tiyi/project/MAS/MAS/reports",
    "directories": {
        "analysis": "analysis",
        "compatibility": "compatibility", 
        "deployment": "deployment",
        "testing": "testing"
    }
}

# 导入报告管理器
try:
    from ..reports import report_manager
    REPORT_MANAGER = report_manager
except ImportError:
    REPORT_MANAGER = None# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/mas.db")

# HuggingFace模型配置
HUGGINGFACE_CONFIG = {
    "cache_dir": "./model_cache/",
    "models": {
        "code_quality": {
            "name": "microsoft/codebert-base",
            "task": "text-classification",
            "description": "代码质量分析模型"
        },
        "security": {
            "name": "microsoft/codebert-base",
            "task": "text-classification",
            "description": "安全漏洞检测模型"
        },
        "performance": {
            "name": "microsoft/codebert-base",
            "task": "text-classification",
            "description": "性能分析模型"
        },
        "conversation": {
            "name": "Qwen/Qwen1.5-7B-Chat",
            "task": "text-generation",
            "description": "用户对话模型"
        },
        "databaseManage": {
            "name": "Qwen/Qwen1.5-7B-Chat",
            "task": "text-classification",
            "description": "数据库管理模型"
        }
        # 移除static_analysis配置，因为我们现在使用传统工具
    }
}
# 静态分析工具配置
STATIC_TOOLS_CONFIG = {
    "pylint": {
        "enabled": True,
        "timeout": 60,
        "args": ["--output-format=json", "--disable=C0111"]
    },
    "bandit": {
        "enabled": True,
        "timeout": 30,
        "args": ["-f", "json"]
    },
    "flake8": {
        "enabled": True,
        "timeout": 30,
        "args": ["--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"]
    },
    "mypy": {
        "enabled": True,
        "timeout": 60,
        "args": ["--ignore-missing-imports", "--show-error-codes"]
    },
    "safety": {
        "enabled": True,
        "timeout": 30,
        "args": ["--json"]
    }
}

def get_config():
    return {
        "database_url": DATABASE_URL,
        "static_tools": STATIC_TOOLS_CONFIG,
        "huggingface": HUGGINGFACE_CONFIG,
        "prompts": PROMPT_MAPPING
    }

def get_model_prompt(task_type: str, model_name: str = None, **kwargs) -> str:
    """获取指定任务和模型的Prompt"""
    return get_prompt(task_type, model_name, **kwargs)

AGENT_CONFIG = get_config()
REPORT_CONFIG = {"format": "excel", "output_dir": f"{PROJECT_ROOT}/reports"}

# Prompt配置的便捷访问
PROMPT_CONFIG = {
    "get_prompt": get_model_prompt,
    "supported_tasks": list_supported_tasks(),
    "mapping": PROMPT_MAPPING
}