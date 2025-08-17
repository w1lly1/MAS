import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/mas.db")

# HuggingFace模型配置
HUGGINGFACE_CONFIG = {
    "cache_dir": os.getenv("HF_CACHE_DIR", f"{PROJECT_ROOT}/.hf_cache"),
    "models": {
        "code_quality": {
            "name": "microsoft/CodeBERT-base",
            "task": "text-classification"
        },
        "security": {
            "name": "huggingface/CodeBERTa-small-v1", 
            "task": "text-classification"
        },
        "performance": {
            "name": "distilbert-base-uncased",
            "task": "text-classification"
        }
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
        "huggingface": HUGGINGFACE_CONFIG
    }

AGENT_CONFIG = get_config()
REPORT_CONFIG = {"format": "excel", "output_dir": f"{PROJECT_ROOT}/reports"}