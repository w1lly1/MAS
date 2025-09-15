"""
MAS测试框架
Multi-Agent System Test Framework
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试配置
TEST_CONFIG = {
    "timeout": 30,  # 默认测试超时时间（秒）
    "retry_count": 3,  # 重试次数
    "log_level": "INFO",
    "temp_dir": "/tmp/mas_tests",
    "mock_models": True,  # 是否使用模拟模型
}

# 测试工具函数
def setup_test_environment():
    """设置测试环境"""
    import tempfile
    import shutil
    
    # 创建临时测试目录
    temp_dir = Path(TEST_CONFIG["temp_dir"])
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    return temp_dir

def cleanup_test_environment():
    """清理测试环境"""
    import shutil
    
    temp_dir = Path(TEST_CONFIG["temp_dir"])
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

__version__ = "1.0.0"
