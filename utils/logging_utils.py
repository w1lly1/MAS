"""
统一的日志工具模块 - 为整个项目提供一致的日志格式
"""
import logging
import datetime
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def log(agent_name: str, level: LogLevel, message: str):
    """
    简化的日志接口
    
    Args:
        agent_name: 智能体名称
        level: 日志级别
        message: 日志消息
    """
    # 创建专门用于简化接口的日志器
    logger = logging.getLogger("mas")
    
    # 如果还没有处理器，添加一个
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    # 根据级别记录日志
    formatted_message = f"{agent_name} - {message}"
    
    if level == LogLevel.DEBUG:
        logger.debug(formatted_message)
    elif level == LogLevel.INFO:
        logger.info(formatted_message)
    elif level == LogLevel.WARNING:
        logger.warning(formatted_message)
    elif level == LogLevel.ERROR:
        logger.error(formatted_message)
