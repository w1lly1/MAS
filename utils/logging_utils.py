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


def log_table(agent_name: str, level: LogLevel, table_str: str, title: str = None, use_logger_prefix: bool = True):
    """Log or print a potentially multi-line table/string.

    Args:
        agent_name: name to include when using logger prefix
        level: LogLevel to use when logging
        table_str: multi-line table text
        title: optional title printed before the table
        use_logger_prefix: if False, emit raw output without timestamp/level/agent prefixes (prints directly).

    When `use_logger_prefix` is False the output is printed directly to stdout to avoid
    the logging formatter prefixing each table line. This is useful when you want
    clean table blocks without timestamps/levels.
    """
    logger = logging.getLogger("mas")
    # ensure handler/formatter present (reuse same init logic)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    # pick logging function
    if level == LogLevel.DEBUG:
        fn = logger.debug
    elif level == LogLevel.INFO:
        fn = logger.info
    elif level == LogLevel.WARNING:
        fn = logger.warning
    else:
        fn = logger.error

    if use_logger_prefix:
        if title:
            fn(f"{agent_name} - {title}")
        for line in str(table_str).splitlines():
            fn(f"{agent_name} - {line}")
    else:
        # Raw print without logger prefixes/formatting — keeps table alignment intact.
        try:
            if title:
                print(title)
            print(str(table_str))
        except Exception:
            # fallback to logger if print fails for any reason
            if title:
                fn(f"{agent_name} - {title}")
            for line in str(table_str).splitlines():
                fn(f"{agent_name} - {line}")
