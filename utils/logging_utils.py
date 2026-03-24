"""
统一的日志工具模块 - 为整个项目提供一致的日志格式
"""
import logging
import os
import sys
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# 全局标志，确保处理器只配置一次
_log_handlers_configured = False

def _ensure_log_handlers(logger):
    """确保日志处理器已正确配置"""
    global _log_handlers_configured
    
    if _log_handlers_configured:
        return
    
    # 检查是否需要配置处理器
    # 即使 logger.handlers 不为空，也需要检查是否是我们添加的处理器
    need_configure = not logger.handlers or not _log_handlers_configured
    
    if need_configure:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 清除可能存在的无效处理器
        logger.handlers.clear()
        
        # 检查是否启用文件日志模式
        log_file = os.getenv('MAS_LOG_FILE')
        
        if log_file:
            # 使用文件处理器
            try:
                # 在创建文件处理器之前，先清空文件内容
                if os.path.exists(log_file):
                    open(log_file, 'w', encoding='utf-8').close()
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # 文件处理器创建失败时回退到控制台
                print(f"[WARN] 无法创建日志文件 {log_file}: {e}")
        
        # 默认总是添加控制台处理器（除非明确禁用）
        if os.getenv('MAS_LOG_CONSOLE') != '0':
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 设置日志级别
        if os.getenv('MAS_DEBUG') == '1':
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # 禁止日志传播到根 logger，避免重复输出
        logger.propagate = False
        
        _log_handlers_configured = True

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
    
    # 确保处理器已配置
    _ensure_log_handlers(logger)
    
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
    # 确保处理器已配置
    _ensure_log_handlers(logger)

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
                print(title, flush=True)
            print(str(table_str), flush=True)
        except Exception:
            # fallback to logger if print fails for any reason
            if title:
                fn(f"{agent_name} - {title}")
            for line in str(table_str).splitlines():
                fn(f"{agent_name} - {line}")
