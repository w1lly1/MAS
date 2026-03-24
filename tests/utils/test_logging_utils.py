import re
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 统一使用一种导入方式
from utils import log, LogLevel

# 每个测试前重置日志配置
def _reset_log_config():
    """重置日志配置，确保测试之间相互独立"""
    import utils.logging_utils as log_module
    log_module._log_handlers_configured = False
    logger = logging.getLogger("mas")
    logger.handlers.clear()
    logger.propagate = True

def test_simple_log_function_info_level(capsys):
    """测试新的简化日志接口 - INFO级别"""
    _reset_log_config()
    
    # 测试INFO级别日志
    log("test_agent", LogLevel.INFO, "这是一个测试信息")
    
    # 验证日志是否正确输出到stdout
    captured = capsys.readouterr()
    assert "test_agent - 这是一个测试信息" in captured.out
    assert "INFO" in captured.out


def test_simple_log_function_warning_level(capsys):
    """测试新的简化日志接口 - WARNING级别"""
    _reset_log_config()
    
    # 测试WARNING级别日志
    log("test_agent", LogLevel.WARNING, "这是一个警告信息")
    
    # 验证日志是否正确输出到stdout
    captured = capsys.readouterr()
    assert "test_agent - 这是一个警告信息" in captured.out
    assert "WARNING" in captured.out


def test_simple_log_function_error_level(capsys):
    """测试新的简化日志接口 - ERROR级别"""
    _reset_log_config()
    
    # 测试ERROR级别日志
    log("test_agent", LogLevel.ERROR, "这是一个错误信息")
    
    # 验证日志是否正确输出到stdout
    captured = capsys.readouterr()
    assert "test_agent - 这是一个错误信息" in captured.out
    assert "ERROR" in captured.out


def test_simple_log_function_debug_level(capsys):
    """测试新的简化日志接口 - DEBUG级别"""
    _reset_log_config()
    
    # 设置环境变量启用DEBUG级别
    original_debug = os.environ.get('MAS_DEBUG')
    os.environ['MAS_DEBUG'] = '1'
    
    try:
        # 测试DEBUG级别日志
        log("test_agent", LogLevel.DEBUG, "这是一个调试信息")
        
        # 验证日志是否正确输出到stdout
        captured = capsys.readouterr()
        assert "test_agent - 这是一个调试信息" in captured.out
        assert "DEBUG" in captured.out
    finally:
        # 恢复环境变量
        if original_debug is not None:
            os.environ['MAS_DEBUG'] = original_debug
        else:
            os.environ.pop('MAS_DEBUG', None)


def test_log_output_format_elements():
    """测试日志输出格式元素"""
    # 检查LogLevel枚举值
    assert LogLevel.DEBUG.value == "debug"
    assert LogLevel.INFO.value == "info"
    assert LogLevel.WARNING.value == "warning"
    assert LogLevel.ERROR.value == "error"


def test_multiple_log_calls(capsys):
    """测试多次调用日志函数"""
    _reset_log_config()
    
    # 多次调用
    log("agent1", LogLevel.INFO, "第一条消息")
    log("agent2", LogLevel.WARNING, "第二条消息")
    log("agent3", LogLevel.ERROR, "第三条消息")
    
    # 验证所有日志都输出到stdout
    captured = capsys.readouterr()
    assert "agent1 - 第一条消息" in captured.out
    assert "agent2 - 第二条消息" in captured.out
    assert "agent3 - 第三条消息" in captured.out


def test_log_with_special_characters(capsys):
    """测试包含特殊字符的日志消息"""
    _reset_log_config()
    
    # 测试包含特殊字符的消息
    special_message = "包含中文、English、数字123、符号!@#$%^&*()的消息"
    log("test_agent", LogLevel.INFO, special_message)
    
    # 验证日志是否正确输出到stdout
    captured = capsys.readouterr()
    assert special_message in captured.out


def test_log_with_empty_message(capsys):
    """测试空消息"""
    _reset_log_config()
    
    # 测试空消息
    log("test_agent", LogLevel.INFO, "")
    
    # 验证日志是否正确输出到stdout
    captured = capsys.readouterr()
    assert "test_agent -" in captured.out


def test_log_with_none_values(capsys):
    """测试None值处理"""
    _reset_log_config()
    
    # 这个测试主要是确保不会抛出异常
    try:
        log(None, LogLevel.INFO, "测试消息")  # agent_name为None
        captured = capsys.readouterr()
        assert "测试消息" in captured.out
        
        log("test_agent", LogLevel.INFO, None)  # message为None
        captured = capsys.readouterr()
        # None 会被转换为字符串 "None"
        
        # 如果没有异常，则测试通过
        assert True
    except Exception as e:
        # 如果有任何异常，则测试失败
        assert False, f"log函数应该能处理None值而不抛出异常，但抛出了: {e}"
