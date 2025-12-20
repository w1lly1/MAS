import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 统一使用一种导入方式
from utils import log, LogLevel

import logging

def test_simple_log_function_info_level(caplog):
    """测试新的简化日志接口 - INFO级别"""
    caplog.set_level(logging.INFO)
    
    # 测试INFO级别日志
    log("test_agent", LogLevel.INFO, "这是一个测试信息")
    
    # 验证日志是否正确记录
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "INFO"
    assert "test_agent - 这是一个测试信息" in record.message


def test_simple_log_function_warning_level(caplog):
    """测试新的简化日志接口 - WARNING级别"""
    caplog.set_level(logging.WARNING)
    
    # 测试WARNING级别日志
    log("test_agent", LogLevel.WARNING, "这是一个警告信息")
    
    # 验证日志是否正确记录
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "test_agent - 这是一个警告信息" in record.message


def test_simple_log_function_error_level(caplog):
    """测试新的简化日志接口 - ERROR级别"""
    caplog.set_level(logging.ERROR)
    
    # 测试ERROR级别日志
    log("test_agent", LogLevel.ERROR, "这是一个错误信息")
    
    # 验证日志是否正确记录
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert "test_agent - 这是一个错误信息" in record.message


def test_simple_log_function_debug_level(caplog):
    """测试新的简化日志接口 - DEBUG级别"""
    # 对于DEBUG级别，需要显式设置logger的级别
    import logging
    logging.getLogger("mas").setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)
    
    # 测试DEBUG级别日志
    log("test_agent", LogLevel.DEBUG, "这是一个调试信息")
    
    # 验证日志是否正确记录
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "DEBUG"
    assert "test_agent - 这是一个调试信息" in record.message


def test_log_output_format_elements():
    """测试日志输出格式元素"""
    # 检查LogLevel枚举值
    assert LogLevel.DEBUG.value == "debug"
    assert LogLevel.INFO.value == "info"
    assert LogLevel.WARNING.value == "warning"
    assert LogLevel.ERROR.value == "error"


def test_multiple_log_calls(caplog):
    """测试多次调用日志函数"""
    caplog.set_level(logging.INFO)
    
    # 多次调用
    log("agent1", LogLevel.INFO, "第一条消息")
    log("agent2", LogLevel.WARNING, "第二条消息")
    log("agent3", LogLevel.ERROR, "第三条消息")
    
    # 验证所有日志都被记录
    assert len(caplog.records) == 3
    
    # 验证每条记录的内容
    assert "agent1 - 第一条消息" in caplog.records[0].message
    assert "agent2 - 第二条消息" in caplog.records[1].message
    assert "agent3 - 第三条消息" in caplog.records[2].message


def test_log_with_special_characters(caplog):
    """测试包含特殊字符的日志消息"""
    caplog.set_level(logging.INFO)
    
    # 测试包含特殊字符的消息
    special_message = "包含中文、English、数字123、符号!@#$%^&*()的消息"
    log("test_agent", LogLevel.INFO, special_message)
    
    # 验证日志是否正确记录
    assert len(caplog.records) == 1
    assert special_message in caplog.records[0].message


def test_log_with_empty_message(caplog):
    """测试空消息"""
    caplog.set_level(logging.INFO)
    
    # 测试空消息
    log("test_agent", LogLevel.INFO, "")
    
    # 验证日志是否正确记录
    assert len(caplog.records) == 1
    assert "test_agent - " in caplog.records[0].message


def test_log_with_none_values():
    """测试None值处理"""
    # 这个测试主要是确保不会抛出异常
    try:
        log(None, LogLevel.INFO, "测试消息")  # agent_name为None
        log("test_agent", None, "测试消息")   # level为None
        log("test_agent", LogLevel.INFO, None)  # message为None
        # 如果没有异常，则测试通过
        assert True
    except Exception:
        # 如果有任何异常，则测试失败
        assert False, "log函数应该能处理None值而不抛出异常"