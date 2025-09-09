# Model Compatibility Testing - Quick Start Guide

## 概述

这个测试框架帮助您系统性地检查AI模型与当前transformers库的兼容性，确保在部署到生产环境之前发现并解决潜在问题。

## 快速开始

### 1. 环境检查
```bash
# 检查基本环境和依赖
python tests/model_compatibility/environment_check.py
```

### 2. 快速兼容性检查
```bash
# 快速测试核心模型的基本兼容性
python tests/model_compatibility/quick_check.py
```

### 3. 完整兼容性测试
```bash
# 运行所有模型的详细兼容性测试
python tests/model_compatibility/compatibility_test_suite.py
```

### 4. 自动化测试（推荐）
```bash
# 运行完整的自动化测试流程
bash tests/model_compatibility/run_tests.sh
```

## 测试特定模型

### 单独测试ChatGLM3-6B
```bash
python tests/model_compatibility/individual_model_tests/test_chatglm3.py
```

### 单独测试Qwen2-7B
```bash
python tests/model_compatibility/individual_model_tests/test_qwen2.py
```

### 单独测试CodeBERT
```bash
python tests/model_compatibility/individual_model_tests/test_codebert.py
```

## 结果解读

### 测试状态说明
- ✅ **pass**: 测试通过，模型完全兼容
- ❌ **fail**: 测试失败，存在兼容性问题
- ⚠️ **warning**: 测试通过但有警告，可能存在小问题
- ⏭️ **skip**: 测试被跳过（通常因为前置条件未满足）

### 兼容性级别
- **fully_compatible**: 完全兼容，可直接用于生产
- **mostly_compatible**: 大部分兼容，可能有小限制
- **partially_compatible**: 部分兼容，需要修复或替代方案
- **incompatible**: 不兼容，不建议使用

## 结果文件

所有测试结果保存在 `tests/model_compatibility/results/` 目录：

- `environment_report.json` - 环境兼容性报告
- `compatibility_report.json` - 完整兼容性分析报告
- `detailed_results.json` - 详细测试结果
- `quick_check.json` - 快速检查结果

## 常见问题和解决方案

### ChatGLM2-6B 兼容性问题
如果遇到ChatGLM2-6B的兼容性问题：

1. **降级transformers**（推荐）:
```bash
pip install transformers==4.27.4
```

2. **切换到ChatGLM3-6B**:
```python
model_name = "THUDM/chatglm3-6b"  # 更好的兼容性
```

3. **使用Qwen2-7B替代**:
```python
model_name = "Qwen/Qwen2-7B-Chat"  # 现代化替代方案
```

### Transformers版本兼容性
- **4.56.0+**: 最新版本，可能与某些模型不兼容
- **4.40-4.55**: 较新版本，大部分现代模型兼容
- **4.36-4.39**: 良好兼容性，推荐版本范围
- **<4.36**: 较旧版本，可能缺少新模型支持

## 生产部署建议

### 兼容性验证流程
1. 运行完整兼容性测试
2. 确认核心模型状态为"fully_compatible"或"mostly_compatible"
3. 对关键功能进行专项测试
4. 在测试环境进行集成测试
5. 逐步部署到生产环境

### 监控和维护
- 定期运行兼容性测试（特别是依赖更新后）
- 监控模型性能指标
- 准备备用模型方案
- 建立回滚机制

## 扩展测试框架

### 添加新模型测试
1. 在 `model_registry.py` 中注册新模型
2. 创建 `tests/model_compatibility/individual_model_tests/test_new_model.py`
3. 在 `compatibility_test_suite.py` 中添加测试模块
4. 运行测试验证

### 自定义测试用例
参考现有测试文件的结构，实现自定义测试逻辑：
- 配置加载测试
- 模型加载测试
- 功能性测试
- 性能测试

## 技术支持

如果遇到问题：
1. 查看 `tests/model_compatibility/results/` 中的详细错误信息
2. 检查 `COMPATIBILITY_REPORT.md` 中的已知问题
3. 参考模型官方文档和GitHub Issues
4. 考虑使用替代模型或降级依赖版本
