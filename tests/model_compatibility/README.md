# Model Compatibility Testing Framework

This framework provides comprehensive testing for AI models compatibility with the current transformers library setup.

## Test Structure

- `environment_check.py` - Check current environment and transformers version
- `model_registry.py` - Registry of all models to test
- `compatibility_test_suite.py` - Main test suite runner
- `individual_model_tests/` - Individual test files for each model
- `results/` - Test results and reports
- `utils/` - Testing utilities and helpers

## Supported Models

### Chat Models
- ChatGLM2-6B
- ChatGLM3-6B
- Qwen2-7B
- Baichuan2-7B
- Yi-6B-Chat

### Code Models
- CodeBERT
- CodeT5
- StarCoder

### Embedding Models
- BGE-large-zh
- Text2vec-large-chinese

## Usage

1. Run environment check: `python environment_check.py`
2. Run full compatibility suite: `python compatibility_test_suite.py`
3. Run individual model test: `python individual_model_tests/test_chatglm3.py`
