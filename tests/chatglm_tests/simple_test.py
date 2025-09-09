#!/usr/bin/env python3
"""
ChatGLM简单测试
逐步诊断问题
"""

import sys
import traceback

print("🚀 开始ChatGLM简单测试")

try:
    print("📦 导入基础库...")
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    print("✅ Transformers导入成功")
    
    print("📝 测试基本配置加载...")
    config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    print(f"✅ 配置加载成功: {type(config)}")
    print(f"📋 模型类型: {config.model_type}")
    print(f"📋 层数: {config.num_layers}")
    
    print("📝 测试tokenizer加载...")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    print(f"✅ Tokenizer加载成功: {type(tokenizer)}")
    
    print("🧪 测试基本编码...")
    test_text = "你好"
    encoded = tokenizer.encode(test_text)
    print(f"✅ 编码成功: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"✅ 解码成功: {decoded}")
    
    print("✅ 基础测试完成")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    print(f"🐛 详细错误:")
    traceback.print_exc()
    sys.exit(1)

print("🎉 所有基础测试通过")
