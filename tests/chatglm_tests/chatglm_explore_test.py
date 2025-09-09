#!/usr/bin/env python3
"""
ChatGLM兼容性探索测试
探索正确的ChatGLM加载和修复方式
"""

import os
import sys
import traceback
from typing import Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
except ImportError as e:
    print(f"❌ 导入库失败: {e}")
    sys.exit(1)


def explore_chatglm_loading():
    """探索ChatGLM的正确加载方式"""
    print("🔍 探索ChatGLM加载方式...")
    
    try:
        # 1. 先加载tokenizer，这样可以访问自定义类
        print("📝 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        print(f"✅ Tokenizer类型: {type(tokenizer)}")
        
        # 2. 获取tokenizer的实际类
        tokenizer_class = type(tokenizer)
        print(f"📝 Tokenizer类: {tokenizer_class}")
        
        # 3. 检查tokenizer的方法
        print("📝 Tokenizer方法:")
        for method in dir(tokenizer):
            if method.startswith('_pad'):
                print(f"  - {method}")
        
        # 4. 测试基本功能
        print("🧪 测试基本编码...")
        test_text = "你好"
        encoded = tokenizer.encode(test_text)
        print(f"✅ 编码成功: {encoded}")
        
        decoded = tokenizer.decode(encoded)
        print(f"✅ 解码成功: {decoded}")
        
        return tokenizer, tokenizer_class
        
    except Exception as e:
        print(f"❌ 探索失败: {e}")
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None, None


def apply_dynamic_fixes(tokenizer_class):
    """动态应用修复"""
    try:
        print("🔧 应用动态修复...")
        
        # 修复tokenizer的_pad方法
        if hasattr(tokenizer_class, '_pad'):
            original_pad = tokenizer_class._pad
            
            def fixed_pad(self, encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None):
                """修复的_pad方法"""
                # 移除padding_side属性如果存在
                if hasattr(self, 'padding_side'):
                    delattr(self, 'padding_side')
                
                return original_pad(self, encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)
            
            tokenizer_class._pad = fixed_pad
            print("✅ Tokenizer _pad方法已修复")
        
        # 修复config的属性访问
        from transformers.configuration_utils import PretrainedConfig
        
        # 获取ChatGLMConfig类
        config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        config_class = type(config)
        
        if hasattr(config_class, '__getattribute__'):
            original_getattribute = config_class.__getattribute__
            
            def fixed_getattribute(self, name):
                """修复的__getattribute__方法"""
                # 属性映射
                if name == 'num_hidden_layers' and hasattr(self, 'num_layers'):
                    return self.num_layers
                elif name == 'intermediate_size' and hasattr(self, 'ffn_hidden_size'):
                    return self.ffn_hidden_size
                
                return original_getattribute(self, name)
            
            config_class.__getattribute__ = fixed_getattribute
            print("✅ Config __getattribute__方法已修复")
        
        return True
        
    except Exception as e:
        print(f"❌ 动态修复失败: {e}")
        return False


def test_model_loading():
    """测试模型加载"""
    try:
        print("📦 测试模型加载...")
        
        # 加载配置
        config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        print(f"✅ 配置加载成功")
        
        # 测试属性访问
        print(f"📋 num_layers: {config.num_layers}")
        print(f"📋 hidden_size: {config.hidden_size}")
        
        # 测试修复后的属性
        try:
            print(f"📋 num_hidden_layers: {config.num_hidden_layers}")
        except Exception as e:
            print(f"❌ num_hidden_layers访问失败: {e}")
        
        # 加载模型
        model = AutoModel.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        model.eval()
        print("✅ 模型加载成功")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None


def test_chat_functionality(tokenizer, model):
    """测试聊天功能"""
    try:
        print("🧪 测试聊天功能...")
        
        # 简单对话测试
        message = "你好"
        print(f"📝 输入: {message}")
        
        response, history = model.chat(
            tokenizer,
            message,
            history=[],
            max_length=100,
            temperature=0.8,
        )
        
        print(f"✅ 回应: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 聊天测试失败: {e}")
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    print("🚀 ChatGLM兼容性探索测试")
    print("=" * 50)
    
    # 1. 探索加载方式
    tokenizer, tokenizer_class = explore_chatglm_loading()
    if not tokenizer:
        return False
    
    # 2. 应用动态修复
    if not apply_dynamic_fixes(tokenizer_class):
        return False
    
    # 3. 测试模型加载
    model = test_model_loading()
    if not model:
        return False
    
    # 4. 测试聊天功能
    if test_chat_functionality(tokenizer, model):
        print("\n✅ 所有测试通过，ChatGLM功能正常")
        return True
    else:
        print("\n❌ 聊天功能测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
