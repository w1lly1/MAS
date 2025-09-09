#!/usr/bin/env python3
"""
ChatGLM配置兼容性修复测试
解决 ChatGLMConfig 与新版 transformers 库的兼容性问题
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
    from transformers.cache_utils import DynamicCache
    from transformers.configuration_utils import PretrainedConfig
except ImportError as e:
    print(f"❌ 导入库失败: {e}")
    sys.exit(1)


class ChatGLMCompatibilityFixer:
    """ChatGLM兼容性修复器"""
    
    @staticmethod
    def apply_tokenizer_fix():
        """应用tokenizer兼容性修复"""
        from transformers import ChatGLMTokenizer
        
        # 保存原始_pad方法
        original_pad = ChatGLMTokenizer._pad
        
        def fixed_pad(self, encoded_inputs, max_length=None, padding_strategy=None, pad_to_multiple_of=None, return_attention_mask=None):
            """修复的_pad方法，过滤不兼容的参数"""
            # 过滤掉padding_side参数
            if hasattr(self, 'padding_side'):
                delattr(self, 'padding_side')
            
            # 调用原始方法
            return original_pad(self, encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)
        
        # 应用修复
        ChatGLMTokenizer._pad = fixed_pad
        print("✅ ChatGLM tokenizer兼容性修复已应用")
    
    @staticmethod
    def apply_config_fix():
        """应用config兼容性修复"""
        try:
            # 动态修补ChatGLMConfig
            from transformers import ChatGLMConfig
            
            # 保存原始__getattribute__方法
            original_getattribute = ChatGLMConfig.__getattribute__
            
            def fixed_getattribute(self, name):
                """修复的__getattribute__方法，添加缺失的属性映射"""
                # 属性映射表
                attribute_mapping = {
                    'num_hidden_layers': 'num_layers',  # ChatGLM使用num_layers而不是num_hidden_layers
                    'hidden_size': 'hidden_size',
                    'num_attention_heads': 'num_attention_heads',
                    'intermediate_size': 'ffn_hidden_size',
                }
                
                # 如果请求的是标准属性名，尝试映射到ChatGLM的属性名
                if name in attribute_mapping:
                    chatglm_attr = attribute_mapping[name]
                    if hasattr(self, chatglm_attr):
                        return original_getattribute(self, chatglm_attr)
                    else:
                        # 如果ChatGLM属性也不存在，提供默认值
                        defaults = {
                            'num_hidden_layers': 28,  # ChatGLM2-6B的默认层数
                            'hidden_size': 4096,
                            'num_attention_heads': 32,
                            'intermediate_size': 13696,
                        }
                        if name in defaults:
                            return defaults[name]
                
                # 调用原始方法
                return original_getattribute(self, name)
            
            # 应用修复
            ChatGLMConfig.__getattribute__ = fixed_getattribute
            print("✅ ChatGLM config兼容性修复已应用")
            
        except Exception as e:
            print(f"⚠️ Config修复应用失败: {e}")
    
    @staticmethod
    def apply_all_fixes():
        """应用所有兼容性修复"""
        ChatGLMCompatibilityFixer.apply_tokenizer_fix()
        ChatGLMCompatibilityFixer.apply_config_fix()


class FixedChatGLMAgent:
    """修复版ChatGLM代理"""
    
    def __init__(self, model_name: str = "THUDM/chatglm2-6b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.config = None
        
    def initialize(self) -> bool:
        """初始化代理"""
        try:
            print("🔧 开始初始化修复版ChatGLM...")
            
            # 应用所有兼容性修复
            print("🔧 应用ChatGLM兼容性修复...")
            ChatGLMCompatibilityFixer.apply_all_fixes()
            
            # 加载配置
            print("📋 加载配置...")
            self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            print(f"✅ 配置加载成功: num_layers={getattr(self.config, 'num_layers', 'N/A')}")
            
            # 加载tokenizer
            print("🔤 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                padding=True,
                truncation=True,
            )
            print("✅ Tokenizer加载成功")
            
            # 加载模型
            print(f"📦 加载模型: {self.model_name}")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            
            # 设置为评估模式
            self.model.eval()
            print("✅ 模型加载成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            print(f"🐛 详细错误: {traceback.format_exc()}")
            return False
    
    def test_tokenizer(self) -> bool:
        """测试tokenizer基本功能"""
        try:
            print("🧪 测试基本tokenizer功能...")
            test_text = "你好，我是ChatGLM助手"
            
            # 编码
            encoded = self.tokenizer.encode(test_text)
            print(f"📝 原文: {test_text}")
            print(f"🔢 编码: {encoded[:10]}..." if len(encoded) > 10 else f"🔢 编码: {encoded}")
            
            # 解码
            decoded = self.tokenizer.decode(encoded)
            print(f"📄 解码: {decoded}")
            
            return True
            
        except Exception as e:
            print(f"❌ Tokenizer测试失败: {e}")
            return False
    
    def generate_response(self, message: str, max_length: int = 512) -> Tuple[bool, Optional[str]]:
        """生成回应"""
        try:
            print(f"🧠 生成回应: {message}")
            
            # 使用chat方法生成回应
            response, history = self.model.chat(
                self.tokenizer,
                message,
                history=[],
                max_length=max_length,
                temperature=0.8,
                top_p=0.8,
                do_sample=True,
            )
            
            return True, response
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            print(f"🐛 详细错误: {traceback.format_exc()}")
            return False, None
    
    def test_conversation(self) -> bool:
        """测试对话功能"""
        test_messages = [
            "你好",
            "你是谁？",
            "请简单介绍代码分析",
            "Hello",
        ]
        
        success_count = 0
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n🧪 测试 {i}: {message}")
            success, response = self.generate_response(message)
            
            if success and response:
                print(f"✅ 回应: {response[:100]}..." if len(response) > 100 else f"✅ 回应: {response}")
                success_count += 1
            else:
                print("❌ 生成失败")
        
        print(f"\n📊 测试结果: {success_count}/{len(test_messages)} 成功")
        return success_count == len(test_messages)


def check_config_attributes():
    """检查配置属性"""
    try:
        print("\n🔍 检查ChatGLM配置属性...")
        config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        
        # 检查常见属性
        attributes_to_check = [
            'num_layers', 'num_hidden_layers', 'hidden_size', 
            'num_attention_heads', 'ffn_hidden_size', 'intermediate_size'
        ]
        
        print("📋 配置属性:")
        for attr in attributes_to_check:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"  ✅ {attr}: {value}")
            else:
                print(f"  ❌ {attr}: 不存在")
        
        # 列出所有属性
        print("\n📋 所有配置属性:")
        for attr in sorted(dir(config)):
            if not attr.startswith('_') and not callable(getattr(config, attr)):
                try:
                    value = getattr(config, attr)
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <无法获取>")
                    
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")


def main():
    """主函数"""
    print("🚀 测试修复版ChatGLM代理（配置修复版）")
    print("=" * 50)
    
    # 检查配置属性
    check_config_attributes()
    
    # 初始化代理
    print("\n🤖 初始化修复版ChatGLM代理")
    agent = FixedChatGLMAgent()
    
    if not agent.initialize():
        print("❌ 初始化失败，退出测试")
        return False
    
    print("✅ 初始化成功")
    
    # 测试tokenizer
    if not agent.test_tokenizer():
        print("❌ Tokenizer测试失败")
        return False
    
    print("✅ 基本功能测试成功")
    
    # 测试对话
    if agent.test_conversation():
        print("\n✅ 修复方案有效")
        return True
    else:
        print("\n❌ 修复方案仍需改进")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
