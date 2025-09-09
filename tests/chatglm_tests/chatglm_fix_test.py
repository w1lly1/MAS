#!/usr/bin/env python3
"""
ChatGLM2-6B Tokenizer 兼容性修复
通过Monkey Patch解决padding_side参数问题
"""

import os
import sys
import asyncio
import traceback
import torch
from typing import Optional, Dict, Any

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def fix_chatglm_compatibility():
    """修复ChatGLM的兼容性问题"""
    print("🔧 应用ChatGLM全面兼容性修复...")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        model_name = "THUDM/chatglm2-6b"
        
        # 1. 修复tokenizer兼容性
        print("🔧 修复tokenizer兼容性...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # 获取ChatGLM tokenizer类
        ChatGLMTokenizer = type(tokenizer)
        
        # 保存原始_pad方法
        if hasattr(ChatGLMTokenizer, '_original_pad'):
            print("⚠️ Tokenizer已修复，跳过")
        else:
            original_pad = ChatGLMTokenizer._pad
            ChatGLMTokenizer._original_pad = original_pad
            
            # 创建兼容的_pad方法
            def compatible_pad(self, encoded_inputs, **kwargs):
                # 移除不兼容的参数
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['padding_side']}
                return ChatGLMTokenizer._original_pad(self, encoded_inputs, **filtered_kwargs)
            
            # 应用monkey patch
            ChatGLMTokenizer._pad = compatible_pad
            print("✅ Tokenizer兼容性修复成功")
        
        # 2. 修复配置兼容性
        print("🔧 修复配置兼容性...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        ChatGLMConfig = type(config)
        
        # 检查是否需要修复num_hidden_layers
        if not hasattr(config, 'num_hidden_layers') and hasattr(config, 'num_layers'):
            print("🔧 添加num_hidden_layers属性映射...")
            
            # 保存原始__getattribute__方法
            if not hasattr(ChatGLMConfig, '_original_getattribute'):
                ChatGLMConfig._original_getattribute = ChatGLMConfig.__getattribute__
                
                def patched_getattribute(self, name):
                    if name == 'num_hidden_layers' and hasattr(self, 'num_layers'):
                        return self.num_layers
                    return ChatGLMConfig._original_getattribute(self, name)
                
                ChatGLMConfig.__getattribute__ = patched_getattribute
                print("✅ 配置兼容性修复成功")
        
        print("✅ ChatGLM兼容性修复成功")
        return tokenizer, config
        
    except Exception as e:
        print(f"❌ 兼容性修复失败: {e}")
        import traceback
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None, None

class FixedChatGLMAgent:
    """修复版ChatGLM代理"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialized = False
        print("🤖 初始化修复版ChatGLM代理")
    
    def initialize(self):
        """初始化模型"""
        try:
            print("🔧 开始初始化修复版ChatGLM...")
            
            # 首先应用全面兼容性修复
            self.tokenizer, config = fix_chatglm_compatibility()
            if not self.tokenizer:
                return False
            
            # 加载模型 - 使用正确的模型类
            from transformers import AutoModelForCausalLM
            
            model_name = "THUDM/chatglm2-6b"
            print(f"📦 加载模型: {model_name}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # 对已加载的模型实例应用修复
            print("🔧 对模型实例应用兼容性修复...")
            if not hasattr(self.model, '_extract_past_from_model_output'):
                def _extract_past_from_model_output(*args, **kwargs):
                    """提取past_key_values，兼容新版transformers的参数"""
                    # 第一个参数应该是outputs
                    outputs = args[0] if args else None
                    
                    if outputs is None:
                        return None
                    
                    if hasattr(outputs, 'past_key_values'):
                        return outputs.past_key_values
                    elif isinstance(outputs, dict) and 'past_key_values' in outputs:
                        return outputs['past_key_values']
                    else:
                        return None
                
                # 将方法绑定到模型实例
                import types
                self.model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, self.model)
                print("✅ 模型实例兼容性修复成功")
            
            print("✅ 模型加载成功")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            return False
    
    def test_basic_functions(self):
        """测试基本功能"""
        if not self.initialized:
            print("❌ 模型未初始化")
            return False
        
        try:
            print("🧪 测试基本tokenizer功能...")
            
            # 测试编码
            test_text = "你好，我是ChatGLM助手"
            tokens = self.tokenizer.encode(test_text, add_special_tokens=True)
            decoded = self.tokenizer.decode(tokens)
            
            print(f"📝 原文: {test_text}")
            print(f"🔢 编码: {tokens[:10]}...")
            print(f"📄 解码: {decoded}")
            
            return True
            
        except Exception as e:
            print(f"❌ 基本功能测试失败: {e}")
            return False
    
    def generate_response(self, user_input: str) -> Optional[str]:
        """生成回应"""
        if not self.initialized:
            print("❌ 模型未初始化")
            return None
        
        try:
            print(f"🧠 生成回应: {user_input}")
            
            # 使用更安全的方式调用ChatGLM
            # 直接使用tokenizer编码和模型生成，避免chat方法的缓存问题
            inputs = self.tokenizer.encode(user_input, return_tensors="pt")
            
            # 使用generate方法而不是chat方法
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,  # 限制生成长度
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False  # 禁用缓存避免缓存相关错误
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            print(f"✅ 生成成功: {response}")
            return response
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            
            # 如果上述方法失败，尝试使用最基本的方法
            try:
                print("🔄 尝试基础生成方法...")
                
                # 简单的文本生成，不使用对话格式
                input_text = f"问：{user_input}\n答："
                inputs = self.tokenizer(input_text, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        use_cache=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 提取答案部分
                if "答：" in response:
                    response = response.split("答：", 1)[-1].strip()
                
                print(f"✅ 基础方法生成成功: {response}")
                return response
                
            except Exception as e2:
                print(f"❌ 基础方法也失败: {e2}")
                return None

def test_fixed_chatglm():
    """测试修复版ChatGLM"""
    print("🚀 测试修复版ChatGLM代理")
    print("=" * 50)
    
    agent = FixedChatGLMAgent()
    
    # 初始化
    if not agent.initialize():
        print("❌ 初始化失败")
        return None
    
    print("✅ 初始化成功")
    
    # 测试基本功能
    if not agent.test_basic_functions():
        print("❌ 基本功能测试失败")
        return None
    
    print("✅ 基本功能测试成功")
    
    # 测试对话
    test_cases = [
        "你好",
        "你是谁？",
        "请简单介绍代码分析",
        "Hello"
    ]
    
    success_count = 0
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 测试 {i}: {test_input}")
        response = agent.generate_response(test_input)
        if response:
            print(f"✅ 回应: {response}")
            success_count += 1
        else:
            print("❌ 生成失败")
    
    print(f"\n📊 测试结果: {success_count}/{len(test_cases)} 成功")
    
    if success_count > 0:
        print("🎉 修复方案有效！")
        return agent
    else:
        print("❌ 修复方案无效")
        return None

if __name__ == "__main__":
    print("🚀 开始ChatGLM兼容性修复测试")
    print("=" * 50)
    
    try:
        # 运行修复测试
        result = test_fixed_chatglm()
        
        if result:
            print("\n🎉 ChatGLM兼容性问题修复成功！")
            print("💡 可以使用这个修复方案更新原始代理")
        else:
            print("\n❌ 修复方案仍需改进")
    except Exception as e:
        print(f"\n💥 程序异常退出: {e}")
        import traceback
        print(f"🐛 详细错误: {traceback.format_exc()}")
