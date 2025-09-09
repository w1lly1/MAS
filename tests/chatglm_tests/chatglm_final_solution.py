#!/usr/bin/env python3
"""
ChatGLM2-6B 最终兼容性解决方案
整合所有修复，提供生产就绪的代理实现
"""

import os
import sys
import torch
from typing import Optional, Dict, Any

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def apply_chatglm_compatibility_fixes():
    """应用ChatGLM2-6B的所有兼容性修复"""
    print("🔧 应用ChatGLM2-6B兼容性修复...")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        model_name = "THUDM/chatglm2-6b"
        
        # 1. 修复tokenizer兼容性
        print("🔧 修复tokenizer兼容性...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # 获取ChatGLM tokenizer类并修复_pad方法
        ChatGLMTokenizer = type(tokenizer)
        if not hasattr(ChatGLMTokenizer, '_original_pad'):
            original_pad = ChatGLMTokenizer._pad
            ChatGLMTokenizer._original_pad = original_pad
            
            def compatible_pad(self, encoded_inputs, **kwargs):
                # 移除不兼容的参数
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['padding_side']}
                return ChatGLMTokenizer._original_pad(self, encoded_inputs, **filtered_kwargs)
            
            ChatGLMTokenizer._pad = compatible_pad
            print("✅ Tokenizer兼容性修复成功")
        
        # 2. 修复配置兼容性
        print("🔧 修复配置兼容性...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        ChatGLMConfig = type(config)
        
        if not hasattr(config, 'num_hidden_layers') and hasattr(config, 'num_layers'):
            if not hasattr(ChatGLMConfig, '_original_getattribute'):
                ChatGLMConfig._original_getattribute = ChatGLMConfig.__getattribute__
                
                def patched_getattribute(self, name):
                    if name == 'num_hidden_layers' and hasattr(self, 'num_layers'):
                        return self.num_layers
                    return ChatGLMConfig._original_getattribute(self, name)
                
                ChatGLMConfig.__getattribute__ = patched_getattribute
                print("✅ 配置兼容性修复成功")
        
        # 3. 预修复模型类的get_masks方法 - 这是关键修复！
        print("🔧 预修复ChatGLM模型类...")
        from transformers import AutoModelForCausalLM
        
        # 加载一个临时模型实例来获取模型类
        temp_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # 获取transformer类并修复get_masks方法
        transformer = temp_model.transformer
        transformer_class = type(transformer)
        
        if not hasattr(transformer_class, '_original_get_masks'):
            print("🔧 修复get_masks方法...")
            original_get_masks = transformer_class.get_masks
            transformer_class._original_get_masks = original_get_masks
            
            def safe_get_masks(self, input_ids, past_key_values, padding_mask=None):
                """安全的get_masks方法，处理None past_key_values"""
                seq_length = input_ids.shape[1]
                
                # 安全地获取past_length
                if past_key_values is None or not past_key_values or past_key_values[0] is None or past_key_values[0][0] is None:
                    past_length = 0
                else:
                    try:
                        past_length = past_key_values[0][0].shape[0]
                    except (AttributeError, IndexError, TypeError):
                        past_length = 0
                
                # 创建attention mask
                import torch
                device = input_ids.device
                dtype = torch.float32
                
                full_attention_mask = torch.ones(
                    input_ids.shape[0], 
                    past_length + seq_length, 
                    past_length + seq_length,
                    device=device,
                    dtype=dtype
                )
                
                # 应用因果掩码 (下三角矩阵)
                full_attention_mask.triu_(diagonal=1)
                full_attention_mask = full_attention_mask < 0.5
                
                if past_length:
                    full_attention_mask = full_attention_mask[..., past_length:, :]
                
                if padding_mask is not None:
                    full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
                    if not past_length:
                        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(-1)
                
                full_attention_mask = full_attention_mask.unsqueeze(1)
                return full_attention_mask
            
            transformer_class.get_masks = safe_get_masks
            print("✅ get_masks方法修复成功")
        
        # 清理临时模型
        del temp_model
        
        return tokenizer, config
        
    except Exception as e:
        print(f"❌ 兼容性修复失败: {e}")
        import traceback
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None, None

class ProductionChatGLMAgent:
    """生产就绪的ChatGLM代理"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialized = False
        print("🤖 初始化生产版ChatGLM代理")
    
    def initialize(self):
        """初始化模型"""
        try:
            print("🔧 开始初始化ChatGLM2-6B模型...")
            
            # 应用兼容性修复
            self.tokenizer, config = apply_chatglm_compatibility_fixes()
            if not self.tokenizer:
                return False
            
            # 加载模型
            from transformers import AutoModelForCausalLM
            
            model_name = "THUDM/chatglm2-6b"
            print(f"📦 加载模型: {model_name}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,  # 使用半精度提高效率
                low_cpu_mem_usage=True      # 减少CPU内存使用
            )
            
            # 修复模型实例的_extract_past_from_model_output方法
            if not hasattr(self.model, '_extract_past_from_model_output'):
                def _extract_past_from_model_output(*args, **kwargs):
                    outputs = args[0] if args else None
                    if outputs is None:
                        return None
                    if hasattr(outputs, 'past_key_values'):
                        return outputs.past_key_values
                    elif isinstance(outputs, dict) and 'past_key_values' in outputs:
                        return outputs['past_key_values']
                    return None
                
                import types
                self.model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, self.model)
                print("✅ 模型兼容性修复完成")
            
            print("✅ 模型初始化成功")
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            return False
    
    def generate_response(self, user_input: str, max_new_tokens: int = 100) -> Optional[str]:
        """生成回应"""
        if not self.initialized:
            print("❌ 模型未初始化")
            return None
        
        try:
            print(f"🧠 生成回应: {user_input}")
            
            # 使用ChatGLM的chat方法（推荐方法）
            try:
                response, history = self.model.chat(
                    self.tokenizer,
                    user_input,
                    history=[],
                    max_length=2048,
                    temperature=0.8,
                    top_p=0.8
                )
                print(f"✅ 使用chat方法生成成功: {response}")
                return response
            
            except Exception as chat_error:
                print(f"⚠️ chat方法失败，尝试generate方法: {chat_error}")
                
                # 备用方法：使用generate
                inputs = self.tokenizer.encode(user_input, return_tensors="pt")
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    pad_token_id = self.tokenizer.pad_token_id
                else:
                    pad_token_id = self.tokenizer.eos_token_id
                
                # 创建注意力掩码
                attention_mask = torch.ones_like(inputs)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.8,
                        pad_token_id=pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True  # 可以尝试启用缓存
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                print(f"✅ 使用generate方法生成成功: {response}")
                return response
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            print(f"🐛 详细错误: {traceback.format_exc()}")
            return None
    
    def test_functionality(self):
        """测试基本功能"""
        if not self.initialized:
            return False
        
        try:
            # 测试tokenizer
            test_text = "你好，世界！"
            tokens = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(tokens)
            print(f"✅ Tokenizer测试成功: {test_text} -> {decoded}")
            
            # 测试简单生成
            response = self.generate_response("你好", max_new_tokens=20)
            if response:
                print(f"✅ 简单生成测试成功: {response}")
                return True
            else:
                print("❌ 简单生成测试失败")
                return False
                
        except Exception as e:
            print(f"❌ 功能测试失败: {e}")
            return False

def main():
    """主测试函数"""
    print("🚀 ChatGLM2-6B 生产解决方案测试")
    print("=" * 60)
    
    # 创建代理实例
    agent = ProductionChatGLMAgent()
    
    # 初始化
    if not agent.initialize():
        print("❌ 初始化失败")
        return
    
    # 基本功能测试
    if not agent.test_functionality():
        print("❌ 基本功能测试失败")
        return
    
    # 对话测试
    test_questions = [
        "你好",
        "你是谁？",
        "请简单介绍Python编程",
        "什么是代码分析？"
    ]
    
    print(f"\n🧪 开始对话测试 ({len(test_questions)} 个问题)")
    print("-" * 40)
    
    successful_responses = 0
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        response = agent.generate_response(question, max_new_tokens=100)
        
        if response and response.strip():
            print(f"✅ 回答: {response}")
            successful_responses += 1
        else:
            print("❌ 生成失败或空响应")
    
    # 结果统计
    print(f"\n📊 测试结果统计")
    print(f"成功响应: {successful_responses}/{len(test_questions)}")
    success_rate = (successful_responses / len(test_questions)) * 100
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("🎉 ChatGLM2-6B兼容性修复成功！")
        print("💡 可以将此解决方案应用到生产环境")
    elif success_rate >= 50:
        print("⚠️ 部分功能正常，建议进一步优化")
    else:
        print("❌ 需要进一步调试和修复")

if __name__ == "__main__":
    main()
