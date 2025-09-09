#!/usr/bin/env python3
"""
ChatGLM2-6B 模型兼容性测试
专门解决 padding_side 参数兼容性问题
"""

import os
import sys
import asyncio
import traceback
from typing import Optional

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_basic_imports():
    """测试基础模块导入"""
    print("🔧 测试基础模块导入...")
    try:
        import torch
        import transformers
        import sentencepiece
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ SentencePiece: {sentencepiece.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_chatglm_tokenizer_direct():
    """直接测试ChatGLM tokenizer"""
    print("🧪 直接测试ChatGLM tokenizer...")
    try:
        from transformers import AutoTokenizer
        
        model_name = "THUDM/chatglm2-6b"
        print(f"📦 加载tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("✅ Tokenizer加载成功")
        
        # 测试基本编码功能
        test_text = "你好，我是ChatGLM助手"
        tokens = tokenizer.encode(test_text, add_special_tokens=True)
        decoded = tokenizer.decode(tokens)
        
        print(f"📝 原文: {test_text}")
        print(f"🔢 编码: {tokens[:10]}...")  # 只显示前10个token
        print(f"📄 解码: {decoded}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Tokenizer测试失败: {e}")
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None

def test_chatglm_model_direct():
    """直接测试ChatGLM模型加载"""
    print("🧪 直接测试ChatGLM模型加载...")
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "THUDM/chatglm2-6b"
        print(f"📦 加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        
        print("✅ 模型加载成功")
        
        # 测试直接对话
        test_input = "你好"
        print(f"📝 测试输入: {test_input}")
        
        # 使用模型的chat方法
        response, history = model.chat(tokenizer, test_input, history=[])
        print(f"🎉 模型回应: {response}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None, None

def test_alternative_pipeline():
    """测试替代的pipeline配置"""
    print("🧪 测试替代pipeline配置...")
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        
        model_name = "THUDM/chatglm2-6b"
        print(f"📦 使用替代方式加载: {model_name}")
        
        # 方法1: 不使用pipeline，直接使用模型
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        
        print("✅ 直接模型加载成功")
        
        # 手动生成
        test_prompt = "用户: 你好\n助手:"
        inputs = tokenizer.encode(test_prompt, return_tensors="pt")
        
        print(f"📝 输入prompt: {test_prompt}")
        print("🤖 开始生成...")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回应部分
        if test_prompt in response:
            ai_response = response.replace(test_prompt, "").strip()
        else:
            ai_response = response.strip()
            
        print(f"🎉 生成成功: {ai_response}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 替代方法失败: {e}")
        print(f"🐛 详细错误: {traceback.format_exc()}")
        return None, None

class ChatGLMWrapper:
    """ChatGLM包装器 - 避免pipeline兼容性问题"""
    
    def __init__(self, model_name: str = "THUDM/chatglm2-6b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.initialized = False
    
    def initialize(self):
        """初始化模型"""
        try:
            print(f"🔧 初始化ChatGLM包装器: {self.model_name}")
            
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            print("✅ Tokenizer加载成功")
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto"
            )
            print("✅ 模型加载成功")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 包装器初始化失败: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 100) -> Optional[str]:
        """生成回应"""
        if not self.initialized:
            print("❌ 模型未初始化")
            return None
        
        try:
            print(f"🧠 生成回应: {prompt[:50]}...")
            
            # 使用ChatGLM的原生chat方法
            response, _ = self.model.chat(
                self.tokenizer, 
                prompt, 
                history=[],
                max_length=max_length,
                temperature=0.8
            )
            
            print(f"✅ 生成成功: {response}")
            return response
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return None

def test_chatglm_wrapper():
    """测试ChatGLM包装器"""
    print("🧪 测试ChatGLM包装器...")
    
    wrapper = ChatGLMWrapper()
    
    if wrapper.initialize():
        print("✅ 包装器初始化成功")
        
        # 测试对话
        test_cases = [
            "你好",
            "你是谁？",
            "请介绍一下代码分析的重要性",
            "How are you?"
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n🧪 测试 {i}: {test_input}")
            response = wrapper.generate_response(test_input)
            if response:
                print(f"✅ 回应: {response}")
            else:
                print("❌ 生成失败")
        
        return wrapper
    else:
        print("❌ 包装器初始化失败")
        return None

async def run_all_tests():
    """运行所有测试"""
    print("🚀 ChatGLM2-6B 兼容性测试开始")
    print("=" * 60)
    
    # 1. 基础导入测试
    if not test_basic_imports():
        print("❌ 基础导入失败，终止测试")
        return
    
    print("\n" + "=" * 60)
    
    # 2. Tokenizer测试
    tokenizer = test_chatglm_tokenizer_direct()
    if not tokenizer:
        print("❌ Tokenizer测试失败")
    
    print("\n" + "=" * 60)
    
    # 3. 直接模型测试
    model, tokenizer = test_chatglm_model_direct()
    if model and tokenizer:
        print("✅ 直接模型调用成功")
    
    print("\n" + "=" * 60)
    
    # 4. 替代方法测试
    alt_model, alt_tokenizer = test_alternative_pipeline()
    if alt_model and alt_tokenizer:
        print("✅ 替代方法成功")
    
    print("\n" + "=" * 60)
    
    # 5. 包装器测试
    wrapper = test_chatglm_wrapper()
    if wrapper:
        print("✅ 包装器测试成功")
        return wrapper
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
    return None

if __name__ == "__main__":
    import torch
    
    print(f"🖥️ 运行环境:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   工作目录: {os.getcwd()}")
    
    # 运行测试
    result = asyncio.run(run_all_tests())
    
    if result:
        print("🎉 找到可用的ChatGLM实现方案！")
    else:
        print("❌ 所有测试方案都失败了")
