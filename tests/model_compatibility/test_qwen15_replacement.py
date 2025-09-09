#!/usr/bin/env python3
"""
Test Qwen1.5-7B-Chat as Qwen2-7B replacement
"""

import sys
import os
from transformers import AutoModel, AutoTokenizer
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_qwen15_replacement():
    """Test Qwen1.5-7B-Chat functionality"""
    
    model_id = "Qwen/Qwen1.5-7B-Chat"
    print(f"🧪 Testing {model_id} as Qwen2 replacement")
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Test basic tokenization
        test_text = "你好，请帮我分析代码质量"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization successful: {len(tokens['input_ids'][0])} tokens")
        
        # Load model (if GPU available)
        if torch.cuda.is_available():
            print("🚀 Loading model on GPU...")
            model = AutoModel.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            print("💻 Loading model on CPU...")
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        
        print("✅ Model loaded successfully")
        
        # Test basic inference
        print("🔬 Testing basic inference...")
        with torch.no_grad():
            outputs = model(**tokens)
            print(f"✅ Inference successful: {outputs.last_hidden_state.shape}")
        
        # Test chat functionality
        print("💬 Testing chat functionality...")
        messages = [
            {"role": "system", "content": "你是一个代码分析助手"},
            {"role": "user", "content": "请分析这段代码的质量"}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            chat_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print("✅ Chat template applied successfully")
        else:
            print("⚠️  Chat template not available, using direct text")
        
        return {
            "model_id": model_id,
            "status": "success",
            "features": {
                "tokenization": True,
                "model_loading": True,
                "inference": True,
                "chat_template": hasattr(tokenizer, 'apply_chat_template')
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": str(model.dtype) if 'model' in locals() else "unknown"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "model_id": model_id,
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    result = test_qwen15_replacement()
    
    # Save results
    import json
    os.makedirs("tests/model_compatibility/results", exist_ok=True)
    with open("tests/model_compatibility/results/qwen15_test.json", 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    if result["status"] == "success":
        print("\n🎉 Qwen1.5-7B-Chat ready as Qwen2 replacement!")
    else:
        print("\n💥 Qwen1.5-7B-Chat test failed")
