#!/usr/bin/env python3
"""
Alternative Models for Qwen2-7B
Test available chat models that can replace Qwen2-7B
"""

import sys
import os
from transformers import AutoConfig, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_model_availability(model_id, model_name):
    """Test if a model is available and accessible"""
    print(f"\n🧪 Testing {model_name} ({model_id})")
    
    try:
        # Test config loading
        print("  📋 Config...", end=" ")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("✅")
        
        # Test tokenizer loading  
        print("  🔤 Tokenizer...", end=" ")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("✅")
        
        # Basic info
        model_type = getattr(config, 'model_type', 'unknown')
        vocab_size = getattr(config, 'vocab_size', 'unknown')
        
        print(f"  📊 Type: {model_type}, Vocab: {vocab_size}")
        
        return {
            "model_id": model_id,
            "model_name": model_name,
            "status": "available",
            "model_type": model_type,
            "vocab_size": vocab_size
        }
        
    except Exception as e:
        print(f"❌ {str(e)[:80]}...")
        return {
            "model_id": model_id,
            "model_name": model_name,
            "status": "failed",
            "error": str(e)
        }

def main():
    """Test alternative chat models"""
    print("🔍 Testing Alternative Chat Models for Qwen2-7B")
    print("=" * 60)
    
    # Alternative Chinese/English chat models
    alternative_models = [
        # ChatGLM series
        ("THUDM/chatglm3-6b", "ChatGLM3-6B"),
        ("THUDM/chatglm2-6b", "ChatGLM2-6B"),
        
        # Baichuan series
        ("baichuan-inc/Baichuan2-7B-Chat", "Baichuan2-7B-Chat"),
        ("baichuan-inc/Baichuan2-13B-Chat", "Baichuan2-13B-Chat"),
        
        # Yi series
        ("01-ai/Yi-6B-Chat", "Yi-6B-Chat"),
        ("01-ai/Yi-9B-Chat", "Yi-9B-Chat"),
        
        # InternLM series
        ("internlm/internlm2-chat-7b", "InternLM2-Chat-7B"),
        
        # Alternative Qwen models
        ("Qwen/Qwen1.5-7B-Chat", "Qwen1.5-7B-Chat"),
        ("Qwen/Qwen-7B-Chat", "Qwen-7B-Chat"),
        
        # Other options
        ("microsoft/DialoGPT-large", "DialoGPT-Large"),
        ("facebook/blenderbot-3B", "BlenderBot-3B"),
    ]
    
    available_models = []
    failed_models = []
    
    for model_id, model_name in alternative_models:
        result = test_model_availability(model_id, model_name)
        
        if result["status"] == "available":
            available_models.append(result)
        else:
            failed_models.append(result)
    
    # Results summary
    print(f"\n📊 Alternative Models Summary")
    print("=" * 40)
    print(f"✅ Available: {len(available_models)}")
    print(f"❌ Failed: {len(failed_models)}")
    
    if available_models:
        print(f"\n🎯 Recommended Alternatives:")
        for i, model in enumerate(available_models[:5], 1):  # Show top 5
            print(f"  {i}. {model['model_name']} ({model['model_id']})")
            print(f"     Type: {model['model_type']}, Vocab: {model['vocab_size']}")
    
    # Save results
    import json
    results = {
        "available_models": available_models,
        "failed_models": failed_models,
        "timestamp": "2025-09-15",
        "tested_for": "Qwen2-7B replacement"
    }
    
    os.makedirs("tests/model_compatibility/results", exist_ok=True)
    with open("tests/model_compatibility/results/alternative_models.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: tests/model_compatibility/results/alternative_models.json")
    
    return available_models

if __name__ == "__main__":
    main()
