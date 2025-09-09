#!/usr/bin/env python3
"""
Quick Compatibility Check Script
Fast compatibility check for critical models
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def quick_environment_check():
    """Quick environment compatibility check"""
    print("🔍 Quick Environment Check")
    print("=" * 40)
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
        
        # Quick version analysis
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        version_number = major * 100 + minor
        
        if version_number >= 456:
            print("⚠️ Very recent transformers - potential compatibility issues")
        elif version_number >= 440:
            print("⚠️ Post-4.40 transformers - ChatGLM2 likely incompatible")
        elif version_number >= 436:
            print("✅ Good compatibility expected")
        else:
            print("⚠️ Older transformers - some newer models may not work")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def quick_model_test(model_id: str, model_name: str):
    """Quick test for a single model"""
    print(f"\n🧪 Quick test: {model_name}")
    
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Test config loading
        print("  📋 Loading config...", end=" ")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("✅")
        
        # Test tokenizer loading
        print("  🔤 Loading tokenizer...", end=" ")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("✅")
        
        # Quick tokenization test
        print("  🧪 Testing tokenization...", end=" ")
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print("✅")
        
        return {
            "model": model_name,
            "status": "pass",
            "message": "Quick test passed"
        }
        
    except Exception as e:
        print(f"❌ {str(e)[:50]}...")
        return {
            "model": model_name,
            "status": "fail",
            "message": str(e)
        }

def main():
    """Main quick check"""
    print("🚀 Quick Model Compatibility Check")
    print("=" * 50)
    
    # Environment check
    if not quick_environment_check():
        return
    
    # Quick model tests
    models_to_test = [
        ("THUDM/chatglm3-6b", "ChatGLM3-6B"),
        ("Qwen/Qwen2-7B-Chat", "Qwen2-7B"),
        ("microsoft/codebert-base", "CodeBERT"),
    ]
    
    results = []
    
    for model_id, model_name in models_to_test:
        result = quick_model_test(model_id, model_name)
        results.append(result)
    
    # Summary
    print(f"\n📊 Quick Check Summary")
    print("=" * 30)
    
    passed = len([r for r in results if r["status"] == "pass"])
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    for result in results:
        icon = "✅" if result["status"] == "pass" else "❌"
        print(f"  {icon} {result['model']}")
    
    if passed == total:
        print("\n🎉 All quick tests passed!")
        print("💡 Run full test suite: python compatibility_test_suite.py")
    else:
        print(f"\n⚠️ Some models failed quick tests")
        print("🔧 Check detailed compatibility with full test suite")
    
    # Save quick results
    os.makedirs("tests/model_compatibility/results", exist_ok=True)
    with open("tests/model_compatibility/results/quick_check.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {"passed": passed, "total": total}
        }, f, indent=2)
    
    print(f"💾 Results saved to: tests/model_compatibility/results/quick_check.json")

if __name__ == "__main__":
    main()
