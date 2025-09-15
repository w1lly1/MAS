#!/usr/bin/env python3
"""
Test Qwen1.5-7B Replacement for Qwen2-7B
验证Qwen1.5-7B作为Qwen2-7B替代方案的兼容性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.model_compatibility.individual_model_tests.test_qwen15 import run_qwen15_tests
from tests.model_compatibility.model_registry import model_registry

def main():
    print("🔄 Testing Qwen1.5-7B as Qwen2-7B Replacement")
    print("=" * 60)
    
    # Check model registration
    try:
        qwen15_model = model_registry.get_model("qwen1.5-7b")
        print(f"✅ Model found in registry: {qwen15_model.name}")
        print(f"   Model ID: {qwen15_model.model_id}")
        print(f"   Min transformers: {qwen15_model.min_transformers_version}")
        print(f"   GPU Memory: {qwen15_model.gpu_memory_gb}GB")
    except Exception as e:
        print(f"❌ Model not found in registry: {e}")
        return False
    
    print("\n" + "=" * 60)
    
    # Run compatibility tests
    results = run_qwen15_tests()
    
    # Analyze results
    passed = len([r for r in results if r.status == "pass"])
    warnings = len([r for r in results if r.status == "warning"])
    failed = len([r for r in results if r.status == "fail"])
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"📈 Final Analysis - Qwen1.5-7B Replacement Test")
    print("=" * 60)
    print(f"Passed:   {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Warnings: {warnings}/{total} ({warnings/total*100:.1f}%)")
    print(f"Failed:   {failed}/{total} ({failed/total*100:.1f}%)")
    
    # Replacement recommendation
    if passed >= 4:  # At least 4 out of 5 tests pass
        print("\n🎉 RECOMMENDATION: Qwen1.5-7B is SUITABLE as Qwen2-7B replacement")
        print("   - Good compatibility with transformers 4.56.0")
        print("   - Core functionality working properly")
        if warnings > 0:
            print(f"   - {warnings} warning(s) noted, but not blocking")
    elif passed >= 2:
        print("\n⚠️  RECOMMENDATION: Qwen1.5-7B has PARTIAL compatibility")
        print("   - Some functionality working")
        print("   - May require additional configuration")
        print("   - Consider as backup option")
    else:
        print("\n❌ RECOMMENDATION: Qwen1.5-7B NOT suitable as replacement")
        print("   - Too many compatibility issues")
        print("   - Look for alternative models")
    
    return passed >= 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
