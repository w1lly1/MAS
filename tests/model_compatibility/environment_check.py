#!/usr/bin/env python3
"""
Environment Compatibility Check
Checks current environment setup and transformers version compatibility
"""

import sys
import subprocess
import importlib.util
from typing import Dict, Optional, List, Tuple
import json
from datetime import datetime

def get_package_version(package_name: str) -> Optional[str]:
    """Get package version if installed"""
    try:
        module = importlib.util.find_spec(package_name)
        if module is None:
            return None
        package = importlib.import_module(package_name)
        return getattr(package, '__version__', 'unknown')
    except ImportError:
        return None

def check_gpu_availability() -> Dict[str, any]:
    """Check GPU availability and CUDA support"""
    gpu_info = {
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_names": []
    }
    
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if gpu_info["cuda_available"]:
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(gpu_info["gpu_count"])]
    except ImportError:
        pass
    
    return gpu_info

def check_critical_packages() -> Dict[str, Optional[str]]:
    """Check versions of critical packages"""
    critical_packages = [
        'transformers',
        'torch',
        'tokenizers',
        'sentencepiece',
        'accelerate',
        'safetensors',
        'datasets',
        'huggingface_hub',
        'numpy',
        'pandas'
    ]
    
    versions = {}
    for package in critical_packages:
        versions[package] = get_package_version(package)
    
    return versions

def analyze_transformers_compatibility() -> Dict[str, any]:
    """Analyze transformers version compatibility with different models"""
    transformers_version = get_package_version('transformers')
    
    if not transformers_version:
        return {"error": "transformers not installed"}
    
    try:
        # Parse version
        version_parts = transformers_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        version_number = major * 100 + minor
    except (ValueError, IndexError):
        return {"error": f"Could not parse transformers version: {transformers_version}"}
    
    compatibility = {
        "version": transformers_version,
        "version_number": version_number,
        "models": {}
    }
    
    # Model compatibility matrix
    model_compatibility = {
        "ChatGLM2-6B": {
            "compatible_versions": "<=4.30.0",
            "issues": ["KV cache incompatibility", "tokenizer padding_side"] if version_number >= 440 else [],
            "recommended_action": "Downgrade to 4.27.4" if version_number >= 440 else "Compatible"
        },
        "ChatGLM3-6B": {
            "compatible_versions": ">=4.36.0",
            "issues": [] if version_number >= 436 else ["Unsupported in older versions"],
            "recommended_action": "Compatible" if version_number >= 436 else "Upgrade transformers"
        },
        "Qwen2-7B": {
            "compatible_versions": ">=4.37.0",
            "issues": [] if version_number >= 437 else ["Unsupported in older versions"],
            "recommended_action": "Compatible" if version_number >= 437 else "Upgrade transformers"
        },
        "Baichuan2-7B": {
            "compatible_versions": ">=4.33.0",
            "issues": [] if version_number >= 433 else ["Unsupported in older versions"],
            "recommended_action": "Compatible" if version_number >= 433 else "Upgrade transformers"
        },
        "CodeBERT": {
            "compatible_versions": ">=4.0.0",
            "issues": [],
            "recommended_action": "Compatible"
        }
    }
    
    for model, info in model_compatibility.items():
        compatibility["models"][model] = {
            **info,
            "status": "compatible" if not info["issues"] else "incompatible"
        }
    
    return compatibility

def generate_environment_report() -> Dict[str, any]:
    """Generate comprehensive environment report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "packages": check_critical_packages(),
        "gpu": check_gpu_availability(),
        "transformers_compatibility": analyze_transformers_compatibility()
    }
    
    return report

def print_report(report: Dict[str, any]):
    """Print formatted environment report"""
    print("🔍 Environment Compatibility Report")
    print("=" * 60)
    
    print(f"\n📅 Generated: {report['timestamp']}")
    print(f"🐍 Python: {report['python_version'].split()[0]}")
    print(f"💻 Platform: {report['platform']}")
    
    # GPU Info
    gpu_info = report['gpu']
    print(f"\n🎮 GPU Information:")
    if gpu_info['cuda_available']:
        print(f"  ✅ CUDA Available: {gpu_info['cuda_version']}")
        print(f"  🔢 GPU Count: {gpu_info['gpu_count']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"  🎯 GPU {i}: {name}")
    else:
        print("  ❌ CUDA Not Available")
    
    # Package Versions
    print(f"\n📦 Package Versions:")
    packages = report['packages']
    for pkg, version in packages.items():
        status = "✅" if version else "❌"
        print(f"  {status} {pkg}: {version or 'Not installed'}")
    
    # Transformers Compatibility
    compat = report['transformers_compatibility']
    if 'error' not in compat:
        print(f"\n🔧 Transformers Compatibility Analysis:")
        print(f"  📊 Version: {compat['version']}")
        
        print(f"\n📋 Model Compatibility:")
        for model, info in compat['models'].items():
            status_icon = "✅" if info['status'] == 'compatible' else "❌"
            print(f"  {status_icon} {model}: {info['recommended_action']}")
            if info['issues']:
                for issue in info['issues']:
                    print(f"    ⚠️ Issue: {issue}")
    else:
        print(f"\n❌ Transformers Analysis Error: {compat['error']}")

def save_report(report: Dict[str, any], filename: str = "environment_report.json"):
    """Save report to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Report saved to: {filename}")

def main():
    """Main function"""
    print("🚀 Starting Environment Compatibility Check...")
    
    report = generate_environment_report()
    print_report(report)
    
    # Save report
    save_report(report, "/var/fpwork/tiyi/project/MAS/MAS/tests/model_compatibility/results/environment_report.json")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    compat = report['transformers_compatibility']
    if 'error' not in compat:
        incompatible_models = [model for model, info in compat['models'].items() 
                             if info['status'] == 'incompatible']
        
        if incompatible_models:
            print(f"  ⚠️ Incompatible models found: {', '.join(incompatible_models)}")
            print(f"  🔧 Consider version adjustments or model alternatives")
        else:
            print(f"  ✅ All models appear compatible with current setup")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Run individual model tests: python compatibility_test_suite.py")
    print(f"  2. Check specific model compatibility in individual_model_tests/")
    print(f"  3. Review recommendations in the compatibility report")

if __name__ == "__main__":
    main()
