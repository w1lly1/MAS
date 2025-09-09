#!/usr/bin/env python3
"""
ChatGLM 兼容性检查和解决方案建议脚本
"""

import sys
import subprocess
import importlib.util

def check_package_version(package_name):
    """检查包版本"""
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'unknown')
    except ImportError:
        return None

def check_environment():
    """检查当前环境"""
    print("🔍 环境检查")
    print("=" * 50)
    
    # Python 版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python 版本: {python_version}")
    
    # 关键包版本
    packages = {
        'transformers': check_package_version('transformers'),
        'torch': check_package_version('torch'),
        'sentencepiece': check_package_version('sentencepiece')
    }
    
    for pkg, version in packages.items():
        status = "✅" if version else "❌"
        print(f"{status} {pkg}: {version or '未安装'}")
    
    return packages

def suggest_solutions(packages):
    """基于环境提供解决方案建议"""
    print("\n💡 解决方案建议")
    print("=" * 50)
    
    transformers_version = packages.get('transformers')
    
    if transformers_version is None:
        print("❌ transformers 未安装")
        print("📋 建议: pip install transformers==4.27.4")
        return
    
    # 解析版本号
    try:
        major, minor = map(int, transformers_version.split('.')[:2])
        version_number = major * 100 + minor
    except:
        print(f"⚠️ 无法解析 transformers 版本: {transformers_version}")
        return
    
    if version_number >= 440:  # 4.40+
        print("⚠️ transformers 版本过高 (>= 4.40)")
        print("📋 推荐解决方案:")
        print("\n🎯 方案一: 降级 transformers (推荐)")
        print("   pip install transformers==4.27.4")
        print("   - 优点: 最简单直接，兼容性最好")
        print("   - 缺点: 可能与其他项目冲突")
        
        print("\n🎯 方案二: 切换到 ChatGLM3-6B")
        print("   model_name = 'THUDM/chatglm3-6b'")
        print("   - 优点: 官方新版本，兼容性更好")
        print("   - 缺点: 需要重新下载模型")
        
        print("\n🎯 方案三: 使用独立环境")
        print("   conda create -n chatglm python=3.10")
        print("   conda activate chatglm")
        print("   pip install transformers==4.27.4 torch sentencepiece")
        
    elif version_number >= 430:  # 4.30-4.39
        print("⚠️ transformers 版本中等偏高 (4.30-4.39)")
        print("📋 建议: 尝试降级到 4.27.4 或使用 ChatGLM3-6B")
        
    else:  # < 4.30
        print("✅ transformers 版本应该兼容")
        print("📋 如果仍有问题，检查其他依赖版本")

def create_quick_fix_script():
    """创建快速修复脚本"""
    script_content = '''#!/bin/bash
# ChatGLM2-6B 快速修复脚本

echo "🚀 ChatGLM2-6B 兼容性修复"
echo "========================="

echo "📦 降级 transformers..."
pip install transformers==4.27.4

echo "🔧 安装依赖..."
pip install sentencepiece>=0.1.99

echo "✅ 修复完成！"
echo "💡 建议重启 Python 环境以确保修复生效"
'''
    
    with open('/var/fpwork/tiyi/project/MAS/MAS/tests/chatglm_tests/quick_fix.sh', 'w') as f:
        f.write(script_content)
    
    print("\n📝 已创建快速修复脚本: tests/chatglm_tests/quick_fix.sh")
    print("运行方法: bash tests/chatglm_tests/quick_fix.sh")

def main():
    """主函数"""
    print("🚀 ChatGLM2-6B 兼容性诊断工具")
    print("=" * 60)
    
    # 检查环境
    packages = check_environment()
    
    # 提供建议
    suggest_solutions(packages)
    
    # 创建修复脚本
    create_quick_fix_script()
    
    print("\n📖 详细报告: tests/chatglm_tests/COMPATIBILITY_REPORT.md")
    print("🔧 测试文件: tests/chatglm_tests/chatglm_final_solution.py")

if __name__ == "__main__":
    main()
