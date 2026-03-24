#!/usr/bin/env python3
"""
测试Code Quality Agent对Python和C/C++的支持
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.language_detector import LanguageDetector, ProgrammingLanguage

def test_language_detection():
    """测试语言检测功能"""
    print("🔍 测试语言检测功能...")
    
    detector = LanguageDetector()
    
    # 测试用例
    test_cases = [
        ("test.py", "def hello():\n    print('Hello World')\n    return True", ProgrammingLanguage.PYTHON),
        ("test.cpp", "#include <iostream>\nusing namespace std;\nint main() {\n    cout << \"Hello World\";\n    return 0;\n}", ProgrammingLanguage.CPP),
        ("test.c", "#include <stdio.h>\nint main() {\n    printf(\"Hello World\\n\");\n    return 0;\n}", ProgrammingLanguage.C),
    ]
    
    all_passed = True
    for filename, content, expected_lang in test_cases:
        detected_lang = detector.detect_language(content, os.path.splitext(filename)[1])
        success = detected_lang == expected_lang
        all_passed = all_passed and success
        print(f"  {filename}: {'✅' if success else '❌'} 期望: {expected_lang.value}, 检测: {detected_lang.value}")
    
    return all_passed

def test_file_extensions():
    """测试文件扩展名支持"""
    print("\n📂 测试文件扩展名支持...")
    
    detector = LanguageDetector()
    extensions = detector.get_supported_extensions()
    
    required_extensions = ['.py', '.cpp', '.cxx', '.cc', '.c', '.h', '.hpp', '.hxx']
    supported = all(ext in extensions for ext in required_extensions)
    
    print(f"  支持的扩展名: {len(extensions)} 个")
    print(f"  关键扩展名支持: {'✅' if supported else '❌'}")
    
    return supported

def main():
    """主测试函数"""
    print("🚀 测试Code Quality Agent语言支持功能\n")
    
    # 运行测试
    detection_test = test_language_detection()
    extension_test = test_file_extensions()
    
    # 总结
    all_tests_passed = detection_test and extension_test
    
    print(f"\n📊 测试总结:")
    print(f"  语言检测: {'✅ 通过' if detection_test else '❌ 失败'}")
    print(f"  扩展名支持: {'✅ 通过' if extension_test else '❌ 失败'}")
    print(f"  整体结果: {'🎉 所有测试通过!' if all_tests_passed else '💥 存在失败测试'}")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)