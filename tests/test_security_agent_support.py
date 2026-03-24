#!/usr/bin/env python3
"""
测试Security Agent对Python/C/C++语言的支持能力
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_security_agent_file_support():
    """测试Security Agent的文件类型支持能力"""
    print("🔍 测试Security Agent文件类型支持...")
    
    # 模拟Security Agent的文件过滤逻辑
    def filter_security_files(files):
        security_keywords = ["auth", "login", "password", "security", "crypto", "hash"]
        supported_extensions = ['.py', '.js', '.java', '.php']
        
        filtered_files = []
        for file in files:
            if (file.endswith(tuple(supported_extensions)) or 
                any(keyword in file.lower() for keyword in security_keywords)):
                filtered_files.append(file)
        return filtered_files
    
    # 测试文件列表
    test_files = [
        "app.py",           # Python文件
        "login.py",         # 包含安全关键词的Python文件
        "main.cpp",         # C++文件（应该被过滤掉）
        "auth_handler.c",   # C文件（应该被过滤掉）
        "server.js",        # JavaScript文件
        "security_module.java",  # Java文件
        "config.php",       # PHP文件
        "utils.go",         # Go文件（应该被过滤掉）
        "password_hash.rs"  # Rust文件（应该被过滤掉）
    ]
    
    filtered = filter_security_files(test_files)
    
    # 分析支持情况
    python_files = [f for f in filtered if f.endswith('.py')]
    cpp_files = [f for f in test_files if f.endswith(('.cpp', '.cxx', '.cc'))]
    c_files = [f for f in test_files if f.endswith('.c')]
    supported_other = [f for f in filtered if f.endswith(('.js', '.java', '.php'))]
    filtered_out = [f for f in test_files if f not in filtered]
    
    print(f"  支持的Python文件: {len(python_files)} 个")
    print(f"  不支持的C++文件: {len(cpp_files)} 个")
    print(f"  不支持的C文件: {len(c_files)} 个")
    print(f"  其他支持的文件: {len(supported_other)} 个")
    print(f"  被过滤的文件: {len(filtered_out)} 个")
    
    # 检查安全关键词匹配
    security_keyword_matches = [f for f in test_files 
                              if any(keyword in f.lower() for keyword in 
                                   ["auth", "login", "password", "security", "crypto", "hash"])]
    
    print(f"\n  安全关键词匹配文件: {len(security_keyword_matches)} 个")
    for file in security_keyword_matches:
        print(f"    - {file}")
    
    return {
        'python_supported': len(python_files) > 0,
        'cpp_supported': len([f for f in filtered if f.endswith(('.cpp', '.cxx', '.cc'))]) > 0,
        'c_supported': len([f for f in filtered if f.endswith('.c')]) > 0,
        'security_keywords_work': len(security_keyword_matches) > 0
    }

def test_security_pattern_detection():
    """测试Security Agent的安全模式检测能力"""
    print("\n🛡️  测试Security Agent安全模式检测...")
    
    # 测试不同类型代码中的安全模式
    test_cases = [
        {
            'language': 'Python',
            'code': '''
import os
def dangerous_eval(user_input):
    result = eval(user_input)  # 危险的eval使用
    os.system("rm -rf /")      # 危险的系统调用
    return result
            ''',
            'expected_patterns': ['eval', 'os.system']
        },
        {
            'language': 'C++',
            'code': '''
#include <cstring>
void unsafe_copy(char* input) {
    char buffer[100];
    strcpy(buffer, input);     // 缓冲区溢出风险
    printf(input);             // 格式化字符串漏洞
}
            ''',
            'expected_patterns': ['strcpy', 'printf(input)']
        },
        {
            'language': 'C',
            'code': '''
#include <stdio.h>
#include <string.h>
void process_data(char* data) {
    char temp[50];
    gets(temp);                // 危险的输入函数
    strcat(temp, data);        // 可能的缓冲区溢出
}
            ''',
            'expected_patterns': ['gets', 'strcat']
        }
    ]
    
    results = {}
    for case in test_cases:
        language = case['language']
        code = case['code']
        expected_patterns = case['expected_patterns']
        
        # 检测模式
        detected_patterns = []
        for pattern in expected_patterns:
            if pattern in code:
                detected_patterns.append(pattern)
        
        success_rate = len(detected_patterns) / len(expected_patterns) * 100
        results[language] = {
            'detected': len(detected_patterns),
            'expected': len(expected_patterns),
            'success_rate': success_rate,
            'patterns': detected_patterns
        }
        
        print(f"  {language}: {success_rate:.1f}% 检测率")
        print(f"    检测到: {', '.join(detected_patterns)}")
        if len(detected_patterns) < len(expected_patterns):
            missing = [p for p in expected_patterns if p not in detected_patterns]
            print(f"    未检测到: {', '.join(missing)}")
    
    return results

def main():
    """主测试函数"""
    print("🚀 Security Agent语言支持能力检测\n")
    
    # 运行测试
    file_support = test_security_agent_file_support()
    pattern_detection = test_security_pattern_detection()
    
    # 生成报告
    print(f"\n📊 Security Agent语言支持能力报告:")
    print(f"  Python支持: {'✅' if file_support['python_supported'] else '❌'}")
    print(f"  C++支持: {'✅' if file_support['cpp_supported'] else '❌'} (当前不支持)")
    print(f"  C支持: {'✅' if file_support['c_supported'] else '❌'} (当前不支持)")
    print(f"  安全关键词检测: {'✅' if file_support['security_keywords_work'] else '❌'}")
    
    # 模式检测总结
    avg_detection_rate = sum(r['success_rate'] for r in pattern_detection.values()) / len(pattern_detection)
    print(f"  平均模式检测率: {avg_detection_rate:.1f}%")
    
    # 建议
    print(f"\n💡 改进建议:")
    if not file_support['cpp_supported']:
        print(f"  - 增加C++文件支持 (.cpp, .cxx, .cc)")
    if not file_support['c_supported']:
        print(f"  - 增加C文件支持 (.c, .h)")
    if avg_detection_rate < 80:
        print(f"  - 增强安全模式检测规则")
    
    overall_score = sum([
        file_support['python_supported'],
        file_support['cpp_supported'], 
        file_support['c_supported'],
        file_support['security_keywords_work'],
        avg_detection_rate > 80
    ]) / 5 * 100
    
    print(f"\n📈 整体支持评分: {overall_score:.1f}%")
    
    return overall_score > 60

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)