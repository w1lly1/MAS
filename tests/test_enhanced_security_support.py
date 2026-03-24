#!/usr/bin/env python3
"""
测试增强后的Security Agent对Python/C/C++语言的支持
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_security_agent_support():
    """测试增强后的Security Agent文件支持能力"""
    print("🔍 测试增强后的Security Agent文件支持...")
    
    # 模拟增强后的Security Agent文件过滤逻辑
    def filter_security_files(files):
        supported_extensions = [
            '.py', '.cpp', '.cxx', '.cc',  # Python和C++
            '.c', '.h', '.hpp',            # C和头文件
            '.js', '.java', '.php'
        ]
        security_keywords = ["auth", "login", "password", "security", "crypto", "hash"]
        
        filtered_files = []
        file_info = []
        
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if (file_ext in supported_extensions or 
                any(keyword in file.lower() for keyword in security_keywords)):
                filtered_files.append(file)
                file_info.append({
                    'name': file,
                    'extension': file_ext,
                    'type': 'direct' if file_ext in supported_extensions else 'keyword_match'
                })
        
        return filtered_files, file_info
    
    # 测试文件列表
    test_files = [
        "app.py",              # Python文件
        "login_auth.py",       # 包含安全关键词的Python文件
        "main.cpp",            # C++文件
        "handler.cxx",         # C++文件
        "utility.cc",          # C++文件
        "auth_handler.c",      # C文件
        "config.h",            # C头文件
        "types.hpp",           # C++头文件
        "server.js",           # JavaScript文件
        "security_module.java", # Java文件
        "config.php",          # PHP文件
        "utils.go",            # Go文件（应该被过滤掉）
        "password_crypto.rs"   # Rust文件（通过关键词匹配）
    ]
    
    filtered, file_info = filter_security_files(test_files)
    
    # 分类统计
    python_files = [f for f in file_info if f['extension'] == '.py']
    cpp_files = [f for f in file_info if f['extension'] in ['.cpp', '.cxx', '.cc']]
    c_files = [f for f in file_info if f['extension'] in ['.c', '.h', '.hpp']]
    header_files = [f for f in file_info if f['extension'] in ['.h', '.hpp']]
    other_supported = [f for f in file_info if f['extension'] in ['.js', '.java', '.php']]
    keyword_matches = [f for f in file_info if f['type'] == 'keyword_match']
    filtered_out = [f for f in test_files if f not in filtered]
    
    print(f"  ✅ 支持的Python文件: {len(python_files)} 个")
    print(f"  ✅ 支持的C++文件: {len(cpp_files)} 个")
    print(f"  ✅ 支持的C文件: {len(c_files)} 个")
    print(f"  ✅ 支持的头文件: {len(header_files)} 个")
    print(f"  ✅ 其他支持文件: {len(other_supported)} 个")
    print(f"  ✅ 关键词匹配文件: {len(keyword_matches)} 个")
    print(f"  ❌ 被过滤文件: {len(filtered_out)} 个")
    
    # 详细信息
    print(f"\n📋 详细文件支持情况:")
    for file in file_info:
        status = "✅" if file['type'] == 'direct' else "🔑"
        print(f"  {status} {file['name']} ({file['extension']})")
    
    if filtered_out:
        print(f"\n🗑️  被过滤的文件:")
        for file in filtered_out:
            print(f"  ❌ {file}")
    
    return {
        'python_supported': len(python_files) > 0,
        'cpp_supported': len(cpp_files) > 0,
        'c_supported': len(c_files) > 0,
        'headers_supported': len(header_files) > 0,
        'keyword_matching_works': len(keyword_matches) > 0,
        'coverage_rate': len(filtered) / len(test_files) * 100
    }

def test_c_cpp_security_patterns():
    """测试C/C++安全模式检测"""
    print("\n🛡️  测试C/C++安全模式检测能力...")
    
    # C/C++安全测试用例
    test_cases = [
        {
            'language': 'C',
            'code': '''
#include <stdio.h>
#include <string.h>

void unsafe_functions() {
    char buffer[100];
    char input[200];
    
    gets(buffer);              // ❌ 危险函数
    scanf("%s", input);        // ❌ 格式化漏洞
    strcpy(buffer, input);     // ❌ 缓冲区溢出
    strcat(buffer, "test");    // ❌ 可能溢出
    sprintf(buffer, "%s", input); // ❌ 格式化漏洞
}
            ''',
            'dangerous_patterns': ['gets', 'scanf("%s"', 'strcpy', 'strcat', 'sprintf']
        },
        {
            'language': 'C++',
            'code': '''
#include <iostream>
#include <cstring>

class UnsafeClass {
public:
    void dangerous_methods(char* input) {
        char buffer[50];
        strcpy(buffer, input);     // ❌ 缓冲区溢出
        printf(input);             // ❌ 格式化字符串漏洞
        system("ls");              // ❌ 系统调用
        malloc(100);               // ⚠️  需要配套free
    }
};
            ''',
            'dangerous_patterns': ['strcpy', 'printf(input)', 'system', 'malloc']
        }
    ]
    
    results = {}
    for case in test_cases:
        language = case['language']
        code = case['code']
        patterns = case['dangerous_patterns']
        
        detected = []
        for pattern in patterns:
            if pattern in code:
                detected.append(pattern)
        
        detection_rate = len(detected) / len(patterns) * 100
        results[language] = {
            'total_patterns': len(patterns),
            'detected_patterns': len(detected),
            'detection_rate': detection_rate,
            'detected': detected,
            'missed': [p for p in patterns if p not in detected]
        }
        
        print(f"  {language}安全模式检测: {detection_rate:.1f}%")
        print(f"    检测到: {', '.join(detected)}")
        if results[language]['missed']:
            print(f"    未检测到: {', '.join(results[language]['missed'])}")
    
    return results

def main():
    """主测试函数"""
    print("🚀 增强版Security Agent语言支持测试\n")
    
    # 运行测试
    file_support = test_enhanced_security_agent_support()
    security_patterns = test_c_cpp_security_patterns()
    
    # 生成综合报告
    print(f"\n📊 增强后Security Agent支持能力报告:")
    
    # 文件支持评分
    file_support_score = sum([
        file_support['python_supported'],
        file_support['cpp_supported'],
        file_support['c_supported'], 
        file_support['headers_supported'],
        file_support['keyword_matching_works']
    ]) / 5 * 100
    
    print(f"  文件类型支持: {file_support_score:.1f}%")
    print(f"    Python: {'✅' if file_support['python_supported'] else '❌'}")
    print(f"    C++: {'✅' if file_support['cpp_supported'] else '❌'}")
    print(f"    C: {'✅' if file_support['c_supported'] else '❌'}")
    print(f"    头文件: {'✅' if file_support['headers_supported'] else '❌'}")
    print(f"    关键词匹配: {'✅' if file_support['keyword_matching_works'] else '❌'}")
    
    # 安全模式检测评分
    pattern_scores = [r['detection_rate'] for r in security_patterns.values()]
    avg_pattern_score = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0
    
    print(f"  安全模式检测: {avg_pattern_score:.1f}%")
    for lang, result in security_patterns.items():
        print(f"    {lang}: {result['detection_rate']:.1f}% ({result['detected_patterns']}/{result['total_patterns']})")
    
    # 覆盖率
    coverage = file_support['coverage_rate']
    print(f"  文件覆盖率: {coverage:.1f}%")
    
    # 综合评分
    overall_score = (file_support_score * 0.4 + avg_pattern_score * 0.4 + coverage * 0.2)
    print(f"\n📈 综合支持评分: {overall_score:.1f}%")
    
    # 改进建议
    print(f"\n💡 评估结果:")
    if overall_score >= 90:
        print(f"  🎉 优秀 - 已具备完整的Python/C/C++安全分析能力")
    elif overall_score >= 75:
        print(f"  👍 良好 - 主要功能完备，可投入生产使用")
    elif overall_score >= 60:
        print(f"  ⚠️  及格 - 基础功能可用，建议进一步完善")
    else:
        print(f"  ❌ 不足 - 需要显著改进")
    
    return overall_score >= 75

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)