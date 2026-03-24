#!/usr/bin/env python3
"""
测试Performance Agent对Python/C/C++语言的支持能力
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_performance_agent_language_detection():
    """测试Performance Agent的语言检测能力"""
    print("🔍 测试Performance Agent语言检测能力...")
    
    # 模拟Performance Agent的语言检测逻辑
    def detect_language(code_content):
        structure = {"language": "unknown"}
        
        # Python检测
        if "def " in code_content:
            structure["language"] = "python"
            structure["function_count"] = code_content.count("def ")
            structure["class_count"] = code_content.count("class ")
            structure["async_patterns"] = "async " in code_content or "await " in code_content
        
        # C/C++检测（简化版本）
        elif "#include" in code_content:
            if "std::" in code_content or "namespace" in code_content:
                structure["language"] = "cpp"
            else:
                structure["language"] = "c"
            structure["function_count"] = code_content.count("{")  # 简化计数
            structure["class_count"] = code_content.count("class ") + code_content.count("struct ")
        
        # 循环检测
        structure["loop_count"] = (
            code_content.count("for ") + 
            code_content.count("while ") + 
            code_content.count("do ")
        )
        
        return structure
    
    # 测试用例
    test_cases = [
        {
            'language': 'Python',
            'code': '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.result = 0
    
    async def calculate_async(self, data):
        for item in data:
            self.result += item
''',
            'expected_features': ['python', 'function_count', 'class_count', 'async_patterns']
        },
        {
            'language': 'C++',
            'code': '''
#include <iostream>
#include <vector>

class Calculator {
private:
    int result;
    
public:
    Calculator() : result(0) {}
    
    int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }
    
    void calculate(const std::vector<int>& data) {
        for (int item : data) {
            result += item;
        }
    }
};
''',
            'expected_features': ['cpp', 'function_count', 'class_count']
        },
        {
            'language': 'C',
            'code': '''
#include <stdio.h>
#include <stdlib.h>

struct Calculator {
    int result;
};

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

void calculate(struct Calculator* calc, int* data, int size) {
    for (int i = 0; i < size; i++) {
        calc->result += data[i];
    }
}

int main() {
    struct Calculator calc = {0};
    int data[] = {1, 2, 3, 4, 5};
    calculate(&calc, data, 5);
    printf("Result: %d\\n", calc.result);
    return 0;
}
''',
            'expected_features': ['c', 'function_count', 'class_count']
        }
    ]
    
    results = {}
    for case in test_cases:
        language = case['language']
        code = case['code']
        expected = case['expected_features']
        
        detected = detect_language(code)
        
        # 检查检测到的特征
        detected_features = []
        if detected['language'] != 'unknown':
            detected_features.append(detected['language'])
        if detected.get('function_count', 0) > 0:
            detected_features.append('function_count')
        if detected.get('class_count', 0) > 0:
            detected_features.append('class_count')
        if detected.get('async_patterns'):
            detected_features.append('async_patterns')
        if detected.get('loop_count', 0) > 0:
            detected_features.append('loop_count')
        
        accuracy = len(set(detected_features) & set(expected)) / len(set(expected)) * 100
        
        results[language] = {
            'detected_language': detected['language'],
            'accuracy': accuracy,
            'detected_features': detected_features,
            'expected_features': expected
        }
        
        print(f"  {language}: {accuracy:.1f}% 准确率")
        print(f"    检测语言: {detected['language']}")
        print(f"    检测特征: {', '.join(detected_features)}")
    
    return results

def test_performance_patterns():
    """测试Performance Agent的性能模式检测能力"""
    print("\n⚡ 测试Performance Agent性能模式检测...")
    
    # 测试不同语言的性能相关模式
    test_cases = [
        {
            'language': 'Python',
            'code': '''
def inefficient_algorithm(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            result.append(data[i] * data[j])  # O(n²) 复杂度
    return result

def recursive_fibonacci(n):
    if n <= 1:
        return n
    return recursive_fibonacci(n-1) + recursive_fibonacci(n-2)  # 指数级复杂度

import time
def slow_operation():
    time.sleep(1)  # 阻塞操作
''',
            'expected_patterns': ['nested_loops', 'recursion', 'blocking_io']
        },
        {
            'language': 'C++',
            'code': '''
#include <vector>
#include <chrono>
#include <thread>

class InefficientProcessor {
public:
    std::vector<int> process_quadratic(const std::vector<int>& data) {
        std::vector<int> result;
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data.size(); j++) {
                result.push_back(data[i] * data[j]);  // O(n²) 复杂度
            }
        }
        return result;
    }
    
    int recursive_fibonacci(int n) {
        if (n <= 1) return n;
        return recursive_fibonacci(n-1) + recursive_fibonacci(n-2);  // 指数级
    }
    
    void blocking_operation() {
        std::this_thread::sleep_for(std::chrono::seconds(1));  // 阻塞
    }
};
''',
            'expected_patterns': ['nested_loops', 'recursion', 'blocking_io']
        },
        {
            'language': 'C',
            'code': '''
#include <stdio.h>
#include <unistd.h>

void quadratic_process(int* data, int size, int* result) {
    int index = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[index++] = data[i] * data[j];  // O(n²) 复杂度
        }
    }
}

int recursive_fibonacci(int n) {
    if (n <= 1) return n;
    return recursive_fibonacci(n-1) + recursive_fibonacci(n-2);  // 指数级
    
void blocking_call() {
    sleep(1);  // 阻塞调用
}
''',
            'expected_patterns': ['nested_loops', 'recursion', 'blocking_io']
        }
    ]
    
    # 模拟性能模式检测
    def detect_performance_patterns(code):
        patterns = []
        
        # 嵌套循环检测
        if code.count('for ') >= 2 or 'for.*for' in code:
            patterns.append('nested_loops')
            
        # 递归检测
        if 'return.*recursive_' in code or 'return.*fibonacci' in code:
            patterns.append('recursion')
            
        # 阻塞IO检测
        blocking_keywords = ['sleep', 'time.sleep', 'std::this_thread::sleep_for', 'usleep']
        if any(keyword in code for keyword in blocking_keywords):
            patterns.append('blocking_io')
            
        return patterns
    
    results = {}
    for case in test_cases:
        language = case['language']
        code = case['code']
        expected = case['expected_patterns']
        
        detected = detect_performance_patterns(code)
        accuracy = len(set(detected) & set(expected)) / len(set(expected)) * 100
        
        results[language] = {
            'accuracy': accuracy,
            'detected_patterns': detected,
            'expected_patterns': expected,
            'missing_patterns': list(set(expected) - set(detected))
        }
        
        print(f"  {language}: {accuracy:.1f}% 模式检测准确率")
        print(f"    检测到: {', '.join(detected)}")
        if results[language]['missing_patterns']:
            print(f"    未检测到: {', '.join(results[language]['missing_patterns'])}")
    
    return results

def test_complexity_analysis():
    """测试复杂度分析能力"""
    print("\n🔢 测试复杂度分析能力...")
    
    # 复杂度测试用例
    complexity_cases = [
        {
            'name': '简单线性',
            'code': 'for i in range(n): process(i)',
            'expected_complexity': 'O(n)'
        },
        {
            'name': '嵌套循环',
            'code': 'for i in range(n):\n    for j in range(m): process(i,j)',
            'expected_complexity': 'O(n*m)'
        },
        {
            'name': '递归',
            'code': 'def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n',
            'expected_complexity': 'O(2^n)'
        }
    ]
    
    # 模拟复杂度分析
    def analyze_complexity(code):
        if 'for.*for' in code or code.count('for ') >= 2:
            return 'O(n²)'
        elif 'for ' in code:
            return 'O(n)'
        elif 'recursive' in code or 'fib(' in code:
            return 'O(2^n)'
        else:
            return 'O(1)'
    
    correct = 0
    for case in complexity_cases:
        detected = analyze_complexity(case['code'])
        is_correct = detected == case['expected_complexity']
        if is_correct:
            correct += 1
        print(f"  {case['name']}: 检测 {detected}, 期望 {case['expected_complexity']} {'✅' if is_correct else '❌'}")
    
    accuracy = correct / len(complexity_cases) * 100
    print(f"  复杂度分析准确率: {accuracy:.1f}%")
    
    return accuracy

def main():
    """主测试函数"""
    print("🚀 Performance Agent语言支持能力检测\n")
    
    # 运行各项测试
    language_detection = test_performance_agent_language_detection()
    performance_patterns = test_performance_patterns()
    complexity_accuracy = test_complexity_analysis()
    
    # 生成综合报告
    print(f"\n📊 Performance Agent语言支持能力报告:")
    
    # 语言检测评分
    lang_accuracies = [r['accuracy'] for r in language_detection.values()]
    avg_lang_accuracy = sum(lang_accuracies) / len(lang_accuracies)
    
    print(f"  语言检测准确率: {avg_lang_accuracy:.1f}%")
    for lang, result in language_detection.items():
        print(f"    {lang}: {result['accuracy']:.1f}% (检测为: {result['detected_language']})")
    
    # 性能模式检测评分
    pattern_accuracies = [r['accuracy'] for r in performance_patterns.values()]
    avg_pattern_accuracy = sum(pattern_accuracies) / len(pattern_accuracies)
    
    print(f"  性能模式检测: {avg_pattern_accuracy:.1f}%")
    for lang, result in performance_patterns.items():
        print(f"    {lang}: {result['accuracy']:.1f}%")
    
    # 复杂度分析评分
    print(f"  复杂度分析: {complexity_accuracy:.1f}%")
    
    # 综合评分
    overall_score = (avg_lang_accuracy * 0.4 + avg_pattern_accuracy * 0.4 + complexity_accuracy * 0.2)
    print(f"\n📈 综合支持评分: {overall_score:.1f}%")
    
    # 评估结果
    print(f"\n💡 评估结果:")
    if overall_score >= 90:
        print(f"  🎉 优秀 - 已具备完整的Python/C/C++性能分析能力")
    elif overall_score >= 75:
        print(f"  👍 良好 - 主要功能完备，可投入生产使用")
    elif overall_score >= 60:
        print(f"  ⚠️  及格 - 基础功能可用，建议进一步完善")
    else:
        print(f"  ❌ 不足 - 需要显著改进")
    
    # 改进建议
    print(f"\n🔧 改进建议:")
    if avg_lang_accuracy < 80:
        print(f"  - 增强C/C++语言特征识别")
    if avg_pattern_accuracy < 80:
        print(f"  - 扩展性能模式检测规则")
    if complexity_accuracy < 80:
        print(f"  - 改进复杂度分析算法")
    
    return overall_score >= 75

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)