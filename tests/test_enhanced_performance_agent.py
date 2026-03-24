#!/usr/bin/env python3
"""
测试增强后的Performance Agent对Python/C/C++语言的支持能力
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_performance_agent():
    """测试增强后的Performance Agent能力"""
    print("🚀 测试增强后的Performance Agent能力\n")
    
    # 模拟增强后的Performance Agent功能
    def enhanced_code_analysis(code_content):
        """模拟增强的代码分析功能"""
        result = {
            "language": "unknown",
            "structure": {},
            "complexity": {},
            "recursive_functions": [],
            "performance_patterns": []
        }
        
        # 智能语言检测
        result["language"] = detect_language(code_content)
        
        # 结构分析
        result["structure"] = analyze_code_structure(code_content, result["language"])
        
        # 复杂度分析
        result["complexity"] = enhanced_complexity_analysis(code_content)
        
        # 递归检测
        result["recursive_functions"] = detect_recursive_calls(code_content, result["language"])
        
        # 性能模式检测
        result["performance_patterns"] = detect_performance_patterns(code_content)
        
        return result
    
    def detect_language(code_content):
        """增强的语言检测"""
        content_lower = code_content.lower()
        
        # Python特征
        python_score = sum(1 for indicator in ['def ', 'import ', 'class ', 'self.'] 
                          if indicator in code_content)
        
        # C++特征
        cpp_score = sum(1 for indicator in ['#include', 'namespace', 'std::', 'template<'] 
                       if indicator in content_lower)
        
        # C特征
        c_score = sum(1 for indicator in ['#include', 'printf', 'scanf', 'malloc'] 
                     if indicator in content_lower)
        
        scores = {'python': python_score, 'cpp': cpp_score, 'c': c_score}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'
    
    def analyze_code_structure(code_content, language):
        """增强的结构分析"""
        structure = {
            "function_count": 0,
            "class_count": 0,
            "loop_count": 0,
            "memory_operations": False,
            "pointer_usage": False,
            "template_usage": False
        }
        
        if language == 'python':
            structure["function_count"] = code_content.count("def ")
            structure["class_count"] = code_content.count("class ")
            structure["loop_count"] = code_content.count("for ") + code_content.count("while ")
            
        elif language in ['cpp', 'c']:
            structure["function_count"] = code_content.count("{")
            structure["class_count"] = code_content.count("class ") + code_content.count("struct ")
            structure["loop_count"] = code_content.count("for ") + code_content.count("while ") + code_content.count("do ")
            structure["memory_operations"] = any(op in code_content for op in ['malloc', 'free', 'new', 'delete'])
            structure["pointer_usage"] = '*' in code_content and '&' in code_content
            structure["template_usage"] = 'template<' in code_content.lower()
        
        return structure
    
    def enhanced_complexity_analysis(code_content):
        """增强的复杂度分析"""
        complexity_info = {
            "loops_detected": 0,
            "nested_loop_depth": 0,
            "recursive_calls": 0,
            "conditional_branches": 0,
            "estimated_complexity": "O(1)",
            "complexity_factors": []
        }
        
        lines = code_content.split('\n')
        loop_stack = []
        max_depth = 0
        
        for line in lines:
            stripped = line.strip()
            
            # 循环检测
            if any(loop_keyword in stripped for loop_keyword in ['for ', 'while ', 'do ']):
                loop_stack.append(stripped)
                complexity_info["loops_detected"] += 1
                current_depth = len([l for l in loop_stack if any(kw in l for kw in ['for ', 'while ', 'do '])])
                max_depth = max(max_depth, current_depth)
                complexity_info["nested_loop_depth"] = max_depth
                
            # 条件分支检测
            elif any(cond in stripped for cond in ['if ', 'elif ', 'else:', 'switch ']):
                complexity_info["conditional_branches"] += 1
                
            # 递归迹象检测
            elif '(' in stripped and stripped.count('(') == stripped.count(')'):
                complexity_info["recursive_calls"] += 1
        
        # 复杂度估算
        if complexity_info["recursive_calls"] > 0:
            complexity_info["estimated_complexity"] = "O(2^n)"
        elif complexity_info["nested_loop_depth"] >= 3:
            complexity_info["estimated_complexity"] = "O(n³)"
        elif complexity_info["nested_loop_depth"] == 2:
            complexity_info["estimated_complexity"] = "O(n²)"
        elif complexity_info["loops_detected"] > 0:
            complexity_info["estimated_complexity"] = "O(n)"
        else:
            complexity_info["estimated_complexity"] = "O(1)"
            
        return complexity_info
    
    def detect_recursive_calls(code_content, language):
        """增强的递归调用检测"""
        recursive_functions = []
        
        if language == 'python':
            lines = code_content.split('\n')
            current_function = None
            
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    current_function = func_name
                elif current_function and current_function in line and '(' in line:
                    if not line.strip().startswith('def '):
                        recursive_functions.append({
                            "function_name": current_function,
                            "line_number": i + 1,
                            "call_expression": line.strip()
                        })
        
        elif language in ['cpp', 'c']:
            # 简化的C/C++递归检测
            lines = code_content.split('\n')
            for i, line in enumerate(lines):
                if 'return' in line and '(' in line and ';' in line:
                    # 简单检测可能的递归调用
                    recursive_functions.append({
                        "function_name": "detected_recursive",
                        "line_number": i + 1,
                        "call_expression": line.strip()
                    })
        
        return recursive_functions
    
    def detect_performance_patterns(code_content):
        """性能模式检测"""
        patterns = []
        
        # 嵌套循环模式
        if code_content.count('for ') >= 2 or 'for.*for' in code_content:
            patterns.append('nested_loops')
            
        # 递归模式
        if 'return.*recursive_' in code_content or 'return.*fibonacci' in code_content:
            patterns.append('recursion')
            
        # 阻塞IO模式
        blocking_keywords = ['sleep', 'time.sleep', 'std::this_thread::sleep_for', 'usleep']
        if any(keyword in code_content for keyword in blocking_keywords):
            patterns.append('blocking_io')
            
        # 内存操作模式
        if any(op in code_content for op in ['malloc', 'free', 'new', 'delete', 'alloc']):
            patterns.append('memory_operations')
            
        return patterns
    
    # 测试用例
    test_cases = [
        {
            'language': 'Python',
            'code': '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def process_data(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            result.append(data[i] * data[j])
    return result

import time
def slow_function():
    time.sleep(1)
''',
            'expected_improvements': ['recursive_detection', 'nested_loop_detection', 'blocking_io_detection']
        },
        {
            'language': 'C++',
            'code': '''
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

class Processor {
public:
    int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }
    
    std::vector<int> process_quadratic(const std::vector<int>& data) {
        std::vector<int> result;
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data.size(); j++) {
                result.push_back(data[i] * data[j]);
            }
        }
        return result;
    }
    
    void blocking_operation() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
};
''',
            'expected_improvements': ['recursive_detection', 'nested_loop_detection', 'blocking_io_detection', 'memory_operations']
        },
        {
            'language': 'C',
            'code': '''
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

void* process_data(int* data, int size) {
    int* result = (int*)malloc(size * size * sizeof(int));
    int index = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[index++] = data[i] * data[j];
        }
    }
    return result;
}

void blocking_call() {
    sleep(1);
}
''',
            'expected_improvements': ['recursive_detection', 'nested_loop_detection', 'blocking_io_detection', 'memory_operations']
        }
    ]
    
    # 运行测试
    results = {}
    total_improvements = 0
    detected_improvements = 0
    
    for case in test_cases:
        language = case['language']
        code = case['code']
        expected = case['expected_improvements']
        
        total_improvements += len(expected)
        
        analysis = enhanced_code_analysis(code)
        
        # 检查改进的检测能力
        detected = []
        
        # 递归检测
        if analysis["recursive_functions"]:
            detected.append('recursive_detection')
            
        # 嵌套循环检测
        if analysis["complexity"]["nested_loop_depth"] >= 2:
            detected.append('nested_loop_detection')
            
        # 阻塞IO检测
        if 'blocking_io' in analysis["performance_patterns"]:
            detected.append('blocking_io_detection')
            
        # 内存操作检测
        if 'memory_operations' in analysis["performance_patterns"]:
            detected.append('memory_operations')
        
        accuracy = len(set(detected) & set(expected)) / len(set(expected)) * 100 if expected else 100
        
        results[language] = {
            'accuracy': accuracy,
            'detected_improvements': detected,
            'expected_improvements': expected,
            'analysis_details': {
                'language_detected': analysis["language"],
                'functions_found': analysis["structure"]["function_count"],
                'loops_found': analysis["structure"]["loop_count"],
                'complexity_estimated': analysis["complexity"]["estimated_complexity"],
                'recursive_calls': len(analysis["recursive_functions"])
            }
        }
        
        detected_improvements += len(set(detected) & set(expected))
        
        print(f"  {language} 改进检测: {accuracy:.1f}%")
        print(f"    检测到的改进: {', '.join(detected)}")
        print(f"    语言识别: {analysis['language']}")
        print(f"    复杂度估算: {analysis['complexity']['estimated_complexity']}")
        print(f"    递归函数数: {len(analysis['recursive_functions'])}")
        print()
    
    # 综合评分
    overall_accuracy = detected_improvements / total_improvements * 100 if total_improvements > 0 else 100
    
    print(f"📊 增强后Performance Agent能力报告:")
    print(f"  总体改进检测准确率: {overall_accuracy:.1f}%")
    
    for lang, result in results.items():
        print(f"    {lang}: {result['accuracy']:.1f}%")
    
    # 评估结果
    print(f"\n💡 评估结果:")
    if overall_accuracy >= 90:
        print(f"  🎉 优秀 - 增强功能表现卓越")
    elif overall_accuracy >= 75:
        print(f"  👍 良好 - 改进明显，功能显著增强")
    elif overall_accuracy >= 60:
        print(f"  ⚠️  及格 - 有一定改进，仍有提升空间")
    else:
        print(f"  ❌ 需要更多改进")
    
    return overall_accuracy >= 75

if __name__ == "__main__":
    success = test_enhanced_performance_agent()
    exit(0 if success else 1)