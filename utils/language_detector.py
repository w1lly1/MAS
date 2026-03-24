"""
统一的语言检测工具
用于检测代码文件的编程语言并提供相应的分析支持
"""
import os
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ProgrammingLanguage(Enum):
    PYTHON = "python"
    CPP = "cpp"
    C = "c"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"

class LanguageDetector:
    """智能编程语言检测器"""
    
    def __init__(self):
        # 各语言的特征标识
        self.language_signatures = {
            ProgrammingLanguage.PYTHON: {
                'extensions': ['.py'],
                'keywords': ['def ', 'import ', 'class ', 'self.', '__init__'],
                'patterns': [r'#!/usr/bin/env python', r'#.*coding:']
            },
            ProgrammingLanguage.CPP: {
                'extensions': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx'],
                'keywords': ['#include', 'namespace', 'std::', 'cout', 'cin', 'template<'],
                'patterns': [r'#include\s+<iostream>', r'using namespace std;']
            },
            ProgrammingLanguage.C: {
                'extensions': ['.c', '.h'],
                'keywords': ['#include', 'printf', 'scanf', 'malloc', 'free'],
                'patterns': [r'#include\s+<stdio.h>', r'int main\s*\(']
            },
            ProgrammingLanguage.JAVASCRIPT: {
                'extensions': ['.js'],
                'keywords': ['function ', 'var ', 'let ', 'const ', 'console.'],
                'patterns': [r'console\.log', r'function\s+\w+\s*\(']
            },
            ProgrammingLanguage.JAVA: {
                'extensions': ['.java'],
                'keywords': ['public class', 'private ', 'protected ', 'static '],
                'patterns': [r'public static void main', r'import java\.']
            },
            ProgrammingLanguage.GO: {
                'extensions': ['.go'],
                'keywords': ['package ', 'func ', 'import ', 'var ', 'const '],
                'patterns': [r'package main', r'func main\s*\(']
            },
            ProgrammingLanguage.RUST: {
                'extensions': ['.rs'],
                'keywords': ['fn ', 'let ', 'mut ', 'use ', 'impl '],
                'patterns': [r'fn main\s*\(', r'use std::']
            }
        }
    
    def detect_language(self, content: str, file_extension: str = None, filename: str = None) -> ProgrammingLanguage:
        """
        检测代码的编程语言
        
        Args:
            content: 代码内容
            file_extension: 文件扩展名
            filename: 文件名
            
        Returns:
            ProgrammingLanguage枚举值
        """
        content_lower = content.lower()
        scores = {}
        
        # 基于文件扩展名的初步判断
        if file_extension:
            for lang, signature in self.language_signatures.items():
                if file_extension.lower() in signature['extensions']:
                    scores[lang] = scores.get(lang, 0) + 10  # 扩展名权重较高
        
        # 基于关键字的评分
        for lang, signature in self.language_signatures.items():
            keyword_score = sum(1 for keyword in signature['keywords'] 
                              if keyword in content)
            scores[lang] = scores.get(lang, 0) + keyword_score
        
        # 如果有得分，返回最高分的语言
        if scores:
            detected_lang = max(scores, key=scores.get)
            # 只有当得分大于阈值时才确认检测结果
            if scores[detected_lang] > 2:
                return detected_lang
        
        return ProgrammingLanguage.UNKNOWN
    
    def get_language_specific_prompts(self, language: ProgrammingLanguage) -> Dict[str, str]:
        """
        获取特定语言的Prompt配置
        
        Args:
            language: 编程语言
            
        Returns:
            包含各种分析类型的Prompt字典
        """
        from infrastructure.config.prompts import (
            PYTHON_QUALITY_GUIDELINES, CPP_QUALITY_GUIDELINES,
            PYTHON_SECURITY_FEATURES, CPP_SECURITY_FEATURES,
            PYTHON_VULNERABILITY_PATTERNS, CPP_VULNERABILITY_PATTERNS,
            PYTHON_PERFORMANCE_CONSIDERATIONS, CPP_PERFORMANCE_CONSIDERATIONS,
            PYTHON_OPTIMIZATION_STRATEGIES, CPP_OPTIMIZATION_STRATEGIES
        )
        
        prompt_configs = {
            ProgrammingLanguage.PYTHON: {
                'quality_guidelines': PYTHON_QUALITY_GUIDELINES,
                'security_features': PYTHON_SECURITY_FEATURES,
                'vulnerability_patterns': PYTHON_VULNERABILITY_PATTERNS,
                'performance_considerations': PYTHON_PERFORMANCE_CONSIDERATIONS,
                'optimization_strategies': PYTHON_OPTIMIZATION_STRATEGIES
            },
            ProgrammingLanguage.CPP: {
                'quality_guidelines': CPP_QUALITY_GUIDELINES,
                'security_features': CPP_SECURITY_FEATURES,
                'vulnerability_patterns': CPP_VULNERABILITY_PATTERNS,
                'performance_considerations': CPP_PERFORMANCE_CONSIDERATIONS,
                'optimization_strategies': CPP_OPTIMIZATION_STRATEGIES
            },
            ProgrammingLanguage.C: {
                'quality_guidelines': CPP_QUALITY_GUIDELINES,  # C和C++质量指南相似
                'security_features': CPP_SECURITY_FEATURES,
                'vulnerability_patterns': CPP_VULNERABILITY_PATTERNS,
                'performance_considerations': CPP_PERFORMANCE_CONSIDERATIONS,
                'optimization_strategies': CPP_OPTIMIZATION_STRATEGIES
            }
        }
        
        return prompt_configs.get(language, {})
    
    def get_supported_extensions(self) -> List[str]:
        """获取所有支持的文件扩展名"""
        extensions = set()
        for signature in self.language_signatures.values():
            extensions.update(signature['extensions'])
        return list(extensions)
    
    def is_supported_language(self, language: ProgrammingLanguage) -> bool:
        """检查是否支持该语言"""
        return language in [ProgrammingLanguage.PYTHON, ProgrammingLanguage.CPP, ProgrammingLanguage.C]

# 全局实例
detector = LanguageDetector()

def detect_file_language(file_path: str, content: str = None) -> Tuple[ProgrammingLanguage, str]:
    """
    检测文件的编程语言
    
    Args:
        file_path: 文件路径
        content: 文件内容（可选，如果不提供会自动读取）
        
    Returns:
        (检测到的语言, 文件扩展名)
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    if content is None:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # 只读取前1000个字符用于检测
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read(1000)
            except Exception:
                content = ""
        except Exception:
            content = ""
    
    detected_language = detector.detect_language(content, file_extension, filename)
    return detected_language, file_extension

def get_language_prompts(language: ProgrammingLanguage) -> Dict[str, str]:
    """
    获取指定语言的Prompt配置
    
    Args:
        language: 编程语言
        
    Returns:
        Prompt配置字典
    """
    return detector.get_language_specific_prompts(language)

def is_language_supported(language: ProgrammingLanguage) -> bool:
    """检查语言是否受支持"""
    return detector.is_supported_language(language)