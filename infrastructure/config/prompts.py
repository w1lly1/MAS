"""
MAS系统AI模型Prompt配置文件
统一管理所有AI模型的提示词，方便维护和模型切换
"""

# ===============================
# 用户沟通代理 Prompts
# ===============================

# DialoGPT 专用对话提示词 (英文格式，因为DialoGPT主要支持英文)
DIALOGPT_CONVERSATION_PROMPT = """User: {user_message}
Assistant: I am MAS code analysis assistant. I can help you with code quality analysis, security detection, and performance optimization."""

# DialoGPT 中文适配提示词 (简化格式)
DIALOGPT_CHINESE_PROMPT = """{user_message}"""

# ChatGLM2 专用对话提示词（中英文双语）
CHATGLM2_CONVERSATION_PROMPT = """用户: {user_message}
助手: 您好！我是MAS代码分析助手，支持中英文对话。我可以帮您进行：
• 代码质量分析 / Code Quality Analysis
• 安全漏洞检测 / Security Vulnerability Detection  
• 性能优化建议 / Performance Optimization
• 多语言代码审查 / Multi-language Code Review

请问有什么可以帮助您的吗？ How can I assist you today?"""

# 通用对话模型提示词（适用于GPT系列、Claude等）
GENERAL_CONVERSATION_PROMPT = """你是MAS多智能体系统的专业AI代码分析助手。

用户说: {user_message}
会话历史: {conversation_history}

请自然、友好地回应用户。你的能力包括：
- 代码质量分析
- 安全漏洞检测  
- 性能优化建议
- 多种编程语言支持

请直接回应，保持对话自然流畅。"""

# ===============================
# 代码质量分析 Prompts
# ===============================

CODE_QUALITY_ANALYSIS_PROMPT = """请分析以下代码的质量，从多个维度进行评估：

代码内容：
```{language}
{code_content}
```

分析维度：
1. 代码规范性（命名、格式、注释）
2. 代码复杂度（循环嵌套、函数长度）
3. 可读性和可维护性
4. 设计模式和架构
5. 潜在的bug风险

请提供：
- 整体质量评分（1-10分）
- 具体问题列表
- 改进建议

以JSON格式返回结果：
{{
    "overall_score": 8.5,
    "issues": [
        {{
            "type": "naming",
            "severity": "medium",
            "line": 10,
            "description": "变量名不够描述性",
            "suggestion": "使用更具描述性的变量名"
        }}
    ],
    "recommendations": ["建议1", "建议2"]
}}"""

# ===============================
# 威胁建模与漏洞检测 Prompts
# ===============================

THREAT_MODELING_PROMPT = """基于代码与上下文进行STRIDE威胁建模:
组件:{system_components}
数据流:{data_flow}
代码:
```
{code_content}
```
输出每类威胁的简要风险与说明(JSON)。"""

VULNERABILITY_DETECTION_PROMPT = """识别代码片段潜在安全漏洞:
```
{code_snippet}
```
输出结构化结果(JSON) 包含: 类型/严重性/说明/位置。"""

# ===============================
# 重构建议 Prompt
# ===============================
REFACTORING_PROMPT = """请分析以下代码并提供重构建议：\n\n代码内容：\n```{language}\n{code_content}\n```\n\n重构分析维度：\n1. 代码重复（DRY原则）\n2. 函数职责单一性\n3. 类设计合理性\n4. 设计模式应用\n5. 代码结构优化\n6. 性能优化机会\n\n请提供 JSON 结构：\n{\n  \"refactoring_suggestions\": [\n    {\n      \"type\": \"extract_method\",\n      \"priority\": \"high\",\n      \"location\": \"行号范围\",\n      \"description\": \"重构描述\",\n      \"before\": \"重构前代码片段\",\n      \"after\": \"重构后代码片段\",\n      \"benefits\": [\"好处1\", \"好处2\"]\n    }\n  ],\n  \"overall_assessment\": \"整体代码质量评估\",\n  \"estimated_effort\": \"预估重构工作量\"\n}"""

# ===============================
# 性能分析细分 Prompts
# ===============================
ALGORITHMIC_ANALYSIS_PROMPT = """请分析以下代码片段的算法效率:\n```\n{code_snippet}\n```\n关注: 循环嵌套深度 / 递归模式 / 数据结构访问 / 排序与搜索方式 / 数学运算复杂度\n输出简要复杂度评估(JSON可解析):\n{\"best_case\": \"O(n)\", \"average_case\": \"O(n log n)\", \"worst_case\": \"O(n^2)\", \"space\": \"O(n)\"}"""

OPTIMIZATION_SUGGESTION_PROMPT = """基于以下性能瓶颈和代码内容生成优化建议:\n代码:\n```\n{current_code}\n```\n性能问题:\n{performance_issues}\n请给出: 立即优化 / 算法改进 / 结构调整 / 监控建议 (JSON列表)"""

# ===============================
# 配置映射：模型类型到Prompt的映射
# ===============================

PROMPT_MAPPING = {
    # 用户沟通模型
    "conversation": {
        # 主要模型 - ChatGLM2-6B（推荐的轻量级中英文模型）
        "THUDM/chatglm2-6b": CHATGLM2_CONVERSATION_PROMPT,
        
        # 备用模型（当前使用，中文支持有限）
        "microsoft/DialoGPT-small": DIALOGPT_CHINESE_PROMPT,
        
        # 新增Qwen模型支持
        "Qwen/Qwen1.5-7B-Chat": GENERAL_CONVERSATION_PROMPT,
        
        # 默认模型
        "default": CHATGLM2_CONVERSATION_PROMPT
    },
    
    # 代码分析模型
    "code_analysis": {
        "microsoft/codebert-base": CODE_QUALITY_ANALYSIS_PROMPT,
        "salesforce/codet5-base": CODE_QUALITY_ANALYSIS_PROMPT,
        "default": CODE_QUALITY_ANALYSIS_PROMPT
    },
    
    # 重构建议模型
    "refactoring": {
        "microsoft/codebert-base": REFACTORING_PROMPT,
        "salesforce/codet5-base": REFACTORING_PROMPT,
        "default": REFACTORING_PROMPT
    },
    
    # 统一命名: performance (包含细分variant)
    "performance": {
        "algorithmic_analysis": ALGORITHMIC_ANALYSIS_PROMPT,
        "optimization": OPTIMIZATION_SUGGESTION_PROMPT,
        "default": ALGORITHMIC_ANALYSIS_PROMPT
    },
    
    # 统一命名: security (包含细分variant)
    "security": {
        "threat_modeling": THREAT_MODELING_PROMPT,
        "vulnerability_detection": VULNERABILITY_DETECTION_PROMPT,
        "default": THREAT_MODELING_PROMPT
    }
}

def get_prompt(task_type: str, model_name: str = None, variant: str = None, **kwargs) -> str:
    """
    根据任务类型/模型名称/可选variant获取Prompt
    
    Args:
        task_type: 任务类型（conversation, code_analysis, security_analysis等）
        model_name: 模型名称，如果为None则使用default
        variant: 可选的变体名称，用于获取特定的Prompt变体
        **kwargs: Prompt格式化参数
    
    Returns:
        格式化后的Prompt字符串
    """
    if task_type not in PROMPT_MAPPING:
        raise ValueError(f"不支持的任务类型: {task_type}")
    prompts = PROMPT_MAPPING[task_type]
    if variant and variant in prompts:
        template = prompts[variant]
    else:
        template = prompts.get(model_name, prompts.get("default"))
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Prompt格式化失败，缺少参数: {e}")

def list_supported_tasks() -> list:
    """返回支持的任务类型列表"""
    return list(PROMPT_MAPPING.keys())

def list_supported_models(task_type: str) -> list:
    """返回指定任务类型支持的模型列表"""
    if task_type not in PROMPT_MAPPING:
        return []
    return list(PROMPT_MAPPING[task_type].keys())
