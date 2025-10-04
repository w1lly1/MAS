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

# 指令微调模型提示词（适用于Flan-T5、Alpaca等）
INSTRUCTION_TUNED_PROMPT = """### 指令
你是一个专业的代码分析AI助手，名为MAS助手。

### 用户输入
{user_message}

### 会话上下文
{conversation_history}

### 回应要求
请自然地回应用户，介绍你的代码分析能力并协助用户需求。

### 回应"""

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
# 安全分析 Prompts
# ===============================

SECURITY_ANALYSIS_PROMPT = """请对以下代码进行安全漏洞分析：

代码文件：{file_path}
代码内容：
```{language}
{code_content}
```

重点检查：
1. SQL注入风险
2. XSS漏洞
3. 命令注入
4. 路径遍历
5. 敏感信息泄露
6. 身份验证绕过
7. 权限控制问题
8. 密码学误用

请提供JSON格式的分析结果：
{{
    "vulnerabilities": [
        {{
            "type": "sql_injection",
            "severity": "high",
            "line": 25,
            "description": "用户输入未经过滤直接拼接到SQL查询中",
            "cwe_id": "CWE-89",
            "recommendation": "使用参数化查询"
        }}
    ],
    "security_score": 7.5,
    "summary": "总体安全状况描述"
}}"""

# ===============================
# 性能分析 Prompts
# ===============================

PERFORMANCE_ANALYSIS_PROMPT = """请分析以下代码的性能特征和潜在瓶颈：

代码文件：{file_path}
代码内容：
```{language}
{code_content}
```

分析重点：
1. 时间复杂度分析
2. 空间复杂度分析
3. 数据库查询效率
4. 循环和递归优化
5. 内存使用模式
6. I/O操作效率
7. 缓存使用策略

请返回JSON格式结果：
{{
    "performance_score": 8.0,
    "bottlenecks": [
        {{
            "type": "database",
            "severity": "high",
            "line": 15,
            "description": "N+1查询问题",
            "impact": "可能导致数据库性能急剧下降",
            "suggestion": "使用批量查询或预加载"
        }}
    ],
    "optimizations": ["优化建议1", "优化建议2"],
    "complexity": {{
        "time": "O(n²)",
        "space": "O(n)"
    }}
}}"""

# ===============================
# 代码理解和总结 Prompts
# ===============================

CODE_SUMMARY_PROMPT = """请分析并总结以下代码的功能和结构：

代码文件：{file_path}
代码内容：
```{language}
{code_content}
```

请提供：
1. 代码主要功能描述
2. 关键类和方法说明
3. 数据流分析
4. 依赖关系
5. 潜在改进点

返回格式：
{{
    "summary": "代码主要功能描述",
    "key_components": [
        {{
            "name": "类名或函数名",
            "type": "class/function",
            "description": "功能描述",
            "complexity": "复杂度评估"
        }}
    ],
    "data_flow": "数据流描述",
    "dependencies": ["依赖1", "依赖2"],
    "improvement_areas": ["改进建议1", "改进建议2"]
}}"""

# ===============================
# 测试代码生成 Prompts
# ===============================

TEST_GENERATION_PROMPT = """请为以下代码生成单元测试：

待测试代码：
```{language}
{code_content}
```

测试要求：
1. 覆盖主要功能路径
2. 包含边界情况测试
3. 错误处理测试
4. 使用适当的测试框架（如pytest、unittest等）
5. 包含mock对象处理外部依赖

请生成完整的测试代码，包含：
- 测试类结构
- 各种测试用例
- 测试数据准备
- 断言验证

测试代码：
```{language}
# 在此处生成测试代码
```"""

# ===============================
# 文档生成 Prompts
# ===============================

DOCUMENTATION_PROMPT = """请为以下代码生成详细的API文档：

代码内容：
```{language}
{code_content}
```

文档要求：
1. 函数/方法签名
2. 参数说明（类型、含义、默认值）
3. 返回值说明
4. 异常说明
5. 使用示例
6. 注意事项

请使用标准的文档格式（如docstring、JSDoc等）生成文档。

生成的文档：
```
# 在此处生成文档
```"""

# ===============================
# 代码重构建议 Prompts
# ===============================

REFACTORING_PROMPT = """请分析以下代码并提供重构建议：

代码内容：
```{language}
{code_content}
```

重构分析维度：
1. 代码重复（DRY原则）
2. 函数职责单一性
3. 类设计合理性
4. 设计模式应用
5. 代码结构优化
6. 性能优化机会

请提供：
{{
    "refactoring_suggestions": [
        {{
            "type": "extract_method",
            "priority": "high",
            "location": "行号范围",
            "description": "重构描述",
            "before": "重构前代码片段",
            "after": "重构后代码片段",
            "benefits": ["好处1", "好处2"]
        }}
    ],
    "overall_assessment": "整体代码质量评估",
    "estimated_effort": "预估重构工作量"
}}"""

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
    
    # 安全分析模型
    "security_analysis": {
        "microsoft/codebert-base": SECURITY_ANALYSIS_PROMPT,
        "default": SECURITY_ANALYSIS_PROMPT
    },
    
    # 性能分析模型
    "performance_analysis": {
        "microsoft/codebert-base": PERFORMANCE_ANALYSIS_PROMPT,
        "default": PERFORMANCE_ANALYSIS_PROMPT
    },
    
    # 重构建议模型
    "refactoring": {
        "microsoft/codebert-base": REFACTORING_PROMPT,
        "salesforce/codet5-base": REFACTORING_PROMPT,
        "default": REFACTORING_PROMPT
    }
}

def get_prompt(task_type: str, model_name: str = None, **kwargs) -> str:
    """
    根据任务类型和模型名称获取对应的Prompt
    
    Args:
        task_type: 任务类型（conversation, code_analysis, security_analysis等）
        model_name: 模型名称，如果为None则使用default
        **kwargs: Prompt格式化参数
    
    Returns:
        格式化后的Prompt字符串
    """
    if task_type not in PROMPT_MAPPING:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    prompts = PROMPT_MAPPING[task_type]
    prompt_template = prompts.get(model_name, prompts["default"])
    
    try:
        return prompt_template.format(**kwargs)
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
