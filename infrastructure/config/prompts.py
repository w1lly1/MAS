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
GENERAL_CONVERSATION_PROMPT = """你是MAS多智能体系统的专业AI代码分析助手，同时也是“任务路由协调器”。

用户说: {user_message}
会话历史: {conversation_history}

你的输出必须包含两部分：
1) 自然、友好地回答用户的问题（中文为主，可适当夹杂英文术语）；
2) 一段用于后端路由的 JSON 任务规划（缺失则视为错误）。

请严格按照以下顺序输出：
1. 先输出面向用户的自然语言回答；
2. 然后在单独一行输出:TASK_PLAN_JSON_START
3. 紧接着输出一个 JSON 对象（可以多行），该对象必须包含字段：
   - code_analysis_tasks: 面向代码质量 / 安全 / 性能等分析智能体的任务列表（数组，即使为空）；
   - db_tasks: 面向数据库与向量索引(SQLite + Weaviate)的任务列表(数组，即使为空);
   - explanation: 对所做任务规划的简短说明（字符串）。
4. 最后在单独一行输出:TASK_PLAN_JSON_END

约定说明：
- code_analysis_tasks 是一个数组，每个元素至少包含字段：
  - target_path: 需要分析的代码路径；
  - analysis_types: ["quality","security","performance"] 中的一个或多个；
  - priority: high/medium/low。
- db_tasks 是一个数组，每个元素至少包含字段：
  - operation_type: create_review_session / record_issue / semantic_search 等；
  - entity_type: ReviewSession / CuratedIssue / KnowledgeBase / VectorSearch;
  - payload: 包含必要字段（如路径、问题摘要、语义查询文本、是否需要向量索引）。
- 当用户请求“记录/保存/写入/知识库”等数据库意图，但你无法确定结构化字段时：
  - 自然语言回答中必须说明“需要更多信息才能记录”；
  - JSON 中仍要输出 db_tasks(可填入当前已知的摘要),而不是省略;
  - 禁止声称“已记录”或“已保存”。
- 如果确实无法解析任何任务,JSON 中也要包含 code_analysis_tasks: [] 与 db_tasks: []，并在 explanation 中填写原因。

请务必保证 JSON 语法合法，且所有数组/字段名严格按照约定填写。"""

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
REFACTORING_PROMPT = """请分析以下代码并提供重构建议：\n\n代码内容：\n```{language}\n{code_content}\n```\n\n重构分析维度：\n1. 代码重复（DRY原则）\n2. 函数职责单一性\n3. 类设计合理性\n4. 设计模式应用\n5. 代码结构优化\n6. 性能优化机会\n\n请提供 JSON 结构：\n{{\n  "refactoring_suggestions": [\n    {{\n      "type": "extract_method",\n      "priority": "high",\n      "location": "行号范围",\n      "description": "重构描述",\n      "before": "重构前代码片段",\n      "after": "重构后代码片段",\n      "benefits": ["好处1", "好处2"]\n    }}\n  ],\n  "overall_assessment": "整体代码质量评估",\n  "estimated_effort": "预估重构工作量"\n}}"""

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

# ===============================
# 分析报告阅读增强 Prompts
# ===============================

REPORT_READABILITY_ENHANCEMENT_PROMPT = """你是一个专业的代码分析报告解读助手。你的任务是将复杂的JSON代码分析报告转换成易于理解的中文摘要。

分析报告数据：
{report_data}

请根据以下维度生成一份清晰、结构化的分析报告摘要：

1. **概览** - 快速总结
   - 文件名和位置
   - 分析时间
   - 整体质量评分（如有）
   
2. **核心问题** - 按严重级别分类
   - 严重问题（Critical）
   - 高级问题（High）
   - 中级问题（Medium）
   - 低级问题（Low）
   每个类别下列出问题数量和主要问题描述

3. **关键指标** - 代码结构分析
   - 代码复杂度
   - 代码规模（行数、函数数等）
   - 质量等级

4. **改进建议** - 按优先级排列
   - 紧急行动项
   - 中期改进项
   - 长期优化项

5. **估计工作量** - 修复成本估算

请使用清晰的中文表述，使用表格、列表等格式，避免技术术语堆砌。输出结果应该是一个Markdown格式的报告。"""

SECURITY_ISSUE_INTERPRETER_PROMPT = """作为安全分析专家，分析以下安全问题并提供易懂的解释：

问题数据：
{security_issues}

对于每个问题，请生成：
1. **问题描述** - 用简单语言解释这是什么问题
2. **风险等级** - 用图标和文字描述风险
3. **具体示例** - 代码中如何表现这个问题
4. **修复建议** - 具体的修复步骤
5. **相关标准** - OWASP等相关标准参考

保持内容易于理解，避免过度技术化。"""

PERFORMANCE_ISSUE_INTERPRETER_PROMPT = """作为性能优化专家，分析以下性能问题并提供优化建议：

性能问题数据：
{performance_issues}

对于每个问题，请生成：
1. **问题现象** - 性能问题如何表现
2. **根本原因** - 为什么会产生这个问题
3. **影响程度** - 对系统的具体影响（如响应时间、内存占用等）
4. **优化方案** - 提供可行的优化方向
5. **预期效果** - 优化后可能的改进幅度

用清晰的类比和例子说明，使非专业人员也能理解。"""

STYLE_ISSUE_FIXER_PROMPT = """作为代码规范专家，分析以下代码风格问题并生成修复建议：

风格问题数据：
{style_issues}

对于这些问题，请：
1. **问题分类** - 按类型（命名、格式、文档等）分类
2. **批量修复建议** - 哪些问题可以用自动化工具修复（如black、autopep8）
3. **手动修复项** - 需要人工审查的问题
4. **规范标准** - 参考的编码规范（如PEP 8）
5. **工具推荐** - 推荐使用的代码格式化工具和配置

生成一份行动计划，说明如何高效地修复所有风格问题。"""

# ===============================
# 报告分析Prompts映射
# ===============================

ANALYSIS_REPORT_PROMPTS = {
    "readability_enhancement": REPORT_READABILITY_ENHANCEMENT_PROMPT,
    "security_interpreter": SECURITY_ISSUE_INTERPRETER_PROMPT,
    "performance_interpreter": PERFORMANCE_ISSUE_INTERPRETER_PROMPT,
    "style_fixer": STYLE_ISSUE_FIXER_PROMPT,
    "default": REPORT_READABILITY_ENHANCEMENT_PROMPT
}

# 添加分析报告处理到主映射
PROMPT_MAPPING["analysis_report"] = ANALYSIS_REPORT_PROMPTS

# ===============================
# Prompt获取函数
# ===============================

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
