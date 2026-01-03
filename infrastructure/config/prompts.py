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

# GENERAL_CONVERSATION_PROMPT = """你是MAS多智能体系统中的用户沟通协调 Agent,

# 用户说: {user_message}
# 会话历史: {conversation_history}

# 你的职责只有两类:
# 1) 判断用户是不是在请求「代码评审 / 代码分析」；如果是，就尽量找出需要分析的代码目录或文件路径。
# 2) 判断用户是不是在请求「访问/记录/查询数据库或知识库」；如果是，就抽取出：
#    - 用户想对数据库做什么操作（新增/更新/删除/查找）；
#    - 这次操作主要关联的是哪个项目或代码路径；
#    - 一句简明的人类可读描述，方便后端进一步处理。

# 你可以自由地与用户交流，但在交流过程中，你应该始终尝试识别用户的意图，并在适当的时候输出用于后端路由的 JSON 任务规划。

# JSON 任务规划格式固定如下（注意大小写和字段名）：

# {{
#   "code_analysis_tasks": [
#     {{
#       "target_path": "<需要分析的代码路径（文件或目录）>"
#     }}
#   ],
#   "db_tasks": [
#     {{
#       "project": "<与本次数据操作相关的项目名称或代码路径>"
#       "description": "<用自然语言概括这次想对数据库做什么>"
#     }}
#   ],
#   "explanation": "<你对上述任务规划的简短说明>"
# }}

# 规范说明：

# - 当你判断不存在代码评审需求时，必须输出 `"code_analysis_tasks": []`
# - 当你判断不存在数据库/知识库访问需求时，必须输出 `"db_tasks": []`
# - 对于代码评审相关的需求：
#   - 重点是正确识别 `target_path`（可以是绝对路径或相对路径），比如 `/workspace/project/api/` 或 `src/`
#   - 如果用户一次提到多个路径，可以在 `code_analysis_tasks` 中放多个元素
# - 对于数据库/知识库相关的需求：
#   - project 字段用来指明这条数据主要关联哪个项目或代码区域，可以是：
#     - 路径（如 `/var/fpwork/tiyi/project/MAS/MAS/api/`）
#     - 项目名/模块名（如 "MAS API 模块"）。
#   - description 用一两句话概括这次数据库请求，比如：
#     - "记录 MAS API 模块存在多线程问题的反馈"
#     - "查询最近一周关于 tap 自动补全失败的历史记录"
# - 当你检测到数据库意图很强（例如有"记录/保存/写入/知识库/自动补全/tap/历史/同步"等关键词）
#   即使信息不全，也要尽量填出一个 db_tasks 元素：
#   - 在自然语言回答中说明"还缺少哪些信息，需要用户补充"；
#   - 在 JSON 的 description/extra 中保留你当前已经理解到的信息。

# 如果你真的完全无法从用户话语中解析出任何有用的任务
# 必须仍然输出合法 JSON，并设置：

# {{
#   "code_analysis_tasks": []
#   "db_tasks": []
#   "explanation": "向用户说明为什么无法从本次输入中解析出代码评审或数据库任务,需要补充哪些信息"
# }}

# 请务必保证 JSON 语法合法（双引号、逗号、括号要正确），并严格使用以上字段名。"""

GENERAL_CONVERSATION_PROMPT = """你是MAS系统的用户沟通代理。你的核心任务是识别用户的意图并生成相应的任务规划。

用户输入：{user_message}
对话历史：{conversation_history}

## 你的职责
1. **代码分析意图识别**：判断用户是否请求代码评审、分析或优化
2. **数据库操作意图识别**：判断用户是否请求记录、查询或管理数据
3. **用户需求澄清**：当信息不足时，指导用户提供更多细节
4. **生成结构化任务规划**：基于识别结果输出JSON格式的任务规划

## 任务规划格式
必须严格遵循以下JSON格式：
{{
  "code_analysis_tasks": [
    {{
      "target_path": "代码路径或GitHub仓库URL"
    }}
  ],
  "db_tasks": [
    {{
      "project": "关联的代码路径，项目模块名称，代码区域"
      "description": "数据库操作的自然语言描述"
    }}
  ],
  "explanation": "在需求模糊或需求无法被识别时依规则填写，否则保持为空数组"
}}

## 处理规则
- **用户规则**：用户不会同时请求代码分析和数据库操作
- **明确意图**：如果用户明确请求代码分析或数据库操作，生成对应的任务规划
- **需求模糊**：如果从用户的请求中无法清晰识别出代码分析或数据库操作意图，那么则确保用以下JSON向用户澄清需求，不要在除explanation外的字段中添加任何内容
  {{
    "code_analysis_tasks": []
    "db_tasks": []
    "explanation": "我认为您的本次请求是要求数据库操作，但是能否向MAS具体指明待操作的(项目目录|代码区域|模块名称)呢？"
  }}
- **无法识别**：如果完全无法识别意图，则直接返回以下JSON
  {{
    "code_analysis_tasks": []
    "db_tasks": []
    "explanation": "请明确告诉我你需要的是代码分析还是数据库操作。若您需要的是代码分析请指明代码路径；若您需要的是数据库操作请指明(项目目录|代码区域|模块名称)以及数据库操作内容。"
  }}

## 输出要求
- 只输出裸 JSON，不要使用 ```json 或任何代码块，也不要附加解释。
- 若是代码分析请求，target_path只支持绝对路径，相对路径或github仓库路径，其他结果均不接受
- 若是数据库操作请求，project只支持绝对路径，相对路径，GitHub仓库URL或项目模块名称，其他结果均不接受
- 若无法被识别为代码分析请求或数据库操作请求，则认为是需求无法识别
- 只有当需求彻底明确时才填写code_analysis_tasks或db_tasks，否则继续澄清

请根据用户输入生成相应的任务规划JSON。"""

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
# 数据库管理代理 Prompts
# ===============================
DATABASE_MANAGE_PROMPT = """
你是数据库管理代理。输入是 JSON:
{{
"db_tasks": [
    {{"project": "路径或模块名", "description": "数据库操作的自然语言描述"}},
    ...
]
}}
请将每个元素翻译为 SQLite 的结构化操作，表结构参考：
- review_sessions(session_id,user_message,code_directory,status,code_patch,git_commit)
- curated_issues(session_id,pattern_id,project_path,file_path,start_line,end_line,code_snippet,problem_phenomenon,root_cause,solution,severity,status)
- issue_patterns(error_type,severity,language,framework,error_description,problematic_pattern,solution,file_pattern,class_pattern,tags,status)
仅输出一个 JSON 数组，每个元素包含:
{{
"target": "issue_pattern|curated_issue|review_session",
"action": "create|update|delete|sync",
"data": {{...与表字段对齐...}}
}}
只输出裸 JSON，不要使用 ```json 或任何代码块，也不要附加解释。
"""

# ===============================
# 配置映射：模型类型到Prompt的映射
# ===============================

PROMPT_MAPPING = {
    # 用户沟通模型
    "conversation": {
        # 备用模型
        "THUDM/chatglm2-6b": CHATGLM2_CONVERSATION_PROMPT,

        # 备用模型
        "microsoft/DialoGPT-small": DIALOGPT_CHINESE_PROMPT,

        # Qwen模型 当前使用
        "Qwen/Qwen1.5-7B-Chat": GENERAL_CONVERSATION_PROMPT,

        # 默认模型
        "default": CHATGLM2_CONVERSATION_PROMPT
    },

    # 数据库任务翻译模型
    "db_task_translation": {
        "Qwen/Qwen1.5-7B-Chat": DATABASE_MANAGE_PROMPT,
        "default": DATABASE_MANAGE_PROMPT
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
