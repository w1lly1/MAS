## MAS (Multi-Agent System) - AI 代码审查与知识管理系统

MAS 是一个基于多智能体的 AI 驱动代码审查与知识管理系统。  
它通过多个专职智能体协同工作，对代码进行质量 / 安全 / 性能 / 静态扫描分析，并支持将问题与经验沉淀到数据库与向量索引中，为后续项目复用提供基础。

---

## 核心能力概览

### AI 驱动对话与任务路由

- 使用 **Qwen/Qwen1.5-7B-Chat** 作为用户沟通 Agent 的核心对话模型。
- 通过 Prompt + 结构化 TASK_PLAN JSON（`code_analysis_tasks` / `db_tasks`）识别用户意图。
- 在 JSON 缺失或不稳定时，使用关键词 fallback 逻辑，区分：
  - “只分析代码”的请求；
  - “只做知识/数据库写入”的请求。

### 多智能体代码分析

- **Code Quality Agent**：复杂度、规范性、可维护性、设计问题等。
- **Security Agent**：常见漏洞、依赖风险、敏感信息暴露等。
- **Performance Agent**：算法复杂度、性能瓶颈、低效 I/O 等。
- **Static Scan Agent**：集成 Pylint / Flake8 / Bandit 等传统静态工具。

### 多层报告与可读性增强

- 每个分析 Agent 输出独立的 JSON 报告。
- **Summary Agent** 按文件聚合结果，生成：
  - `consolidated_*.json`（单文件综合报告）
  - `run_summary.json`（运行级整体汇总）
- **Readability Enhancement Agent**：
  - 将 JSON 报告转换为 Markdown；
  - 生成面向人的问题列表与分析摘要。

### 知识与历史运行管理（进行中）

- 用户沟通 Agent 已能根据用户请求生成 `db_tasks`，并通过 Mock DB Agent 验证路由与结构。
- 规划中将引入 **DBVectorIndexAgent**：
  - 写入 SQLite 作为“事实库”；
  - 将关键信息写入 Weaviate 作为“向量索引层”；
  - 支持对历史问题与知识条目的语义检索。

---

## 项目结构（简要）

MAS/
├── mas.py                       # CLI 主入口
├── README.md                    # 项目文档（本文件）
├── requirements.txt             # 依赖管理
│
├── api/                         # CLI 与对话模式集成
│   ├── main.py                  # 命令行实现（login, results 等）
│   └── cli_mode_prompt_design.md# CLI review/store 模式与 Prompt 设计（规划）
│
├── core/
│   ├── agents_integration.py    # 多智能体分析集成与调度
│   └── agents/
│       ├── base_agent.py
│       ├── agent_manager.py
│       ├── ai_driven_user_communication_agent.py      # 用户沟通/路由 Agent
│       ├── ai_driven_code_quality_agent.py            # 代码质量 Agent
│       ├── ai_driven_security_agent.py                # 安全分析 Agent
│       ├── ai_driven_performance_agent.py             # 性能分析 Agent
│       ├── static_scan_agent.py                       # 传统静态扫描 Agent
│       ├── analysis_result_summary_agent.py           # 结果汇总 Agent
│       └── ai_driven_readability_enhancement_agent.py # 可读性增强 Agent
│
├── infrastructure/
│   ├── reports/
│   │   └── report_manager.py    # 报告管理与工具
│   ├── config/
│   │   ├── prompts.py           # Prompt 模板与映射
│   │   ├── ai_agents.py         # Agent 配置访问器
│   │   └── ai_agent_config.json # 统一 AI/Agent 配置
│   └── database/
│       ├── design.md            # SQLite + Weaviate 架构设计
│       ├── models.py            # 数据模型（演进中）
│       └── service.py           # 数据库服务（演进中）
│
├── reports/
│   └── analysis/
│       └── <run_id>/            # 每次分析运行的完整报告目录
│           ├── dispatch_report_*.json
│           ├── agents/          # 各智能体独立报告
│           ├── consolidated/    # 按文件聚合的综合报告
│           ├── readability_enhancement/  # Markdown 可读性报告
│           └── run_summary.json          # 运行级汇总报告
│
└── tests/                       # 测试与实验（部分仍在演进）---

## 架构总览

### CLI 层（`mas.py` + `api/`）

- 当前支持的主要入口：
  - `python mas.py login`  
    启动交互式对话模式。
  - `python mas.py login --target-dir /path/to/code`  
    启动对话并立即对该目录进行代码分析。
  - `python mas.py results <run_id>`  
    查看指定 run 的汇总信息与报告路径（视实现情况）。
- 规划中的显式模式（详见 `api/cli_mode_prompt_design.md`）：
  - `mas review -d <dir>`：显式“代码评审模式”，优先触发多智能体分析。
  - `mas store --message/--from-file`：显式“知识存储模式”，优先生成 `db_tasks` 写入知识库/数据库。

> 当前版本主要基于 `login` 模式；`review` / `store` 为后续演进方向，用于降低对 LLM 意图识别的依赖。

### 用户沟通与任务路由层

文件：`core/agents/ai_driven_user_communication_agent.py`

- 职责：
  - 与用户进行自然语言对话（使用 Qwen/Qwen1.5-7B-Chat）。
  - 通过 `GENERAL_CONVERSATION_PROMPT` 生成 Prompt，并要求模型输出：
    - 面向用户的自然语言回答；
    - `TASK_PLAN_JSON_START` / `TASK_PLAN_JSON_END` 包裹的任务规划 JSON。
  - 解析任务规划 JSON：
    - `code_analysis_tasks`：代码分析任务（目标路径、分析类型等）。
    - `db_tasks`：数据库 / 知识库存储或语义检索任务。
  - 当 JSON 缺失或不合法时：
    - 记录警告日志；
    - 使用关键词与路径的 fallback 逻辑区分：
      - 代码分析场景；
      - 数据库存储场景，并构造最小 `db_tasks`。

- 调试辅助：
  - 环境变量 `MAS_MOCK_CODE_ANALYSIS=1`：
    - 在触发代码分析时，仅打印规划出的 `code_tasks`，而不真正启动多智能体分析；
    - 方便验证 Prompt 与路由逻辑。

### 多智能体分析层

目录：`core/agents/`

- **Code Quality Agent**：  
  负责复杂度分析、命名/结构规范、潜在设计问题等。
- **Security Agent**：  
  负责常见安全漏洞、依赖安全与敏感信息识别。
- **Performance Agent**：  
  负责性能瓶颈与算法复杂度分析。
- **Static Scan Agent**：  
  集成传统静态工具（如 Pylint / Flake8 / Bandit）做语法与风格检查。
- **Analysis Result Summary Agent**：  
  聚合多 Agent 输出，按文件生成 `consolidated_*.json`，按 run 生成 `run_summary.json`。
- **Readability Enhancement Agent**：  
  将 JSON 报告转为 Markdown，输出更易阅读的分析总结。

### 数据库与向量索引层（规划中）

设计文档：`core/db_agent_integration_plan.md`、`infrastructure/database/design.md`

- 目标：
  - SQLite 作为权威事实库，存储：
    - 分析 run 元信息；
    - 已人工确认的重要问题（CuratedIssue）；
    - 知识条目与模式（KnowledgeBase / IssuePattern）。
  - Weaviate 作为向量索引层，存储：
    - 问题与知识条目的向量；
    - Run 级总结的向量；
    - 支持自然语言语义检索。

- 当前阶段：
  - 用户沟通 Agent 已能生成 `db_tasks` 并通过 Mock DB Agent 打印出来，验证路由是否正确。
  - 真正的 `DBVectorIndexAgent` 尚在开发计划中，将在后续替换 Mock。

---

## 核心工作流

### 代码分析主流程

1. **触发分析**
   - CLI：`python mas.py login -d /path/to/code`；
   - 或在交互会话中输入“分析目录 /path/to/code”。

2. **扫描与任务派发**
   - 扫描目录中的 Python 文件；
   - 为每个文件生成 requirement；
   - 输出 `dispatch_report_*.json`，分配给各分析 Agent。

3. **多智能体并发分析**
   - Code Quality / Security / Performance / Static Scan 各自生成 JSON 报告，存放在：
     - `reports/analysis/<run_id>/agents/<agent_type>/...`

4. **结果汇总与可读性增强**
   - Summary Agent 聚合为：
     - `consolidated_*.json`（按文件的综合报告）；
     - `run_summary.json`（整个 run 的总体统计）。
   - Readability Enhancement Agent 将上述 JSON 转换为 Markdown：
     - `readability_enhancement/consolidated/*.md` 等。

5. **查看结果**
   - 遍历 `reports/analysis/<run_id>/` 目录查看 JSON / Markdown；
   - 或通过 CLI（如 `mas.py results <run_id>`，视实际实现）。

### 知识存储工作流（当前 Mock + 规划）

1. **用户表达意图**
   - 如：“请把这次 bug 记录到知识库里”或“请将以上信息记入数据库”。

2. **任务规划**
   - 用户沟通 Agent 根据 Prompt 和 fallback 逻辑：
     - 识别为 DB 写入意图；
     - 生成 `db_tasks`，包含 `operation_type` / `entity_type` / `payload` 等。

3. **路由到 DB Agent**
   - 当前阶段：Mock DB Agent 打印收到的 `db_tasks`，用于验证结构与路由；
   - 后续阶段：DBVectorIndexAgent 写入 SQLite + Weaviate，并提供读/写/检索接口。

---

## 基本使用

### 环境要求

- Python 3.12+
- PyTorch 2.8.0+
- Transformers 4.56.0+
- 建议 8GB 以上内存（更推荐 16GB+ 以稳定运行 Qwen1.5-7B-Chat）

### 安装

git clone <repository-url>
cd MAS

python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt### 启动交互会话与分析

# 启动对话模式
python mas.py login

# 启动对话并立即分析指定目录
python mas.py login --target-dir /path/to/your/code调试 Prompt / 路由时可使用 Mock 模式：

MAS_MOCK_CODE_ANALYSIS=1 python mas.py login
# 拦截后端 Code/Security/Performance 分析，仅打印规划出的任务---

## 设计文档索引

- `core/db_agent_integration_plan.md`  
  数据库向量 Agent（DBVectorIndexAgent）分阶段集成与职责划分。
- `infrastructure/database/design.md`  
  SQLite + Weaviate 的实体建模、一致性策略与向量写入方案。
- `api/cli_mode_prompt_design.md`  
  CLI review/store 模式设计与 Prompt 策略。

---

### 开发环境

git clone <repository-url>
cd MAS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt