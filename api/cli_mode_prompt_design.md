# MAS CLI 模式与 Prompt 策略设计（review / store）

## 1. 背景与目标

为了减少对大模型意图识别的过度依赖，并提升命令行场景下的可控性与可测试性，本设计引入两种显式的 CLI 模式：

- `mas review`: 代码评审/分析模式
- `mas store`: 数据存储/知识库模式

两种模式都通过 **用户沟通 Agent（AIDrivenUserCommunicationAgent）** 进入对话，但在：

- **加载的 Prompt**
- **默认任务路由倾向（code_analysis_tasks vs db_tasks）**

上存在明显差异，从而赋予 Agent 不同的“身份”和“工作模式”。

---

## 2. CLI 模式与交互流程

### 2.1 review 模式（代码评审）

- 推荐入口：
  - `mas review -d <target_dir>`
- 交互特性：
  - 启动后进入对话模式，用户可以多轮提问，例如：
    - “这次分析的重点是安全问题。”
    - “帮我解释一下 main.py 的主要风险点。”
  - 在该模式下：
    - **默认优先触发代码分析任务**（quality/security/performance）；
    - 仅在用户明确要求时，才会产生 DB 相关任务（如：把本次结果记入知识库）。

### 2.2 store 模式（数据存储）

- 推荐入口：
  - `mas store --message "..."` 或 `mas store --from-file <path>`
- 交互特性：
  - 启动后进入对话模式，用户可以多轮补充元信息，例如：
    - “这是哪个项目里的问题？”
    - “这类问题的严重级别如何？”
  - 在该模式下：
    - **默认优先产生 DB 相关任务**（记录问题、知识条目、历史经验）；
    - 仅在用户明确提出分析需求时，才触发代码分析 Agent。

---

## 3. Prompt 策略设计

### 3.1 利用现有 `get_prompt` 与 `PROMPT_MAPPING`

`infrastructure/config/prompts.py` 已提供：

- `PROMPT_MAPPING["conversation"]`
- `get_prompt(task_type: str, model_name: str = None, variant: str = None, **kwargs)`

我们采用 **variant / 伪 model_name 键** 来区分三种对话 Prompt：

- `conversation_default`：现有 login / 自由对话模式
- `conversation_review`：review 模式专用 Prompt
- `conversation_store`：store 模式专用 Prompt

实现方式示意：

- 在 `PROMPT_MAPPING["conversation"]` 中新增键：
  - `"conversation_review"` → REVIEW 模式 Prompt
  - `"conversation_store"` → STORE 模式 Prompt
- CLI 调用时，通过：
  - `get_prompt(task_type="conversation", model_name="conversation_review", ...)`
  - 或 `get_prompt(task_type="conversation", variant="conversation_review", ...)`
  来切换具体 Prompt。

### 3.2 review 模式 Prompt 要点

- 身份设定：
  - 你是“代码审查 / 多维分析专家”，优先做质量、安全、性能分析。
- 结构化输出倾向：
  - `code_analysis_tasks` 必须是主角，包含：
    - 目标路径（来自 CLI `-d`）
    - 分析类型列表（quality/security/performance）
    - 优先级
  - `db_tasks` 仅在用户明确说“记录 / 保存 / 写入 / 知识库”等意图时才非空。
- 行为约束：
  - 不要随意声称“已记录到数据库”，除非明确产生了结构化 `db_tasks`。

### 3.3 store 模式 Prompt 要点

- 身份设定：
  - 你是“知识库管理员 / 经验归纳专家”，优先帮助用户把信息整理并写入知识库。
- 结构化输出倾向：
  - `db_tasks` 是主角，包含：
    - `operation_type`（如 `record_issue_feedback`、`create_knowledge_item`）
    - `entity_type`（`KnowledgeBase` / `CuratedIssue` 等）
    - `payload`（包括摘要、路径、严重级别、标签等）
  - `code_analysis_tasks` 默认为空；
    - 只有用户明确希望“顺便分析代码 / 检查问题”时，才添加相应任务。
- 行为约束：
  - 允许在自然语言回答中说“已为你准备好写入任务”，但仍需通过后端实现真正写入；
  - 任务 JSON 必须稳定、可解析。

---

## 4. 用户沟通 Agent 行为切换

### 4.1 会话模式注入

在 `AIDrivenUserCommunicationAgent` 中，为每个会话维护一个 `mode` 字段，例如：

- `"default"`（login / 自由对话）
- `"review"`
- `"store"`

CLI 在创建会话或首次发消息前，将模式写入：

- `session_memory[session_id]["mode"] = "review" | "store"`

### 4.2 Prompt 选择逻辑

在 `process_ai_conversation(...)` 构建 Prompt 时：

- 读取当前会话的 `mode`；
- 根据模式选择不同的 `model_name` / `variant` 传给 `get_prompt`：
  - `mode == "review"` → `model_name="conversation_review"`
  - `mode == "store"` → `model_name="conversation_store"`
  - 否则使用默认。

### 4.3 路由偏好与 Fallback 协同

- review 模式：
  - 即便 TASK_PLAN_JSON 缺失，fallback 时也应该：
    - 更容易触发代码分析；
    - 不自动生成 DB 任务，除非检测到明确“记录/保存”意图。
- store 模式：
  - 即便 TASK_PLAN_JSON 缺失，也应：
    - 优先构造 `db_tasks`（类似当前 `_infer_db_tasks_from_message`）；
    - 不主动启动 `_start_code_analysis`。

---

## 5. 实施步骤规划

### 5.1 设计与文档阶段

- **步骤 1**：把本文件保存为 `api/cli_mode_prompt_design.md`（或等价命名），并纳入版本控制。
- **步骤 2**：在 `core/db_agent_integration_plan.md` 中简单引用本设计，说明 CLI 模式与 DB Agent 的关系（可选）。

### 5.2 CLI 命令层改造

- **步骤 3**：在 `mas.py` 或主 CLI 入口中：
  - 引入子命令 `review` 与 `store`；
  - 为每个子命令明确解析参数（路径/消息/文件）。
- **步骤 4**：为每个子命令创建独立的会话 ID 和模式：
  - `session_id = f\"cli_review_{timestamp}\"` / `cli_store_...`；
  - 在调用用户沟通 Agent 之前，写入 `mode`。

### 5.3 Prompt 与 Agent 集成

- **步骤 5**：在 `infrastructure/config/prompts.py` 中：
  - 为 `PROMPT_MAPPING["conversation"]` 新增 `conversation_review` / `conversation_store` 模板；
  - 内容基于第 3 节的要点编写。
- **步骤 6**：在 `AIDrivenUserCommunicationAgent.process_ai_conversation` 中：
  - 读取当前 `session_id` 的 `mode`；
  - 按模式选择相应 Prompt；
  - 保持 TASK_PLAN + fallback 逻辑兼容。

### 5.4 路由与测试

- **步骤 7**：为 `review` 模式编写测试/手动用例：
  - `mas review -d /path/to/api` → 触发代码分析 + 生成报告；
  - 确认不会误生成 DB 任务，除非用户明确要求写库。
- **步骤 8**：为 `store` 模式编写测试/手动用例：
  - `mas store --message \"...请存入知识库\"` → 只产生 `db_tasks`，不触发代码分析；
  - 在 Mock DB Agent 输出中验证结构化任务正确。
- **步骤 9**：更新 README 或单独 CLI 文档，说明三种用法：
  - `mas login`（自由对话/向后兼容）；
  - `mas review`（明确的代码评审）；
  - `mas store`（知识库存储）。

---

如果你希望，我可以在你确认后，按这份规划实际创建 `api/cli_mode_prompt_design.md` 文件并初步填入对应的 Prompt 模板和 Agent 接口改造草案。  

C++ 规则检查：已执行；当前仓库无 C++ 源文件，因此无相关问题可报告。