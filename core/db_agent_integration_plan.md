# MAS 数据库向量 Agent 分步实施计划

## 1. 背景与目标

根据 `infrastructure/database/design.md` 的架构设计，本计划分阶段引入一个专职的数据库向量/知识 Agent，并扩展现有用户沟通 Agent 的职责，使其能够识别和路由数据库相关需求。总体目标是：

> 用户沟通 Agent → 数据库向量 Agent → SQLite + Weaviate

在不破坏现有代码分析能力的前提下，让用户自然语言提出的“知识库 / 历史记录 / 语义检索”类需求，能够被正确拆分并落到数据库与向量索引层。

---

## 2. 角色职责划分

### 2.1 用户沟通 Agent（AIDrivenUserCommunicationAgent）

- 负责与用户进行自然语言对话；
- 解析用户需求，区分：
  - 代码质量 / 安全 / 性能分析任务；
  - 数据库相关任务（会话管理、知识库维护、语义检索等）；
- 输出结构化的任务规划：
  - `code_analysis_tasks`: 面向代码分析智能体的任务列表；
  - `db_tasks`: 面向 DB Agent 的任务列表；
- 将 `db_tasks` 转发给数据库向量 Agent（阶段一为 Mock，阶段三替换为真实实现）。

### 2.2 数据库向量 Agent（建议命名：DBVectorIndexAgent）

- 接收来自用户沟通 Agent 的 `db_tasks`；
- 依据 `infrastructure/database/design.md`：
  - 所有结构化数据写入、更新、删除首先操作 SQLite（事实库）；
  - 再根据需要同步到 Weaviate（语义索引层）；
- 对上层暴露语义化的数据库服务接口，例如：
  - 创建 / 查询审核会话（ReviewSession）；
  - 创建 / 查询人工确认的问题实例（CuratedIssue）；
  - 创建 / 更新 / 查询知识条目（IssuePattern / KnowledgeBase）；
  - 基于自然语言 / 向量的知识检索（KnowledgeItem / AnalysisSummary 等）。

### 2.3 其他分析 Agent（代码质量 / 安全 / 性能 / 静态扫描等）

- 保持现有职责，不直接感知 SQLite / Weaviate；
- 其分析结果可以在后续通过 用户沟通 Agent → DBVectorIndexAgent 的链路被写入数据库和向量索引。

---

## 3. 阶段一：扩展用户沟通 Agent Prompt + 引入 Mock DB Agent [x]

**目标**：在不改动真实数据库逻辑的前提下，让用户沟通 Agent 能够：

- 识别“普通代码分析请求”和“数据库相关请求”；
- 以结构化形式输出 DB 任务，并将其“转发”给一个 Mock DB Agent 接口。

### 3.1 更新用户沟通 Prompt（`infrastructure/config/prompts.py`）

- 在 `GENERAL_CONVERSATION_PROMPT` 或针对对话模型的 prompt 中，明确新增能力描述：
  - 解析用户需求后，需要区分：
    - 代码质量 / 安全 / 性能分析任务；
    - 数据库相关任务（会话管理、知识库维护、语义检索等）。
  - 要求模型输出一个可被程序解析的结构化结果（例如 JSON），其中包含：
    - `code_analysis_tasks`: 针对代码分析 Agent 的任务列表；
    - `db_tasks`: 针对 DB Agent 的任务列表（仅抽象描述，包含操作类型、实体、过滤条件、是否需要语义检索等）。
- 约定清晰的输出格式，例如：
  - 先正常回答用户；
  - 再在 `TASK_PLAN_JSON_START` / `TASK_PLAN_JSON_END` 标记之间输出 JSON。

### 3.2 在用户沟通 Agent 内部接入 Mock DB Agent 接口

- 在 `core/agents/ai_driven_user_communication_agent.py` 中：
  - 为 DB 相关任务预留一个调用接口（例如：`self.db_agent.handle_tasks(db_tasks=..., session_id=...)`）；
  - 当前阶段实现为 Mock：
    - 不真正触发 SQLite / Weaviate 操作；
    - 只记录调用日志（如打印到控制台）或返回固定的“模拟结果”，用于验证路由是否正确。
- Mock 接口的输入结构应与后续真实 DB Agent 的签名保持兼容（例如统一使用 `db_tasks: List[Dict[str, Any]]`）。

### 3.3 在配置中为 Mock DB Agent 预留占位

- 在 `infrastructure/config/ai_agent_config.json` / `infrastructure/config/ai_agents.py` 中：
  - 预留一个新的 Agent 类型条目（例如 `"db_vector_agent"`），指向 Mock 实现；
  - 便于在阶段三时将其替换为真实 DBVectorIndexAgent。

---

## 4. 阶段二：验证用户沟通 Agent 的意图识别与路由能力 [ ]

**目标**：在只存在 Mock DB Agent 的情况下，验证用户沟通 Agent 是否能按预期把请求正确拆分并路由。

### 4.1 覆盖核心场景的对话用例

- **纯代码分析请求**  
  示例：“帮我分析这个项目的代码质量，路径在 /path/to/project”  
  期望：只产生 `code_analysis_tasks`，`db_tasks` 为空。

- **纯数据库请求**  
  示例：“帮我把这次分析的结果保存到知识库里，以后类似问题能提前提示我。”  
  期望：产生合适的 `db_tasks`，`code_analysis_tasks` 为空或最小化。

- **混合请求**  
  示例：“分析这个仓库的安全问题，并把严重问题记入知识库，方便以后相似项目复用经验。”  
  期望：同时生成代码分析任务和 DB 任务。

### 4.2 验证结构化输出格式的稳定性

- 针对用户沟通 Agent 的调用结果编写单元测试或集成测试：
  - 断言 `db_tasks` 中包含：
    - 操作类型（如 `create_knowledge_item` / `search_similar_issues` 等）；
    - 目标实体（`ReviewSession` / `CuratedIssue` / `KnowledgeBase` / `VectorSearch` 等）；
    - 可选的 `semantic_query_text` 或“需要写入向量索引”的标志。
- 配合 Mock DB Agent，验证：
  - 是否收到了结构正确的 `db_tasks`；
  - 在日志或测试断言中可以清楚看到路由行为。

### 4.3 人工对话走查

- 从 CLI 或 Web 入口发起几轮真实对话，人工确认：
  - 用户体验仍然自然；
  - 在日志中可以清晰看到代码任务与 DB 任务的拆分与路由。

---

## 5. 阶段三：实现真实 DB Agent（DBVectorIndexAgent），并替换 Mock 实现 [ ]

**目标**：创建一个正式命名的 DB Agent，实现数据库服务接口，负责协调 SQLite 与 Weaviate，并替换阶段一中的 Mock 版本。

### 5.1 职责边界

- 接收来自用户沟通 Agent 结构化的 `db_tasks`；
- 依据 `infrastructure/database/design.md`：
  - 所有结构化数据写入、更新、删除首先操作 SQLite；
  - 再根据需要触发 Weaviate 向量对象的创建 / 更新 / 删除；
  - 遵循“SQLite 是权威源、Weaviate 是派生索引”的原则，保证最终一致性。

### 5.2 对上层暴露的接口设计

- 可以选择统一入口：
  - `handle_tasks(db_tasks: List[DbTask]) -> DbTaskResults`
- 或按实体细分出更语义化的方法，例如：
  - `create_review_session(...)`
  - `create_curated_issue(...)`
  - `create_or_update_knowledge_item(...)`
  - `semantic_search_knowledge(query_text, filters)`
  - `semantic_search_runs(...)`
- 内部实现中：
  - 调用 SQLite 的 Service/Repository 接口（如 `DatabaseService`）；
  - 根据记录类型和配置，决定是否调用 embedding / 向量逻辑；
  - 通过 Weaviate 适配层（如 `WeaviateVectorService`）写入或查询向量索引。

### 5.3 Embedding / 向量权重策略的实现方式

- 短期内可直接在 DBVectorIndexAgent 内部：
  - 基于 `design.md` 中的模板（如 `[ErrorType] {error_type} ...`）拼接 `text_payload`；
  - 调用已有的 `embed_fn` 或简单的 embedding 实现生成向量；
  - 写入 Weaviate 时显式传入 `vector` 字段。
- 中长期可以再抽象出独立的 Embedding Agent：
  - DBVectorIndexAgent 只负责“告诉它要为哪条记录生成什么类型的向量”；
  - Embedding Agent 负责模型选择、模板拼接与向量生成。

### 5.4 替换 Mock 实现

- 在配置层（`ai_agent_config.json` / `ai_agents.py`）中，将原有 Mock DB Agent 的实现指针替换为 DBVectorIndexAgent；
- 确保接口签名兼容，无需改动用户沟通 Agent 的调用代码。

---

## 6. 阶段四：端到端流程验证（用户沟通 Agent → DB Agent → 数据库） [ ]

**目标**：验证从用户输入到 SQLite + Weaviate 的完整链路是否按预期工作，并满足设计文档中的一致性与语义检索要求。

### 6.1 写入链路测试

- 场景示例：  
  用户通过对话描述一个新发现的错误模式，希望写入知识库，并在未来相似问题中被检索到。

- 预期：
  - 用户沟通 Agent 生成 `db_tasks`，包括创建 `KnowledgeBase` 条目与需要向量索引的标志；
  - DBVectorIndexAgent 调用 SQLite Service 创建记录，返回主键；
  - DBVectorIndexAgent 基于记录内容生成 embedding，调用 Weaviate 写入 `KnowledgeItem` 对象；
  - 日志中可以看到 SQLite 与 Weaviate 操作顺序正确，且主键对应一致。

### 6.2 查询 / 语义检索测试

- 场景示例：  
  用户用自然语言询问：“有没有类似的历史 SQL 超时问题经验？”

- 预期：
  - 用户沟通 Agent 将请求识别为语义检索，生成包含 `semantic_query_text` 的 `db_tasks`；
  - DBVectorIndexAgent 调用 embedding / Weaviate 执行 `nearVector` 或等效查询；
  - 返回的结果中包含正确的 `sqlite_id` / 关联主键；
  - DBVectorIndexAgent 再通过 SQLite 查询完整记录，汇总为结构化结果；
  - 用户最终在对话中看到基于历史知识的解释与建议。

### 6.3 异常与降级测试

- Weaviate 不可用时：
  - DBVectorIndexAgent 能否优雅降级为仅使用 SQLite 查询 / 写入；
  - 并向上层返回“向量索引暂不可用”的状态，而不是完全失败。
- SQLite 写失败时：
  - 不应继续写 Weaviate；
  - 应向上游返回错误，提示用户或调用方处理。

---

## 7. 测试与验收标准

- **阶段一 / 二**：
  - Prompt 输出的 JSON 结构稳定，字段名与类型固定；
  - Mock DB Agent 能持续收到结构化 `db_tasks`，不会因为模型“发挥”导致结构崩溃；
  - 用户对话体验未明显退化。
- **阶段三**：
  - DBVectorIndexAgent 所有接口具备基本单元测试（SQLite 写入、Weaviate 写入 / 更新 / 删除）；
  - 与 `vector_sync.py` / 现有 Weaviate 服务的交互符合设计。
- **阶段四**：
  - 写入 / 检索 / 异常三个路径的端到端用例全部通过；
  - 在 Weaviate 不可用时系统仍能工作（退化到仅 SQLite 模式）。

---

## 8. 后续演进方向（可选）

- 引入独立的 Embedding Agent，将向量策略从 DBVectorIndexAgent 中解耦；
- 为更多实体（如 run 级分析摘要、代码片段 CodeChunk）引入向量索引；
- 在用户沟通 Agent 中增加“确认 / 回滚”机制，用于危险的数据库写入操作。