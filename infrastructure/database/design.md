## MAS 数据库与向量检索设计总览

本设计说明了在 MAS 中如何同时使用 **SQLite** 和 **Weaviate**，以及分步落地路线，方便后续实现与维护。

---

## 一、总体架构：SQLite + Weaviate 的分工

- **SQLite：结构化“事实库” (Source of Truth)**  
  - 存储：`UserRequirement`（审核会话）、`AnalysisResult`（人工确认的问题实例）、`KnowledgeBase`（错误模式知识条目）等结构化记录。  
  - 职责：事务与一致性、审计与统计查询（按时间 / 会话 / 严重度 / 文件等维度聚合）。  

- **Weaviate：语义“索引库”（向量检索层）**  
  - 存储：从上述记录中抽取的文本 + 向量 (embedding) + 少量过滤字段（如 `severity`、`language`）。  
  - 职责：对“相似错误 / 相似代码 / 相似项目”的快速语义检索，然后通过主键回到 SQLite 获取完整结构化信息。  

> **设计原则：**  
> SQLite 是权威数据源，Weaviate 是从 SQLite 派生出来的语义索引；两者通过主键（如 `id` / `kb_code` / `run_id`）关联。

---

## 二、分步实施路线（建议按顺序推进）

**当前状态（TODO）**  
- [x] 步骤 1：完善 SQLite 的增删改查能力  
- [x] 步骤 2：测试 SQLite 的增删改查  
- [x] 步骤 3：引入 Weaviate 的基础 schema + CRUD（不含 embedding）  
- [x] 步骤 4：测试 Weaviate 的基础 CRUD（不含 embedding）  
- [x] 步骤 5：基于大模型 Agent 的 embedding 接口设计与最小实现  
- [x] 步骤 6：mock 大模型行为的 embedding 测试（Weaviate 向量写入与检索）  
- [x] 步骤 7：为 SQLite 拓展向 Weaviate 的同步接口（基于 embedding Agent）  
- [ ] 步骤 8：将数据库与向量接口开放给 MAS 并进行系统级测试（含语义质量评估）  

### 1. 完善 SQLite 的增删改查能力

- 围绕三类核心实体设计清晰的 CRUD 接口（在 `DatabaseService` 或独立 Repository 中实现）：
  - **审核会话 (`UserRequirement` / `review_sessions`)**  
    - 新建会话：保存用户问题、目标代码目录、补丁 / 提交信息等。  
    - 查询会话：按 `session_id` / 时间范围筛选。  
  - **人工确认的问题实例 (`AnalysisResult` / `curated_issues`)**  
    - 新建实例：绑定会话 + 代码位置 + 现象 / 原因 / 方案 + 严重度。  
    - 查询实例：按会话、文件路径、行号范围、状态等过滤。  
  - **错误模式知识条目 (`KnowledgeBase` / `issue_patterns`)**  
    - 新增 / 更新条目：`error_type`、`severity`、`error_description`、`problematic_pattern`、`solution`、`kb_code` 等。  
    - 查询：按类型、严重度、语言 / 框架、标签组合过滤。
- 明确删除策略：  
  - 对知识条目和问题实例优先使用“软删” (`status=deprecated/obsolete`)，必要时由后台任务做硬删清理。

### 2. 测试 SQLite 的增删改查

- 为三张表分别编写单元测试：  
  - 创建 → 查询 → 更新 → 删除 / 软删的完整闭环。  
- 编写跨表测试用例：  
  - 一个会话下创建多条问题实例，验证关系和级联删除是否符合预期。  
- 使用内存 SQLite 或独立测试数据库，避免污染生产数据。

### 3. 引入 Weaviate：schema + 基础 CRUD（不含 embedding）

  - 设计 Weaviate schema（概念示例）：  
    - `KnowledgeItem`：对应 `KnowledgeBase` 中的错误模式条目。  
      - 属性：`sqlite_id` / `kb_code`、`error_type`、`severity`、`created_at` 等。  
    - `AnalysisSummary`：对应 run / 文件级报告摘要。  
      - 属性：`run_id`、`readable_file`、摘要统计字段。  
    - （可选）`CodeChunk`：代码片段级别的索引。  
      - 属性：`file_path`、`language`、`run_id`、`start_line`、`end_line` 等。  
  - 为上述 class 实现基础 CRUD：创建 / 更新 / 删除 / 按主键查询。  
  - 明确此阶段 **不要求实现完整 embedding 策略**，只需保证：  
    - Weaviate 的 schema 中预留向量字段，由上层显式传入向量；  
    - 或在未传入向量时，允许使用简单的默认策略（如 `_build_issue_pattern_text` + `embed_fn`）作为临时兜底。  
  - 当前代码中，`WeaviateVectorService.create_knowledge_item(..., vector=...)` 已支持显式传入向量；如未传 `vector` 且配置了 `embed_fn`，会使用内部拼接文本生成默认向量，这可以视为 Step5 引入大模型 Agent 之前的占位实现。

### 4. 测试 Weaviate 的基础 CRUD（不含 embedding）

- 单元测试：  
  - 使用 mock Weaviate client，验证调用参数、错误处理和重试逻辑。  
- 集成测试：  
  - 启动本地或容器化 Weaviate 实例，执行小规模数据插入 / 更新 / 删除 / 语义检索：  
    - 验证更新是否覆盖旧向量；  
    - 删除后是否不可见；  
    - `nearVector/nearText` 返回的对象中主键字段是否正确。

### 5. 基于大模型 Agent 的 embedding 接口设计与最小实现

- **目标**：  
  - 不在业务代码中直接决定“哪些字段参与 embedding、怎么拼接文本、用哪个模型”；  
  - 而是将 SQLite / Weaviate 的接口统一暴露给一个专职的大模型 Agent，由它来：  
    - 决定如何从结构化记录生成 embedding 文本/向量；  
    - 决定何时对 Weaviate 进行插入 / 更新 / 删除等操作。
- **Agent 角色（示例：`DBEmbeddingAgent` / `KnowledgeEncodingAgent`）**：  
  - 输入：来自 SQLite 的结构化记录（`IssuePattern` / `CuratedIssue` / 报告摘要等）。  
  - 输出：  
    - `text_payload`：用于向量化的串联文本（带标签，如 `[ErrorType] {error_type}` 等）；  
    - `vector`：由大模型（或其工具）生成的 embedding 向量；  
    - 可选：需要对 Weaviate 做的 CRUD 操作描述（create / update / delete + 主键 / 过滤条件）。
- **接口设计建议**：  
  - 在基础设施层暴露一个“对大模型友好”的 API，例如：  
    - 向 Agent 提供 IssuePattern/CuratedIssue 的结构化视图与字段说明；  
    - 提供一个“应用索引操作”的接口：接收 Agent 规划好的 Weaviate 操作清单并执行。  
  - 职责划分：  
    - SQLite / Weaviate adapter：只负责“按指令执行 CRUD / 写入向量”；  
    - 大模型 Agent：负责“理解记录 + 决定 embedding 策略 + 规划 CRUD 操作”。  
- **当前阶段实现（无真实新模型时）**：  
  - 先用一个本地的 `MockDBEmbeddingAgent` 替代真实大模型：  
    - 内部采用固定的文本拼接模板（类似 `_build_issue_pattern_text`）；  
    - 调用选定的 embedding 模型或假 embedding 函数，生成向量；  
    - 返回“建议写入 Weaviate 的操作”（如 `create_knowledge_item` 所需字段与向量）。  
  - 未来接入真实大模型时，只需替换 Agent 实现，而不改动 SQLite/Weaviate 适配层。

### 6. mock 大模型行为的 embedding 测试（Weaviate 向量写入与检索）

- **单元测试（Agent 视角）**：  
  - 为 `MockDBEmbeddingAgent` 准备多组输入记录（不同 `error_type` / `severity` / 有无 `solution` 等），断言：  
    - 生成的 `text_payload` 覆盖所有关键字段，标签格式符合约定；  
    - 当字段缺失时不会抛异常，而是以空字符串占位；  
    - 返回的“操作清单”中，Weaviate CRUD 操作与主键/过滤条件正确。
- **集成测试（Weaviate + mock Agent）**：  
  - 使用测试 SQLite 数据库 / 内存数据，构造若干 `IssuePattern` / 报告摘要；  
  - 通过同步管线调用 `MockDBEmbeddingAgent`，拿到向量与 Weaviate 操作：  
    - 将数据/向量写入本地或容器化 Weaviate 实例；  
    - 使用 `nearVector/nearText` 做小规模检索，验证：  
      - 更新是否覆盖旧向量；  
      - 删除/失效标记是否生效；  
      - top-k 结果中包含预期的条目（简单语义 sanity check）。  
  - 此阶段不追求“最优语义效果”，重点是验证：  
    - **大模型 Agent → Weaviate 的协议设计是可行的**；  
    - 后续更换为真实大模型时，只需要替换 Agent，不动基础设施层。

### 7. 为 SQLite 拓展向 Weaviate 的同步接口（基于 embedding Agent）

- **原则：SQLite 先写，Weaviate 之后同步，最终一致性。**
- 建议引入一个清晰的同步层（如 `VectorIndexService` 或 outbox 表）：  
  - 写 SQLite 成功后，记录一条“待同步事件”（新增 / 更新 / 删除）；  
  - 后台 worker 依次读取事件：  
    - 组装用于 embedding 的文本；  
    - 生成向量；  
    - 调用 Weaviate 的对应写操作；  
    - 标记事件完成或重试。  
- 所有事件需保持幂等：同一事件重试不会破坏数据一致性。

### 8. 将数据库与向量接口开放给 MAS 并测试（含语义质量评估）

- 在 `infrastructure/database` 中仅暴露“语义化”的服务接口，而非 ORM / Weaviate 细节，例如：  
  - `ReviewSessionService`、`CuratedIssueService`、`KnowledgeBaseService`、`VectorSearchService`。  
- 在 MAS 各 Agent / CLI 中只依赖这些服务接口：  
  - 创建会话 / 记录问题实例 / 查询知识条目 / 语义搜索相似问题等。  
- 编写系统级集成测试：  
  - 从 CLI 发起一次真实的审核会话 → 人工录入问题实例和知识条目 → 再次分析时，新智能体可以利用历史记录给出预警。  
  - 在 Weaviate 不可用时，验证系统能优雅退化到仅使用 SQLite 的模式。

---

## 三、数据写入与查询工作流（结合 SQLite 与 Weaviate）

### 1. 写入 / 更新工作流（以 KnowledgeBase 为例）

1. **写入 SQLite（主流程）**  
   - 调用 `DatabaseService.update_knowledge_base(...)` 新增一条知识条目，获得主键 `id = K`。  
2. **派发向量同步任务（同步或异步）**  
   - 组装一段描述该条目的文本（见下文 embedding 策略）；  
   - 使用选定模型生成向量；  
   - 向 Weaviate 的 `KnowledgeItem` class 写入 object：  
     - `id` / `sqlite_id` 使用 `K` 或 `"knowledge_base:K"` 等规则；  
     - properties 含 `error_type`、`severity` 等；  
     - `vector` 字段保存 embedding。  
3. **更新 / 删除**  
   - 更新：先改 SQLite，再按同样主键更新 Weaviate object 的属性和向量；  
   - 删除：删除 SQLite 记录后，同步删除对应 Weaviate object，或设置 `is_deleted=true` 并在检索时过滤。  

同样的模式也可用于：  
- `AnalysisResult` 的 run / 文件级摘要；  
- 代码片段（`CodeChunk`）等。

### 2. 查询 / 检索工作流

#### 场景 A：知识库语义检索

1. 用户通过 CLI / Agent 提出自然语言问题（如“循环里的 SQL 超时问题，有没有类似经验？”）。  
2. 使用 embedding 模型将该问题转成向量 `v_query`。  
3. 在 Weaviate 的 `KnowledgeItem` 上执行 `nearVector/nearText` 查询，并可选择性添加过滤条件（如 `severity="high"`）。  
4. 得到 top-k 候选，每条包含 `sqlite_id`、`error_type`、`severity` 等；  
5. 使用 `sqlite_id` 回到 SQLite 查询完整 `KnowledgeBase` 记录或相关问题实例；  
6. AI Agent 基于这些记录生成最终人类可读答案。

#### 场景 B：历史 run 的相似项目检索

1. 对当前项目提取简介（README、文件结构、语言统计等），生成向量。  
2. 在 Weaviate 的 `AnalysisSummary` 上做 `nearVector` 检索，获取相似 run 的 `run_id`。  
3. 通过 `run_id` 在报告目录 / SQLite 中查出对应的 `run_summary.json` 和 `consolidated` 报告。  
4. 汇总“类似项目常见问题”，辅助当前分析或作为 Prompt 上下文。

---

## 四、Embedding 策略概要

### 0. 角色分工与当前阶段结论

- **WeaviateVectorService**：仅作为向量存储适配器，负责 schema、对象 CRUD 与向量读写，不负责“业务上哪些字段更重要”。  
- **embedding 生成**：由外部注入的 `embed_fn` 或上层服务显式传入 `vector` 完成，Weaviate 只负责保存与检索。  
- **知识编码服务 / Agent（建议引入）**：  
  - 从用户沟通 Agent 与各分析 Agent 接收结构化摘要（如 `error_type`、`severity`、`description`、`pattern`、`solution` 等）；  
  - 根据配置决定哪些字段参与向量文本、哪些仅作为过滤字段，并选择合适的 embedding 模型；  
  - 统一调用 `WeaviateVectorService.create_knowledge_item(..., vector=vector)` 等接口，将“经过业务权重设计”的向量写入 Weaviate。  
- **当前阶段策略**：  
  - Weaviate 层已支持通过 `create_knowledge_item(..., vector=...)` 使用上层提供的向量；  
  - `_build_issue_pattern_text` 作为默认/兜底策略存在，但后续推荐使用独立的“知识编码服务”来集中管理 embedding 策略与权重。

### 1. 总体原则

- 尽量为“可互相对比”的内容使用 **统一的 embedding 模型与维度**，便于跨实体检索。  
- 单条记录可以有一个或多个 embedding 视角：  
  - 如错误模式可拆为“问题向量”和“解决方案向量”。  
- 文本结构尽量“自带上下文”：  
  - 例如：`[error_type=SQLInjection][severity=high] 这里是错误描述...`，有助于模型理解语义。

### 2. 针对具体表的策略

- **KnowledgeBase / issue_patterns**（错误模式 + 解决方案）  
  - 简单起步：每条记录生成一个向量，拼接文本示例：  
    - `[ErrorType] {error_type}`  
    - `[Severity] {severity}`  
    - `[Description] {error_description}`  
    - `[Pattern] {problematic_pattern}`  
    - `[Solution] {solution}`  
  - 进阶：  
    - `problem_vector`：`error_type + error_description + problematic_pattern`；  
    - `solution_vector`：`solution + error_description`；  
    - 可在 Weaviate 中用两个 class 或一个 class 多向量字段来承载。

- **分析结果 / 报告摘要（run / 文件级）**  
  - run 级 summary 文本示例：  
    - `[ProjectPath] {target_directory}`  
    - `[RunSummary] 严重问题: {critical_count}, 高: {high_count}, 中: {medium_count}, 低: {low_count}`  
    - `[TopIssues] 1) {issue_1} 2) {issue_2} ...`  
  - 文件级 consolidated：  
    - `[File] {readable_file}`  
    - `[Issues] - ({severity}) {description} at line {line} - ...`  

- **代码片段（可选）**  
  - 对长代码进行 chunking：如每 50–150 行一片，10–20 行重叠。  
  - 文本模板：  
    - `[File] path/to/file.py`  
    - `[Language] python`  
    - `[ChunkIndex] 3`  
    - `[Code] ...代码片段...`  
  - 每个 chunk 生成 `code_vector`，用于“相似实现 / 相似 bug 模式”检索。

### 3. 模型选择与调用

- 代码相关（代码片段、带代码的错误描述）：  
  - 可使用 `microsoft/codebert-base` 或其他 code 专用模型（如 CodeT5、code-sentence-transformers）。  
- 以自然语言为主（中文错误描述、报告摘要）：  
  - 可使用中文表现较好的 sentence-transformer，或在 Qwen 上增加 embedding 头。  
- 调用方式：  
  - 若用 Weaviate text2vec 模块：只需传原始文本，让 Weaviate 自动生成向量；  
  - 若在 MAS 内部统一 embedding：使用 Transformers 生成向量，再通过 Weaviate 的自带向量模式写入 `vector` 字段。

---

## 五、读写流程小结

- **写链路**  
  1. Agent / 服务层首先调用 `DatabaseService` 写入 SQLite；  
  2. 写成功后，向 `VectorIndexService` 或 outbox 表记录一条同步任务；  
  3. 后台任务：加载记录 → 组装文本 → 生成 embedding → 写入 / 更新 / 删除 Weaviate object。  

- **读链路**  
  - 普通列表和统计：直接查询 SQLite；  
  - 语义检索场景（相似错误 / 相似项目 / 相似代码）：  
    1. 在 Weaviate 上做向量检索得到候选主键；  
    2. 用主键回 SQLite 查询结构化数据；  
    3. 由 MAS 的智能体组合结果生成最终回答或预警。  

这样可以在保证 **SQLite 的可靠性与完整性** 的前提下，引入 **Weaviate 的语义检索能力**，并通过渐进式步骤完成从“本地 CRUD”到“多智能体 + 语义知识库”的演进。 


