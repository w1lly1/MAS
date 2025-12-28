## MAS 数据库 ER 图与 Weaviate 映射概览

本文件总结当前 SQLite 表结构（ER）以及与 Weaviate 向量索引之间的映射关系，方便后续维护与同步设计（Step 5 及以后）。

---

## 一、SQLite ER 关系图

当前 SQLite 使用三张核心表：`review_sessions`、`curated_issues`、`issue_patterns`。

### 1. ER 文字图

```text
┌──────────────────────┐            ┌──────────────────────┐
│   review_sessions    │ 1       *  │    curated_issues    │
│   (审核会话)          │──────────▶│   (问题实例)          │
└──────────────────────┘            └──────────────────────┘
                                      ▲
                                      │ *
                                      │
                                      │ 1
                              ┌──────────────────────┐
                              │    issue_patterns    │
                              │    (错误模式知识条目)  │
                              └──────────────────────┘
```

### 2. 各表字段摘要

#### 表：`review_sessions`

- **主键与标识**
  - `id: Integer, PK`
  - `session_id: String(255), index`
- **内容与上下文**
  - `user_message: Text`
  - `code_directory: String(500)`
  - `code_patch: Text, nullable`
  - `git_commit: String(100), nullable`
- **状态与时间**
  - `status: String(50), default="open"`
  - `created_at: DateTime(timezone=True), default=UTC now`
  - `updated_at: DateTime(timezone=True), default+onupdate=UTC now`
- **关系**
  - `curated_issues`：一对多，指向 `curated_issues.session_id`

#### 表：`curated_issues`

- **主键**
  - `id: Integer, PK`
- **外键 / 关联**
  - `session_id: Integer, FK -> review_sessions.id, index, not null`
  - `pattern_id: Integer, FK -> issue_patterns.id, index, nullable`
- **代码定位信息**
  - `project_path: String(500), nullable`
  - `file_path: String(1000), index`
  - `start_line: Integer`
  - `end_line: Integer`
  - `code_snippet: Text`
- **问题三要素**
  - `problem_phenomenon: Text`
  - `root_cause: Text`
  - `solution: Text`
- **元信息**
  - `severity: String(50)`
  - `status: String(50), default="open"`
  - `created_at: DateTime(timezone=True), default=UTC now`
  - `updated_at: DateTime(timezone=True), default+onupdate=UTC now`

#### 表：`issue_patterns`

- **主键与标识**
  - `id: Integer, PK`
  - `kb_code: String(50), unique, index, nullable`
  - `title: String(255)`
- **分类与适用范围**
  - `error_type: String(255), index`
  - `severity: String(50), index`
  - `language: String(50), nullable`
  - `framework: String(100), nullable`
- **模式与解决方案描述**
  - `error_description: Text`
  - `problematic_pattern: Text`
  - `solution: Text`
- **简单匹配字段**
  - `file_pattern: String(255), nullable`
  - `class_pattern: String(255), nullable`
- **其他元信息**
  - `tags: Text, nullable`
  - `status: String(50), default="active"`
  - `created_at: DateTime(timezone=True), default=UTC now`
  - `updated_at: DateTime(timezone=True), default+onupdate=UTC now`
- **关系**
  - `issue_instances`：一对多，指向 `curated_issues.pattern_id`

---

## 二、IssuePattern 与 Weaviate 的映射

Weaviate 目前只为 `issue_patterns` 提供一个主索引类 `KnowledgeItem`，用于语义检索。

### 1. Weaviate Class：`KnowledgeItem`

由 `WeaviateVectorService.ensure_knowledge_schema()` 创建，核心 schema 如下：

```text
class: KnowledgeItem
vectorizer: "none"   # 不自动生成向量，由客户端传入

properties:
  - sqlite_id: int          # 映射到 SQLite issue_patterns.id
  - kb_code: text           # 映射 issue_patterns.kb_code
  - error_type: text        # 映射 issue_patterns.error_type
  - severity: text          # 映射 issue_patterns.severity
  - status: text            # 映射 issue_patterns.status
  - language: text          # 映射 issue_patterns.language
  - framework: text         # 映射 issue_patterns.framework
```

### 2. 字段映射一览

| SQLite 表/字段                 | Weaviate 属性      | 说明                                      |
|--------------------------------|--------------------|------------------------------------------|
| `issue_patterns.id`            | `sqlite_id`        | 主键映射，用于回到 SQLite                 |
| `issue_patterns.kb_code`       | `kb_code`          | 知识条目标识                              |
| `issue_patterns.error_type`    | `error_type`       | 错误类型分类                              |
| `issue_patterns.severity`      | `severity`         | 严重度，用于检索过滤/排序                  |
| `issue_patterns.status`        | `status`           | active/deprecated/draft 等               |
| `issue_patterns.language`      | `language`         | 主要适用语言                              |
| `issue_patterns.framework`     | `framework`        | 主要适用框架                              |
| （多字段拼接为文本）           | 向量字段 (vector)  | 参与 `_build_issue_pattern_text` 生成文本    |

### 3. 向量生成路径

- 由 `WeaviateVectorService.create_knowledge_item(...)` 负责写入：
  - 如调用方提供 `vector: List[float]`：
    - 直接写入 Weaviate，不再拼接文本。
  - 如未提供向量但配置了 `embed_fn`：
    - 使用 `_build_issue_pattern_text(props)` 将以下字段拼接为文本：
      - `kb_code`, `error_type`, `severity`, `language`, `framework`,
      - `error_description`, `problematic_pattern`, `solution`
    - 将拼接文本传给 `embed_fn` 获得 embedding，再写入 Weaviate。

### 4. 主键关联与检索流程

- **写入时**
  1. 在 SQLite 的 `issue_patterns` 中创建或更新记录，获得 `id = K`。
  2. 调用 `create_knowledge_item(sqlite_id=K, ...)` 在 Weaviate 中写入对应 `KnowledgeItem`。
- **检索时**
  1. 在 Weaviate 中基于向量执行检索，得到若干 `KnowledgeItem`（含 `sqlite_id` 等属性）。
  2. 使用 `sqlite_id` 回到 SQLite 的 `issue_patterns` 表中查询完整记录及其关联的 `curated_issues`。

---

## 三、后续扩展建议（简要）

- 如需为 `curated_issues`、run 级报告、代码片段等建立向量索引，可新增 Weaviate Class：
  - 例如：`AnalysisSummary`、`CodeChunk` 等，并遵循与 `sqlite_id`/`run_id` 类似的主键关联策略。
- 当前文件可以作为 Step 5（SQLite → Weaviate 同步层）实现时的参考基础文档。 


