# 强化用户沟通模型识别方案

## 1. 目标

本方案只聚焦 MAS 的用户沟通模型识别能力，不修改 DB 执行层语义、不重做删除事务、不扩展到数据库存储实现。目标是让用户沟通模型在 LLM 输出不稳定的情况下，仍能稳定识别三类结果：

- `db`
- `code`
- `unknown`

并且在识别层完成以下能力：

- 保持裸 JSON 零翻译输出
- 对不稳定输出进行验证与修复
- 使用规则而不是 explanation 驱动路由
- 对多意图冲突进行安全澄清
- 对识别结果进行可观测记录
- 通过回归测试锁定不稳定输出场景

## 2. 范围

### 2.1 包含内容

- `GENERAL_CONVERSATION_PROMPT` 的输出 contract 收敛
- `AIDrivenUserCommunicationAgent` 的解析、归一化、修复与路由规则
- 用户沟通层的 observability 日志
- 识别能力相关的单元测试与少量真实模型回归测试

### 2.2 明确排除

- DB 执行层
- 删除动作事务
- 删除确认状态机
- Weaviate / SQLite 数据写入实现
- 任务持久化结构重构
- 其它智能体的业务逻辑改造

## 3. 设计原则

### 3.1 外稀疏、内规范

- 外层给模型：保持输出尽量稀疏，避免多余字段污染语义
- 内层给系统：进入 Python 后统一 canonicalize 为权威结构，再进行验证与路由

### 3.2 路由必须规则化

- 路由不能依赖 explanation 文案
- 路由必须只依据 canonical contract 字段
- `intent`、任务存在性、冲突状态决定 next_action

### 3.3 安全优先

- 多意图冲突必须澄清
- 解析失败不能默默误路由
- 不稳定输出应尽量修复，但不能扩大语义
- 路径类字段只识别，不补全绝对路径

## 4. Canonical Contract

### 4.1 外层输出 contract

用户沟通模型输出必须为裸 JSON，并且只允许以下顶层字段：

- `intent`
- `db_tasks`
- `code_analysis_tasks`
- `explanation`

### 4.2 `intent` 取值

只允许：

- `db`
- `code`
- `unknown`

任何别名都必须归一为上述值，例如：

- `database` -> `db`
- `sql` -> `db`
- `data` -> `db`

### 4.3 任务字段原则

- `db_tasks`：只表达数据库意图相关的候选信息
- `code_analysis_tasks`：只表达代码分析意图相关的候选信息
- `explanation`：只用于简要说明，不参与路由决策

### 4.4 冲突原则

如果 `db_tasks` 和 `code_analysis_tasks` 同时存在，或语义明显混合，则：

- `intent` 置为 `unknown`
- 清空可执行任务
- 进入澄清分支

## 5. LLM 层策略

### 5.1 Prompt 目标

Prompt 的作用是让模型“提议结构化意图”，不是让模型决定最终动作。

### 5.2 Prompt 约束

- 输出必须是裸 JSON
- 顶层字段必须稳定
- 不允许输出 `database` 等别名
- 多意图冲突时输出 `unknown`
- 路径字段不要求模型做文件系统补全

### 5.3 Prompt 角色边界

Prompt 只负责：

- 意图识别
- 候选任务提议
- 基本说明

Prompt 不负责：

- 最终路由
- 安全决策
- 执行确认
- 语义扩写

## 6. Python 侧识别管线

### 6.1 总体流程

用户输入进入 `AIDrivenUserCommunicationAgent` 后，识别流程应按以下顺序进行：

1. 调用模型生成原始输出
2. 尝试严格解析 JSON
3. 对解析失败做受控修复
4. 对修复后结果做 canonicalize
5. 对任务与 intent 做规则校验
6. 对冲突做安全回退
7. 用规则函数决定 next_action
8. 记录结构化观测日志

### 6.2 解析层

解析层需要处理：

- 正常 JSON
- 带 Markdown 围栏的 JSON
- 缺少引号的轻微损坏输出
- 噪声前后缀
- 形如 `intent: database` 的半结构化输出

### 6.3 修复层

修复层允许的动作只有：

- 字段归一
- intent 别名归一
- 轻量文本兜底推断
- 安全冲突回退

修复层禁止：

- 擅自增加任务语义
- 擅自把 unknown 升级为确定执行
- 擅自补全路径到绝对路径
- 擅自把 explanation 当作路由依据

### 6.4 路由层

路由必须由规则函数决定：

- `intent == db` -> DB 路由
- `intent == code` -> 代码分析路由
- `intent == unknown` -> 澄清
- 任务存在但 intent 缺失时，可用任务存在性作为次级规则
- explanation 不参与最终动作判定

## 7. 观测与诊断

### 7.1 必须记录的路由信息

- `session_id`
- `intent_before_repair`
- `intent_after_repair`
- `next_action`
- `route_reason`
- `repair_applied`
- `repair_reasons`
- `mixed_intent_detected`

### 7.2 观测目标

要能区分以下三类问题：

- 模型原始输出问题
- 解析器问题
- 路由规则问题

### 7.3 观测原则

- 日志必须结构化
- 日志必须能回放
- 日志必须能支持回归定位
- 不要依赖人工阅读自然语言日志判断核心状态

## 8. 规则要求

### 8.1 多意图冲突

如果一个输入同时包含代码分析与数据库意图：

- 不自动并行
- 不自动优先级裁决
- 直接澄清

### 8.2 路径处理

识别层不做以下事情：

- 不补全相对路径为绝对路径
- 不替用户猜测目录根
- 不把路径识别和文件系统探测混在一起

### 8.3 安全回退

当出现以下情况时必须保守处理：

- JSON 解析失败且无法可靠修复
- intent 无法归一
- 混合任务冲突
- 输出内容与 contract 冲突

保守处理的默认动作是：

- `intent = unknown`
- 清空任务
- 提示用户补充信息

## 9. 测试策略

### 9.1 单元测试

必须覆盖：

- `database` / `db` / `sql` 的 intent 归一
- 非严格 JSON 的解析修复
- 路由不依赖 explanation
- 混合意图冲突澄清
- 路径不补全策略

### 9.2 真实模型回归

保留少量真实模型样本，用于验证：

- Qwen 输出 `database` 时能稳定落到 `db`
- 输出缺字段时能安全修复
- 输出混合语义时能澄清
- 输出损坏 JSON 时不会路由中断

### 9.3 回归门禁

识别能力改动必须满足：

- 单测通过
- 真实模型回归通过
- 路由日志可观测
- contract 一致性检查通过

## 10. 推荐实现文件

- `core/agents/ai_driven_user_communication_agent.py` — 解析、修复、路由、观测
- `infrastructure/config/prompts.py` — conversation prompt 的 canonical contract
- `tests/agents/test_ai_driven_user_communication_agent.py` — 识别回归
- `tests/integration/` — 可选的链路回归

## 11. 实施顺序

### Phase A
先冻结 contract 和边界，确认外层输出仍保持裸 JSON 零翻译。
固定 user_comm 的外层 contract：只允许 intent、db_tasks、code_analysis_tasks、explanation 四个顶层字段。

明确双层职责：LLM 只提议意图与候选任务，Python 侧负责验证、修复、路由与安全回退。

明确排除范围：不改 DB 执行层、不改删除事务、不改确认状态机。

将冲突策略定为“强制澄清”，避免多意图自动并行或自动优先级裁决。

将路径策略定为“只识别不补全”，识别层不做绝对路径推断。
### Phase B
收敛 prompt，明确 intent 归一规则和冲突澄清规则。
收紧 infrastructure/config/prompts.py 中的 GENERAL_CONVERSATION_PROMPT，保持裸 JSON 输出。

继续强制 intent 只允许 db|code|unknown，并归一 database/sql/data 为 db。

补上多意图与不确定输入的明确指引：冲突时输出 unknown 并要求用户补充。

保持 prompt 作为弱约束，不把正确性押在 prompt 稳定性上。
### Phase C
重构 Python 侧解析、修复、canonicalize、路由函数。
在 core/agents/ai_driven_user_communication_agent.py 里把解析、归一化、修复、路由收敛成一条明确链路。

将 _parse_task_plan_from_response、_normalize_task_plan、_extract_intent_fallback、_validate_and_repair_task_plan 组合成 validate-and-repair pipeline。

将路由决策改成纯规则函数，不再依赖 explanation 文案。

遇到混合任务或冲突任务时，统一进入澄清分支，不让系统自行猜测。

保留最小容错，但只做合同修补，不做语义扩写。
### Phase D
增加结构化 observability 日志。
增加结构化路由日志，记录 session_id、原始 intent、修复后 intent、next_action、repair_reasons、冲突标记。

增加识别失败率、修复命中率、澄清分支命中率的可观察信号。

区分“模型原输出问题”和“代码修复问题”，方便后续定位。

保持日志足够可追踪，但避免把所有原文无节制扩散到每条路径。
### Phase E
补充单元测试与真实模型回归样本。
在 tests/agents/test_ai_driven_user_communication_agent.py 增加 intent 别名归一测试，覆盖 database、db、sql。

增加非严格 JSON 测试，覆盖无引号、缺字段、围栏、噪声前后缀。

增加规则路由测试，确认 explanation 里即使出现“数据库”字样，也不会改变 next_action。

增加混合意图测试，确认同时出现代码与数据库语义时必须澄清。

增加路径不补全测试，确认 user_comm 只识别，不做绝对路径推断。

保留少量真实模型回归，用来验证 Qwen 输出漂移下的稳定识别。
### Phase F
发布门禁

把识别能力改动的门禁固定为单测 + 少量真实模型回归。

真实模型样本固定为核心 case，不把 DB 执行链路纳入本次门禁。

先保证识别稳定，再考虑未来是否引入 dry-run 模式。
## 12. 预期结果

完成后，系统应该具备以下行为：

- `intent=database` 不会再丢路由
- explanation 再强也不能改变 next_action
- 混合意图会被安全澄清
- 损坏 JSON 不会直接导致继续对话
- 路由过程可被日志追踪和测试锁定

## 13. 非目标

本方案不解决：

- DB 删除执行实现
- 删除确认状态机
- 数据持久化优化
- 多智能体架构重构
- 模型替换
