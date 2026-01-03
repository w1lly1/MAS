from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

Base = declarative_base()

# 从用户问题 / 审核会话角度出发
class ReviewSession(Base):
    """
    用户发起的一次代码审核 / 问题求助会话。
    主要用于记录本次人工审核的上下文信息（是谁、在哪个项目、什么时候提的问题）。
    """

    # 物理表名：后续可以通过 Alembic 等方式做迁移
    __tablename__ = "review_sessions"

    id = Column(Integer, primary_key=True, index=True)  # 内部自增主键
    session_id = Column(String(255), index=True)  # CLI 会话标识或用户标识，用于将多次请求串联起来
    user_message = Column(Text)  # 用户最初的自然语言问题描述或审核需求说明
    code_directory = Column(String(500))  # 需要审核的代码根目录或本地路径
    code_patch = Column(Text, nullable=True)  # 与本次审核相关的补丁内容（如 git diff），可选
    git_commit = Column(String(100), nullable=True)  # 触发本次审核的基线 commit（如 HEAD 的 SHA）
    status = Column(String(50), default="open")  # 会话状态：open / in_review / completed / cancelled 等
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )  # 记录创建时间（会话发起时间）
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )  # 最近一次更新会话记录的时间

    # 关系：一个会话可以产生多个人工确认的问题实例（ CuratedIssue ）
    curated_issues = relationship(
        "CuratedIssue",
        back_populates="session",
        cascade="all, delete-orphan",
    )


# 从人工确认的分析结果角度出发 —— 具体问题实例，绑定到某段代码
class CuratedIssue(Base):
    """
    人工确认的“问题实例”记录：
    - 发生在某个项目的某个文件、某段代码上
    - 包含问题现象、根因分析和本次解决方案
    未来用于在代码变更时对比历史问题，做潜在风险提醒。
    """

    __tablename__ = "curated_issues"

    id = Column(Integer, primary_key=True, index=True)  # 内部自增主键

    # 关联维度
    session_id = Column(
        Integer,
        ForeignKey("review_sessions.id"),
        index=True,
        nullable=False,
    )  # 关联的用户审核会话（ReviewSession.id）
    pattern_id = Column(
        Integer,
        ForeignKey("issue_patterns.id"),
        index=True,
        nullable=True,
    )  # 可选：关联到抽象错误模式知识条目（IssuePattern.id）

    # 代码定位信息
    project_path = Column(
        String(500), nullable=True
    )  # 当时项目路径/仓库说明（冗余保存，便于跨会话检索）
    file_path = Column(
        String(1000), index=True
    )  # 发现问题的源文件相对路径（相对于 code_directory / repo 根）
    start_line = Column(Integer)  # 问题相关代码片段的起始行号（包含）
    end_line = Column(Integer)  # 问题相关代码片段的结束行号（包含）
    code_snippet = Column(
        Text
    )  # 发现问题时的代码片段快照（用于后续做向量化匹配和历史对比）

    # 人工总结的三要素
    problem_phenomenon = Column(Text)  # 问题现象：在什么场景下表现出什么异常 / 风险
    root_cause = Column(Text)  # 导致问题的根本原因分析（结合业务 / 架构 / 实现）
    solution = Column(Text)  # 针对本次问题实例采取的具体解决方案

    # 元信息
    severity = Column(
        String(50)
    )  # 问题严重程度：critical / high / medium / low / info 等
    status = Column(
        String(50), default="open"
    )  # 当前问题状态：open / resolved / obsolete 等
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )  # 记录创建时间（第一次被登记为问题的时间）
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )  # 最近一次更新该问题实例记录的时间

    # ORM 关系：回指会话与模式（便于在 Python 端导航）
    session = relationship("ReviewSession", back_populates="curated_issues")
    pattern = relationship("IssuePattern", back_populates="issue_instances")


# 错误模式 + 解决方案 —— 可跨项目复用的知识条目（模式级知识库）
class IssuePattern(Base):
    """
    抽象层面的“错误模式 + 解决方案”知识条目：
    - 总结某类常见问题的共性现象、根因与通用解决方案
    - 可以被多个具体问题实例（AnalysisResult）引用和复用
    未来可与 Weaviate 等向量数据库结合，用于语义检索和模式匹配。
    """

    __tablename__ = "issue_patterns"

    # 内部自增主键
    id = Column(Integer, primary_key=True, index=True)
    # 错误模式的简短标题，便于在列表/对话中展示
    title = Column(String(255))

    # 分类与适用范围
    # 错误类型分类，例如 SQLInjection / XSS / PerformanceLoop 等
    error_type = Column(
        String(255), index=True
    )
    # 推荐的严重程度，用于排序和筛选
    severity = Column(
        String(50), index=True
    )
    # 主要适用的编程语言，如 python / java / cpp 等
    language = Column(
        String(50), nullable=True
    )
    # 主要适用的框架，如 django / spring / fastapi 等
    framework = Column(
        String(100), nullable=True
    )

    # 模式与解决方案描述
    # 对该错误模式的详细描述（从现象和风险角度进行说明）
    error_description = Column(
        Text
    )
    # 典型的易出问题代码写法 / 模式（可包含伪代码/正则/结构描述）
    problematic_pattern = Column(
        Text
    )
    # 通用解决方案建议（可包括步骤、代码示例、注意事项等）
    solution = Column(
        Text
    )

    # 兼容旧有使用场景的匹配字段（可用于文件/类名的简单模式匹配）
    # 匹配文件路径的简单模式（如 "*/views/*.py"），可选
    file_pattern = Column(
        String(255), nullable=True
    )
    # 匹配类名/模块名的简单模式（如 "*Service"），可选
    class_pattern = Column(
        String(255), nullable=True
    )

    # 额外标签信息，建议以 JSON 字符串或逗号分隔字符串形式存储
    tags = Column(
        Text, nullable=True
    )
    # 知识条目状态：active / deprecated / draft 等
    status = Column(
        String(50), default="active"
    )
    # 条目创建时间（首次录入知识库）
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    # 最近一次更新该知识条目的时间
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # 关系：一个模式可以对应多个具体问题实例
    issue_instances = relationship(
        "CuratedIssue",
        back_populates="pattern",
        cascade="all, delete-orphan",
    )


