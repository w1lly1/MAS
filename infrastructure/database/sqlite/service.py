from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import Base, ReviewSession, CuratedIssue, IssuePattern

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = BASE_DIR / "infrastructure" / "database" / "mas.db"


class DatabaseService:
    def __init__(self, database_url: str = f"sqlite:///{DEFAULT_DB_PATH}"):
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ====== ReviewSession CRUD ======
    async def create_review_session(
        self,
        session_id: str,
        user_message: str,
        code_directory: str,
        code_patch: Optional[str] = None,
        git_commit: Optional[str] = None,
        status: str = "open",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> int:
        """
        创建一条审核会话（ReviewSession）记录。

        当前实现仅填充 `ReviewSession` 的部分关键字段：
        - session_id: CLI 会话或用户标识
        - user_message: 用户原始问题描述
        - code_directory: 本次审核的目标代码目录
        - code_patch/git_commit: 可选的补丁或基线提交信息

        后续如需支持 tags 等字段，可以在调用方或这里扩展。
        """
        with self.get_session() as db:
            session = ReviewSession(
                session_id=session_id,
                user_message=user_message,
                code_directory=code_directory,
                code_patch=code_patch,
                git_commit=git_commit,
                status=status,
            )
            # 如有需要，允许外部覆盖默认时间戳
            if created_at is not None:
                session.created_at = created_at
            if updated_at is not None:
                session.updated_at = updated_at

            db.add(session)
            db.commit()
            db.refresh(session)  # 刷新以获取自增主键等信息
            return session.id

    async def get_review_sessions(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        列出审核会话（ReviewSession）列表。

        - 可按状态筛选（如 open / completed）
        - 支持简单的分页（limit / offset）
        返回值为字典列表，便于在上层序列化或展示。
        """
        with self.get_session() as db:
            query = db.query(ReviewSession)
            if status:
                query = query.filter(ReviewSession.status == status)
            query = query.order_by(ReviewSession.created_at.desc())
            if offset:
                query = query.offset(offset)
            if limit is not None and limit > 0:
                query = query.limit(limit)
            sessions = query.all()
            return [
                {
                    "id": item.id,
                    "session_id": item.session_id,
                    "user_message": item.user_message,
                    "code_directory": item.code_directory,
                    "status": item.status,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                }
                for item in sessions
            ]

    async def get_review_session_by_id(self, db_id: int) -> Optional[Dict[str, Any]]:
        """按主键获取单条 ReviewSession。"""
        with self.get_session() as db:
            item = (
                db.query(ReviewSession)
                .filter(ReviewSession.id == db_id)
                .one_or_none()
            )
            if not item:
                return None
            return {
                "id": item.id,
                "session_id": item.session_id,
                "user_message": item.user_message,
                "code_directory": item.code_directory,
                "status": item.status,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }

    async def get_review_session_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """按 session_id 获取单条 ReviewSession（用于去重）。"""
        with self.get_session() as db:
            item = (
                db.query(ReviewSession)
                .filter(ReviewSession.session_id == session_id)
                .order_by(ReviewSession.created_at.desc())
                .first()
            )
            if not item:
                return None
            return {
                "id": item.id,
                "session_id": item.session_id,
                "user_message": item.user_message,
                "code_directory": item.code_directory,
                "status": item.status,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }

    async def update_review_session_status(
        self,
        db_id: int,
        status: str,
    ) -> bool:
        """
        更新审核会话的状态字段（软更新，不删除记录）。

        返回是否成功更新（未找到记录则返回 False）。
        """
        with self.get_session() as db:
            session_obj = (
                db.query(ReviewSession)
                .filter(ReviewSession.id == db_id)
                .one_or_none()
            )
            if not session_obj:
                return False
            session_obj.status = status
            db.commit()
            return True

    async def delete_review_session(self, db_id: int) -> bool:
        """
        硬删一条审核会话记录（ReviewSession），同时依赖 ORM 级联删除相关的 CuratedIssue。

        返回是否成功删除（未找到记录则返回 False）。
        """
        with self.get_session() as db:
            session_obj = (
                db.query(ReviewSession)
                .filter(ReviewSession.id == db_id)
                .one_or_none()
            )
            if not session_obj:
                return False
            db.delete(session_obj)
            db.commit()
            return True

    async def delete_all_review_sessions(self) -> int:
        """硬删全部 ReviewSession（级联删除 CuratedIssue）。"""
        with self.get_session() as db:
            count = db.query(ReviewSession).delete()
            db.commit()
            return int(count or 0)

    # ====== CuratedIssue CRUD ======
    async def create_curated_issue(
        self,
        session_id: int,
        file_path: str,
        start_line: int,
        end_line: int,
        code_snippet: str,
        problem_phenomenon: str,
        root_cause: str,
        solution: str,
        severity: str = "medium",
        status: str = "open",
        project_path: Optional[str] = None,
        pattern_id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> int:
        """
        创建一条人工确认的问题实例（CuratedIssue）。

        该记录绑定到：
        - 某个审核会话（session_id）
        - 某个文件的特定行号区间（file_path, start_line, end_line）
        并记录问题现象 / 根因 / 解决方案等信息。
        """
        with self.get_session() as db:
            issue = CuratedIssue(
                session_id=session_id,
                pattern_id=pattern_id,
                project_path=project_path,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code_snippet=code_snippet,
                problem_phenomenon=problem_phenomenon,
                root_cause=root_cause,
                solution=solution,
                severity=severity,
                status=status,
            )
            if created_at is not None:
                issue.created_at = created_at
            if updated_at is not None:
                issue.updated_at = updated_at

            db.add(issue)
            db.commit()
            db.refresh(issue)
            return issue.id

    async def get_curated_issues(
        self,
        session_db_id: Optional[int] = None,
        file_path: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        按条件列出人工确认的问题实例（CuratedIssue）。

        支持筛选条件：
        - session_db_id: 只看某个审核会话下的问题
        - file_path: 只看某个文件中的问题
        - status: 只看特定状态（如 open / resolved / obsolete）
        """
        with self.get_session() as db:
            query = db.query(CuratedIssue)
            if session_db_id is not None:
                query = query.filter(CuratedIssue.session_id == session_db_id)
            if file_path is not None:
                query = query.filter(CuratedIssue.file_path == file_path)
            if status is not None:
                query = query.filter(CuratedIssue.status == status)

            issues = query.order_by(CuratedIssue.created_at.desc()).all()
            return [
                {
                    "id": item.id,
                    "session_id": item.session_id,
                    "pattern_id": item.pattern_id,
                    "project_path": item.project_path,
                    "file_path": item.file_path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "severity": item.severity,
                    "status": item.status,
                    "problem_phenomenon": item.problem_phenomenon,
                    "root_cause": item.root_cause,
                    "solution": item.solution,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                }
                for item in issues
            ]

    async def get_curated_issue_by_id(self, issue_id: int) -> Optional[Dict[str, Any]]:
        """按主键获取单条 CuratedIssue。"""
        with self.get_session() as db:
            item = (
                db.query(CuratedIssue)
                .filter(CuratedIssue.id == issue_id)
                .one_or_none()
            )
            if not item:
                return None
            return {
                "id": item.id,
                "session_id": item.session_id,
                "pattern_id": item.pattern_id,
                "project_path": item.project_path,
                "file_path": item.file_path,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "severity": item.severity,
                "status": item.status,
                "problem_phenomenon": item.problem_phenomenon,
                "root_cause": item.root_cause,
                "solution": item.solution,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }

    async def get_curated_issue_by_session_and_pattern(
        self, session_id: str, pattern_id: int
    ) -> Optional[Dict[str, Any]]:
        """按 session_id + pattern_id 获取 CuratedIssue（用于去重）。"""
        with self.get_session() as db:
            item = (
                db.query(CuratedIssue)
                .filter(
                    CuratedIssue.session_id == session_id,
                    CuratedIssue.pattern_id == pattern_id,
                )
                .order_by(CuratedIssue.created_at.desc())
                .first()
            )
            if not item:
                return None
            return {
                "id": item.id,
                "session_id": item.session_id,
                "pattern_id": item.pattern_id,
                "project_path": item.project_path,
                "file_path": item.file_path,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "severity": item.severity,
                "status": item.status,
                "problem_phenomenon": item.problem_phenomenon,
                "root_cause": item.root_cause,
                "solution": item.solution,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }

    async def update_curated_issue_status(
        self,
        issue_id: int,
        status: str,
    ) -> bool:
        """
        更新某条问题实例的状态，用于软删除或标记已解决等。

        返回是否成功更新。
        """
        with self.get_session() as db:
            issue_obj = (
                db.query(CuratedIssue)
                .filter(CuratedIssue.id == issue_id)
                .one_or_none()
            )
            if not issue_obj:
                return False
            issue_obj.status = status
            db.commit()
            return True

    async def delete_curated_issue(self, issue_id: int) -> bool:
        """
        硬删一条人工确认的问题实例（CuratedIssue）。

        返回是否成功删除（未找到记录则返回 False）。
        """
        with self.get_session() as db:
            issue_obj = (
                db.query(CuratedIssue)
                .filter(CuratedIssue.id == issue_id)
                .one_or_none()
            )
            if not issue_obj:
                return False
            db.delete(issue_obj)
            db.commit()
            return True

    async def delete_all_curated_issues(self) -> int:
        """硬删全部 CuratedIssue。"""
        with self.get_session() as db:
            count = db.query(CuratedIssue).delete()
            db.commit()
            return int(count or 0)

    # ====== IssuePattern CRUD ======
    async def create_issue_pattern(
        self,
        error_type: str,
        error_description: str,
        problematic_pattern: str,
        solution: str,
        severity: str = "medium",
        title: Optional[str] = None,
        language: Optional[str] = None,
        framework: Optional[str] = None,
        file_pattern: str = "",
        class_pattern: str = "",
        tags: Optional[str] = None,
        status: str = "active",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> int:
        """
        新增一条错误模式 + 解决方案知识条目（IssuePattern）。

        当前仅填充 IssuePattern(issue_patterns) 的核心字段：
        - error_type / severity: 模式分类与严重程度
        - error_description: 对模式的详细描述
        - problematic_pattern: 典型易错写法
        - file_pattern / class_pattern: 可选的简单文件/类名匹配模式
        - solution: 通用修复建议

        后续可以在此基础上扩展 title/language/framework/tags 等字段，
        或在调用方封装更高层的知识录入逻辑。
        """
        with self.get_session() as db:
            pattern = IssuePattern(
                title=title,
                error_type=error_type,
                severity=severity,
                language=language,
                framework=framework,
                error_description=error_description,
                problematic_pattern=problematic_pattern,
                solution=solution,
                file_pattern=file_pattern,
                class_pattern=class_pattern,
                tags=tags,
                status=status,
            )
            if created_at is not None:
                pattern.created_at = created_at
            if updated_at is not None:
                pattern.updated_at = updated_at

            db.add(pattern)
            db.commit()
            db.refresh(pattern)
            return pattern.id

    async def update_issue_pattern(
        self,
        pattern_id: int,
        error_type: Optional[str] = None,
        error_description: Optional[str] = None,
        problematic_pattern: Optional[str] = None,
        file_pattern: Optional[str] = None,
        class_pattern: Optional[str] = None,
        solution: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> bool:
        """
        按主键更新一条错误模式（IssuePattern）的部分字段。

        仅对传入非 None 的参数进行更新，返回是否成功更新。
        """
        with self.get_session() as db:
            pattern = (
                db.query(IssuePattern)
                .filter(IssuePattern.id == pattern_id)
                .one_or_none()
            )
            if not pattern:
                return False

            if error_type is not None:
                pattern.error_type = error_type
            if error_description is not None:
                pattern.error_description = error_description
            if problematic_pattern is not None:
                pattern.problematic_pattern = problematic_pattern
            if file_pattern is not None:
                pattern.file_pattern = file_pattern
            if class_pattern is not None:
                pattern.class_pattern = class_pattern
            if solution is not None:
                pattern.solution = solution
            if severity is not None:
                pattern.severity = severity
            if status is not None:
                pattern.status = status

            db.commit()
            return True

    async def get_issue_patterns(
        self,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取当前知识库中的错误模式列表。

        返回值为面向上层逻辑的简化字典结构，主要包含：
        - error_type / error_description / problematic_pattern
        - file_pattern / class_pattern / solution / severity

        注意：如需使用 title/tags 等扩展字段，可在此方法中补充映射。
        """
        with self.get_session() as db:
            query = db.query(IssuePattern)
            if status is not None:
                query = query.filter(IssuePattern.status == status)
            patterns = query.all()
            return [
                {
                    "id": item.id,
                    "error_type": item.error_type,
                    "error_description": item.error_description,
                    "problematic_pattern": item.problematic_pattern,
                    "file_pattern": item.file_pattern,
                    "class_pattern": item.class_pattern,
                    "solution": item.solution,
                    "severity": item.severity,
                    "status": item.status,
                    "language": item.language,
                    "framework": item.framework,
                }
                for item in patterns
            ]

    async def delete_issue_pattern(self, pattern_id: int) -> bool:
        """
        硬删一条错误模式（IssuePattern）记录。

        注意：由于 IssuePattern 与 CuratedIssue 之间存在 ORM 关系，
        若启用了级联删除，将一并删除关联的问题实例。
        """
        with self.get_session() as db:
            pattern = (
                db.query(IssuePattern)
                .filter(IssuePattern.id == pattern_id)
                .one_or_none()
            )
            if not pattern:
                return False
            db.delete(pattern)
            db.commit()
            return True

    async def delete_all_issue_patterns(self) -> int:
        """硬删全部 IssuePattern。"""
        with self.get_session() as db:
            count = db.query(IssuePattern).delete()
            db.commit()
            return int(count or 0)

    async def get_issue_pattern_by_id(self, pattern_id: int) -> Optional[Dict[str, Any]]:
        """
        根据主键获取单条 IssuePattern 记录，返回结构化字典表示。
        """
        with self.get_session() as db:
            pattern = (
                db.query(IssuePattern)
                .filter(IssuePattern.id == pattern_id)
                .one_or_none()
            )
            if not pattern:
                return None
            return {
                "id": pattern.id,
                "error_type": pattern.error_type,
                "severity": pattern.severity,
                "status": pattern.status,
                "language": pattern.language,
                "framework": pattern.framework,
                "error_description": pattern.error_description,
                "problematic_pattern": pattern.problematic_pattern,
                "solution": pattern.solution,
            }


