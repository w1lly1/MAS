from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, UserRequirement, KnowledgeBase, AnalysisResult
from typing import List, Optional, Dict, Any

class DatabaseService:
    def __init__(self, database_url: str = "sqlite:///./mas.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def get_session(self) -> Session:
        return self.SessionLocal()
        
    async def save_user_requirement(self, session_id: str, user_message: str, 
                                  code_directory: str, code_patch: Optional[str] = None,
                                  git_commit: Optional[str] = None) -> int:
        """保存用户需求"""
        with self.get_session() as db:
            requirement = UserRequirement(
                session_id=session_id,
                user_message=user_message,
                code_directory=code_directory,
                code_patch=code_patch,
                git_commit=git_commit
            )
            db.add(requirement)
            db.commit()
            db.refresh(requirement)
            return requirement.id
            
    async def save_analysis_result(self, requirement_id: int, agent_type: str, 
                                 result_data: Dict[str, Any], status: str):
        """保存分析结果"""
        with self.get_session() as db:
            result = AnalysisResult(
                requirement_id=requirement_id,
                agent_type=agent_type,
                result_data=result_data,
                status=status
            )
            db.add(result)
            db.commit()
            
    async def update_knowledge_base(self, error_type: str, error_description: str,
                                  problematic_pattern: str, file_pattern: str = "",
                                  class_pattern: str = "", solution: str = "",
                                  severity: str = "medium"):
        """更新知识库"""
        with self.get_session() as db:
            knowledge = KnowledgeBase(
                error_type=error_type,
                error_description=error_description,
                problematic_pattern=problematic_pattern,
                file_pattern=file_pattern,
                class_pattern=class_pattern,
                solution=solution,
                severity=severity
            )
            db.add(knowledge)
            db.commit()
            
    async def get_knowledge_patterns(self) -> List[Dict[str, Any]]:
        """获取知识库中的模式"""
        with self.get_session() as db:
            knowledge_items = db.query(KnowledgeBase).all()
            return [
                {
                    "error_type": item.error_type,
                    "error_description": item.error_description,
                    "problematic_pattern": item.problematic_pattern,
                    "file_pattern": item.file_pattern,
                    "class_pattern": item.class_pattern,
                    "solution": item.solution,
                    "severity": item.severity
                }
                for item in knowledge_items
            ]