from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class UserRequirement(Base):
    __tablename__ = "user_requirements"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    user_message = Column(Text)
    code_directory = Column(String(500))
    code_patch = Column(Text, nullable=True)
    git_commit = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, index=True)
    error_type = Column(String(255), index=True)
    error_description = Column(Text)
    problematic_pattern = Column(Text)
    file_pattern = Column(String(255))
    class_pattern = Column(String(255))
    solution = Column(Text)
    severity = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    requirement_id = Column(Integer)
    agent_type = Column(String(100))
    result_data = Column(JSON)
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)