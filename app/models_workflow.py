from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean, Text, Float
from datetime import datetime
import uuid
from app.database import Base

class WorkflowTemplate(Base):
    __tablename__ = "workflow_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String, default="1.0")
    definition = Column(JSON, nullable=False)  # Complete workflow definition
    is_active = Column(Boolean, default=True)
    executions_count = Column(Integer, default=0)
    avg_execution_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    template_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    status = Column(String, default="running")  # running, completed, failed, stopped, cancelled
    input_data = Column(JSON, nullable=False)
    output = Column(JSON, nullable=True)
    step_results = Column(JSON, nullable=True)  # Results from each step
    error = Column(Text, nullable=True)
    steps_completed = Column(Integer, default=0)
    steps_total = Column(Integer, default=0)
    execution_time_ms = Column(Float, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
