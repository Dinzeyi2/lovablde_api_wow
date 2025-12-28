from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)

class Model(Base):
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # recommendation, classification, etc.
    status = Column(String, default="training")  # training, ready, failed
    file_path = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)
    prediction_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Algorithm(Base):
    __tablename__ = "algorithms"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    code = Column(Text, nullable=False)
    language = Column(String, default="python")  # python, javascript
    status = Column(String, default="deployed")
    execution_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    endpoint = Column(String, nullable=False)
    model_id = Column(String, nullable=True)
    algorithm_id = Column(String, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    status = Column(String, nullable=False)  # success, error
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
