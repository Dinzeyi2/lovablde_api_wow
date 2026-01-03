import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database URL from environment variable or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./algoapi.db")

# PostgreSQL fix for Railway
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=True  # ← ADD THIS TO SEE SQL QUERIES IN LOGS
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Function to create tables
def create_tables():
    """Create all tables in the database"""
    from app.models import APIKey, Model, Algorithm, UsageLog
    from app.models_workflow import WorkflowTemplate, WorkflowExecution
    
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully")
