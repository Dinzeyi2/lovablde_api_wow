"""
Database Models for Algorithm Usage Tracking
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Index
from datetime import datetime
import uuid


# Import your existing Base from your main database module
# Adjust this import based on your existing structure
try:
    from app.database import Base
except:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()


class AlgorithmUsage(Base):
    """Track algorithm API usage and performance"""
    
    __tablename__ = "algorithm_usage"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: f"algo_{uuid.uuid4().hex[:12]}")
    
    # User tracking
    user_id = Column(String, index=True, nullable=True)  # Nullable for anonymous usage
    api_key = Column(String, index=True, nullable=True)
    
    # Algorithm details
    category = Column(String, nullable=False, index=True)  # "graph", "optimization", etc
    algorithm = Column(String, nullable=False, index=True)  # "dijkstra", "knapsack", etc
    variant = Column(String, nullable=True)  # For algorithms with variants (e.g., "01", "fractional")
    
    # Performance metrics
    execution_time_ms = Column(Float, nullable=False)
    input_size = Column(Integer, nullable=True)  # Number of nodes, items, etc
    
    # Input/Output tracking (for debugging and analytics)
    input_hash = Column(String, index=True)  # Hash of input for duplicate detection
    output_summary = Column(JSON, nullable=True)  # Store key output metrics
    
    # Status tracking
    success = Column(Boolean, default=True, index=True)
    error_message = Column(String, nullable=True)
    error_type = Column(String, nullable=True)  # "validation", "timeout", "computation"
    
    # Request metadata
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_user_category_algo', 'user_id', 'category', 'algorithm'),
        Index('idx_created_success', 'created_at', 'success'),
        Index('idx_category_created', 'category', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AlgorithmUsage {self.category}/{self.algorithm} user={self.user_id}>"


class AlgorithmQuota(Base):
    """Track user algorithm usage quotas"""
    
    __tablename__ = "algorithm_quotas"
    
    id = Column(String, primary_key=True, default=lambda: f"quota_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, unique=True, index=True, nullable=False)
    
    # Quota limits (monthly)
    total_calls_limit = Column(Integer, default=1000)
    total_calls_used = Column(Integer, default=0)
    
    # Per-category limits
    graph_calls_limit = Column(Integer, default=500)
    graph_calls_used = Column(Integer, default=0)
    
    optimization_calls_limit = Column(Integer, default=300)
    optimization_calls_used = Column(Integer, default=0)
    
    ml_calls_limit = Column(Integer, default=200)
    ml_calls_used = Column(Integer, default=0)
    
    # Billing period
    period_start = Column(DateTime, default=datetime.utcnow)
    period_end = Column(DateTime)
    
    # Subscription tier
    tier = Column(String, default="free")  # "free", "starter", "pro", "enterprise"
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AlgorithmQuota user={self.user_id} tier={self.tier}>"
    
    def has_quota(self, category: str = None) -> bool:
        """Check if user has remaining quota"""
        if self.total_calls_used >= self.total_calls_limit:
            return False
        
        if category:
            category_attr = f"{category}_calls_used"
            limit_attr = f"{category}_calls_limit"
            if hasattr(self, category_attr) and hasattr(self, limit_attr):
                used = getattr(self, category_attr, 0)
                limit = getattr(self, limit_attr, 0)
                return used < limit
        
        return True
    
    def increment_usage(self, category: str):
        """Increment usage counters"""
        self.total_calls_used += 1
        
        category_attr = f"{category}_calls_used"
        if hasattr(self, category_attr):
            current = getattr(self, category_attr, 0)
            setattr(self, category_attr, current + 1)


# Migration script to create tables
def create_tables(engine):
    """Create all algorithm tables"""
    Base.metadata.create_all(bind=engine, tables=[
        AlgorithmUsage.__table__,
        AlgorithmQuota.__table__
    ])
    print("✅ Algorithm tables created successfully")


if __name__ == "__main__":
    print("Database models defined:")
    print(f"  - {AlgorithmUsage.__tablename__}")
    print(f"  - {AlgorithmQuota.__tablename__}")
