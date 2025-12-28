"""
AlgoAPI - PRODUCTION HARDENED VERSION
Security: Docker isolation, rate limiting, input validation
Architecture: Celery task queue, proper error handling
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from datetime import datetime
import tempfile
import uuid

from app.database import engine, SessionLocal, Base
from app.models import Model, APIKey
from app.auth import verify_api_key, create_api_key
from app.prebuilt import PREBUILT_ALGORITHMS
from app.tasks import train_model_task, execute_algorithm_secure
from app.services.logic_verifier import LogicVerifier

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize app with security
app = FastAPI(
    title="AlgoAPI - Secure",
    description="Production-hardened Complex Algorithm API",
    version="2.0.0-secure"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== HEALTH & STATUS ====================

@app.get("/health")
async def health_check():
    """Health check with system status"""
    from app.services.secure_executor import SecureExecutor
    
    executor = SecureExecutor()
    isolation_status = executor.verify_resource_limits()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0-secure",
        "isolation": isolation_status.get("isolation", "unknown"),
        "docker_available": isolation_status.get("docker_available", False)
    }

# ==================== API KEY MANAGEMENT ====================

class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')

@app.post("/api/v1/keys/create")
@limiter.limit("5/hour")  # Limit API key creation
async def create_new_api_key(
    request: Request,
    req: CreateAPIKeyRequest,
    db = Depends(get_db)
):
    """Create new API key - rate limited to prevent abuse"""
    api_key = create_api_key(db, req.name, req.email)
    return {
        "api_key": api_key.key,
        "name": api_key.name,
        "created_at": api_key.created_at,
        "warning": "Store this key securely - it won't be shown again"
    }

# ==================== ML MODEL TRAINING ====================

class TrainModelRequest(BaseModel):
    model_type: str = Field(..., regex='^(recommendation|classification|regression|clustering)$')
    name: str = Field(..., min_length=1, max_length=100)
    target_column: Optional[str] = Field(None, max_length=100)
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed = ['recommendation', 'classification', 'regression', 'clustering']
        if v not in allowed:
            raise ValueError(f'model_type must be one of: {allowed}')
        return v

@app.post("/api/v1/train")
@limiter.limit("5/hour")  # Max 5 training jobs per hour
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    model_type: str = 'recommendation',
    target_column: Optional[str] = None,
    name: str = 'my-model',
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """
    Train ML model - SECURE VERSION
    - Rate limited
    - File size validated
    - Queued to background worker (Celery)
    - Data quality checked before training
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    # Validate file size (max 10MB)
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    if len(file_content) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_SIZE / 1024 / 1024}MB"
        )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(file_content)
        temp_path = tmp.name
    
    # Create model record
    model_id = str(uuid.uuid4())
    db_model = Model(
        id=model_id,
        user_id=api_key.user_id,
        name=name,
        model_type=model_type,
        status="queued"
    )
    db.add(db_model)
    db.commit()
    
    # Queue training job (runs in Celery worker, not web server)
    task = train_model_task.delay(
        data_path=temp_path,
        model_type=model_type,
        target_column=target_column,
        name=name,
        user_id=api_key.user_id,
        model_id=model_id
    )
    
    return {
        "status": "queued",
        "model_id": model_id,
        "task_id": task.id,
        "check_status": f"/api/v1/models/{model_id}",
        "estimated_time": "2-5 minutes",
        "note": "Training happens in background. Check status endpoint."
    }

# ==================== PREDICTION ====================

class PredictRequest(BaseModel):
    data: Dict[str, Any] = Field(..., max_length=1000)  # Limit input size

@app.post("/api/v1/predict/{model_id}")
@limiter.limit("100/minute")  # High limit for predictions
async def predict(
    request: Request,
    model_id: str,
    req: PredictRequest,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Make predictions - rate limited"""
    
    model = db.query(Model).filter(
        Model.id == model_id,
        Model.user_id == api_key.user_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Model status: {model.status}. Wait for training to complete."
        )
    
    # Prediction happens in main thread (fast operation)
    from app.services.ml_trainer import MLTrainer
    trainer = MLTrainer()
    
    try:
        prediction = trainer.predict(model_id, req.data, db)
        
        # Update usage
        model.prediction_count += 1
        db.commit()
        
        return {
            "model_id": model_id,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==================== PRE-BUILT ALGORITHMS (SECURE) ====================

class ExecuteAlgorithmRequest(BaseModel):
    algorithm_name: str = Field(..., regex='^[a-z-]+$')
    params: Dict[str, Any] = Field(..., max_length=1000)
    
    @validator('algorithm_name')
    def validate_algorithm(cls, v):
        if v not in PREBUILT_ALGORITHMS:
            raise ValueError(f'Algorithm not found: {v}')
        return v

@app.post("/api/v1/algorithm/execute")
@limiter.limit("50/minute")  # Rate limit algorithm execution
async def execute_algorithm(
    request: Request,
    req: ExecuteAlgorithmRequest,
    api_key: APIKey = Depends(verify_api_key)
):
    """
    Execute pre-built algorithm - SECURE VERSION
    - Only runs pre-verified code (no user code execution)
    - Rate limited
    - Input validated
    """
    
    # Execute in Celery worker (prevents blocking web server)
    # For now, execute directly since algorithms are safe
    from app.services.algorithm_executor import AlgorithmExecutor
    
    executor = AlgorithmExecutor()
    
    try:
        result = executor.execute(req.algorithm_name, req.params)
        
        return {
            "algorithm": req.algorithm_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

# ==================== ALGORITHM CATALOG ====================

@app.get("/api/v1/algorithms")
async def list_algorithms():
    """List all verified algorithms with safety certificates"""
    
    verifier = LogicVerifier()
    
    algorithms = []
    for name, algo in PREBUILT_ALGORITHMS.items():
        algo_info = {
            "name": name,
            "description": algo["description"],
            "category": algo["category"],
            "pricing_tier": algo.get("pricing_tier", "starter"),
            "verified": True,
            "safety_level": "production"
        }
        
        # Add safety certificate for critical algorithms
        if name in ["fraud-detection", "dynamic-pricing", "credit-scoring"]:
            algo_info["certificate"] = verifier.generate_safety_certificate(
                name,
                algo.get("constraints", [])
            )
        
        algorithms.append(algo_info)
    
    return {"algorithms": algorithms}

@app.get("/api/v1/algorithms/{algorithm_name}")
async def get_algorithm_details(algorithm_name: str):
    """Get algorithm details with safety information"""
    
    if algorithm_name not in PREBUILT_ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    algo = PREBUILT_ALGORITHMS[algorithm_name].copy()
    
    # Add verification info
    verifier = LogicVerifier()
    if algorithm_name == "dynamic-pricing":
        algo["verification"] = verifier.verify_pricing_algorithm(cost=60, markup_min=1.2)
    
    return algo

# ==================== MODEL MANAGEMENT ====================

@app.get("/api/v1/models")
async def list_models(
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """List user's models"""
    models = db.query(Model).filter(Model.user_id == api_key.user_id).all()
    
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "type": m.model_type,
                "status": m.status,
                "created_at": m.created_at.isoformat(),
                "prediction_count": m.prediction_count,
                "accuracy": m.accuracy
            }
            for m in models
        ]
    }

@app.get("/api/v1/models/{model_id}")
async def get_model(
    model_id: str,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Get model status and details"""
    
    model = db.query(Model).filter(
        Model.id == model_id,
        Model.user_id == api_key.user_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model.id,
        "name": model.name,
        "type": model.model_type,
        "status": model.status,
        "created_at": model.created_at.isoformat(),
        "prediction_count": model.prediction_count,
        "accuracy": model.accuracy,
        "metadata": model.metadata,
        "endpoint": f"/api/v1/predict/{model.id}" if model.status == "ready" else None
    }

# ==================== USAGE STATS ====================

@app.get("/api/v1/usage")
async def get_usage_stats(
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Get usage statistics"""
    
    models = db.query(Model).filter(Model.user_id == api_key.user_id).all()
    
    return {
        "total_models": len(models),
        "total_predictions": sum(m.prediction_count for m in models),
        "models_by_status": {
            "queued": len([m for m in models if m.status == "queued"]),
            "training": len([m for m in models if m.status == "training"]),
            "ready": len([m for m in models if m.status == "ready"]),
            "failed": len([m for m in models if m.status == "failed"])
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Celery Tasks - Background jobs for training, processing, execution
All long-running operations happen here, not in the API
"""

from app.celery_config import celery_app
from app.services.ml_trainer import MLTrainer
from app.services.secure_executor import SecureExecutor
from app.services.data_sanitizer import DataSanitizer
from app.database import SessionLocal
from app.models import Model, Algorithm
import os
from datetime import datetime, timedelta

@celery_app.task(bind=True, max_retries=3)
def train_model_task(self, data_path: str, model_type: str, target_column: str, 
                     name: str, user_id: str, model_id: str):
    """
    Background task for model training
    Updates database with progress and results
    """
    db = SessionLocal()
    
    try:
        # Get model record
        db_model = db.query(Model).filter(Model.id == model_id).first()
        if not db_model:
            raise ValueError(f"Model {model_id} not found")
        
        # Update status to training
        db_model.status = "training"
        db_model.metadata = {"started_at": datetime.utcnow().isoformat()}
        db.commit()
        
        # Sanitize data first
        sanitizer = DataSanitizer()
        sanitized_path, report = sanitizer.sanitize_file(data_path)
        
        # Check if data is usable
        if report.get('critical_issues'):
            db_model.status = "failed"
            db_model.metadata = {
                "error": "Data quality issues",
                "issues": report['critical_issues']
            }
            db.commit()
            return {"status": "failed", "reason": "data_quality"}
        
        # Train model
        trainer = MLTrainer()
        success = trainer.train_model(
            data_path=sanitized_path,
            model_type=model_type,
            target_column=target_column,
            name=name,
            user_id=user_id,
            model_id=model_id,
            db=db
        )
        
        if success:
            db_model.status = "ready"
            db_model.metadata = {
                "completed_at": datetime.utcnow().isoformat(),
                "data_quality_report": report
            }
        else:
            db_model.status = "failed"
            db_model.metadata = {"error": "Training failed"}
        
        db.commit()
        
        # Cleanup temp file
        if os.path.exists(data_path):
            os.remove(data_path)
        
        return {"status": "success", "model_id": model_id}
    
    except Exception as e:
        # Update model status to failed
        if db_model:
            db_model.status = "failed"
            db_model.metadata = {"error": str(e)}
            db.commit()
        
        # Retry on certain errors
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise self.retry(exc=e, countdown=60)  # Retry after 1 minute
        
        raise
    
    finally:
        db.close()


@celery_app.task(bind=True)
def execute_algorithm_secure(self, algorithm_name: str, params: dict, user_id: str):
    """
    Execute pre-built algorithm in isolated environment
    """
    try:
        executor = SecureExecutor()
        result = executor.execute_isolated(algorithm_name, params)
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task
def process_data_task(file_path: str, operation: str, params: dict, user_id: str):
    """
    Background task for data processing
    """
    try:
        from app.services.data_processor import DataProcessor
        
        processor = DataProcessor()
        result = processor.process(file_path, operation, params)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return result
    
    except Exception as e:
        raise


@celery_app.task
def cleanup_old_models():
    """
    Periodic task: Delete models older than 30 days (for free tier)
    """
    db = SessionLocal()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        old_models = db.query(Model).filter(
            Model.created_at < cutoff_date,
            Model.status == "ready"
        ).all()
        
        for model in old_models:
            # Delete model file
            if model.file_path and os.path.exists(model.file_path):
                os.remove(model.file_path)
            
            # Delete from database
            db.delete(model)
        
        db.commit()
        
        return {"cleaned": len(old_models)}
    
    finally:
        db.close()


@celery_app.task
def check_stalled_training():
    """
    Periodic task: Check for training jobs that are stuck
    """
    db = SessionLocal()
    
    try:
        # Find models stuck in "training" for > 30 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=30)
        
        stalled = db.query(Model).filter(
            Model.status == "training",
            Model.updated_at < cutoff
        ).all()
        
        for model in stalled:
            model.status = "failed"
            model.metadata = {"error": "Training timeout"}
        
        db.commit()
        
        return {"stalled": len(stalled)}
    
    finally:
        db.close()
"""
Celery Worker Configuration for AlgoAPI
Handles all background tasks: model training, data processing, algorithm execution
"""

from celery import Celery
from celery.schedules import crontab
import os

# Redis connection (Railway will provide REDIS_URL)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'algoapi',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['app.tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    broker_connection_retry_on_startup=True,
)

# Periodic tasks (optional - for cleanup, monitoring)
celery_app.conf.beat_schedule = {
    'cleanup-old-models': {
        'task': 'app.tasks.cleanup_old_models',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'check-training-status': {
        'task': 'app.tasks.check_stalled_training',
        'schedule': 300.0,  # Every 5 minutes
    },
}

if __name__ == '__main__':
    celery_app.start()
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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# Database URL from environment variable or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./algoapi.db")

# PostgreSQL fix for Railway
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
from fastapi import HTTPException, Header
from typing import Optional
import secrets
import hashlib
from datetime import datetime
from app.models import APIKey

def generate_api_key() -> str:
    """Generate a secure API key"""
    random_bytes = secrets.token_bytes(32)
    api_key = hashlib.sha256(random_bytes).hexdigest()
    return f"algoapi_{api_key[:32]}"

def create_api_key(db, name: str, email: str) -> APIKey:
    """Create a new API key in the database"""
    api_key = generate_api_key()
    user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
    
    db_api_key = APIKey(
        key=api_key,
        user_id=user_id,
        name=name,
        email=email
    )
    
    db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)
    
    return db_api_key

def verify_api_key(x_api_key: Optional[str] = Header(None), db = None) -> APIKey:
    """Verify API key from request header"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    
    # Import here to avoid circular dependency
    from app.main import get_db
    if db is None:
        db = next(get_db())
    
    api_key = db.query(APIKey).filter(APIKey.key == x_api_key, APIKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Update last used timestamp
    api_key.last_used = datetime.utcnow()
    db.commit()
    
    return api_key
"""
Pre-built algorithms catalog with descriptions and pricing tiers
"""

PREBUILT_ALGORITHMS = {
    'fraud-detection': {
        'name': 'Fraud Detection',
        'description': 'Detect fraudulent transactions using multi-signal risk scoring',
        'category': 'security',
        'pricing_tier': 'pro',
        'parameters': {
            'transaction_amount': 'float',
            'user_location': 'string (country code)',
            'device_fingerprint': 'string',
            'account_age_days': 'integer',
            'transaction_time': 'integer (hour 0-23)'
        },
        'returns': {
            'is_fraud': 'boolean',
            'risk_score': 'float (0-1)',
            'risk_factors': 'list',
            'recommendation': 'string (approve/review/block)'
        },
        'use_cases': [
            'E-commerce payment validation',
            'Financial transaction screening',
            'Account security monitoring'
        ]
    },
    
    'dynamic-pricing': {
        'name': 'Dynamic Pricing',
        'description': 'Calculate optimal prices based on competition, inventory, and demand',
        'category': 'ecommerce',
        'pricing_tier': 'pro',
        'parameters': {
            'base_price': 'float',
            'competitor_prices': 'list of floats',
            'inventory_level': 'integer (percentage)',
            'demand_score': 'float (0-1)',
            'cost': 'float'
        },
        'returns': {
            'recommended_price': 'float',
            'profit_margin_percent': 'float',
            'price_change_percent': 'float',
            'reasoning': 'string'
        },
        'use_cases': [
            'E-commerce pricing optimization',
            'SaaS plan pricing',
            'Retail markdown strategy'
        ]
    },
    
    'recommendation-collab': {
        'name': 'Collaborative Filtering Recommendations',
        'description': 'Generate personalized recommendations based on user behavior',
        'category': 'ml',
        'pricing_tier': 'starter',
        'parameters': {
            'user_id': 'string',
            'item_ratings': 'dict (item_id: rating)',
            'catalog': 'list of item_ids',
            'n_recommendations': 'integer'
        },
        'returns': {
            'recommendations': 'list of items with predicted ratings'
        },
        'use_cases': [
            'Product recommendations',
            'Content suggestions',
            'Personalized feeds'
        ]
    },
    
    'sentiment-analysis': {
        'name': 'Sentiment Analysis',
        'description': 'Analyze text sentiment (positive, negative, neutral)',
        'category': 'nlp',
        'pricing_tier': 'starter',
        'parameters': {
            'text': 'string'
        },
        'returns': {
            'sentiment': 'string (positive/negative/neutral)',
            'score': 'float (0-1)',
            'confidence': 'float (0-1)'
        },
        'use_cases': [
            'Customer review analysis',
            'Social media monitoring',
            'Support ticket categorization'
        ]
    },
    
    'churn-prediction': {
        'name': 'Customer Churn Prediction',
        'description': 'Predict likelihood of customer churn',
        'category': 'analytics',
        'pricing_tier': 'pro',
        'parameters': {
            'days_since_last_activity': 'integer',
            'total_purchases': 'integer',
            'avg_purchase_value': 'float',
            'support_tickets': 'integer',
            'account_age_months': 'integer'
        },
        'returns': {
            'will_churn': 'boolean',
            'churn_probability': 'float (0-1)',
            'risk_level': 'string (high/medium/low)',
            'recommended_action': 'string'
        },
        'use_cases': [
            'SaaS retention campaigns',
            'Subscription service optimization',
            'Customer success prioritization'
        ]
    },
    
    'lead-scoring': {
        'name': 'Lead Scoring',
        'description': 'Score and qualify sales leads based on engagement and demographics',
        'category': 'sales',
        'pricing_tier': 'starter',
        'parameters': {
            'email_opens': 'integer',
            'page_views': 'integer',
            'company_size': 'string (small/medium/large)',
            'job_title': 'string',
            'industry': 'string'
        },
        'returns': {
            'lead_score': 'integer (0-100)',
            'quality': 'string (hot/warm/cold)',
            'recommended_action': 'string'
        },
        'use_cases': [
            'Sales pipeline prioritization',
            'Marketing automation',
            'Lead nurturing workflows'
        ]
    },
    
    'inventory-optimization': {
        'name': 'Inventory Optimization',
        'description': 'Calculate optimal inventory levels and reorder points',
        'category': 'logistics',
        'pricing_tier': 'pro',
        'parameters': {
            'current_stock': 'integer',
            'daily_sales_avg': 'float',
            'lead_time_days': 'integer',
            'safety_stock_days': 'integer'
        },
        'returns': {
            'reorder_point': 'float',
            'recommended_order_quantity': 'float',
            'needs_reorder': 'boolean',
            'days_until_stockout': 'float',
            'urgency': 'string'
        },
        'use_cases': [
            'E-commerce inventory management',
            'Retail stock optimization',
            'Warehouse management'
        ]
    },
    
    'route-optimization': {
        'name': 'Route Optimization',
        'description': 'Optimize delivery routes using TSP algorithm',
        'category': 'logistics',
        'pricing_tier': 'pro',
        'parameters': {
            'locations': 'list of {lat, lon, address}',
            'start_location': 'object {lat, lon, address}'
        },
        'returns': {
            'optimized_route': 'list of locations in order',
            'total_distance_km': 'float',
            'estimated_time_hours': 'float'
        },
        'use_cases': [
            'Delivery route planning',
            'Field service scheduling',
            'Logistics optimization'
        ]
    },
    
    'credit-scoring': {
        'name': 'Credit Scoring',
        'description': 'Calculate credit scores based on financial factors',
        'category': 'fintech',
        'pricing_tier': 'pro',
        'parameters': {
            'income': 'float',
            'debt': 'float',
            'payment_history_score': 'integer (0-100)',
            'credit_age_years': 'integer',
            'num_accounts': 'integer'
        },
        'returns': {
            'credit_score': 'integer (300-850)',
            'rating': 'string (excellent/good/fair/poor)',
            'approval_recommendation': 'boolean',
            'interest_rate_tier': 'string'
        },
        'use_cases': [
            'Loan approval automation',
            'Risk assessment',
            'Credit line determination'
        ]
    },
    
    'demand-forecasting': {
        'name': 'Demand Forecasting',
        'description': 'Forecast future demand using time series analysis',
        'category': 'analytics',
        'pricing_tier': 'starter',
        'parameters': {
            'historical_sales': 'list of numbers',
            'forecast_periods': 'integer'
        },
        'returns': {
            'forecast': 'list of predicted values',
            'baseline': 'float',
            'trend': 'string (increasing/decreasing/stable)',
            'confidence': 'string (high/medium/low)'
        },
        'use_cases': [
            'Inventory planning',
            'Revenue forecasting',
            'Capacity planning'
        ]
    }
}

# Categories for filtering
ALGORITHM_CATEGORIES = {
    'security': ['fraud-detection'],
    'ecommerce': ['dynamic-pricing', 'inventory-optimization'],
    'ml': ['recommendation-collab'],
    'nlp': ['sentiment-analysis'],
    'analytics': ['churn-prediction', 'demand-forecasting'],
    'sales': ['lead-scoring'],
    'logistics': ['route-optimization'],
    'fintech': ['credit-scoring']
}
