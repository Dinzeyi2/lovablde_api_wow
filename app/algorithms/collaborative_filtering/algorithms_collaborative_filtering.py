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
from app.models_workflow import WorkflowTemplate, WorkflowExecution
from app.auth import verify_api_key, generate_api_key
from app.routes import workflows 
from app.routes import algorithms
from app.routes import anomaly_detection
from app.prebuilt import PREBUILT_ALGORITHMS
from app.tasks import train_model_task, execute_algorithm_secure
from app.services.logic_verifier import LogicVerifier
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.routes import collaborative_filtering
from app.routes import anomaly_detection
from app.routes import load_balancing
from app.routes import shell_company_detection
from app.routes import pathfinding

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize app with security
app = FastAPI(
    title="AlgoAPI - Secure",
    description="Production-hardened Complex Algorithm API",
    version="2.0.1-secure"
)
# Include workflow routes
app.include_router(workflows.router)
app.include_router(algorithms.router)
app.include_router(collaborative_filtering.router)
app.include_router(anomaly_detection.router)
app.include_router(load_balancing.router) 
app.include_router(shell_company_detection.router)
app.include_router(pathfinding.router)

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AlgoAPI Production - All algorithms operational", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
