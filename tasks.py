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
