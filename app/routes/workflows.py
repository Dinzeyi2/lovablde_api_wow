"""
Workflow API Routes
Endpoints for creating and executing multi-step workflows
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth import verify_api_key
from app.database import SessionLocal
from app.models import APIKey
from app.models_workflow import WorkflowTemplate, WorkflowExecution
from app.services.workflow_engine import WorkflowEngine
from workflows_prebuilt import PREBUILT_WORKFLOWS

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])
limiter = Limiter(key_func=get_remote_address)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== PYDANTIC MODELS ====================

class CreateWorkflowTemplateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field(default="1.0", max_length=20)
    steps: List[Dict[str, Any]] = Field(..., min_items=1, max_items=50)
    output_format: Optional[Dict[str, Any]] = None

class ExecuteWorkflowRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    execution_mode: str = Field(default="sync", pattern="^(sync|async)$")
    input_data: Dict[str, Any] = Field(..., max_properties=100)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

# ==================== WORKFLOW TEMPLATE MANAGEMENT ====================

@router.post("/templates")
@limiter.limit("10/hour")
async def create_workflow_template(
    request: Request,
    req: CreateWorkflowTemplateRequest,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """
    Create a new workflow template
    
    A workflow template defines a sequence of algorithm executions
    that can be run multiple times with different inputs.
    """
    
    # Create template
    template = WorkflowTemplate(
        user_id=api_key.user_id,
        name=req.name,
        description=req.description,
        version=req.version,
        definition={
            "name": req.name,
            "description": req.description,
            "version": req.version,
            "steps": req.steps,
            "output_format": req.output_format
        }
    )
    
    db.add(template)
    db.commit()
    db.refresh(template)
    
    return {
        "template_id": template.id,
        "name": template.name,
        "version": template.version,
        "steps_count": len(req.steps),
        "created_at": template.created_at.isoformat()
    }

@router.get("/templates")
async def list_workflow_templates(
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """List all workflow templates for the user"""
    
    # Get user templates
    user_templates = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.user_id == api_key.user_id,
        WorkflowTemplate.is_active == True
    ).all()
    
    templates = [
        {
            "template_id": t.id,
            "name": t.name,
            "description": t.description,
            "version": t.version,
            "steps_count": len(t.definition.get("steps", [])),
            "executions_count": t.executions_count,
            "avg_execution_time_ms": t.avg_execution_time_ms,
            "created_at": t.created_at.isoformat()
        }
        for t in user_templates
    ]
    
    # Add prebuilt workflows
    prebuilt = [
        {
            "template_id": wf_id,
            "name": wf["name"],
            "description": wf["description"],
            "category": wf["category"],
            "steps_count": len(wf["steps"]),
            "is_prebuilt": True
        }
        for wf_id, wf in PREBUILT_WORKFLOWS.items()
    ]
    
    return {
        "user_templates": templates,
        "prebuilt_workflows": prebuilt,
        "total_user": len(templates),
        "total_prebuilt": len(prebuilt)
    }

@router.get("/templates/{template_id}")
async def get_workflow_template(
    template_id: str,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Get details of a specific workflow template"""
    
    # Check if it's a prebuilt workflow
    if template_id in PREBUILT_WORKFLOWS:
        wf = PREBUILT_WORKFLOWS[template_id]
        return {
            "template_id": template_id,
            "name": wf["name"],
            "description": wf["description"],
            "category": wf["category"],
            "steps": wf["steps"],
            "output_format": wf.get("output_format"),
            "is_prebuilt": True
        }
    
    # Check user templates
    template = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == template_id,
        WorkflowTemplate.user_id == api_key.user_id
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Workflow template not found")
    
    return {
        "template_id": template.id,
        "name": template.name,
        "description": template.description,
        "version": template.version,
        "definition": template.definition,
        "executions_count": template.executions_count,
        "avg_execution_time_ms": template.avg_execution_time_ms,
        "created_at": template.created_at.isoformat(),
        "is_prebuilt": False
    }

@router.put("/templates/{template_id}")
@limiter.limit("20/hour")
async def update_workflow_template(
    request: Request,
    template_id: str,
    req: CreateWorkflowTemplateRequest,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Update an existing workflow template"""
    
    template = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == template_id,
        WorkflowTemplate.user_id == api_key.user_id
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Workflow template not found")
    
    # Update template
    template.name = req.name
    template.description = req.description
    template.version = req.version
    template.definition = {
        "name": req.name,
        "description": req.description,
        "version": req.version,
        "steps": req.steps,
        "output_format": req.output_format
    }
    template.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "template_id": template.id,
        "message": "Template updated successfully"
    }

@router.delete("/templates/{template_id}")
@limiter.limit("10/hour")
async def delete_workflow_template(
    request: Request,
    template_id: str,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Delete a workflow template"""
    
    template = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == template_id,
        WorkflowTemplate.user_id == api_key.user_id
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Workflow template not found")
    
    # Soft delete
    template.is_active = False
    db.commit()
    
    return {
        "template_id": template_id,
        "message": "Template deleted successfully"
    }

# ==================== WORKFLOW EXECUTION ====================

@router.post("/execute")
@limiter.limit("50/minute")
async def execute_workflow(
    request: Request,
    req: ExecuteWorkflowRequest,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """
    Execute a workflow (sync or async)
    
    Sync mode: Returns result immediately (max 30 seconds)
    Async mode: Returns execution_id, check status later
    """
    
    # Initialize workflow engine
    engine = WorkflowEngine(db=db)
    
    try:
        # Execute workflow
        result = engine.execute_workflow(
            template_id=req.template_id,
            input_data=req.input_data,
            execution_mode=req.execution_mode,
            user_id=api_key.user_id
        )
        
        # Update template statistics
        template = db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == req.template_id
        ).first()
        
        if template:
            template.executions_count += 1
            db.commit()
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@router.get("/executions/{execution_id}")
async def get_workflow_execution(
    execution_id: str,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Get the status and results of a workflow execution"""
    
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == api_key.user_id
    ).first()
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    result = {
        "execution_id": execution.id,
        "template_id": execution.template_id,
        "status": execution.status,
        "started_at": execution.started_at.isoformat(),
        "steps_completed": execution.steps_completed,
        "steps_total": execution.steps_total
    }
    
    if execution.status == "running":
        result["progress_percent"] = (execution.steps_completed / execution.steps_total * 100) if execution.steps_total > 0 else 0
    
    if execution.status in ["completed", "failed", "stopped"]:
        result["completed_at"] = execution.completed_at.isoformat() if execution.completed_at else None
        result["execution_time_ms"] = execution.execution_time_ms
        result["output"] = execution.output
        result["step_results"] = execution.step_results
        
        if execution.error:
            result["error"] = execution.error
    
    return result

@router.get("/executions")
async def list_workflow_executions(
    template_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """List workflow executions with optional filters"""
    
    query = db.query(WorkflowExecution).filter(
        WorkflowExecution.user_id == api_key.user_id
    )
    
    if template_id:
        query = query.filter(WorkflowExecution.template_id == template_id)
    
    if status:
        query = query.filter(WorkflowExecution.status == status)
    
    executions = query.order_by(WorkflowExecution.started_at.desc()).limit(limit).all()
    
    return {
        "executions": [
            {
                "execution_id": e.id,
                "template_id": e.template_id,
                "status": e.status,
                "started_at": e.started_at.isoformat(),
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "steps_completed": e.steps_completed,
                "steps_total": e.steps_total
            }
            for e in executions
        ],
        "total": len(executions)
    }

@router.post("/executions/{execution_id}/cancel")
@limiter.limit("20/hour")
async def cancel_workflow_execution(
    request: Request,
    execution_id: str,
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Cancel a running workflow execution"""
    
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.user_id == api_key.user_id
    ).first()
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    if execution.status != "running":
        raise HTTPException(status_code=400, detail="Only running executions can be cancelled")
    
    execution.status = "cancelled"
    execution.completed_at = datetime.utcnow()
    db.commit()
    
    return {
        "execution_id": execution_id,
        "status": "cancelled",
        "message": "Workflow execution cancelled"
    }

# ==================== WORKFLOW ANALYTICS ====================

@router.get("/analytics")
async def get_workflow_analytics(
    api_key: APIKey = Depends(verify_api_key),
    db = Depends(get_db)
):
    """Get workflow usage analytics"""
    
    # Get template count
    template_count = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.user_id == api_key.user_id,
        WorkflowTemplate.is_active == True
    ).count()
    
    # Get execution counts by status
    executions = db.query(WorkflowExecution).filter(
        WorkflowExecution.user_id == api_key.user_id
    ).all()
    
    status_counts = {}
    for execution in executions:
        status_counts[execution.status] = status_counts.get(execution.status, 0) + 1
    
    # Calculate average execution time
    completed = [e for e in executions if e.status == "completed" and e.execution_time_ms]
    avg_time = sum(e.execution_time_ms for e in completed) / len(completed) if completed else 0
    
    return {
        "total_templates": template_count,
        "total_executions": len(executions),
        "executions_by_status": status_counts,
        "avg_execution_time_ms": round(avg_time, 2),
        "most_used_templates": _get_most_used_templates(db, api_key.user_id)
    }

def _get_most_used_templates(db, user_id: str):
    """Get top 5 most executed templates"""
    templates = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.user_id == user_id,
        WorkflowTemplate.is_active == True
    ).order_by(WorkflowTemplate.executions_count.desc()).limit(5).all()
    
    return [
        {
            "template_id": t.id,
            "name": t.name,
            "executions_count": t.executions_count
        }
        for t in templates
    ]
