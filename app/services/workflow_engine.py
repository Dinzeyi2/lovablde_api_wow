"""
Workflow Engine - Execute multi-step algorithm workflows
Chains your 31 production algorithms into complete business processes
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
from app.services.algorithm_executor import AlgorithmExecutor
from app.models_workflow import WorkflowExecution, WorkflowTemplate
import re

class WorkflowEngine:
    """
    Execute multi-step workflows that chain algorithms together
    
    Example workflow:
    1. Check fraud
    2. IF fraud_score > 0.7 THEN stop
    3. Calculate dynamic price
    4. Calculate tax
    5. Return final total
    """
    
    def __init__(self, db=None):
        self.db = db
        self.executor = AlgorithmExecutor()
        self.execution_state = {}
    
    def execute_workflow(
        self,
        template_id: str,
        input_data: Dict[str, Any],
        execution_mode: str = "sync",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow from a template
        
        Args:
            template_id: ID of workflow template
            input_data: Input parameters for workflow
            execution_mode: 'sync' or 'async'
            user_id: User executing workflow
            
        Returns:
            Workflow execution result
        """
        
        # Load template
        template = self._load_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = {
            "execution_id": execution_id,
            "template_id": template_id,
            "user_id": user_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "input_data": input_data,
            "steps_completed": 0,
            "steps_total": len(template["steps"]),
            "current_step": None
        }
        
        # Save execution to DB
        if self.db:
            self._save_execution(execution)
        
        # Initialize state
        self.execution_state = {
            "input_data": input_data,
            "steps": {},
            "variables": {},
            "output": None
        }
        
        try:
            # Execute each step
            for step_idx, step in enumerate(template["steps"]):
                step_id = step.get("step_id", f"step_{step_idx}")
                
                # Update current step
                execution["current_step"] = step_id
                execution["steps_completed"] = step_idx
                
                # Execute step
                step_result = self._execute_step(step, execution_id)
                
                # Store step result
                self.execution_state["steps"][step_id] = {
                    "status": "completed",
                    "output": step_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Handle conditional logic
                if step.get("conditions"):
                    action = self._evaluate_conditions(step["conditions"], step_result)
                    
                    if action == "stop_workflow":
                        execution["status"] = "stopped"
                        execution["completed_at"] = datetime.utcnow().isoformat()
                        self.execution_state["output"] = {
                            "status": "stopped",
                            "reason": "Condition triggered workflow stop",
                            "step": step_id,
                            "result": step_result
                        }
                        break
                    
                    elif action.startswith("goto_step:"):
                        # Jump to specific step (not implemented in MVP)
                        pass
            
            # Workflow completed successfully
            if execution["status"] == "running":
                execution["status"] = "completed"
                execution["completed_at"] = datetime.utcnow().isoformat()
                
                # Format output
                if template.get("output_format"):
                    self.execution_state["output"] = self._format_output(
                        template["output_format"],
                        self.execution_state
                    )
                else:
                    # Default: return all step outputs
                    self.execution_state["output"] = {
                        step_id: step_data["output"]
                        for step_id, step_data in self.execution_state["steps"].items()
                    }
            
            # Save final execution state
            execution["output"] = self.execution_state["output"]
            execution["step_results"] = [
                {
                    "step_id": step_id,
                    "status": step_data["status"],
                    "output": step_data["output"],
                    "timestamp": step_data["timestamp"]
                }
                for step_id, step_data in self.execution_state["steps"].items()
            ]
            
            if self.db:
                self._update_execution(execution)
            
            return {
                "execution_id": execution_id,
                "template_id": template_id,
                "status": execution["status"],
                "started_at": execution["started_at"],
                "completed_at": execution.get("completed_at"),
                "output": self.execution_state["output"],
                "steps_executed": execution["steps_completed"],
                "steps_total": execution["steps_total"]
            }
        
        except Exception as e:
            # Mark as failed
            execution["status"] = "failed"
            execution["completed_at"] = datetime.utcnow().isoformat()
            execution["error"] = str(e)
            
            if self.db:
                self._update_execution(execution)
            
            raise
    
    def _execute_step(self, step: Dict[str, Any], execution_id: str) -> Any:
        """
        Execute a single workflow step
        
        Step types:
        - algorithm: Run one of the 31 algorithms
        - custom_logic: Simple operations (sum, multiply, etc.)
        - external_api: Call external API
        - conditional: If/then logic
        - wait: Delay execution
        """
        
        step_type = step.get("type", "algorithm")
        
        if step_type == "algorithm":
            return self._execute_algorithm_step(step)
        
        elif step_type == "custom_logic":
            return self._execute_custom_logic(step)
        
        elif step_type == "external_api":
            return self._execute_external_api(step)
        
        elif step_type == "conditional":
            return self._execute_conditional(step)
        
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    def _execute_algorithm_step(self, step: Dict[str, Any]) -> Any:
        """Execute algorithm step"""
        
        algorithm_name = step.get("algorithm")
        if not algorithm_name:
            raise ValueError("Algorithm name required for algorithm step")
        
        # Resolve input parameters (may reference previous steps)
        input_params = self._resolve_parameters(step.get("input", {}))
        
        # Execute algorithm
        result = self.executor.execute(algorithm_name, input_params)
        
        # Remove metadata
        if "_metadata" in result:
            del result["_metadata"]
        
        return result
    
    def _execute_custom_logic(self, step: Dict[str, Any]) -> Any:
        """Execute custom logic (sum, multiply, etc.)"""
        
        operation = step.get("operation")
        
        if operation == "sum":
            fields = self._resolve_parameters(step.get("fields", []))
            return sum(float(f) for f in fields if f is not None)
        
        elif operation == "multiply":
            fields = self._resolve_parameters(step.get("fields", []))
            result = 1
            for f in fields:
                if f is not None:
                    result *= float(f)
            return result
        
        elif operation == "average":
            fields = self._resolve_parameters(step.get("fields", []))
            valid_fields = [float(f) for f in fields if f is not None]
            return sum(valid_fields) / len(valid_fields) if valid_fields else 0
        
        elif operation == "max":
            fields = self._resolve_parameters(step.get("fields", []))
            return max(float(f) for f in fields if f is not None)
        
        elif operation == "min":
            fields = self._resolve_parameters(step.get("fields", []))
            return min(float(f) for f in fields if f is not None)
        
        else:
            raise ValueError(f"Unknown custom operation: {operation}")
    
    def _execute_external_api(self, step: Dict[str, Any]) -> Any:
        """Execute external API call"""
        import httpx
        
        method = step.get("method", "GET")
        url = self._resolve_parameters(step.get("url"))
        headers = self._resolve_parameters(step.get("headers", {}))
        body = self._resolve_parameters(step.get("body", {}))
        
        timeout = step.get("timeout", 30)
        
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                response = client.get(url, headers=headers)
            elif method == "POST":
                response = client.post(url, headers=headers, json=body)
            elif method == "PUT":
                response = client.put(url, headers=headers, json=body)
            elif method == "DELETE":
                response = client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        
        try:
            return response.json()
        except:
            return {"status_code": response.status_code, "text": response.text}
    
    def _execute_conditional(self, step: Dict[str, Any]) -> Any:
        """Execute conditional step"""
        
        conditions = step.get("conditions", [])
        
        for condition in conditions:
            if self._evaluate_condition(condition.get("if")):
                then_action = condition.get("then")
                
                if then_action == "continue":
                    return {"action": "continue"}
                elif then_action == "stop_workflow":
                    return {"action": "stop_workflow"}
                elif then_action.startswith("goto_step:"):
                    return {"action": "goto_step", "target": then_action.split(":")[1]}
        
        # Default: continue
        return {"action": "continue"}
    
    def _resolve_parameters(self, params: Any) -> Any:
        """
        Resolve parameter values that reference previous steps
        
        Supports:
        - {{input_data.field}} - from original input
        - {{steps.step_id.output.field}} - from step output
        - {{variables.var_name}} - from variables
        """
        
        if isinstance(params, str):
            # Check if it's a template variable
            if "{{" in params and "}}" in params:
                return self._resolve_template_variable(params)
            return params
        
        elif isinstance(params, dict):
            return {
                key: self._resolve_parameters(value)
                for key, value in params.items()
            }
        
        elif isinstance(params, list):
            return [self._resolve_parameters(item) for item in params]
        
        else:
            return params
    
    def _resolve_template_variable(self, template: str) -> Any:
        """Resolve {{variable}} syntax"""
        
        # Extract variable path
        match = re.search(r'\{\{(.+?)\}\}', template)
        if not match:
            return template
        
        var_path = match.group(1).strip()
        parts = var_path.split('.')
        
        # Navigate to value
        if parts[0] == "input_data":
            value = self.execution_state["input_data"]
            for part in parts[1:]:
                value = value.get(part) if isinstance(value, dict) else None
        
        elif parts[0] == "steps":
            step_id = parts[1]
            value = self.execution_state["steps"].get(step_id, {}).get("output", {})
            for part in parts[2:]:
                if part == "output":
                    continue
                value = value.get(part) if isinstance(value, dict) else None
        
        elif parts[0] == "variables":
            value = self.execution_state["variables"]
            for part in parts[1:]:
                value = value.get(part) if isinstance(value, dict) else None
        
        else:
            value = None
        
        # Replace in template
        if "{{" in template and "}}" in template:
            # Full replacement
            return value
        else:
            # Partial replacement (string interpolation)
            return template.replace(f"{{{{{var_path}}}}}", str(value))
    
    def _evaluate_conditions(self, conditions: List[Dict], step_result: Any) -> str:
        """Evaluate conditional logic and return action"""
        
        for condition in conditions:
            condition_expr = condition.get("if")
            
            if self._evaluate_condition_expr(condition_expr, step_result):
                then_action = condition.get("then")
                
                if then_action:
                    return then_action
        
        # No condition matched
        return "continue"
    
    def _evaluate_condition_expr(self, expr: str, context: Any) -> bool:
        """
        Evaluate condition expression
        
        Examples:
        - "output.fraud_score > 0.7"
        - "output.price < 100"
        - "output.status == 'approved'"
        """
        
        # Simple parser for basic comparisons
        operators = {
            ">": lambda a, b: float(a) > float(b),
            "<": lambda a, b: float(a) < float(b),
            ">=": lambda a, b: float(a) >= float(b),
            "<=": lambda a, b: float(a) <= float(b),
            "==": lambda a, b: str(a) == str(b),
            "!=": lambda a, b: str(a) != str(b)
        }
        
        for op, func in operators.items():
            if op in expr:
                left, right = expr.split(op, 1)
                left = left.strip()
                right = right.strip().strip("'\"")
                
                # Get left value from context
                if left.startswith("output."):
                    field = left.replace("output.", "")
                    left_value = context.get(field) if isinstance(context, dict) else None
                else:
                    left_value = left
                
                # Compare
                try:
                    return func(left_value, right)
                except:
                    return False
        
        return False
    
    def _evaluate_condition(self, expr: str) -> bool:
        """Evaluate condition with current state"""
        
        # Resolve any template variables
        expr_resolved = self._resolve_parameters(expr)
        
        # Simple evaluation
        return self._evaluate_condition_expr(expr_resolved, {})
    
    def _format_output(self, output_format: Dict[str, Any], state: Dict) -> Dict[str, Any]:
        """Format workflow output according to template"""
        
        return self._resolve_parameters(output_format)
    
    def _load_template(self, template_id: str) -> Optional[Dict]:
        """Load workflow template from database or prebuilt"""
        
        # Try to load from database
        if self.db:
            template_record = self.db.query(WorkflowTemplate).filter(
                WorkflowTemplate.id == template_id
            ).first()
            
            if template_record:
                return json.loads(template_record.definition)
        
        # Try prebuilt templates
        from workflows_prebuilt import PREBUILT_WORKFLOWS
        return PREBUILT_WORKFLOWS.get(template_id)
    
    def _save_execution(self, execution: Dict):
        """Save execution record to database"""
        
        if not self.db:
            return
        
        db_execution = WorkflowExecution(
            id=execution["execution_id"],
            template_id=execution["template_id"],
            user_id=execution["user_id"],
            status=execution["status"],
            input_data=execution["input_data"],
            output=execution.get("output"),
            started_at=datetime.fromisoformat(execution["started_at"])
        )
        
        self.db.add(db_execution)
        self.db.commit()
    
    def _update_execution(self, execution: Dict):
        """Update execution record in database"""
        
        if not self.db:
            return
        
        db_execution = self.db.query(WorkflowExecution).filter(
            WorkflowExecution.id == execution["execution_id"]
        ).first()
        
        if db_execution:
            db_execution.status = execution["status"]
            db_execution.output = execution.get("output")
            db_execution.step_results = execution.get("step_results")
            
            if execution.get("completed_at"):
                db_execution.completed_at = datetime.fromisoformat(execution["completed_at"])
            
            if execution.get("error"):
                db_execution.error = execution["error"]
            
            self.db.commit()
