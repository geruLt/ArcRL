"""
Example Adapter Implementation (Incomplete)

This is an intentionally incomplete adapter for testing the validator.
It demonstrates common issues that the validator should catch:
- Missing some required endpoints
- Some methods have no implementation (just pass)
- Missing proper inheritance in some cases

Use this file to test the validator's feedback system.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_adapter import BaseAdapter
from typing import Dict, List, Any, Optional


class ExampleAdapter(BaseAdapter):
    """
    Example adapter implementation - intentionally incomplete.
    
    Missing implementations:
    - batch_add_tasks (no implementation)
    - generate_replay (missing entirely)
    
    This allows testing the validator's error detection.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        super().__init__(host, port)
        self.tasks = []
        self.current_task = None
        self.training_started = False
    
    # =========================================================================
    # GET Endpoints - All implemented
    # =========================================================================
    
    def get_root(self) -> Dict[str, Any]:
        """GET / - Root endpoint"""
        return {
            "name": "Example Adapter",
            "version": "1.0.0",
            "endpoints": [
                "GET /",
                "GET /status",
                "GET /get_tasks",
                "GET /health",
                "POST /start_training",
                "POST /set_task"
            ]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """GET /status - Get current status"""
        return {
            "current_task": self.current_task,
            "training_started": self.training_started,
            "task_count": len(self.tasks),
            "state": "running" if self.training_started else "idle"
        }
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """GET /get_tasks - Get all available tasks"""
        return self.tasks
    
    def health_check(self) -> Dict[str, Any]:
        """GET /health - Health check endpoint"""
        return {
            "status": "healthy",
            "adapter": "ExampleAdapter"
        }
    
    # =========================================================================
    # POST Endpoints - Some missing/incomplete
    # =========================================================================
    
    def start_training(self, request_data: Optional[Dict] = None) -> Dict[str, Any]:
        """POST /start_training - Start training"""
        self.training_started = True
        return {
            "success": True,
            "message": "Training started"
        }
    
    def complete_training(self, request_data: Optional[Dict] = None) -> Dict[str, Any]:
        """POST /complete_training - Mark training as complete"""
        self.training_started = False
        return {
            "success": True,
            "message": "Training completed"
        }
    
    def set_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /set_task - Set current task(s)"""
        task_ids = request_data.get("task_ids", [])
        weights = request_data.get("weights", None)
        
        self.current_task = {
            "task_ids": task_ids,
            "weights": weights
        }
        
        return {
            "success": True,
            "task_ids": task_ids
        }
    
    def add_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /add_task - Add a new task"""
        task = request_data.get("task", {})
        task_id = len(self.tasks)
        task["id"] = task_id
        self.tasks.append(task)
        
        return {
            "success": True,
            "task_id": task_id
        }
    
    def batch_add_tasks(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /batch_add_task - Add multiple tasks
        
        NOTE: This method is intentionally not implemented (just pass)
        to demonstrate validator feedback for empty implementations.
        """
        pass
    
    # NOTE: generate_replay is intentionally missing entirely
    # to demonstrate validator feedback for missing methods


# This is not inheriting from BaseAdapter - validator should warn
class StandaloneAdapter:
    """
    A standalone adapter that doesn't inherit from BaseAdapter.
    The validator should warn about this.
    """
    
    def get_status(self):
        return {"status": "ok"}
    
    def get_tasks(self):
        return []
