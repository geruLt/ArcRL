"""
Base Adapter Abstract Class

This module defines the abstract interface that all ArcRL adapters must implement.
Adapters are responsible for exposing a REST API that allows the orchestrator to
control the training curriculum.

Required Endpoints:
    GET  /           - Root endpoint with usage info
    GET  /status     - Get curriculum status and training metrics
    GET  /get_tasks  - Get all available tasks
    GET  /health     - Health check
    POST /start_training    - Start training (releases wait lock)
    POST /complete_training - Mark training as completed
    POST /set_task          - Set task(s) with optional weights
    POST /add_task          - Add a new task
    POST /batch_add_task    - Add multiple tasks
    POST /generate_replay   - Generate replay file
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


# ============================================================================
# Response Models (used for type hints and validation)
# ============================================================================

@dataclass
class StatusResponse:
    """Response model for /status endpoint"""
    current_task: Union[int, List[int]]
    next_task: Optional[Union[int, List[int]]]
    pending_change: bool
    task_space_size: int
    sampling_mode: str  # "single" or "weighted"
    weights: Optional[List[float]]
    state: str  # "pre-training", "training", or "completed"
    global_step: Optional[int] = None
    agent_step: Optional[int] = None
    max_agent_step: Optional[int] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class TaskInfo:
    """Task information model"""
    task_id: int
    predicate: str
    kwargs: Dict[str, Any]
    has_embedding: bool
    sampling_weight: Optional[float] = None


@dataclass
class GetTasksResponse:
    """Response model for /get_tasks endpoint"""
    tasks: List[TaskInfo]
    total: int


@dataclass
class TaskChangeResponse:
    """Response model for /set_task endpoint"""
    success: bool
    message: str
    error: str
    current_task: Union[int, List[int]]
    next_task: Optional[Union[int, List[int]]]
    pending_change: bool


@dataclass
class AddTaskResponse:
    """Response model for /add_task endpoint"""
    success: bool
    message: str
    error: str
    new_task_id: Optional[int]
    old_task_space_size: Optional[int]
    new_task_space_size: Optional[int]
    saved_to_disk: bool
    has_embedding: bool


@dataclass
class BatchAddTaskResponse:
    """Response model for /batch_add_task endpoint"""
    success: bool
    message: str
    total_requested: int
    total_added: int
    total_failed: int
    results: List[AddTaskResponse]
    old_task_space_size: Optional[int]
    new_task_space_size: Optional[int]
    saved_to_disk: bool


@dataclass
class StartTrainingResponse:
    """Response model for /start_training endpoint"""
    success: bool
    message: str
    training_started: bool


@dataclass
class CompleteTrainingResponse:
    """Response model for /complete_training endpoint"""
    success: bool
    message: str
    training_completed: bool


@dataclass
class GenerateReplayResponse:
    """Response model for /generate_replay endpoint"""
    success: bool
    message: str
    error: str
    replay_file: Optional[str]
    task_id: Optional[int]
    seed: Optional[int]


@dataclass
class HealthResponse:
    """Response model for /health endpoint"""
    status: str  # "healthy" or "unhealthy"
    message: str


# ============================================================================
# Request Models
# ============================================================================

@dataclass
class TaskChangeRequest:
    """Request model for /set_task endpoint"""
    task_ids: Union[int, List[int]]
    weights: Optional[List[float]] = None


@dataclass
class AddTaskRequest:
    """Request model for /add_task endpoint"""
    # Simple mode: predicate + kwargs
    predicate: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None
    # Advanced mode: full TaskSpec serialization
    task_spec_dict: Optional[Dict[str, Any]] = None
    # Optional for both modes
    embedding: Optional[str] = None  # Base64 encoded numpy array
    sampling_weight: Optional[float] = 1.0


@dataclass
class BatchAddTaskRequest:
    """Request model for /batch_add_task endpoint"""
    tasks: List[AddTaskRequest]


@dataclass
class GenerateReplayRequest:
    """Request model for /generate_replay endpoint"""
    task_id: Optional[int] = None
    seed: Optional[int] = None
    save_dir: Optional[str] = None


# ============================================================================
# Base Adapter Abstract Class
# ============================================================================

class BaseAdapter(ABC):
    """
    Abstract base class for ArcRL adapters.
    
    An adapter provides a REST API that allows the orchestrator to control
    the training curriculum. Implementations must provide all required endpoints.
    
    Required Class Attributes:
        REQUIRED_ENDPOINTS: List of endpoint definitions that must be implemented
    
    Example Usage:
        class MyAdapter(BaseAdapter):
            def get_root(self) -> Dict:
                return {"message": "Welcome to MyAdapter"}
            
            def get_status(self) -> StatusResponse:
                return StatusResponse(...)
            # ... implement all other methods
    """
    
    # Define required endpoints for validation
    REQUIRED_ENDPOINTS = [
        {"method": "GET", "path": "/", "handler": "get_root", "description": "Root endpoint with usage info"},
        {"method": "GET", "path": "/status", "handler": "get_status", "description": "Get curriculum status and training metrics"},
        {"method": "GET", "path": "/get_tasks", "handler": "get_tasks", "description": "Get all available tasks"},
        {"method": "GET", "path": "/health", "handler": "get_health", "description": "Health check"},
        {"method": "POST", "path": "/start_training", "handler": "start_training", "description": "Start training (releases wait lock)"},
        {"method": "POST", "path": "/complete_training", "handler": "complete_training", "description": "Mark training as completed"},
        {"method": "POST", "path": "/set_task", "handler": "set_task", "description": "Set task(s) with optional weights"},
        {"method": "POST", "path": "/add_task", "handler": "add_task", "description": "Add a new task"},
        {"method": "POST", "path": "/batch_add_task", "handler": "batch_add_task", "description": "Add multiple tasks"},
        {"method": "POST", "path": "/generate_replay", "handler": "generate_replay", "description": "Generate replay file"},
    ]
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the adapter.
        
        Args:
            host: Host to bind the API server to
            port: Port to bind the API server to
        """
        self.host = host
        self.port = port
    
    # ========================================================================
    # Required GET Endpoints
    # ========================================================================
    
    @abstractmethod
    def get_root(self) -> Dict[str, Any]:
        """
        GET / - Root endpoint with usage information.
        
        Returns:
            Dict containing API usage information, available endpoints, etc.
        """
        pass
    
    @abstractmethod
    def get_status(self) -> StatusResponse:
        """
        GET /status - Get curriculum status and training metrics.
        
        Returns:
            StatusResponse with current task, training state, metrics, etc.
        """
        pass
    
    @abstractmethod
    def get_tasks(self) -> GetTasksResponse:
        """
        GET /get_tasks - Get all available tasks.
        
        Returns:
            GetTasksResponse with list of all tasks and their details.
        """
        pass
    
    @abstractmethod
    def get_health(self) -> HealthResponse:
        """
        GET /health - Health check endpoint.
        
        Returns:
            HealthResponse indicating if the adapter is healthy.
        """
        pass
    
    # ========================================================================
    # Required POST Endpoints
    # ========================================================================
    
    @abstractmethod
    def start_training(self) -> StartTrainingResponse:
        """
        POST /start_training - Start training (releases wait lock).
        
        This endpoint signals that training should begin. Implementations
        should release any locks that are holding the training process.
        
        Returns:
            StartTrainingResponse indicating success/failure.
        """
        pass
    
    @abstractmethod
    def complete_training(self) -> CompleteTrainingResponse:
        """
        POST /complete_training - Mark training as completed.
        
        This endpoint signals that training has finished.
        
        Returns:
            CompleteTrainingResponse indicating success/failure.
        """
        pass
    
    @abstractmethod
    def set_task(self, request: TaskChangeRequest) -> TaskChangeResponse:
        """
        POST /set_task - Set task(s) with optional weights.
        
        Args:
            request: TaskChangeRequest with task IDs and optional weights
            
        Returns:
            TaskChangeResponse indicating success/failure and current state.
        """
        pass
    
    @abstractmethod
    def add_task(self, request: AddTaskRequest) -> AddTaskResponse:
        """
        POST /add_task - Add a new task to the task space.
        
        Args:
            request: AddTaskRequest with task definition
            
        Returns:
            AddTaskResponse with new task ID and status.
        """
        pass
    
    @abstractmethod
    def batch_add_task(self, request: BatchAddTaskRequest) -> BatchAddTaskResponse:
        """
        POST /batch_add_task - Add multiple tasks at once.
        
        Args:
            request: BatchAddTaskRequest with list of tasks
            
        Returns:
            BatchAddTaskResponse with results for each task.
        """
        pass
    
    @abstractmethod
    def generate_replay(self, request: GenerateReplayRequest) -> GenerateReplayResponse:
        """
        POST /generate_replay - Generate a replay file for a task.
        
        Args:
            request: GenerateReplayRequest with task ID and options
            
        Returns:
            GenerateReplayResponse with replay file path.
        """
        pass
    
    # ========================================================================
    # Optional: Server Management
    # ========================================================================
    
    def start_server(self):
        """
        Start the API server.
        
        Default implementation raises NotImplementedError.
        Subclasses should implement this to start their chosen web framework.
        """
        raise NotImplementedError(
            "Subclasses must implement start_server() to start the API server"
        )
    
    def stop_server(self):
        """
        Stop the API server.
        
        Default implementation raises NotImplementedError.
        Subclasses should implement this to stop their chosen web framework.
        """
        raise NotImplementedError(
            "Subclasses must implement stop_server() to stop the API server"
        )
    
    # ========================================================================
    # Validation Helper
    # ========================================================================
    
    @classmethod
    def validate_implementation(cls, adapter_class) -> Dict[str, Any]:
        """
        Validate that an adapter class implements all required methods.
        
        Args:
            adapter_class: The adapter class to validate
            
        Returns:
            Dict with validation results:
                - is_valid: bool
                - missing_methods: List of missing method names
                - implemented_methods: List of implemented method names
                - errors: List of error messages
        """
        results = {
            "is_valid": True,
            "missing_methods": [],
            "implemented_methods": [],
            "errors": [],
            "endpoint_coverage": {}
        }
        
        for endpoint in cls.REQUIRED_ENDPOINTS:
            handler_name = endpoint["handler"]
            
            # Check if method exists
            if not hasattr(adapter_class, handler_name):
                results["missing_methods"].append(handler_name)
                results["is_valid"] = False
                results["errors"].append(
                    f"Missing method: {handler_name} for {endpoint['method']} {endpoint['path']}"
                )
                results["endpoint_coverage"][endpoint["path"]] = {
                    "implemented": False,
                    "method": endpoint["method"],
                    "handler": handler_name,
                    "description": endpoint["description"]
                }
            else:
                method = getattr(adapter_class, handler_name)
                
                # Check if it's still abstract
                if getattr(method, '__isabstractmethod__', False):
                    results["missing_methods"].append(handler_name)
                    results["is_valid"] = False
                    results["errors"].append(
                        f"Method {handler_name} is still abstract (not implemented)"
                    )
                    results["endpoint_coverage"][endpoint["path"]] = {
                        "implemented": False,
                        "method": endpoint["method"],
                        "handler": handler_name,
                        "description": endpoint["description"]
                    }
                else:
                    results["implemented_methods"].append(handler_name)
                    results["endpoint_coverage"][endpoint["path"]] = {
                        "implemented": True,
                        "method": endpoint["method"],
                        "handler": handler_name,
                        "description": endpoint["description"]
                    }
        
        return results
