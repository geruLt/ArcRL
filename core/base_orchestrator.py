"""
Base Orchestrator Abstract Class

This module defines the abstract interface that all ArcRL orchestrators must implement.
Orchestrators are responsible for managing the training curriculum using a 5-agent
decision-making architecture.

Required Agent Flow:
    1. Bootcamp Designer    - Creates initial curriculum at training start (runs once)
    2. Training Diagnostician - Analyzes training metrics at each checkpoint
    3. Curriculum Architect   - Generates new curriculum based on analysis
    4. Task Fixer            - Fixes or adjusts failed/problematic tasks
    5. Training Summarizer   - Updates training state summary

The main loop follows this pattern:
    
    Bootcamp Designer (once at start)
           ↓
    Training Diagnostician (each checkpoint)
           ↓
    Curriculum Architect
           ↓
    Task Fixer
           ↓
    Training Summarizer
           ↓
    (loop back to Training Diagnostician at next checkpoint)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os


# ============================================================================
# Configuration Model
# ============================================================================

@dataclass
class OrchestratorConfig:
    """
    Configuration for orchestrators.
    
    All orchestrators must use this configuration structure.
    Configurations can be loaded from JSON files.
    """
    
    # LLM Configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_api_key_env: str = "OPENAI_API_KEY"
    
    # Orchestrator Configuration
    check_interval: int = 3000000  # Steps between curriculum updates
    api_host: str = "localhost"
    api_port: int = 8000
    
    # Monitoring Configuration
    verbose_logging: bool = True
    save_decisions_to_file: bool = False
    decisions_log_path: str = "orchestrator_decisions.json"
    
    # Stall Detection
    max_stalled_checks_at_zero: int = 50
    max_stalled_checks_running: int = 30
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OrchestratorConfig':
        """Create config from dictionary (flattened or nested)"""
        # Handle nested format (like orchestrator_config.json)
        if 'llm' in config_dict or 'orchestrator' in config_dict or 'monitoring' in config_dict:
            llm = config_dict.get('llm', {})
            orch = config_dict.get('orchestrator', {})
            mon = config_dict.get('monitoring', {})
            
            return cls(
                llm_model=llm.get('model', cls.llm_model),
                llm_temperature=llm.get('temperature', cls.llm_temperature),
                llm_api_key_env=llm.get('api_key_env', cls.llm_api_key_env),
                check_interval=orch.get('check_interval', cls.check_interval),
                api_host=orch.get('api_host', cls.api_host),
                api_port=orch.get('api_port', cls.api_port),
                verbose_logging=mon.get('verbose_logging', cls.verbose_logging),
                save_decisions_to_file=mon.get('save_decisions_to_file', cls.save_decisions_to_file),
                decisions_log_path=mon.get('decisions_log_path', cls.decisions_log_path),
            )
        else:
            # Handle flat format
            return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'OrchestratorConfig':
        """Load config from JSON file"""
        with open(file_path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary format (for JSON serialization)"""
        return {
            "llm": {
                "model": self.llm_model,
                "temperature": self.llm_temperature,
                "api_key_env": self.llm_api_key_env
            },
            "orchestrator": {
                "check_interval": self.check_interval,
                "api_host": self.api_host,
                "api_port": self.api_port
            },
            "monitoring": {
                "verbose_logging": self.verbose_logging,
                "save_decisions_to_file": self.save_decisions_to_file,
                "decisions_log_path": self.decisions_log_path
            },
            "stall_detection": {
                "max_stalled_checks_at_zero": self.max_stalled_checks_at_zero,
                "max_stalled_checks_running": self.max_stalled_checks_running
            }
        }
    
    def to_json_file(self, file_path: str):
        """Save config to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


# ============================================================================
# Agent Result Models
# ============================================================================

@dataclass
class AgentResult:
    """Base result from any agent"""
    success: bool
    reasoning: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BootcampDesignerResult(AgentResult):
    """Result from Bootcamp Designer agent"""
    selected_tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_ids: List[int] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)


@dataclass
class TrainingDiagnosticianResult(AgentResult):
    """Result from Training Diagnostician agent"""
    should_update: bool = False
    action: str = "continue"  # "continue", "increase_difficulty", "adjust_weaknesses"
    analysis: Dict[str, Any] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    latest_training_return: float = 0.0


@dataclass
class CurriculumArchitectResult(AgentResult):
    """Result from Curriculum Architect agent"""
    task_ids: List[int] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    curriculum_changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskFixerResult(AgentResult):
    """Result from Task Fixer agent"""
    fixed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    removed_tasks: List[int] = field(default_factory=list)
    added_tasks: List[int] = field(default_factory=list)


@dataclass
class TrainingSummarizerResult(AgentResult):
    """Result from Training Summarizer agent"""
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# Base Orchestrator Abstract Class
# ============================================================================

class BaseOrchestrator(ABC):
    """
    Abstract base class for ArcRL orchestrators.
    
    An orchestrator manages the training curriculum using a 5-agent architecture:
    1. Bootcamp Designer - Initial curriculum setup
    2. Training Diagnostician - Performance analysis
    3. Curriculum Architect - Curriculum generation
    4. Task Fixer - Problem resolution
    5. Training Summarizer - State summarization
    
    Required Class Attributes:
        REQUIRED_AGENTS: List of agent names that must be implemented
        REQUIRED_FLOW: The expected agent execution flow
    
    Example Usage:
        class MyOrchestrator(BaseOrchestrator):
            def ask_bootcamp_designer(self, all_tasks) -> BootcampDesignerResult:
                # Implement initial task selection
                ...
            
            def ask_training_diagnostician(self, metrics) -> TrainingDiagnosticianResult:
                # Implement performance analysis
                ...
            # ... implement all other agents
    """
    
    # Define required agents
    REQUIRED_AGENTS = [
        {
            "name": "Bootcamp Designer",
            "handler": "ask_bootcamp_designer",
            "description": "Creates initial curriculum at training start",
            "runs_once": True
        },
        {
            "name": "Training Diagnostician",
            "handler": "ask_training_diagnostician",
            "description": "Analyzes training metrics at each checkpoint",
            "runs_once": False
        },
        {
            "name": "Curriculum Architect",
            "handler": "ask_curriculum_architect",
            "description": "Generates new curriculum based on analysis",
            "runs_once": False
        },
        {
            "name": "Task Fixer",
            "handler": "ask_task_fixer",
            "description": "Fixes or adjusts failed/problematic tasks",
            "runs_once": False
        },
        {
            "name": "Training Summarizer",
            "handler": "ask_training_summarizer",
            "description": "Updates training state summary",
            "runs_once": False
        }
    ]
    
    # Define the expected flow
    REQUIRED_FLOW = [
        "ask_bootcamp_designer",          # Once at start
        "ask_training_diagnostician",     # Loop start
        "ask_curriculum_architect",
        "ask_task_fixer",
        "ask_training_summarizer",        # Loop end -> back to diagnostician
    ]
    
    # Required main loop methods
    REQUIRED_MAIN_METHODS = [
        {
            "name": "run",
            "description": "Main entry point - initializes and starts the monitoring loop",
            "is_async": True
        },
        {
            "name": "initialize",
            "description": "Setup phase - connect to adapter, get tasks, run bootcamp designer",
            "is_async": False
        },
        {
            "name": "run_checkpoint",
            "description": "Execute one checkpoint iteration (diagnostician -> architect -> fixer -> summarizer)",
            "is_async": False
        }
    ]
    
    def __init__(self, config: OrchestratorConfig, experiment_name: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: OrchestratorConfig with all settings
            experiment_name: Optional name for this experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Base URL for adapter API
        self.base_url = f"http://{config.api_host}:{config.api_port}"
        
        # State tracking
        self.iteration = 0
        self.last_check_step = 0
        self.stalled_checks = 0
        self.last_training_step = 0
        
        # Training info
        self.agent_step = 0
        self.max_agent_step = 0
        self.training_progress_percent = 0.0
        
        # Current curriculum
        self.current_task_ids: List[int] = []
        self.current_weights: List[float] = []
        self.current_tasks: List[Dict] = []
        
        # Training state summary (maintained by Training Summarizer)
        self.training_state_summary = "Training has not started yet. No curriculum applied."
        
        # WandB info
        self.wandb_entity: Optional[str] = None
        self.wandb_project: Optional[str] = None
        self.wandb_run_name: Optional[str] = None
    
    # ========================================================================
    # Required Agent Methods
    # ========================================================================
    
    @abstractmethod
    def ask_bootcamp_designer(self, all_tasks: List[Dict]) -> BootcampDesignerResult:
        """
        Agent 1: Bootcamp Designer
        
        Creates the initial curriculum at training start. This agent runs ONCE
        at the beginning of training to set up the starting curriculum.
        
        Args:
            all_tasks: List of all available tasks from the adapter
            
        Returns:
            BootcampDesignerResult with selected tasks and weights
        """
        pass
    
    @abstractmethod
    def ask_training_diagnostician(
        self,
        metrics: Dict[str, Any],
        current_tasks: List[Dict]
    ) -> TrainingDiagnosticianResult:
        """
        Agent 2: Training Diagnostician
        
        Analyzes training metrics at each checkpoint to determine if
        curriculum changes are needed.
        
        Args:
            metrics: Training metrics from WandB or other source
            current_tasks: Currently active tasks
            
        Returns:
            TrainingDiagnosticianResult with analysis and recommendations
        """
        pass
    
    @abstractmethod
    def ask_curriculum_architect(
        self,
        analysis: TrainingDiagnosticianResult,
        all_tasks: List[Dict],
        metrics: Dict[str, Any]
    ) -> CurriculumArchitectResult:
        """
        Agent 3: Curriculum Architect
        
        Generates new curriculum based on the diagnostician's analysis.
        Only called when diagnostician recommends an update.
        
        Args:
            analysis: Result from Training Diagnostician
            all_tasks: List of all available tasks
            metrics: Training metrics
            
        Returns:
            CurriculumArchitectResult with new curriculum
        """
        pass
    
    @abstractmethod
    def ask_task_fixer(
        self,
        failed_tasks: List[Dict],
        curriculum: CurriculumArchitectResult
    ) -> TaskFixerResult:
        """
        Agent 4: Task Fixer
        
        Fixes or adjusts tasks that failed or are problematic.
        Called after curriculum architect to handle any issues.
        
        Args:
            failed_tasks: Tasks that failed to create or had issues
            curriculum: Result from Curriculum Architect
            
        Returns:
            TaskFixerResult with fixed tasks
        """
        pass
    
    @abstractmethod
    def ask_training_summarizer(
        self,
        previous_summary: str,
        analysis: TrainingDiagnosticianResult,
        curriculum_result: Optional[CurriculumArchitectResult],
        current_step: int
    ) -> TrainingSummarizerResult:
        """
        Agent 5: Training Summarizer
        
        Updates the training state summary after each checkpoint.
        Maintains a running narrative of training progress.
        
        Args:
            previous_summary: Previous training state summary
            analysis: Result from Training Diagnostician
            curriculum_result: Result from Curriculum Architect (if update occurred)
            current_step: Current training step
            
        Returns:
            TrainingSummarizerResult with updated summary
        """
        pass
    
    # ========================================================================
    # Required Main Loop Methods
    # ========================================================================
    
    @abstractmethod
    async def run(self):
        """
        Main entry point for the orchestrator.
        
        This method should:
        1. Call initialize() to set up
        2. Run the monitoring loop that calls run_checkpoint() at each interval
        3. Handle graceful shutdown
        
        The monitoring loop should:
        - Check training status every N seconds
        - Call run_checkpoint() when check_interval steps have passed
        - Stop when training completes or is interrupted
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the orchestrator.
        
        This method should:
        1. Connect to the adapter API
        2. Get status and available tasks
        3. Run the Bootcamp Designer to create initial curriculum
        4. Apply the initial curriculum
        5. Start training if not already started
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def run_checkpoint(self, metrics: Dict[str, Any]) -> bool:
        """
        Execute one checkpoint iteration.
        
        This method implements the main agent flow:
        1. Training Diagnostician - Analyze metrics
        2. Curriculum Architect - Generate new curriculum (if needed)
        3. Task Fixer - Fix any issues
        4. Training Summarizer - Update summary
        
        Args:
            metrics: Training metrics for this checkpoint
            
        Returns:
            bool: True if curriculum was updated, False otherwise
        """
        pass
    
    # ========================================================================
    # Required Adapter Communication Methods
    # ========================================================================
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get status from the adapter API.
        
        Returns:
            Dict with status information from GET /status
        """
        pass
    
    @abstractmethod
    def get_all_tasks(self) -> List[Dict]:
        """
        Get all tasks from the adapter API.
        
        Returns:
            List of task dictionaries from GET /get_tasks
        """
        pass
    
    @abstractmethod
    def set_tasks(self, task_ids: List[int], weights: Optional[List[float]] = None):
        """
        Set tasks on the adapter via POST /set_task.
        
        Args:
            task_ids: List of task IDs to set
            weights: Optional weights for weighted sampling
        """
        pass
    
    @abstractmethod
    def start_training(self) -> Dict[str, Any]:
        """
        Start training via POST /start_training.
        
        Returns:
            Response from the adapter
        """
        pass
    
    # ========================================================================
    # Validation Helper
    # ========================================================================
    
    @classmethod
    def validate_implementation(cls, orchestrator_class) -> Dict[str, Any]:
        """
        Validate that an orchestrator class implements all required methods.
        
        Args:
            orchestrator_class: The orchestrator class to validate
            
        Returns:
            Dict with validation results
        """
        results = {
            "is_valid": True,
            "missing_agents": [],
            "implemented_agents": [],
            "missing_main_methods": [],
            "implemented_main_methods": [],
            "errors": [],
            "agent_coverage": {},
            "main_method_coverage": {}
        }
        
        # Check agent methods
        for agent in cls.REQUIRED_AGENTS:
            handler_name = agent["handler"]
            
            if not hasattr(orchestrator_class, handler_name):
                results["missing_agents"].append(handler_name)
                results["is_valid"] = False
                results["errors"].append(
                    f"Missing agent method: {handler_name} ({agent['name']})"
                )
                results["agent_coverage"][handler_name] = {
                    "implemented": False,
                    "name": agent["name"],
                    "description": agent["description"],
                    "runs_once": agent["runs_once"]
                }
            else:
                method = getattr(orchestrator_class, handler_name)
                
                if getattr(method, '__isabstractmethod__', False):
                    results["missing_agents"].append(handler_name)
                    results["is_valid"] = False
                    results["errors"].append(
                        f"Agent method {handler_name} is still abstract"
                    )
                    results["agent_coverage"][handler_name] = {
                        "implemented": False,
                        "name": agent["name"],
                        "description": agent["description"],
                        "runs_once": agent["runs_once"]
                    }
                else:
                    results["implemented_agents"].append(handler_name)
                    results["agent_coverage"][handler_name] = {
                        "implemented": True,
                        "name": agent["name"],
                        "description": agent["description"],
                        "runs_once": agent["runs_once"]
                    }
        
        # Check main methods
        for method_info in cls.REQUIRED_MAIN_METHODS:
            method_name = method_info["name"]
            
            if not hasattr(orchestrator_class, method_name):
                results["missing_main_methods"].append(method_name)
                results["is_valid"] = False
                results["errors"].append(
                    f"Missing main method: {method_name}"
                )
                results["main_method_coverage"][method_name] = {
                    "implemented": False,
                    "description": method_info["description"],
                    "is_async": method_info["is_async"]
                }
            else:
                method = getattr(orchestrator_class, method_name)
                
                if getattr(method, '__isabstractmethod__', False):
                    results["missing_main_methods"].append(method_name)
                    results["is_valid"] = False
                    results["errors"].append(
                        f"Main method {method_name} is still abstract"
                    )
                    results["main_method_coverage"][method_name] = {
                        "implemented": False,
                        "description": method_info["description"],
                        "is_async": method_info["is_async"]
                    }
                else:
                    results["implemented_main_methods"].append(method_name)
                    results["main_method_coverage"][method_name] = {
                        "implemented": True,
                        "description": method_info["description"],
                        "is_async": method_info["is_async"]
                    }
        
        # Check for required adapter communication methods
        adapter_methods = ["get_status", "get_all_tasks", "set_tasks", "start_training"]
        for method_name in adapter_methods:
            if not hasattr(orchestrator_class, method_name):
                results["errors"].append(f"Missing adapter method: {method_name}")
                results["is_valid"] = False
            else:
                method = getattr(orchestrator_class, method_name)
                if getattr(method, '__isabstractmethod__', False):
                    results["errors"].append(f"Adapter method {method_name} is still abstract")
                    results["is_valid"] = False
        
        return results
    
    @classmethod
    def validate_config(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that a configuration dictionary has all required fields.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            Dict with validation results
        """
        results = {
            "is_valid": True,
            "missing_fields": [],
            "present_fields": [],
            "errors": [],
            "warnings": []
        }
        
        # Required fields (nested format)
        required_structure = {
            "llm": ["model", "temperature", "api_key_env"],
            "orchestrator": ["check_interval", "api_host", "api_port"],
            "monitoring": ["verbose_logging"]
        }
        
        for section, fields in required_structure.items():
            if section not in config_dict:
                results["missing_fields"].append(section)
                results["errors"].append(f"Missing config section: {section}")
                results["is_valid"] = False
            else:
                results["present_fields"].append(section)
                for field in fields:
                    if field not in config_dict[section]:
                        results["missing_fields"].append(f"{section}.{field}")
                        results["errors"].append(f"Missing config field: {section}.{field}")
                        results["is_valid"] = False
                    else:
                        results["present_fields"].append(f"{section}.{field}")
        
        # Check for reasonable values
        if "orchestrator" in config_dict:
            interval = config_dict["orchestrator"].get("check_interval", 0)
            if interval < 100000:
                results["warnings"].append(
                    f"check_interval={interval} seems very low (recommended: 1000000+)"
                )
            
            port = config_dict["orchestrator"].get("api_port", 0)
            if port < 1024 or port > 65535:
                results["errors"].append(f"Invalid api_port: {port}")
                results["is_valid"] = False
        
        if "llm" in config_dict:
            temp = config_dict["llm"].get("temperature", 0)
            if temp < 0 or temp > 2:
                results["warnings"].append(
                    f"temperature={temp} is outside typical range (0-2)"
                )
        
        return results
