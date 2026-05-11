"""
Shared State Management for Orchestrator Dashboard

This module provides a thread-safe interface for the orchestrator to write
state updates that the dashboard can read in real-time.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading


class StateManager:
    """
    Manages shared state between orchestrator and dashboard.
    
    Uses JSON file for persistence with file locking for thread safety.
    Supports per-experiment state files to avoid mixing different runs.
    """
    
    def __init__(self, state_file: str = "mcp_app/data/orchestrator_state.json", experiment_name: Optional[str] = None):
        """
        Initialize StateManager with optional experiment name.
        
        Args:
            state_file: Base path for state file (used if experiment_name is None)
            experiment_name: Unique experiment identifier. If provided, creates
                           experiment-specific state file: orchestrator_state_{name}.json
        """
        if experiment_name:
            # Create experiment-specific state file
            base_dir = Path(state_file).parent
            self.state_file = base_dir / f"orchestrator_state_{experiment_name}.json"
            self.experiment_name = experiment_name
        else:
            # Use default state file
            self.state_file = Path(state_file)
            self.experiment_name = None
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()  # Use RLock for reentrant locking
        
        # Initialize state file if doesn't exist
        if not self.state_file.exists():
            self._init_state()
    
    def _init_state(self):
        """Initialize empty state structure"""
        initial_state = {
            "experiment": {
                "name": self.experiment_name or "default",
                "created_at": datetime.now().isoformat(),
                "state_file": str(self.state_file)
            },
            "training": {
                "state": "pre-training",
                "current_step": 0,
                "start_time": None,
                "target_step": None,
                "wandb_entity": None,
                "wandb_project": None,
                "wandb_run_name": None
            },
            "orchestrator": {
                "status": "initializing",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "iteration": 0,
                "stalled_checks": 0
            },
            "current_tasks": [],
            "checkpoints": [],
            "llm_calls": [],
            "system_health": {
                "api_connected": False,
                "api_latency_ms": None,
                "openai_connected": False,
                "wandb_connected": False,
                "last_health_check": None
            },
            "config": {},
            "metrics": {
                "total_llm_calls": 0,
                "total_checkpoints": 0,
                "total_curriculum_updates": 0,
                "total_cost_usd": 0.0,
                "avg_llm_response_time": 0.0
            },
            "logs": []
        }
        self._write_state(initial_state)
    
    def _read_state(self) -> Dict:
        """Thread-safe read of state"""
        with self.lock:
            try:
                with open(self.state_file, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        print(f"WARNING: Empty state file {self.state_file}, reinitializing")
                        sys.stdout.flush()
                        self._init_state()
                        with open(self.state_file, 'r') as f:
                            return json.load(f)
                    return json.loads(content)
            except FileNotFoundError:
                print(f"WARNING: State file {self.state_file} not found, creating new")
                sys.stdout.flush()
                self._init_state()
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: JSONDecodeError in {self.state_file}: {e}")
                print(f"  Error at line {e.lineno}, column {e.colno}")
                print(f"  Message: {e.msg}")
                sys.stdout.flush()
                # Try to save corrupted file for debugging
                backup_file = self.state_file.parent / f"{self.state_file.stem}_corrupted_{int(time.time())}.json"
                try:
                    import shutil
                    shutil.copy2(self.state_file, backup_file)
                    print(f"  Saved corrupted file to: {backup_file}")
                except Exception as backup_err:
                    print(f"  Failed to backup corrupted file: {backup_err}")
                sys.stdout.flush()
                # Reinitialize
                self._init_state()
                with open(self.state_file, 'r') as f:
                    return json.load(f)
    
    def _write_state(self, state: Dict):
        """Thread-safe write of state with atomic operation"""
        with self.lock:
            # Use atomic write: write to temp file, then rename
            # This prevents corruption if process is killed mid-write
            temp_file = self.state_file.parent / f"{self.state_file.name}.tmp"
            try:
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=2)
                    f.flush()  # Ensure data is written to disk
                    import os
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomic rename (replaces existing file)
                temp_file.rename(self.state_file)
            except Exception as e:
                print(f"ERROR shared_state: Failed to write state: {e}")
                sys.stdout.flush()
                # Clean up temp file if it exists
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def update_training_state(
        self,
        state: Optional[str] = None,
        current_step: Optional[int] = None,
        agent_step: Optional[int] = None,
        max_agent_step: Optional[int] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        target_step: Optional[int] = None
    ):
        """Update training state information"""
        full_state = self._read_state()
        
        if state:
            full_state["training"]["state"] = state
        if current_step is not None:
            full_state["training"]["current_step"] = current_step
        if agent_step is not None:
            full_state["training"]["agent_step"] = agent_step
        if max_agent_step is not None:
            full_state["training"]["max_agent_step"] = max_agent_step
        if wandb_entity:
            full_state["training"]["wandb_entity"] = wandb_entity
        if wandb_project:
            full_state["training"]["wandb_project"] = wandb_project
        if wandb_run_name:
            full_state["training"]["wandb_run_name"] = wandb_run_name
        if target_step is not None:
            full_state["training"]["target_step"] = target_step
        
        # Set start time if transitioning to training
        if state == "training" and not full_state["training"]["start_time"]:
            full_state["training"]["start_time"] = datetime.now().isoformat()
        
        full_state["orchestrator"]["last_update"] = datetime.now().isoformat()
        self._write_state(full_state)
    
    def update_orchestrator_status(
        self,
        status: Optional[str] = None,
        iteration: Optional[int] = None,
        stalled_checks: Optional[int] = None,
        check_interval: Optional[int] = None
    ):
        """Update orchestrator status"""
        full_state = self._read_state()
        
        if status:
            full_state["orchestrator"]["status"] = status
        if iteration is not None:
            full_state["orchestrator"]["iteration"] = iteration
        if check_interval is not None:
            full_state["orchestrator"]["check_interval"] = check_interval
        if stalled_checks is not None:
            full_state["orchestrator"]["stalled_checks"] = stalled_checks
        
        full_state["orchestrator"]["last_update"] = datetime.now().isoformat()
        self._write_state(full_state)
    
    def update_current_tasks(self, tasks: List[Dict], weights: Optional[List[float]] = None):
        """Update current active tasks"""
        full_state = self._read_state()
        
        task_data = []
        for i, task in enumerate(tasks):
            task_info = {
                "task_id": task.get("task_id"),
                "predicate": task.get("predicate"),
                "kwargs": task.get("kwargs", {}),
                "weight": weights[i] if weights and i < len(weights) else None
            }
            task_data.append(task_info)
        
        full_state["current_tasks"] = task_data
        full_state["orchestrator"]["last_update"] = datetime.now().isoformat()
        self._write_state(full_state)
    
    def update_all_tasks(self, tasks: List[Dict]):
        """Update the complete list of all tasks ever created (for resume functionality)"""
        full_state = self._read_state()
        
        # Store minimal task info needed for recreation
        task_data = []
        for task in tasks:
            task_info = {
                "task_id": task.get("task_id"),
                "predicate": task.get("predicate"),
                "kwargs": task.get("kwargs", {}),
                "weight": task.get("sampling_weight", 1.0)
            }
            task_data.append(task_info)
        
        full_state["all_tasks"] = task_data
        self._write_state(full_state)
    
    def add_checkpoint(
        self,
        step: int,
        iteration: int,
        tasks: List[Dict],
        decision: Dict,
        updated: bool
    ):
        """Add a checkpoint record"""
        full_state = self._read_state()
        
        checkpoint = {
            "step": step,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "tasks": tasks,
            "decision": decision,
            "curriculum_updated": updated
        }
        
        full_state["checkpoints"].append(checkpoint)
        full_state["metrics"]["total_checkpoints"] += 1
        if updated:
            full_state["metrics"]["total_curriculum_updates"] += 1
        
        full_state["orchestrator"]["last_update"] = datetime.now().isoformat()
        self._write_state(full_state)
    
    def add_llm_call(
        self,
        call_type: str,
        prompt: str,
        response: Dict,
        duration_seconds: float,
        model: str,
        checkpoint_step: Optional[int] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd: Optional[float] = None
    ):
        """Record an LLM API call"""
        full_state = self._read_state()
        
        llm_call = {
            "timestamp": datetime.now().isoformat(),
            "type": call_type,  # "initial_selection", "performance_analysis", "curriculum_generation"
            "checkpoint_step": checkpoint_step,
            "model": model,
            "prompt": prompt,
            "response": response,
            "duration_seconds": duration_seconds,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd or 0.0
        }
        
        full_state["llm_calls"].append(llm_call)
        full_state["metrics"]["total_llm_calls"] += 1
        if cost_usd:
            full_state["metrics"]["total_cost_usd"] += cost_usd
        
        # Update average response time
        calls = full_state["metrics"]["total_llm_calls"]
        old_avg = full_state["metrics"]["avg_llm_response_time"]
        new_avg = ((old_avg * (calls - 1)) + duration_seconds) / calls
        full_state["metrics"]["avg_llm_response_time"] = new_avg
        
        full_state["orchestrator"]["last_update"] = datetime.now().isoformat()
        self._write_state(full_state)
    
    def update_system_health(
        self,
        api_connected: Optional[bool] = None,
        api_latency_ms: Optional[float] = None,
        openai_connected: Optional[bool] = None,
        wandb_connected: Optional[bool] = None
    ):
        """Update system health status"""
        full_state = self._read_state()
        
        if api_connected is not None:
            full_state["system_health"]["api_connected"] = api_connected
        if api_latency_ms is not None:
            full_state["system_health"]["api_latency_ms"] = api_latency_ms
        if openai_connected is not None:
            full_state["system_health"]["openai_connected"] = openai_connected
        if wandb_connected is not None:
            full_state["system_health"]["wandb_connected"] = wandb_connected
        
        full_state["system_health"]["last_health_check"] = datetime.now().isoformat()
        self._write_state(full_state)
    
    def add_log(self, level: str, message: str, details: Optional[Dict] = None):
        """Add a log entry"""
        full_state = self._read_state()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,  # "info", "warning", "error"
            "message": message,
            "details": details or {}
        }
        
        full_state["logs"].append(log_entry)
        
        # Keep only last 1000 logs to prevent unbounded growth
        if len(full_state["logs"]) > 1000:
            full_state["logs"] = full_state["logs"][-1000:]
        
        self._write_state(full_state)
    
    def set_config(self, config: Dict):
        """Store configuration"""
        full_state = self._read_state()
        full_state["config"] = config
        self._write_state(full_state)
    
    def get_state(self) -> Dict:
        """Get full current state"""
        return self._read_state()
    
    def get_training_state(self) -> Dict:
        """Get just training state"""
        return self._read_state()["training"]
    
    def get_current_tasks(self) -> List[Dict]:
        """Get current active tasks"""
        return self._read_state()["current_tasks"]
    
    def get_checkpoints(self) -> List[Dict]:
        """Get all checkpoints"""
        return self._read_state()["checkpoints"]
    
    def get_llm_calls(self, limit: Optional[int] = None) -> List[Dict]:
        """Get LLM calls, optionally limited to most recent N"""
        calls = self._read_state()["llm_calls"]
        if limit:
            return calls[-limit:]
        return calls
    
    def get_logs(self, limit: Optional[int] = None, level: Optional[str] = None) -> List[Dict]:
        """Get logs, optionally filtered by level and limited to most recent N"""
        logs = self._read_state()["logs"]
        
        if level:
            logs = [log for log in logs if log["level"] == level]
        
        if limit:
            return logs[-limit:]
        return logs
    
    def get_metrics(self) -> Dict:
        """Get summary metrics"""
        return self._read_state()["metrics"]
    
    def clear_old_data(self, keep_checkpoints: int = 50, keep_llm_calls: int = 100):
        """Clear old data to prevent state file from growing too large"""
        full_state = self._read_state()
        
        if len(full_state["checkpoints"]) > keep_checkpoints:
            full_state["checkpoints"] = full_state["checkpoints"][-keep_checkpoints:]
        
        if len(full_state["llm_calls"]) > keep_llm_calls:
            full_state["llm_calls"] = full_state["llm_calls"][-keep_llm_calls:]
        
        self._write_state(full_state)
    
    @staticmethod
    def list_experiments(data_dir: str = "mcp_app/data") -> List[Dict]:
        """
        List all available experiments by scanning for state files.
        
        Returns:
            List of dicts with experiment info: name, state_file, created_at, etc.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return []
        
        experiments = []
        
        # Find all orchestrator_state*.json files
        for state_file in data_path.glob("orchestrator_state*.json"):
            try:
                with open(state_file, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        print(f"WARNING: Skipping empty state file: {state_file.name}")
                        sys.stdout.flush()
                        continue
                    state = json.loads(content)
                    
                exp_info = {
                    "name": state.get("experiment", {}).get("name", "default"),
                    "state_file": str(state_file),
                    "created_at": state.get("experiment", {}).get("created_at"),
                    "wandb_run": state.get("training", {}).get("wandb_run_name"),
                    "current_step": state.get("training", {}).get("current_step", 0),
                    "training_state": state.get("training", {}).get("state", "unknown"),
                    "total_checkpoints": state.get("metrics", {}).get("total_checkpoints", 0),
                    "total_llm_calls": state.get("metrics", {}).get("total_llm_calls", 0)
                }
                experiments.append(exp_info)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping corrupted state file {state_file.name}: {e.msg} at line {e.lineno}, col {e.colno}")
                sys.stdout.flush()
                continue
            except Exception as e:
                # Skip malformed state files
                print(f"WARNING: Skipping state file {state_file.name}: {e}")
                sys.stdout.flush()
                continue
        
        # Sort by created_at (most recent first)
        experiments.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return experiments
    
    def validate_state_file(self) -> bool:
        """
        Validate that the state file contains valid JSON.
        
        Returns:
            True if valid, False if corrupted
        """
        try:
            with open(self.state_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    return False
                json.loads(content)
                return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        except Exception:
            return False
