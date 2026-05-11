"""
ArcRL Validator - Static Validation Tool for Adapters and Orchestrators

This Streamlit app provides a user interface for validating adapter and orchestrator
implementations against the base abstract classes. It performs static analysis
without running any servers or making any network calls.

Features:
    - Drag and drop Python files for validation
    - Validates adapter implementations against BaseAdapter
    - Validates orchestrator implementations against BaseOrchestrator
    - Validates configuration files against expected schema
    - Shows completion percentage and detailed feedback
    - Highlights missing implementations and structural issues

Usage:
    streamlit run tools/validator.py
"""

import streamlit as st
import ast
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import inspect

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_adapter import BaseAdapter
from core.base_orchestrator import BaseOrchestrator, OrchestratorConfig


# Page configuration
st.set_page_config(
    page_title="ArcRL Validator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* ============================================================ */
    /* NUCLEAR OPTION: ABSOLUTE ZERO SPACING - FORCE TO TOP       */
    /* ============================================================ */
    
    /* Kill ALL top padding/margin at every level */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 0.5rem !important;
        margin-top: 0 !important;
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    .main {
        background-color: #0e1117;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Force first child to top */
    .main > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    .main > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Kill spacing on ALL structural divs */
    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Zero spacing for ALL headers */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }
    
    h1 {
        color: #ffffff;
        font-size: 1.8rem;
    }
    
    /* Zero spacing for paragraphs and markdown */
        p, [data-testid="stMarkdown"], [data-testid="stMarkdown"] p {
            margin-top: 0 !important;
            padding-top: 0 !important;
            margin-bottom: 0.3rem !important;
    }
    
    /* Tabs immediately follow content */
    .stTabs {
        margin-top: 0.05rem !important;
        padding-top: 0 !important;
    }
    
    /* Element containers compressed */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    
    /* Compact metrics */
    .stMetric {
        background-color: #1e2130;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #2e3241;
    }
    .stMetric label {
        font-size: 0.75rem;
        margin-bottom: 2px;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Compact validation banners */
    .validation-success {
        background-color: #1a472a;
        padding: 6px 10px;
        border-radius: 4px;
        border-left: 3px solid #00ff00;
        margin: 3px 0;
        font-size: 0.85rem;
    }
    
    .validation-error {
        background-color: #4a1a1a;
        padding: 6px 10px;
        border-radius: 4px;
        border-left: 3px solid #ff4444;
        margin: 3px 0;
        font-size: 0.85rem;
    }
    
    .validation-warning {
        background-color: #4a3a1a;
        padding: 6px 10px;
        border-radius: 4px;
        border-left: 3px solid #ffaa00;
        margin: 3px 0;
        font-size: 0.85rem;
    }
    
    .progress-good {
        color: #00ff00;
    }
    
    .progress-partial {
        color: #ffaa00;
    }
    
    .progress-bad {
        color: #ff4444;
    }
    
    /* Compact headers */
    h1 {
        color: #ffffff;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }
    h2 {
        color: #ffffff;
        font-size: 1.3rem;
        margin-bottom: 0.4rem;
        margin-top: 0.5rem;
    }
    h3 {
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
        margin-top: 0.3rem;
    }
    
    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        width: 280px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    section[data-testid="stSidebar"] .stButton button {
        padding: 0.4rem 0.6rem;
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] .stButton {
        margin-bottom: 0.6rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.3rem;
    }
    
    /* Remove excessive vertical spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Compact dividers */
    hr {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Static Analysis Classes
# ============================================================================

@dataclass
class MethodInfo:
    """Information about a method found in source code"""
    name: str
    is_async: bool
    decorators: List[str]
    has_body: bool  # True if method has real implementation (not just pass/...)
    line_number: int
    docstring: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about a class found in source code"""
    name: str
    bases: List[str]
    methods: List[MethodInfo]
    line_number: int
    docstring: Optional[str] = None


class StaticAnalyzer:
    """Performs static analysis on Python source code"""
    
    @staticmethod
    def analyze_file(source_code: str) -> Dict[str, Any]:
        """
        Analyze Python source code and extract class/method information.
        
        Args:
            source_code: Python source code as string
            
        Returns:
            Dict with analysis results
        """
        result = {
            "success": True,
            "error": None,
            "classes": [],
            "imports": [],
            "has_syntax_error": False
        }
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            result["success"] = False
            result["error"] = f"Syntax error at line {e.lineno}: {e.msg}"
            result["has_syntax_error"] = True
            return result
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    result["imports"].append(f"{module}.{alias.name}")
        
        # Extract classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_info = StaticAnalyzer._analyze_class(node)
                result["classes"].append(class_info)
        
        return result
    
    @staticmethod
    def _analyze_class(node: ast.ClassDef) -> ClassInfo:
        """Analyze a class definition node"""
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = StaticAnalyzer._analyze_method(item)
                methods.append(method_info)
        
        return ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            line_number=node.lineno,
            docstring=docstring
        )
    
    @staticmethod
    def _analyze_method(node) -> MethodInfo:
        """Analyze a method definition node"""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
        
        # Check if method has real implementation
        has_body = StaticAnalyzer._has_implementation(node)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        return MethodInfo(
            name=node.name,
            is_async=is_async,
            decorators=decorators,
            has_body=has_body,
            line_number=node.lineno,
            docstring=docstring
        )
    
    @staticmethod
    def _has_implementation(node) -> bool:
        """Check if a function/method has real implementation (not just pass/...)"""
        body = node.body
        
        # Skip docstring if present
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Str, ast.Constant)):
            body = body[1:]
        
        if not body:
            return False
        
        # Check for pass statement
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return False
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Handles ... (Ellipsis)
                if stmt.value.value is ...:
                    return False
            if isinstance(stmt, ast.Raise):
                # raise NotImplementedError is considered not implemented
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                        return False
        
        return True


# ============================================================================
# Validator Classes
# ============================================================================

class AdapterValidator:
    """Validates adapter implementations against BaseAdapter"""
    
    @staticmethod
    def validate(source_code: str) -> Dict[str, Any]:
        """
        Validate an adapter implementation.
        
        Args:
            source_code: Python source code of the adapter
            
        Returns:
            Validation results
        """
        result = {
            "is_valid": False,
            "completion_percentage": 0.0,
            "errors": [],
            "warnings": [],
            "implemented": [],
            "missing": [],
            "endpoint_coverage": {},
            "class_found": False,
            "inherits_base": False,
            "syntax_valid": True
        }
        
        # Analyze source
        analysis = StaticAnalyzer.analyze_file(source_code)
        
        if not analysis["success"]:
            result["syntax_valid"] = False
            result["errors"].append(analysis["error"])
            return result
        
        # Find adapter class
        adapter_classes = []
        for cls in analysis["classes"]:
            if "BaseAdapter" in cls.bases or any("BaseAdapter" in b for b in cls.bases):
                adapter_classes.append(cls)
                result["inherits_base"] = True
        
        if not adapter_classes:
            # Check if any class might be an adapter without explicit inheritance
            for cls in analysis["classes"]:
                if "adapter" in cls.name.lower():
                    adapter_classes.append(cls)
                    result["warnings"].append(
                        f"Class '{cls.name}' looks like an adapter but doesn't inherit from BaseAdapter"
                    )
        
        if not adapter_classes:
            result["errors"].append("No adapter class found (should inherit from BaseAdapter)")
            return result
        
        result["class_found"] = True
        
        # Use the first adapter class found
        adapter_class = adapter_classes[0]
        
        # Check required endpoints
        required_handlers = {ep["handler"]: ep for ep in BaseAdapter.REQUIRED_ENDPOINTS}
        
        implemented_methods = {m.name: m for m in adapter_class.methods}
        
        implemented_count = 0
        total_required = len(required_handlers)
        
        for handler_name, endpoint in required_handlers.items():
            if handler_name in implemented_methods:
                method = implemented_methods[handler_name]
                
                if method.has_body:
                    result["implemented"].append(handler_name)
                    result["endpoint_coverage"][endpoint["path"]] = {
                        "implemented": True,
                        "method": endpoint["method"],
                        "handler": handler_name,
                        "description": endpoint["description"],
                        "line": method.line_number
                    }
                    implemented_count += 1
                else:
                    result["missing"].append(handler_name)
                    result["endpoint_coverage"][endpoint["path"]] = {
                        "implemented": False,
                        "method": endpoint["method"],
                        "handler": handler_name,
                        "description": endpoint["description"],
                        "reason": "Method exists but has no implementation (pass/...)"
                    }
                    result["errors"].append(
                        f"Method '{handler_name}' has no implementation for {endpoint['method']} {endpoint['path']}"
                    )
            else:
                result["missing"].append(handler_name)
                result["endpoint_coverage"][endpoint["path"]] = {
                    "implemented": False,
                    "method": endpoint["method"],
                    "handler": handler_name,
                    "description": endpoint["description"],
                    "reason": "Method not found"
                }
                result["errors"].append(
                    f"Missing method '{handler_name}' for {endpoint['method']} {endpoint['path']}"
                )
        
        # Calculate completion
        result["completion_percentage"] = (implemented_count / total_required) * 100 if total_required > 0 else 0
        result["is_valid"] = implemented_count == total_required and result["inherits_base"]
        
        return result


class OrchestratorValidator:
    """Validates orchestrator implementations against BaseOrchestrator"""
    
    @staticmethod
    def validate(source_code: str) -> Dict[str, Any]:
        """
        Validate an orchestrator implementation.
        
        Args:
            source_code: Python source code of the orchestrator
            
        Returns:
            Validation results
        """
        result = {
            "is_valid": False,
            "completion_percentage": 0.0,
            "errors": [],
            "warnings": [],
            "implemented_agents": [],
            "missing_agents": [],
            "implemented_main": [],
            "missing_main": [],
            "agent_coverage": {},
            "main_coverage": {},
            "class_found": False,
            "inherits_base": False,
            "syntax_valid": True,
            "flow_valid": False
        }
        
        # Analyze source
        analysis = StaticAnalyzer.analyze_file(source_code)
        
        if not analysis["success"]:
            result["syntax_valid"] = False
            result["errors"].append(analysis["error"])
            return result
        
        # Find orchestrator class
        orchestrator_classes = []
        for cls in analysis["classes"]:
            if "BaseOrchestrator" in cls.bases or any("BaseOrchestrator" in b for b in cls.bases):
                orchestrator_classes.append(cls)
                result["inherits_base"] = True
        
        if not orchestrator_classes:
            # Check if any class might be an orchestrator without explicit inheritance
            for cls in analysis["classes"]:
                if "orchestrator" in cls.name.lower():
                    orchestrator_classes.append(cls)
                    result["warnings"].append(
                        f"Class '{cls.name}' looks like an orchestrator but doesn't inherit from BaseOrchestrator"
                    )
        
        if not orchestrator_classes:
            result["errors"].append("No orchestrator class found (should inherit from BaseOrchestrator)")
            return result
        
        result["class_found"] = True
        
        # Use the first orchestrator class found
        orch_class = orchestrator_classes[0]
        
        implemented_methods = {m.name: m for m in orch_class.methods}
        
        # Check agent methods
        agent_handlers = {agent["handler"]: agent for agent in BaseOrchestrator.REQUIRED_AGENTS}
        agent_implemented = 0
        
        for handler_name, agent in agent_handlers.items():
            if handler_name in implemented_methods:
                method = implemented_methods[handler_name]
                
                if method.has_body:
                    result["implemented_agents"].append(handler_name)
                    result["agent_coverage"][handler_name] = {
                        "implemented": True,
                        "name": agent["name"],
                        "description": agent["description"],
                        "runs_once": agent["runs_once"],
                        "line": method.line_number
                    }
                    agent_implemented += 1
                else:
                    result["missing_agents"].append(handler_name)
                    result["agent_coverage"][handler_name] = {
                        "implemented": False,
                        "name": agent["name"],
                        "description": agent["description"],
                        "runs_once": agent["runs_once"],
                        "reason": "Method exists but has no implementation"
                    }
                    result["errors"].append(
                        f"Agent method '{handler_name}' ({agent['name']}) has no implementation"
                    )
            else:
                result["missing_agents"].append(handler_name)
                result["agent_coverage"][handler_name] = {
                    "implemented": False,
                    "name": agent["name"],
                    "description": agent["description"],
                    "runs_once": agent["runs_once"],
                    "reason": "Method not found"
                }
                result["errors"].append(
                    f"Missing agent method '{handler_name}' ({agent['name']})"
                )
        
        # Check main methods
        main_methods = {m["name"]: m for m in BaseOrchestrator.REQUIRED_MAIN_METHODS}
        main_implemented = 0
        
        for method_name, method_info in main_methods.items():
            if method_name in implemented_methods:
                method = implemented_methods[method_name]
                
                if method.has_body:
                    result["implemented_main"].append(method_name)
                    result["main_coverage"][method_name] = {
                        "implemented": True,
                        "description": method_info["description"],
                        "is_async": method_info["is_async"],
                        "line": method.line_number
                    }
                    main_implemented += 1
                    
                    # Check if async matches expectation
                    if method_info["is_async"] and not method.is_async:
                        result["warnings"].append(
                            f"Method '{method_name}' should be async but isn't"
                        )
                else:
                    result["missing_main"].append(method_name)
                    result["main_coverage"][method_name] = {
                        "implemented": False,
                        "description": method_info["description"],
                        "is_async": method_info["is_async"],
                        "reason": "Method exists but has no implementation"
                    }
                    result["errors"].append(
                        f"Main method '{method_name}' has no implementation"
                    )
            else:
                result["missing_main"].append(method_name)
                result["main_coverage"][method_name] = {
                    "implemented": False,
                    "description": method_info["description"],
                    "is_async": method_info["is_async"],
                    "reason": "Method not found"
                }
                result["errors"].append(
                    f"Missing main method '{method_name}'"
                )
        
        # Check adapter communication methods
        adapter_methods = ["get_status", "get_all_tasks", "set_tasks", "start_training"]
        for method_name in adapter_methods:
            if method_name not in implemented_methods:
                result["errors"].append(f"Missing adapter communication method: {method_name}")
            elif not implemented_methods[method_name].has_body:
                result["errors"].append(f"Adapter method '{method_name}' has no implementation")
        
        # Validate flow (check if run method calls agents in correct order)
        if "run" in implemented_methods:
            result["flow_valid"] = OrchestratorValidator._check_agent_flow(source_code)
            if not result["flow_valid"]:
                result["warnings"].append(
                    "Could not verify agent flow order in run() method. "
                    "Ensure agents are called in order: Bootcamp Designer → Training Diagnostician → "
                    "Curriculum Architect → Task Fixer → Training Summarizer"
                )
        
        # Calculate completion
        total_agents = len(agent_handlers)
        total_main = len(main_methods)
        total = total_agents + total_main
        implemented = agent_implemented + main_implemented
        
        result["completion_percentage"] = (implemented / total) * 100 if total > 0 else 0
        result["is_valid"] = (
            agent_implemented == total_agents and 
            main_implemented == total_main and 
            result["inherits_base"]
        )
        
        return result
    
    @staticmethod
    def _check_agent_flow(source_code: str) -> bool:
        """
        Check if the run method calls agents in the correct order.
        This is a simple heuristic check.
        """
        # Look for method calls in order
        expected_order = [
            "ask_bootcamp_designer",
            "ask_training_diagnostician",
            "ask_curriculum_architect",
            "ask_task_fixer",
            "ask_training_summarizer"
        ]
        
        # Find positions of each method call
        positions = {}
        for method in expected_order:
            pos = source_code.find(method)
            if pos != -1:
                positions[method] = pos
        
        # Check if at least the main loop agents are present and in order
        loop_agents = expected_order[1:]  # Skip bootcamp designer (runs once)
        loop_positions = [positions.get(m, -1) for m in loop_agents]
        
        # Check if all loop agents are present
        if -1 in loop_positions:
            return False
        
        # Check if they're in order
        return loop_positions == sorted(loop_positions)


class ConfigValidator:
    """Validates configuration files against expected schema"""
    
    @staticmethod
    def validate(config_content: str) -> Dict[str, Any]:
        """
        Validate a configuration file.
        
        Args:
            config_content: JSON configuration content
            
        Returns:
            Validation results
        """
        result = {
            "is_valid": False,
            "completion_percentage": 0.0,
            "errors": [],
            "warnings": [],
            "present_fields": [],
            "missing_fields": [],
            "field_coverage": {},
            "parse_success": True
        }
        
        # Parse JSON
        try:
            config = json.loads(config_content)
        except json.JSONDecodeError as e:
            result["parse_success"] = False
            result["errors"].append(f"JSON parse error: {e}")
            return result
        
        # Use BaseOrchestrator's config validation
        validation = BaseOrchestrator.validate_config(config)
        
        result["is_valid"] = validation["is_valid"]
        result["errors"] = validation["errors"]
        result["warnings"] = validation["warnings"]
        result["present_fields"] = validation["present_fields"]
        result["missing_fields"] = validation["missing_fields"]
        
        # Calculate completion based on required fields
        required_fields = [
            "llm", "llm.model", "llm.temperature", "llm.api_key_env",
            "orchestrator", "orchestrator.check_interval", 
            "orchestrator.api_host", "orchestrator.api_port",
            "monitoring", "monitoring.verbose_logging"
        ]
        
        present_count = len([f for f in required_fields if f in validation["present_fields"]])
        result["completion_percentage"] = (present_count / len(required_fields)) * 100
        
        # Build field coverage
        for field in required_fields:
            result["field_coverage"][field] = {
                "present": field in validation["present_fields"],
                "required": True
            }
        
        return result


# ============================================================================
# Streamlit App
# ============================================================================

class ValidatorApp:
    """Main Streamlit application"""
    
    def __init__(self):
        # Initialize session state
        if 'adapter_file' not in st.session_state:
            st.session_state.adapter_file = None
        if 'orchestrator_file' not in st.session_state:
            st.session_state.orchestrator_file = None
        if 'config_file' not in st.session_state:
            st.session_state.config_file = None
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = {}
        # Note: theme is controlled by Streamlit's Settings modal (theme.base)
    
    def render_sidebar(self):
        """Render sidebar with file upload"""
        st.sidebar.title("📁 File Upload")
        
        # Add spacing around file uploaders
        st.sidebar.markdown("""<style>
            [data-testid="stFileUploader"] {
                margin-top: 12px !important;
                margin-bottom: 12px !important;
                padding: 8px !important;
            }
            [data-testid="stFileUploader"] section > input + div {
                min-height: 60px !important;
            }
        </style>""", unsafe_allow_html=True)
        
        # Adapter file
        st.sidebar.markdown("🔌 **Adapter** (.py)")
        adapter_file = st.sidebar.file_uploader(
            "Upload adapter", type=['py'], key='adapter_upload', label_visibility="collapsed"
        )
        if adapter_file:
            st.session_state.adapter_file = adapter_file.read().decode('utf-8')
            st.sidebar.caption(f"✅ {adapter_file.name}")
        
        st.sidebar.markdown("")  # Spacer
        
        # Orchestrator file  
        st.sidebar.markdown("🎭 **Orchestrator** (.py)")
        orchestrator_file = st.sidebar.file_uploader(
            "Upload orchestrator", type=['py'], key='orchestrator_upload', label_visibility="collapsed"
        )
        if orchestrator_file:
            st.session_state.orchestrator_file = orchestrator_file.read().decode('utf-8')
            st.sidebar.caption(f"✅ {orchestrator_file.name}")
        
        st.sidebar.markdown("")  # Spacer
        
        # Config file
        st.sidebar.markdown("⚙️ **Config** (.json)")
        config_file = st.sidebar.file_uploader(
            "Upload config", type=['json'], key='config_upload', label_visibility="collapsed"
        )
        if config_file:
            st.session_state.config_file = config_file.read().decode('utf-8')
            st.sidebar.caption(f"✅ {config_file.name}")
        
        st.sidebar.markdown("")
        # Detect theme and inject appropriate CSS
        try:
            theme_base = st.get_option("theme.base")
            # If theme.base returns None, check if backgroundColor indicates light theme
            if theme_base is None:
                try:
                    bg_color = st.get_option("theme.backgroundColor")
                    # If bg color is light/white-ish, use light theme
                    if bg_color and bg_color.lower() in ["#ffffff", "#fff", "white", ""]:
                        theme_base = "light"
                    else:
                        theme_base = "light"  # Default to light when None
                except:
                    theme_base = "light"  # Default to light
        except Exception:
            theme_base = "light"  # Default to light instead of dark

        if theme_base == "light" or theme_base is None:
            light_css = """
            <style>
                /* ============================================================ */
                /* NUCLEAR OPTION: ABSOLUTE ZERO SPACING - Arctic Light       */
                /* ============================================================ */
                
                /* Kill ALL top padding/margin at every level */
                .main .block-container {
                    padding-top: 0 !important;
                    padding-bottom: 0.5rem !important;
                    margin-top: 0 !important;
                    background-color: #F8FDFF !important;
                    max-width: 100% !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                }
                
                .main {
                    background-color: #F8FDFF !important;
                    color: #0b2132 !important;
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }
                
                /* Force first child to top */
                .main > div:first-child {
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }
                
                .main > div {
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }
                
                /* Kill spacing on ALL structural divs */
                div[data-testid="stVerticalBlock"] {
                    gap: 0 !important;
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }
                
                div[data-testid="stVerticalBlock"] > div:first-child {
                    padding-top: 0 !important;
                    margin-top: 0 !important;
                }
                
                /* Zero spacing for ALL headers */
                h1, h2, h3, h4, h5, h6 {
                    margin-top: 0 !important;
                    padding-top: 0 !important;
                    margin-bottom: 0.1rem !important;
                }
                
                /* Zero spacing for paragraphs and markdown */
                p, [data-testid="stMarkdown"], [data-testid="stMarkdown"] p {
                    margin-top: 0 !important;
                    padding-top: 0 !important;
                    margin-bottom: 0.05rem !important;
                }
                
                /* Tabs immediately follow content */
                .stTabs {
                    margin-top: 0.05rem !important;
                    padding-top: 0 !important;
                }
                
                /* Sidebar */
                section[data-testid="stSidebar"] {
                    background-color: #EBF8FF !important;
                }
                section[data-testid="stSidebar"] .block-container {
                    background-color: #EBF8FF !important;
                }
                section[data-testid="stSidebar"] * {
                    color: #062033 !important;
                }
                
                /* Compact metrics */
                .stMetric {
                    background-color: #F0FBFF !important;
                    padding: 8px 12px !important;
                    border-radius: 6px !important;
                    border: 1px solid #D7EEF9 !important;
                    color: #072033 !important;
                }
                .stMetric label {
                    font-size: 0.75rem !important;
                    margin-bottom: 2px !important;
                }
                .stMetric [data-testid="stMetricValue"] {
                    font-size: 1.2rem !important;
                    font-weight: 600 !important;
                }
                div[data-testid="stMetricValue"] {
                    color: #062033 !important;
                }
                div[data-testid="stMetricLabel"] {
                    color: #0A4A6A !important;
                }
                div[data-testid="stMetricDelta"] {
                    color: #0077B6 !important;
                }
                
                /* Compact validation banners */
                .validation-success {
                    background-color: #E8F8F2 !important;
                    padding: 6px 10px !important;
                    border-radius: 4px !important;
                    border-left: 3px solid #00A86B !important;
                    margin: 3px 0 !important;
                    color: #063028 !important;
                    font-size: 0.85rem !important;
                }
                .validation-error {
                    background-color: #FFF3F4 !important;
                    padding: 6px 10px !important;
                    border-radius: 4px !important;
                    border-left: 3px solid #D6453A !important;
                    margin: 3px 0 !important;
                    color: #2b1214 !important;
                    font-size: 0.85rem !important;
                }
                .validation-warning {
                    background-color: #FFFAF0 !important;
                    padding: 6px 10px !important;
                    border-radius: 4px !important;
                    border-left: 3px solid #FF9900 !important;
                    margin: 3px 0 !important;
                    color: #2b1d00 !important;
                    font-size: 0.85rem !important;
                }
                
                /* Progress indicators */
                .progress-good { color: #0077B6 !important; }
                .progress-partial { color: #0A6A9A !important; }
                .progress-bad { color: #D6453A !important; }
                
                /* Progress bars */
                .stProgress > div > div > div {
                    background-color: #0077B6 !important;
                }
                .stProgress > div > div {
                    background-color: #D7EEF9 !important;
                }
                
                /* Compact headers */
                h1 {
                    color: #062033 !important;
                    font-size: 1.8rem !important;
                    margin-bottom: 0.5rem !important;
                    margin-top: 0 !important;
                }
                h2 {
                    color: #062033 !important;
                    font-size: 1.3rem !important;
                    margin-bottom: 0.4rem !important;
                    margin-top: 0.5rem !important;
                }
                h3, h4, h5, h6 {
                    color: #062033 !important;
                    font-size: 1.1rem !important;
                    margin-bottom: 0.3rem !important;
                    margin-top: 0.3rem !important;
                }
                
                /* Compact sidebar */
                section[data-testid="stSidebar"] {
                    width: 280px !important;
                }
                section[data-testid="stSidebar"] .block-container {
                    padding-top: 1rem !important;
                    padding-bottom: 0.5rem !important;
                }
                section[data-testid="stSidebar"] h1 {
                    font-size: 1.2rem !important;
                    margin-bottom: 0.5rem !important;
                }
                section[data-testid="stSidebar"] h2 {
                    font-size: 1rem !important;
                    margin-bottom: 0.3rem !important;
                }
                section[data-testid="stSidebar"] .stButton button {
                    padding: 0.4rem 0.6rem !important;
                    font-size: 0.85rem !important;
                }
                
                /* Remove excessive vertical spacing */
                .element-container {
                    margin-bottom: 0.5rem !important;
                }
                hr {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
                }
                
                /* ========================================================== */
                /* MULTI-COLUMN LAYOUT SUPPORT - Arctic Light Theme          */
                /* ========================================================== */
                /* Ensure columns have proper spacing and backgrounds */
                div[data-testid="column"] {
                    background-color: transparent !important;
                    padding: 0.3rem !important;
                }
                
                /* Column content should be visible */
                div[data-testid="column"] > div {
                    background-color: transparent !important;
                }
                
                /* Compact multi-column metrics */
                div[data-testid="column"] .stMetric {
                    margin-bottom: 0.4rem !important;
                }
                
                /* Validation banners in columns */
                div[data-testid="column"] .validation-success,
                div[data-testid="column"] .validation-error,
                div[data-testid="column"] .validation-warning {
                    margin: 0.2rem 0 !important;
                }
                
                /* ========================================================== */
                /* TABS - Maximum specificity with all selector patterns     */
                /* ========================================================== */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 10px !important;
                    background-color: transparent !important;
                }
                
                /* All possible tab selectors to guarantee override */
                .stTabs [data-baseweb="tab"],
                .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"],
                .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"],
                .stTabs [data-baseweb="tab-list"] > button,
                div[data-baseweb="tab-list"] button,
                button[data-baseweb="tab"],
                [role="tab"],
                .stTabs button[role="tab"] {
                    background-color: #F1FAFF !important;
                    color: #052935 !important;
                    border-radius: 8px !important;
                    padding: 8px 16px !important;
                    border: 1px solid #D7EEF9 !important;
                }
                
                /* Selected tab - all possible selectors */
                .stTabs [aria-selected="true"],
                .stTabs [data-baseweb="tab"][aria-selected="true"],
                .stTabs [data-baseweb="tab-list"] button[aria-selected="true"],
                button[data-baseweb="tab"][aria-selected="true"],
                [role="tab"][aria-selected="true"],
                .stTabs button[role="tab"][aria-selected="true"] {
                    background-color: #D7EEF9 !important;
                    color: #002335 !important;
                    border: 1px solid #B0DDF3 !important;
                    box-shadow: 0 1px 3px rgba(0,119,182,0.1) inset !important;
                }
                
                /* Tab highlight and border elements */
                .stTabs [data-baseweb="tab-highlight"] {
                    background-color: #0077B6 !important;
                }
                .stTabs [data-baseweb="tab-border"] {
                    background-color: #D7EEF9 !important;
                }
                
                /* Buttons */
                .stButton > button {
                    background-color: #F0FBFF !important;
                    color: #062033 !important;
                    border: 1px solid #D7EEF9 !important;
                    border-radius: 6px !important;
                }
                .stButton > button:hover {
                    background-color: #D7EEF9 !important;
                    border-color: #B0DDF3 !important;
                }
                .stButton > button[kind="primary"] {
                    background-color: #0077B6 !important;
                    color: #FFFFFF !important;
                    border: 1px solid #005A8C !important;
                }
                .stButton > button[kind="primary"]:hover {
                    background-color: #005A8C !important;
                }
                
                /* File uploader */
                section[data-testid="stFileUploader"] {
                    background-color: #F0FBFF !important;
                    border: 1px dashed #B0DDF3 !important;
                    border-radius: 8px !important;
                }
                section[data-testid="stFileUploader"] label {
                    color: #062033 !important;
                }
                
                /* Radio buttons */
                div[role="radiogroup"] label {
                    color: #062033 !important;
                }
                div[role="radio"] {
                    border-color: #0077B6 !important;
                }
                div[role="radio"][aria-checked="true"] {
                    background-color: #0077B6 !important;
                }
                
                /* Dividers */
                hr {
                    border-color: #D7EEF9 !important;
                }
                
                /* Captions & small text */
                .caption, small, .stCaption {
                    color: #0A4A6A !important;
                }
                
                /* Code blocks & pre */
                .stMarkdown code {
                    background-color: #EBF8FF !important;
                    color: #062033 !important;
                    border: 1px solid #D7EEF9 !important;
                }
                pre {
                    background-color: #EBF8FF !important;
                    color: #062033 !important;
                    border: 1px solid #D7EEF9 !important;
                }
                .stCodeBlock {
                    background-color: #EBF8FF !important;
                }
                
                /* Tables */
                table {
                    color: #062033 !important;
                }
                thead tr th {
                    background-color: #D7EEF9 !important;
                    color: #002335 !important;
                }
                tbody tr {
                    background-color: #F8FDFF !important;
                }
                tbody tr:nth-child(even) {
                    background-color: #F0FBFF !important;
                }
                tbody tr:hover {
                    background-color: #EBF8FF !important;
                }
                
                /* Expander */
                .streamlit-expanderHeader {
                    background-color: #F0FBFF !important;
                    color: #062033 !important;
                    border: 1px solid #D7EEF9 !important;
                }
                
                /* Alert boxes */
                .stAlert {
                    background-color: #F0FBFF !important;
                    color: #062033 !important;
                    border: 1px solid #D7EEF9 !important;
                }
                div[data-baseweb="notification"] {
                    background-color: #F0FBFF !important;
                    border: 1px solid #D7EEF9 !important;
                }
                
                /* Success/Error/Warning/Info alerts */
                .stSuccess {
                    background-color: #E8F8F2 !important;
                    color: #063028 !important;
                }
                .stError {
                    background-color: #FFF3F4 !important;
                    color: #2b1214 !important;
                }
                .stWarning {
                    background-color: #FFFAF0 !important;
                    color: #2b1d00 !important;
                }
                .stInfo {
                    background-color: #EBF8FF !important;
                    color: #062033 !important;
                }
                
                /* Markdown elements */
                .stMarkdown {
                    color: #062033 !important;
                }
                .stMarkdown a {
                    color: #0077B6 !important;
                }
                .stMarkdown a:hover {
                    color: #005A8C !important;
                }
                
                /* Ensure all text is readable */
                p, span, div, label {
                    color: #062033 !important;
                }
            </style>
            """
            st.markdown(light_css, unsafe_allow_html=True)
        elif theme_base == "dark":
            # Dark theme - inject tab styles only
            dark_css = """
            <style>
                /* Dark theme tab styles */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 10px !important;
                }
                
                .stTabs [data-baseweb="tab"],
                .stTabs [data-baseweb="tab-list\"] button[data-baseweb="tab"],
                button[data-baseweb="tab"],
                [role="tab"] {
                    background-color: #1e2130 !important;
                    color: #ffffff !important;
                    border-radius: 5px !important;
                    padding: 10px 20px !important;
                }
                
                .stTabs [aria-selected="true"],
                .stTabs [data-baseweb="tab"][aria-selected="true"],
                button[data-baseweb="tab"][aria-selected="true"],
                [role="tab"][aria-selected="true"] {
                    background-color: #2e3241 !important;
                    color: #ffffff !important;
                }
            </style>
            """
            st.markdown(dark_css, unsafe_allow_html=True)
        
        # Theme debug info (small caption)
        theme_display = theme_base if theme_base else "light (default)"
        st.sidebar.caption(f"🎨 Theme: {theme_display}")
        
        # Buttons
        if st.sidebar.button("🔍 Validate All", type="primary", use_container_width=True):
            self._run_validation()
        
        if st.sidebar.button("🗑️ Clear All", use_container_width=True):
            st.session_state.adapter_file = None
            st.session_state.orchestrator_file = None
            st.session_state.config_file = None
            st.session_state.validation_results = {}
            st.rerun()
    
    def _run_validation(self):
        """Run validation on all uploaded files"""
        results = {}
        
        if st.session_state.adapter_file:
            results['adapter'] = AdapterValidator.validate(st.session_state.adapter_file)
        
        if st.session_state.orchestrator_file:
            results['orchestrator'] = OrchestratorValidator.validate(st.session_state.orchestrator_file)
        
        if st.session_state.config_file:
            results['config'] = ConfigValidator.validate(st.session_state.config_file)
        
        st.session_state.validation_results = results
    
    def _render_cheatsheet(self):
        """Render a compact one-page cheatsheet in 2-column layout"""
        # Add CSS to make tables more compact and reduce column gap
        st.markdown("""<style>
            .cheatsheet-section table {margin-bottom: 0 !important;}
            .cheatsheet-section p {margin-bottom: 0.3rem !important;}
            .cheatsheet-section .stMarkdown {margin-bottom: 0 !important;}
            div[data-testid="column"] > div {gap: 0 !important;}
            div[data-testid="stHorizontalBlock"] {gap: 0.5rem !important;}
        </style>""", unsafe_allow_html=True)
        
        st.subheader("📖 Cheatsheet")
        
        # Main 2-column layout: Left = Adapter, Right = (Agents+Methods / Config)
        left_col, right_col = st.columns([1, 1.2], gap="small")
        
        # LEFT COLUMN: Adapter Endpoints (full height)
        with left_col:
            st.markdown("**🔌 Adapter Endpoints (10 required)**")
            st.markdown("""| Method | Endpoint | Handler |
|:------:|:---------|:--------|
| GET | `/` | `get_root` |
| GET | `/status` | `get_status` |
| GET | `/get_tasks` | `get_tasks` |
| GET | `/health` | `health_check` |
| POST | `/start_training` | `start_training` |
| POST | `/complete_training` | `complete_training` |
| POST | `/set_task` | `set_task` |
| POST | `/add_task` | `add_task` |
| POST | `/batch_add_task` | `batch_add_tasks` |
| POST | `/generate_replay` | `generate_replay` |""")
        
        # RIGHT COLUMN: Top row (Agents | Methods) + Bottom row (Config)
        with right_col:
            # Top row: 2 sub-columns for Agents and Methods
            agents_col, methods_col = st.columns([1.3, 1], gap="small")
            
            with agents_col:
                st.markdown("**🎭 Orchestrator Agents (5)**")
                st.markdown("""| # | Agent | Handler |
|:-:|:------|:--------|
| 1 | Bootcamp Designer | `ask_bootcamp_designer` |
| 2 | Training Diagnostician | `ask_training_diagnostician` |
| 3 | Curriculum Architect | `ask_curriculum_architect` |
| 4 | Task Fixer | `ask_task_fixer` |
| 5 | Training Summarizer | `ask_training_summarizer` |""")
                st.caption("*Flow: BD→TD→CA→TF→TS→(loop)*")
            
            with methods_col:
                st.markdown("**📝 Main Methods (3)**")
                st.markdown("""| Method | Async | Description |
|:-------|:-----:|:------------|
| `run` | ✓ | Main loop |
| `initialize` | | Setup |
| `run_checkpoint` | | Checkpoint |""")
            
            # Bottom row: Config Structure (full width of right column)
            st.markdown("**⚙️ Config Structure**")
            st.markdown("""```json
{"llm": {"model": "gpt-4o", "temperature": 0.7, "api_key_env": "OPENAI_API_KEY"},
 "orchestrator": {"check_interval": 3000000, "api_host": "localhost", "api_port": 8000},
 "monitoring": {"verbose_logging": true}}
```""")
        
        st.caption("⬅️ Upload files in the sidebar and click **Validate All** to check your implementation")

    def render_home(self):
        """Render home tab with cheatsheet and results"""
        results = st.session_state.validation_results
        
        if not results:
            # Show compact cheatsheet when no results
            self._render_cheatsheet()
            return
        
        # Calculate totals
        total_completion = 0
        total_count = 0
        all_valid = True
        
        for key, res in results.items():
            total_completion += res.get('completion_percentage', 0)
            total_count += 1
            if not res.get('is_valid', False):
                all_valid = False
        
        avg_completion = total_completion / total_count if total_count > 0 else 0
        
        st.subheader("📊 Validation Results")
        
        # Summary cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Completion",
                f"{avg_completion:.1f}%",
                delta="Valid ✓" if all_valid else "Issues Found"
            )
        
        with col2:
            files_valid = sum(1 for r in results.values() if r.get('is_valid', False))
            st.metric(
                "Files Validated",
                f"{files_valid}/{total_count}",
                delta="All Pass" if files_valid == total_count else f"{total_count - files_valid} Issues"
            )
        
        with col3:
            total_errors = sum(len(r.get('errors', [])) for r in results.values())
            total_warnings = sum(len(r.get('warnings', [])) for r in results.values())
            st.metric(
                "Issues Found",
                f"{total_errors} errors",
                delta=f"{total_warnings} warnings"
            )
        
        st.divider()
        
        # Individual file summaries
        st.subheader("📁 File Status")
        
        for key, res in results.items():
            icon = "✅" if res.get('is_valid', False) else "❌"
            pct = res.get('completion_percentage', 0)
            
            if pct >= 100:
                color = "progress-good"
            elif pct >= 50:
                color = "progress-partial"
            else:
                color = "progress-bad"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {icon} {key.title()}")
            
            with col2:
                st.markdown(f"<span class='{color}'>{pct:.1f}% Complete</span>", unsafe_allow_html=True)
            
            with col3:
                errors = len(res.get('errors', []))
                warnings = len(res.get('warnings', []))
                st.text(f"🔴 {errors} errors | 🟡 {warnings} warnings")
            
            st.progress(pct / 100)
            st.divider()
    
    def render_adapter_results(self):
        """Render adapter validation results"""
        st.header("🔌 Adapter Validation")
        
        results = st.session_state.validation_results.get('adapter')
        
        if not results:
            if st.session_state.adapter_file:
                st.info("Click 'Validate All' to run validation")
            else:
                st.info("Upload an adapter file to validate")
            return
        
        # Status banner
        if results['is_valid']:
            st.success("✅ Adapter implementation is valid!")
        else:
            st.error(f"❌ Adapter implementation is incomplete ({results['completion_percentage']:.1f}% complete)")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Completion", f"{results['completion_percentage']:.1f}%")
        with col2:
            st.metric("Implemented", len(results.get('implemented', [])))
        with col3:
            st.metric("Missing", len(results.get('missing', [])))
        with col4:
            st.metric("Errors", len(results.get('errors', [])))
        
        st.divider()
        
        # Endpoint coverage: 3 REAL COLUMNS with MULTIPLE endpoints each
        st.subheader("📡 Endpoint Coverage")
        
        endpoints = list(results.get('endpoint_coverage', {}).items())
        col1, col2, col3 = st.columns(3)
        
        # Split 10 endpoints into 3 columns: 4, 3, 3
        col1_endpoints = endpoints[0:4]   # First 4
        col2_endpoints = endpoints[4:7]   # Next 3
        col3_endpoints = endpoints[7:10]  # Last 3
        
        # Column 1: endpoints 0-3
        with col1:
            for path, info in col1_endpoints:
                icon = "✅" if info['implemented'] else "❌"
                st.markdown(f"**{icon} {info['method']} {path}**")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
        
        # Column 2: endpoints 4-6
        with col2:
            for path, info in col2_endpoints:
                icon = "✅" if info['implemented'] else "❌"
                st.markdown(f"**{icon} {info['method']} {path}**")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
        
        # Column 3: endpoints 7-9
        with col3:
            for path, info in col3_endpoints:
                icon = "✅" if info['implemented'] else "❌"
                st.markdown(f"**{icon} {info['method']} {path}**")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
        
        # Errors and warnings in 2 columns
        if results.get('errors') or results.get('warnings'):
            st.divider()
            err_col, warn_col = st.columns(2)
            
            with err_col:
                if results.get('errors'):
                    st.subheader("🔴 Errors")
                    for error in results['errors']:
                        st.error(error)
            
            with warn_col:
                if results.get('warnings'):
                    st.subheader("🟡 Warnings")
                    for warning in results['warnings']:
                        st.warning(warning)
        
        # Implementation guide
        if results.get('missing'):
            st.divider()
            st.subheader("📝 Implementation Guide")
            st.markdown("To complete your adapter, implement these methods:")
            
            for handler in results['missing']:
                endpoint = next(
                    (e for e in BaseAdapter.REQUIRED_ENDPOINTS if e['handler'] == handler),
                    None
                )
                if endpoint:
                    st.markdown(f"""
```python
def {handler}(self, ...):
    \"\"\"
    {endpoint['method']} {endpoint['path']} - {endpoint['description']}
    \"\"\"
    # Your implementation here
    pass
```
                    """)
    
    def render_orchestrator_results(self):
        """Render orchestrator validation results"""
        st.header("🎭 Orchestrator Validation")
        
        results = st.session_state.validation_results.get('orchestrator')
        
        if not results:
            if st.session_state.orchestrator_file:
                st.info("Click 'Validate All' to run validation")
            else:
                st.info("Upload an orchestrator file to validate")
            return
        
        # Status banner
        if results['is_valid']:
            st.success("✅ Orchestrator implementation is valid!")
        else:
            st.error(f"❌ Orchestrator implementation is incomplete ({results['completion_percentage']:.1f}% complete)")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Completion", f"{results['completion_percentage']:.1f}%")
        with col2:
            st.metric("Agents", f"{len(results.get('implemented_agents', []))}/5")
        with col3:
            st.metric("Main Methods", f"{len(results.get('implemented_main', []))}/3")
        with col4:
            flow_status = "✓" if results.get('flow_valid') else "?"
            st.metric("Flow Valid", flow_status)
        
        st.divider()
        
        # Combined coverage: 3 REAL COLUMNS with MULTIPLE items each
        # Column 1: Agents 1-2, Column 2: Agents 3-4, Column 3: Agent 5 + All 3 Methods
        st.subheader("🤖 Agent Coverage + ⚡ Main Methods")
        
        st.markdown("""
        **5-Agent Loop:** Bootcamp Designer (once) → Training Diagnostician → Curriculum Architect → Task Fixer → Training Summarizer → (loop)
        """)
        
        agents_list = list(results.get('agent_coverage', {}).items())
        methods_list = list(results.get('main_coverage', {}).items())
        
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Agents 0-1 (first 2 agents)
        with col1:
            st.markdown("**Agents (1-2):**")
            for handler, info in agents_list[0:2]:
                icon = "✅" if info['implemented'] else "❌"
                once_badge = "🔹" if info.get('runs_once') else "🔄"
                st.markdown(f"{icon} {once_badge} **{info['name']}**")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
        
        # Column 2: Agents 2-3 (next 2 agents)
        with col2:
            st.markdown("**Agents (3-4):**")
            for handler, info in agents_list[2:4]:
                icon = "✅" if info['implemented'] else "❌"
                once_badge = "🔹" if info.get('runs_once') else "🔄"
                st.markdown(f"{icon} {once_badge} **{info['name']}**")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
        
        # Column 3: Agent 5 + All 3 Main Methods
        with col3:
            st.markdown("**Agent (5):**")
            if len(agents_list) > 4:
                handler, info = agents_list[4]
                icon = "✅" if info['implemented'] else "❌"
                once_badge = "🔹" if info.get('runs_once') else "🔄"
                st.markdown(f"{icon} {once_badge} **{info['name']}**")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
            
            st.markdown("**Main Methods:**")
            for method_name, info in methods_list:
                icon = "✅" if info['implemented'] else "❌"
                async_badge = "⚡" if info.get('is_async') else ""
                st.markdown(f"{icon} {async_badge} `{method_name}`")
                st.caption(f"{info['description']}")
                if not info['implemented']:
                    st.caption(f"⚠️ {info.get('reason', 'Not implemented')}")
                st.markdown("---")
        
        # Errors and warnings
        if results.get('errors'):
            st.divider()
            st.subheader("🔴 Errors")
            for error in results['errors']:
                st.error(error)
        
        if results.get('warnings'):
            st.divider()
            st.subheader("🟡 Warnings")
            for warning in results['warnings']:
                st.warning(warning)
        
        # Implementation guide
        if results.get('missing_agents') or results.get('missing_main'):
            st.divider()
            st.subheader("📝 Implementation Guide")
            
            if results.get('missing_agents'):
                st.markdown("**Missing Agent Methods:**")
                for handler in results['missing_agents']:
                    agent = next(
                        (a for a in BaseOrchestrator.REQUIRED_AGENTS if a['handler'] == handler),
                        None
                    )
                    if agent:
                        st.markdown(f"""
```python
def {handler}(self, ...):
    \"\"\"
    {agent['name']} - {agent['description']}
    {'Runs once at start' if agent['runs_once'] else 'Runs at each checkpoint'}
    \"\"\"
    # Your implementation here
    pass
```
                        """)
            
            if results.get('missing_main'):
                st.markdown("**Missing Main Methods:**")
                for method_name in results['missing_main']:
                    method_info = next(
                        (m for m in BaseOrchestrator.REQUIRED_MAIN_METHODS if m['name'] == method_name),
                        None
                    )
                    if method_info:
                        async_kw = "async " if method_info['is_async'] else ""
                        st.markdown(f"""
```python
{async_kw}def {method_name}(self, ...):
    \"\"\"
    {method_info['description']}
    \"\"\"
    # Your implementation here
    pass
```
                        """)
    
    def render_config_results(self):
        """Render configuration validation results"""
        st.header("⚙️ Configuration Validation")
        
        results = st.session_state.validation_results.get('config')
        
        if not results:
            if st.session_state.config_file:
                st.info("Click 'Validate All' to run validation")
            else:
                st.info("Upload a configuration file to validate")
            return
        
        # Status banner
        if results['is_valid']:
            st.success("✅ Configuration is valid!")
        else:
            st.error(f"❌ Configuration is incomplete ({results['completion_percentage']:.1f}% complete)")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Completion", f"{results['completion_percentage']:.1f}%")
        with col2:
            st.metric("Present Fields", len(results.get('present_fields', [])))
        with col3:
            st.metric("Missing Fields", len(results.get('missing_fields', [])))
        
        st.divider()
        
        # Field coverage in 3 columns
        st.subheader("📋 Required Fields")
        
        fields_list = list(results.get('field_coverage', {}).items())
        field_col1, field_col2, field_col3 = st.columns(3)
        
        # Distribute fields across 3 columns
        for idx, (field, info) in enumerate(fields_list):
            if idx % 3 == 0:
                target_col = field_col1
            elif idx % 3 == 1:
                target_col = field_col2
            else:
                target_col = field_col3
            
            with target_col:
                icon = "✅" if info['present'] else "❌"
                
                if info['present']:
                    st.markdown(f"<div class='validation-success'>{icon} <code>{field}</code></div>",
                              unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='validation-error'>{icon} <code>{field}</code> - Required</div>",
                              unsafe_allow_html=True)
                st.markdown("---")
        
        # Errors and warnings
        if results.get('errors'):
            st.divider()
            st.subheader("🔴 Errors")
            for error in results['errors']:
                st.error(error)
        
        if results.get('warnings'):
            st.divider()
            st.subheader("🟡 Warnings")
            for warning in results['warnings']:
                st.warning(warning)
        
        # Example config
        if results.get('missing_fields'):
            st.divider()
            st.subheader("📝 Example Configuration")
            st.markdown("Here's the expected configuration structure:")
            
            example_config = OrchestratorConfig().to_dict()
            st.json(example_config)
    
    def render_reference(self):
        """Render reference documentation tab"""
        st.header("📚 Reference Documentation")
        
        tab1, tab2, tab3 = st.tabs(["Adapter API", "Orchestrator Agents", "Configuration"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔌 Required Adapter Endpoints**")
                st.markdown("""
                Your adapter must implement these REST API endpoints:
                
                | Method | Endpoint | Description |
                |--------|----------|-------------|
                | GET | `/` | Root endpoint with usage info |
                | GET | `/status` | Get curriculum status and training metrics |
                | GET | `/get_tasks` | Get all available tasks |
                | GET | `/health` | Health check |
                | POST | `/start_training` | Start training (releases wait lock) |
                | POST | `/complete_training` | Mark training as completed |
                | POST | `/set_task` | Set task(s) with optional weights |
                | POST | `/add_task` | Add a new task |
                | POST | `/batch_add_task` | Add multiple tasks |
                | POST | `/generate_replay` | Generate replay file |
                """)
            
            with col2:
                st.markdown("**Example Adapter Structure**")
                st.code("""
from core.base_adapter import BaseAdapter, StatusResponse, ...

class MyAdapter(BaseAdapter):
    def __init__(self, host="0.0.0.0", port=8000):
        super().__init__(host, port)
        # Your initialization
    
    def get_root(self) -> Dict[str, Any]:
        return {"message": "Welcome to MyAdapter", "endpoints": [...]}
    
    def get_status(self) -> StatusResponse:
        return StatusResponse(
            current_task=self.current_task,
            state=self.training_state,
            ...
        )
    
    # ... implement all required methods
                """, language="python")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎭 Required Orchestrator Agents**")
                st.markdown("""
                Your orchestrator must implement these 5 agents in the following flow:
                
                ```
                ┌─────────────────────┐
                │  Bootcamp Designer  │ ← Runs ONCE at start
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────────┐
                │  Training Diagnostician  │ ← Loop starts here
                └──────────┬──────────────┘
                           ↓
                ┌─────────────────────────┐
                │   Curriculum Architect   │
                └──────────┬──────────────┘
                           ↓
                ┌─────────────────────────┐
                │       Task Fixer         │
                └──────────┬──────────────┘
                           ↓
                ┌─────────────────────────┐
                │   Training Summarizer    │
                └──────────┬──────────────┘
                           ↓
                      (loop back)
                ```
                """)
            
            with col2:
                st.markdown("**Agent Details**")
                for agent in BaseOrchestrator.REQUIRED_AGENTS:
                    with st.expander(f"**{agent['name']}** - `{agent['handler']}`"):
                        st.markdown(f"**Description:** {agent['description']}")
                        st.markdown(f"**Runs:** {'Once at start' if agent['runs_once'] else 'At each checkpoint'}")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚙️ Configuration Structure**")
                st.markdown("""
                Your configuration file must include these sections:
                """)
                
                config = OrchestratorConfig()
                st.json(config.to_dict())
            
            with col2:
                st.markdown("**Field Descriptions**")
                st.markdown("""
                - `llm.model`: The LLM model to use (e.g., "gpt-4o")
                - `llm.temperature`: Temperature for LLM responses (0-2)
                - `llm.api_key_env`: Environment variable name for API key
                - `orchestrator.check_interval`: Steps between curriculum updates
                - `orchestrator.api_host`: Host for adapter API
                - `orchestrator.api_port`: Port for adapter API
                - `monitoring.verbose_logging`: Enable detailed logging
                """)
    
    def run(self):
        """Main entry point"""
        st.title("ArcRL Validator")
        st.markdown("*Static validation tool for adapters and orchestrators*")
        
        # Render sidebar
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Home",
            "🔌 Adapter",
            "🎭 Orchestrator",
            "⚙️ Config",
            "📚 Reference"
        ])
        
        with tab1:
            self.render_home()
        
        with tab2:
            self.render_adapter_results()
        
        with tab3:
            self.render_orchestrator_results()
        
        with tab4:
            self.render_config_results()
        
        with tab5:
            self.render_reference()


if __name__ == "__main__":
    app = ValidatorApp()
    app.run()
