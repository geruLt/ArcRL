"""
Streamlit Dashboard for Neural MMO Training Orchestrator

Real-time visualization of training progress, LLM decisions, and system health.
Also works as an archive viewer for past experiments.

Usage:
    # Run with streamlit (recommended):
    streamlit run tools/dashboard.py
    
    # Run directly with Python:
    python tools/dashboard.py
    # This automatically launches streamlit and allows browsing past experiments
"""

import streamlit as st
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import wandb
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from shared_state import StateManager


# Page configuration
st.set_page_config(
    page_title="ArcRL Dashboard",
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
    
    h1 {
        color: #ffffff;
        font-size: 1.8rem;
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
    
    /* Element containers compressed */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
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
        color: #a0a0a0;
        margin-bottom: 2px;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Status indicators */
    .status-running {
        color: #00ff00;
    }
    .status-stopped {
        color: #ff4444;
    }
    .status-initializing {
        color: #ffaa00;
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
        width: 300px;
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
    section[data-testid="stSidebar"] .stMetric {
        padding: 6px 8px;
        margin-bottom: 0.4rem;
    }
    section[data-testid="stSidebar"] .stMetric label {
        font-size: 0.7rem;
    }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-size: 1rem;
    }
    section[data-testid="stSidebar"] .stButton button {
        padding: 0.4rem 0.6rem;
        font-size: 0.85rem;
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
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Compact dataframes */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Streamlit dataframe styling - Dark theme */
    [data-testid="stDataFrame"] {
        --dataframe-base-color: #ffffff !important;
        --dataframe-base-background-color: #1e2130 !important;
        --dataframe-header-background-color: #2e3241 !important;
        --dataframe-header-color: #ffffff !important;
        --dataframe-stripe-background-color: #262836 !important;
        --dataframe-row-hover-background-color: #2e3241 !important;
        --dataframe-border-color: #2e3241 !important;
    }
</style>
""", unsafe_allow_html=True)


class Dashboard:
    """Main dashboard class"""
    
    def __init__(self):
        # Initialize session state for experiment selection
        if 'selected_experiment' not in st.session_state:
            st.session_state.selected_experiment = None
        
        # Get selected experiment from session state or URL params
        experiment_name = st.session_state.selected_experiment
        
        # Initialize state manager with selected experiment
        if experiment_name:
            self.state_manager = StateManager(experiment_name=experiment_name)
        else:
            # Try to load most recent experiment
            experiments = StateManager.list_experiments()
            if experiments:
                experiment_name = experiments[0]['name']
                st.session_state.selected_experiment = experiment_name
                self.state_manager = StateManager(experiment_name=experiment_name)
            else:
                # No experiments found, use default
                self.state_manager = StateManager()
        
        # Initialize session state for refresh control
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5  # seconds
        
        # Apply theme-specific CSS
        self._apply_theme_css()
    
    def _apply_theme_css(self):
        """Detect theme and apply appropriate CSS"""
        try:
            theme_base = st.get_option("theme.base")
            # If theme.base returns None, default to light
            if theme_base is None:
                try:
                    bg_color = st.get_option("theme.backgroundColor")
                    if bg_color and bg_color.lower() in ["#ffffff", "#fff", "white", ""]:
                        theme_base = "light"
                    else:
                        theme_base = "light"
                except:
                    theme_base = "light"
        except Exception:
            theme_base = "light"

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
                .main {
                    background-color: #F8FDFF !important;
                    color: #0b2132 !important;
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
                    color: #0A4A6A !important;
                    margin-bottom: 2px !important;
                }
                .stMetric [data-testid="stMetricValue"] {
                    font-size: 1.2rem !important;
                    font-weight: 600 !important;
                    color: #062033 !important;
                }
                div[data-testid="stMetricDelta"] {
                    color: #0077B6 !important;
                }
                
                /* Status indicators */
                .status-running { color: #00A86B !important; }
                .status-stopped { color: #D6453A !important; }
                .status-initializing { color: #FF9900 !important; }
                
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
                    width: 300px !important;
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
                section[data-testid="stSidebar"] .stMetric {
                    padding: 6px 8px !important;
                    margin-bottom: 0.4rem !important;
                }
                section[data-testid="stSidebar"] .stMetric label {
                    font-size: 0.7rem !important;
                }
                section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
                    font-size: 1rem !important;
                }
                
                /* Remove excessive spacing */
                .element-container {
                    margin-bottom: 0.5rem !important;
                }
                hr {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
                }
                
                /* Compact expanders */
                .streamlit-expanderHeader {
                    padding: 0.5rem !important;
                    font-size: 0.9rem !important;
                }
                
                /* Compact dataframes */
                .dataframe {
                    font-size: 0.85rem !important;
                }
                
                /* Streamlit dataframe styling - Light theme */
                [data-testid="stDataFrame"] {
                    --dataframe-base-color: #062033 !important;
                    --dataframe-base-background-color: #F8FDFF !important;
                    --dataframe-header-background-color: #D7EEF9 !important;
                    --dataframe-header-color: #002335 !important;
                    --dataframe-stripe-background-color: #F0FBFF !important;
                    --dataframe-row-hover-background-color: #EBF8FF !important;
                    --dataframe-border-color: #D7EEF9 !important;
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
                
                /* All possible tab selectors */
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
                
                /* Selected tab */
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
                
                /* Progress bars */
                .stProgress > div > div > div {
                    background-color: #0077B6 !important;
                }
                .stProgress > div > div {
                    background-color: #D7EEF9 !important;
                }
                
                /* Dataframes/Tables */
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
                
                /* Dividers */
                hr {
                    border-color: #D7EEF9 !important;
                }
                
                /* Code blocks */
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
                
                /* Expanders */
                .streamlit-expanderHeader {
                    background-color: #F0FBFF !important;
                    color: #062033 !important;
                    border: 1px solid #D7EEF9 !important;
                }
                
                /* Alerts */
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
                
                /* Selectbox/Dropdown */
                div[role="listbox"] {
                    background-color: #F8FDFF !important;
                }
                div[role="option"] {
                    color: #062033 !important;
                }
                div[role="option"]:hover {
                    background-color: #EBF8FF !important;
                }
                
                /* Captions */
                .caption, small, .stCaption {
                    color: #0A4A6A !important;
                }
                
                /* Markdown & text */
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
                .stTabs [data-baseweb="tab-list"] {
                    gap: 10px !important;
                }
                
                .stTabs [data-baseweb="tab"],
                .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"],
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
    
    def load_state(self) -> Dict:
        """Load current orchestrator state"""
        try:
            return self.state_manager.get_state()
        except Exception as e:
            st.error(f"Failed to load state: {e}")
            return {}
    
    def format_timestamp(self, iso_timestamp: Optional[str]) -> str:
        """Format ISO timestamp to readable format"""
        if not iso_timestamp:
            return "N/A"
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return iso_timestamp
    
    def format_duration(self, start_iso: Optional[str]) -> str:
        """Calculate duration from start timestamp"""
        if not start_iso:
            return "N/A"
        try:
            start = datetime.fromisoformat(start_iso)
            duration = datetime.now() - start
            
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            seconds = int(duration.total_seconds() % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except:
            return "N/A"
    
    def render_sidebar(self, state: Dict):
        """Render sidebar with experiment selector and controls"""
        # Experiment selector
        st.sidebar.title("🧪 Experiments")
        
        experiments = StateManager.list_experiments()
        
        if experiments:
            # Create options for dropdown (no count display)
            exp_options = {}
            for exp in experiments:
                name = exp['name']
                wandb_run = exp.get('wandb_run') or 'Unknown'
                step = exp.get('current_step', 0)
                status = exp.get('status', 'unknown')
                
                # Handle None or short wandb_run names
                if len(wandb_run) > 30:
                    wandb_display = f"{wandb_run[:30]}..."
                else:
                    wandb_display = wandb_run
                
                # Add status indicator
                status_emoji = "🟢" if status == "monitoring" else "🔴" if status == "stopped" else "⚪"
                label = f"{status_emoji} {name} ({wandb_display} - {step:,} steps)"
                exp_options[label] = name
            
            # Get current selection
            current_exp = st.session_state.selected_experiment
            
            # Find index of current experiment
            current_index = 0
            if current_exp:
                for i, (label, name) in enumerate(exp_options.items()):
                    if name == current_exp:
                        current_index = i
                        break
            
            # Dropdown selector (no metadata shown after selection)
            selected_label = st.sidebar.selectbox(
                "Select Experiment",
                options=list(exp_options.keys()),
                index=current_index,
                help="Switch between different experiment runs"
            )
            
            selected_name = exp_options[selected_label]
            
            # If selection changed, update and reload
            if selected_name != st.session_state.selected_experiment:
                st.session_state.selected_experiment = selected_name
                st.rerun()
        else:
            st.sidebar.warning("📂 No experiments found")
            st.sidebar.info("Start an orchestrator to create a new experiment, or check the data directory.")
        
        st.sidebar.divider()
        st.sidebar.title("📊 Analytics Summary")
        
        metrics = state.get('metrics', {})
        training = state.get('training', {})
        orchestrator = state.get('orchestrator', {})
        
        # Training metrics in 2 columns
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                "Training Step",
                f"{training.get('current_step', 0):,}"
            )
            st.metric(
                "Checkpoints",
                metrics.get('total_checkpoints', 0)
            )
        
        with col2:
            st.metric(
                "LLM Calls",
                metrics.get('total_llm_calls', 0)
            )
            st.metric(
                "Curriculum Updates",
                metrics.get('total_curriculum_updates', 0)
            )
        
        if metrics.get('total_cost_usd', 0) > 0:
            st.sidebar.metric(
                "Total Cost",
                f"${metrics.get('total_cost_usd', 0):.2f}"
            )
        
        # Auto-refresh controls
        st.sidebar.divider()
        st.sidebar.subheader("🔄 Refresh Settings")
        
        # Determine if current experiment is live
        orchestrator = state.get('orchestrator', {})
        orch_status = orchestrator.get('status', 'unknown')
        is_live = orch_status in ['monitoring', 'initializing']
        
        if not is_live:
            st.sidebar.info("⏸️ Auto-refresh disabled for archived experiments")
            st.session_state.auto_refresh = False
        else:
            st.session_state.auto_refresh = st.sidebar.checkbox(
                "Auto-refresh (Live Mode)",
                value=st.session_state.auto_refresh
            )
        
        if st.session_state.auto_refresh and is_live:
            st.session_state.refresh_interval = st.sidebar.slider(
                "Interval (seconds)",
                min_value=2,
                max_value=30,
                value=st.session_state.refresh_interval
            )
        
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Last update time
        last_update = orchestrator.get('last_update')
        if last_update:
            st.sidebar.caption(f"Last update: {self.format_timestamp(last_update)}")
    
    def render_live_overview(self, state: Dict):
        """Tab 1: Live Training Overview"""
        st.header("🎯 Live Training Overview")
        
        training = state.get('training', {})
        orchestrator = state.get('orchestrator', {})
        current_tasks = state.get('current_tasks', [])
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Detect if experiment is archived/stopped
        orch_status = orchestrator.get('status', 'unknown')
        is_archived = orch_status in ['stopped', 'completed', 'error']
        
        with col1:
            training_state = training.get('state', 'unknown')
            # Override training state if orchestrator is stopped
            if is_archived and training_state == 'training':
                training_state = 'stopped'
            
            state_colors = {
                'training': '🟢',
                'completed': '🔵',
                'pre-training': '🟡',
                'stopped': '🔴',
                'unknown': '⚪'
            }
            st.metric(
                "Training State",
                f"{state_colors.get(training_state, '⚪')} {training_state}"
            )
        
        with col2:
            # Prefer agent_step/max_agent_step if available, fall back to current_step/target_step
            agent_step = training.get('agent_step')
            max_agent_step = training.get('max_agent_step')
            current_step = training.get('current_step', 0)
            target_step = training.get('target_step')
            
            # Use agent_step metrics if available (more accurate)
            if agent_step is not None and max_agent_step and max_agent_step > 0:
                progress_pct = (agent_step / max_agent_step) * 100
                st.metric(
                    "Agent Progress",
                    f"{agent_step:,} / {max_agent_step:,}",
                    delta=f"{progress_pct:.1f}%"
                )
            elif target_step:
                progress_pct = (current_step / target_step) * 100
                st.metric(
                    "Progress",
                    f"{current_step:,} / {target_step:,}",
                    delta=f"{progress_pct:.1f}%"
                )
            else:
                st.metric("Current Step", f"{current_step:,}")
        
        with col3:
            uptime = self.format_duration(training.get('start_time'))
            st.metric("Training Uptime", uptime)
        
        with col4:
            orch_status = orchestrator.get('status', 'unknown')
            status_colors = {
                'monitoring': '🟢',
                'initializing': '🟡',
                'stopped': '🔴',
                'error': '❌'
            }
            st.metric(
                "Orchestrator",
                f"{status_colors.get(orch_status, '⚪')} {orch_status}"
            )
        
        # Progress bar (prefer agent_step metrics if available)
        if agent_step is not None and max_agent_step and max_agent_step > 0:
            progress = agent_step / max_agent_step
            st.progress(min(progress, 1.0))
        elif training.get('target_step'):
            progress = current_step / training['target_step']
            st.progress(min(progress, 1.0))
        
        # Curriculum update information (if agent_step metrics available)
        if agent_step is not None and max_agent_step and max_agent_step > 0:
            # Calculate progress percentage for this section
            training_progress = agent_step / max_agent_step
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                # Calculate curriculum updates remaining
                check_interval = orchestrator.get('check_interval')
                
                if is_archived:
                    # For archived experiments, show Complete
                    st.metric(
                        "Curriculum Updates Left",
                        "Complete",
                        help="Experiment has been stopped or completed"
                    )
                elif not check_interval or check_interval == 0:
                    st.metric(
                        "Curriculum Updates Left",
                        "N/A",
                        help="Check interval not configured in orchestrator"
                    )
                else:
                    max_global_steps = max_agent_step * 3.4
                    remaining_global_steps = max(0, max_global_steps - current_step)
                    updates_remaining = int(remaining_global_steps // check_interval)
                    
                    st.metric(
                        "Curriculum Updates Left",
                        f"~{updates_remaining}",
                        help=f"Approximate remaining opportunities to update curriculum (checks every {check_interval:,} global steps; 1 agent_step ≈ 3.4 global_steps)"
                    )
            
            with col6:
                # Show current iteration
                current_iteration = orchestrator.get('iteration', 0)
                st.metric("Current Iteration", f"{current_iteration}")
            
            with col7:
                # Show training phase based on progress
                if is_archived:
                    phase = "🏁 Completed"
                elif training_progress < 0.1:
                    phase = "🌱 Early Training"
                elif training_progress < 0.3:
                    phase = "📈 Early-Mid Training"
                elif training_progress < 0.6:
                    phase = "🎯 Mid Training"
                elif training_progress < 0.85:
                    phase = "🚀 Late Training"
                else:
                    phase = "🏁 Final Phase"
                st.metric("Training Phase", phase)
            
            with col8:
                # Show next checkpoint
                check_interval = orchestrator.get('check_interval')
                
                if is_archived:
                    st.metric(
                        "Next Checkpoint In",
                        "Complete",
                        help="Experiment has been stopped or completed"
                    )
                elif check_interval and check_interval > 0:
                    # Calculate next checkpoint in global_step space
                    next_checkpoint = ((current_step // check_interval) + 1) * check_interval
                    steps_to_checkpoint = next_checkpoint - current_step
                    st.metric(
                        "Next Checkpoint In",
                        f"{steps_to_checkpoint:,} global steps",
                        help=f"Next curriculum evaluation at global step {next_checkpoint:,}"
                    )
                else:
                    st.metric(
                        "Next Checkpoint In",
                        "N/A",
                        help="Check interval not configured"
                    )
        
        st.divider()
        
        # Expanded task weight visualization
        if current_tasks and any(t.get('weight') for t in current_tasks):
            st.subheader("📊 Task Weight Distribution")
            task_data = []
            for task in current_tasks:
                task_data.append({
                    'Predicate': task.get('predicate', 'N/A'),
                    'Weight': task.get('weight', 0.0)
                })
            df = pd.DataFrame(task_data)
            
            fig = px.bar(
                df,
                x='Predicate',
                y='Weight',
                title='Task Weight Distribution',
                color='Weight',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=350,
                template='plotly_dark',
                showlegend=False,
                margin=dict(l=40, r=20, t=50, b=60),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    def render_performance_metrics(self, state: Dict):
        """Tab 2: Performance Metrics"""
        
        # Get Wandb run information
        training = state.get('training', {})
        wandb_entity = training.get('wandb_entity')
        wandb_project = training.get('wandb_project')
        wandb_run_name = training.get('wandb_run_name')
        
        # Header with inline Wandb link
        if wandb_entity and wandb_project and wandb_run_name:
            wandb_url = f"https://wandb.ai/{wandb_entity}/{wandb_project}"
            st.markdown(f"## 📈 Performance Metrics [See details 🔗]({wandb_url})")
        else:
            st.header("📈 Performance Metrics")
        
        # Display Wandb run details if available
        if wandb_entity and wandb_project and wandb_run_name:
            # Fetch and display metrics
            try:
                # Initialize wandb API
                api = wandb.Api()
                
                # Try to find the run
                runs = api.runs(f"{wandb_entity}/{wandb_project}")
                matching_run = None
                
                for run in runs:
                    if run.name == wandb_run_name or run.id == wandb_run_name:
                        matching_run = run
                        break
                
                if matching_run:
                    # Get available metrics (keys from history)
                    st.subheader("📊 Metric Visualization")
                    
                    # Get all available keys
                    history_keys = list(matching_run.history().columns)
                    
                    # Filter to numeric metrics (exclude system columns)
                    metric_keys = [k for k in history_keys if not k.startswith('_') and k not in ['_step', '_timestamp', '_runtime']]
                    
                    if metric_keys:
                        # Dropdown to select metric
                        selected_metric = st.selectbox(
                            "Select Metric:",
                            options=sorted(metric_keys),
                            help="Choose a metric to visualize"
                        )
                        
                        if selected_metric:
                            # Fetch history for selected metric
                            with st.spinner(f"Loading {selected_metric}..."):
                                history = matching_run.history(keys=[selected_metric, '_step'])
                                
                                if not history.empty and selected_metric in history.columns:
                                    # Apply smoothing with your preferred settings
                                    df_plot = self._apply_smoothing(
                                        history, 
                                        selected_metric,
                                        exclude_outliers=True,
                                        ema_alpha=0.99
                                    )
                                    
                                    # Create plot
                                    fig = go.Figure()
                                    
                                    # Raw data (light, transparent)
                                    fig.add_trace(go.Scatter(
                                        x=df_plot['_step'],
                                        y=df_plot['raw'],
                                        mode='lines',
                                        name='Raw',
                                        line=dict(color='lightgray', width=1),
                                        opacity=0.3
                                    ))
                                    
                                    # Smoothed data (prominent)
                                    fig.add_trace(go.Scatter(
                                        x=df_plot['_step'],
                                        y=df_plot['smoothed'],
                                        mode='lines',
                                        name='Smoothed (EMA 0.99)',
                                        line=dict(color='#00d4ff', width=2)
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{selected_metric} over Training Steps",
                                        xaxis_title="Training Step",
                                        yaxis_title=selected_metric,
                                        template='plotly_dark',
                                        height=320,
                                        margin=dict(l=40, r=20, t=40, b=40),
                                        title_font_size=13,
                                        hovermode='x unified',
                                        legend=dict(
                                            yanchor="top",
                                            y=0.99,
                                            xanchor="left",
                                            x=0.01,
                                            font=dict(size=10)
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display stats
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Current", f"{df_plot['smoothed'].iloc[-1]:.4f}")
                                    with col2:
                                        st.metric("Mean", f"{df_plot['raw'].mean():.4f}")
                                    with col3:
                                        st.metric("Max", f"{df_plot['raw'].max():.4f}")
                                    with col4:
                                        st.metric("Min", f"{df_plot['raw'].min():.4f}")
                                else:
                                    st.warning(f"No data available for {selected_metric}")
                    else:
                        st.info("No metrics found in this run yet. The run may be still initializing.")
                else:
                    st.warning(f"Could not find run '{wandb_run_name}' in project {wandb_project}. The run may still be initializing.")
                    st.caption("Available runs will appear here once training starts logging to Wandb.")
                    
            except Exception as e:
                st.error(f"Error fetching Wandb data: {e}")
                st.caption("Make sure you're logged in to Wandb. Run `wandb login` in terminal if needed.")
        else:
            st.warning("Wandb run information not available. Make sure the training has started and Wandb is configured.")
            st.write(f"- Entity: {wandb_entity or 'Not set'}")
            st.write(f"- Project: {wandb_project or 'Not set'}")
            st.write(f"- Run Name: {wandb_run_name or 'Not set'}")
    
    def _apply_smoothing(self, df: pd.DataFrame, metric_col: str, exclude_outliers: bool = True, ema_alpha: float = 0.99) -> pd.DataFrame:
        """Apply time-weighted EMA smoothing with outlier exclusion"""
        result = pd.DataFrame()
        result['_step'] = df['_step']
        result['raw'] = df[metric_col]
        
        # Exclude outliers if requested
        if exclude_outliers:
            # Use IQR method to identify outliers
            Q1 = df[metric_col].quantile(0.25)
            Q3 = df[metric_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Replace outliers with NaN
            values = df[metric_col].copy()
            values[(values < lower_bound) | (values > upper_bound)] = np.nan
        else:
            values = df[metric_col].copy()
        
        # Apply time-weighted EMA (using alpha = 0.99)
        # EMA formula: S_t = alpha * S_{t-1} + (1 - alpha) * x_t
        smoothed = []
        ema = values.iloc[0] if not pd.isna(values.iloc[0]) else 0
        
        for val in values:
            if not pd.isna(val):
                ema = ema_alpha * ema + (1 - ema_alpha) * val
            smoothed.append(ema)
        
        result['smoothed'] = smoothed
        
        return result
    
    def render_llm_decision_log(self, state: Dict):
        """Tab 3: LLM Decision Log"""
        st.header("🤖 LLM Decision Log")
        
        llm_calls = state.get('llm_calls', [])
        
        if not llm_calls:
            st.info("No LLM calls recorded yet")
            return
        
        # Summary stats with proper GPT-4o pricing
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_calls = len(llm_calls)
        avg_duration = sum(c.get('duration_seconds', 0) for c in llm_calls) / total_calls if total_calls > 0 else 0
        
        # Calculate totals for input and output tokens separately
        total_input_tokens = sum(c.get('tokens_in', 0) for c in llm_calls)
        total_output_tokens = sum(c.get('tokens_out', 0) for c in llm_calls)
        
        # GPT-4o pricing: $2.50 per 1M input tokens, $10.00 per 1M output tokens
        input_cost = (total_input_tokens / 1_000_000) * 2.50
        output_cost = (total_output_tokens / 1_000_000) * 10.0
        total_cost = input_cost + output_cost
        
        with col1:
            st.metric("Total Calls", total_calls)
        with col2:
            st.metric("Avg Duration", f"{avg_duration:.2f}s")
        with col3:
            st.metric("Input Tokens", f"{total_input_tokens:,}")
        with col4:
            st.metric("Output Tokens", f"{total_output_tokens:,}")
        with col5:
            st.metric("Total Cost", f"${total_cost:.2f}")
        
        st.divider()
        
        # Filter by type
        call_types = list(set(c.get('type', 'unknown') for c in llm_calls))
        selected_type = st.selectbox(
            "Filter by type:",
            ['All'] + call_types
        )
        
        filtered_calls = llm_calls
        if selected_type != 'All':
            filtered_calls = [c for c in llm_calls if c.get('type') == selected_type]
        
        # Display calls in reverse order (newest first)
        st.subheader(f"Showing {len(filtered_calls)} calls")
        
        for i, call in enumerate(reversed(filtered_calls)):
            with st.expander(
                f"#{len(filtered_calls) - i} - {call.get('type', 'unknown')} - "
                f"{self.format_timestamp(call.get('timestamp'))} "
                f"({call.get('duration_seconds', 0):.2f}s)",
                expanded=(i == 0)  # Expand most recent
            ):
                # Show prompt FIRST (most important for understanding)
                if call.get('prompt'):
                    st.markdown("### 📝 LLM Input (Prompt)")
                    with st.expander("View Full Prompt", expanded=True):
                        st.code(call['prompt'], language="markdown")
                
                # Then show response
                st.markdown("### 🤖 LLM Output (Response)")
                response = call.get('response', {})
                
                # Show reasoning prominently if available
                if isinstance(response, dict) and 'reasoning' in response:
                    st.markdown("**Chain-of-Thought Reasoning:**")
                    reasoning = response['reasoning']
                    if isinstance(reasoning, dict):
                        for key, value in reasoning.items():
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                            st.write(value)
                            st.markdown("---")
                    else:
                        st.write(reasoning)
                    
                    st.markdown("**Full Response JSON:**")
                    st.json(response, expanded=False)
                else:
                    st.json(response, expanded=True)
                
                # Details sidebar
                st.markdown("### 📊 Call Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model:** {call.get('model', 'N/A')}")
                    
                    # Handle None checkpoint_step
                    checkpoint_step = call.get('checkpoint_step')
                    if checkpoint_step is not None:
                        st.write(f"**Step:** {checkpoint_step:,}")
                    else:
                        st.write("**Step:** N/A")
                    
                    st.write(f"**Duration:** {call.get('duration_seconds', 0):.2f}s")
                
                with col2:
                    if call.get('tokens_in'):
                        st.write(f"**Tokens In:** {call.get('tokens_in', 0):,}")
                    if call.get('tokens_out'):
                        st.write(f"**Tokens Out:** {call.get('tokens_out', 0):,}")
                    if call.get('cost_usd'):
                        st.write(f"**Cost:** ${call.get('cost_usd', 0):.4f}")

    
    def render_task_history(self, state: Dict):
        """Tab 4: Task History & Evolution"""
        st.header("📜 Task History & Evolution")
        
        checkpoints = state.get('checkpoints', [])
        current_tasks = state.get('current_tasks', [])
        
        if not checkpoints and not current_tasks:
            st.info("No checkpoint history yet")
            return
        
        # Task evolution over time
        st.subheader("Task Evolution")
        
        # Extract task IDs at each checkpoint
        checkpoint_tasks = []
        for cp in checkpoints:
            tasks = cp.get('tasks', [])
            task_ids = [t.get('task_id') for t in tasks if t.get('task_id') is not None]
            checkpoint_tasks.append({
                'step': cp.get('step', 0),
                'iteration': cp.get('iteration', 0),
                'task_count': len(task_ids),
                'updated': cp.get('curriculum_updated', False)
            })
        
        df = pd.DataFrame(checkpoint_tasks)
        
        if not df.empty:
            # Metrics row at top
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Total Checkpoints", len(checkpoints))
            with met_col2:
                st.metric("Curriculum Updates", df['updated'].sum())
            with met_col3:
                st.metric("Avg Tasks/Checkpoint", f"{df['task_count'].mean():.1f}")
            
            st.divider()
            
            # Two graphs side by side: Tasks Over Time + Task Predicates Over Time
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                # Task count over time (legend INSIDE graph top-right)
                fig = px.line(
                    df,
                    x='step',
                    y='task_count',
                    title='Tasks Over Time',
                    markers=True
                )
                fig.add_scatter(
                    x=df[df['updated']]['step'],
                    y=df[df['updated']]['task_count'],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='star'),
                    name='Curriculum Updated'
                )
                fig.update_layout(
                    height=400,
                    template='plotly_dark',
                    xaxis_title='Training Step',
                    yaxis_title='# Tasks',
                    margin=dict(l=40, r=20, t=40, b=40),
                    title_font_size=14,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(0,0,0,0.5)"
                    )
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            
            with graph_col2:
                # Task Predicates Over Time (moved to right column)
                st.markdown("**📊 Task Predicates Over Time**")
                
                # Build predicate time series data
                predicate_data = self._build_predicate_timeseries(checkpoints, current_tasks)
                
                if predicate_data and predicate_data['predicates']:
                    # Create multiselect for predicates
                    all_predicates = sorted(predicate_data['predicates'].keys())
                    
                    # Initialize session state for selected predicates if not exists
                    if 'selected_predicates' not in st.session_state:
                        st.session_state.selected_predicates = all_predicates[:min(5, len(all_predicates))]
                    
                    # Filter session state to only include predicates that still exist
                    valid_selected = [p for p in st.session_state.selected_predicates if p in all_predicates]
                    if not valid_selected:
                        valid_selected = all_predicates[:min(5, len(all_predicates))]
                    
                    selected_predicates = st.multiselect(
                        "Select Predicates:",
                        options=all_predicates,
                        default=valid_selected,
                        help="Toggle predicate lines",
                        key="predicates_selector"
                    )
                    
                    # Update session state
                    st.session_state.selected_predicates = selected_predicates
                    
                    if selected_predicates:
                        # Create line plot
                        fig_pred = go.Figure()
                        
                        # Color palette
                        colors = px.colors.qualitative.Plotly
                        
                        for idx, pred in enumerate(selected_predicates):
                            pred_series = predicate_data['predicates'][pred]
                            color = colors[idx % len(colors)]
                            
                            fig_pred.add_trace(go.Scatter(
                                x=pred_series['steps'],
                                y=pred_series['counts'],
                                mode='lines+markers',
                                name=pred,
                                line=dict(color=color, width=2),
                                marker=dict(size=6),
                                hovertemplate=f'<b>{pred}</b><br>Step: %{{x}}<br>Count: %{{y}}<extra></extra>'
                            ))
                        
                        fig_pred.update_layout(
                            title="Predicate Counts",
                            xaxis_title="Training Step",
                            yaxis_title="# Tasks",
                            template='plotly_dark',
                            height=400,
                            hovermode='x unified',
                            margin=dict(l=40, r=20, t=40, b=40),
                            title_font_size=14,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(0,0,0,0.5)",
                                font=dict(size=9)
                            )
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': True})
                        st.caption(f"{len(selected_predicates)}/{len(all_predicates)} predicates shown")
                    else:
                        st.info("Select predicates above")
                else:
                    st.info("Need checkpoint data")
        
        st.divider()
    
    def _build_predicate_timeseries(self, checkpoints: List[Dict], current_tasks: List[Dict]) -> Optional[Dict]:
        """Build time series data for each predicate across checkpoints"""
        # Combine checkpoints with current tasks
        all_stages = []
        for cp in checkpoints:
            all_stages.append({
                'step': cp.get('step', 0),
                'iteration': cp.get('iteration', 0),
                'tasks': cp.get('tasks', [])
            })
        
        # Add current tasks as the latest stage
        if current_tasks:
            training_state = self.state_manager.get_state()
            current_step = training_state.get('training', {}).get('current_step', 0)
            all_stages.append({
                'step': current_step or 'Current',
                'iteration': len(all_stages),
                'tasks': current_tasks
            })
        
        if not all_stages:
            return None
        
        # Track all unique predicates and their counts at each checkpoint
        predicate_timeline = {}
        steps = []
        
        for stage in all_stages:
            step = stage['step']
            tasks = stage['tasks']
            steps.append(step)
            
            # Count tasks by predicate at this checkpoint
            predicate_counts = {}
            for task in tasks:
                pred = task.get('predicate', 'Unknown')
                predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
            
            # Update timeline for each predicate
            for pred in predicate_counts:
                if pred not in predicate_timeline:
                    predicate_timeline[pred] = {'steps': [], 'counts': []}
            
            # Fill in counts for all predicates (0 if not present at this step)
            for pred in predicate_timeline:
                predicate_timeline[pred]['steps'].append(step)
                predicate_timeline[pred]['counts'].append(predicate_counts.get(pred, 0))
        
        return {
            'predicates': predicate_timeline,
            'steps': steps
        }
    
    def render_system_health(self, state: Dict):
        """Tab 5: System Health & Logs"""
        st.header("🏥 System Health & Logs")
        
        health = state.get('system_health', {})
        orch_status = state.get('orchestrator_status', '')
        
        # Connection status
        st.subheader("Connection Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Training API - check if api_connected OR if orchestrator is stopped
            api_connected = health.get('api_connected', False)
            is_stopped = orch_status in ['stopped', 'completed', 'error']
            
            if api_connected:
                api_status = "🟢 Connected"
            elif is_stopped:
                api_status = "🟡 Available (stopped)"
            else:
                api_status = "🔴 Disconnected"
            
            st.metric("Training API", api_status)
            if health.get('api_latency_ms'):
                st.caption(f"Latency: {health.get('api_latency_ms'):.1f}ms")
        
        with col2:
            # OpenAI API - check OPENAI_API_KEY env var
            openai_key = os.environ.get('OPENAI_API_KEY')
            openai_connected = health.get('openai_connected', False)
            
            if openai_connected or openai_key:
                openai_status = "🟢 Available"
            else:
                openai_status = "⚪ Unknown"
            
            st.metric("OpenAI API", openai_status)
        
        with col3:
            # Wandb - check if wandb config exists in state
            wandb_config = state.get('config', {}).get('wandb', {})
            wandb_connected = health.get('wandb_connected', False)
            
            if wandb_connected or wandb_config:
                wandb_status = "🟢 Available"
            else:
                wandb_status = "⚪ Unknown"
            
            st.metric("Wandb", wandb_status)
        
        st.divider()
        
        # Logs
        st.subheader("📋 System Logs")
        
        logs = state.get('logs', [])
        
        if not logs:
            st.info("No logs yet")
            return
        
        # Filter by level
        log_levels = list(set(log.get('level', 'info') for log in logs))
        selected_level = st.selectbox(
            "Filter by level:",
            ['All'] + log_levels
        )
        
        filtered_logs = logs
        if selected_level != 'All':
            filtered_logs = [log for log in logs if log.get('level') == selected_level]
        
        # Display logs (most recent first)
        for log in reversed(filtered_logs[-50:]):  # Last 50
            level = log.get('level', 'info')
            icon = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌'
            }.get(level, 'ℹ️')
            
            timestamp = self.format_timestamp(log.get('timestamp'))
            message = log.get('message', '')
            
            st.text(f"{icon} [{timestamp}] {message}")
    
    def render_configuration(self, state: Dict):
        """Tab 6: Configuration & Control"""
        st.header("⚙️ Configuration & Control")
        
        config = state.get('config', {})
        
        if not config:
            st.info("No configuration loaded")
            return
        
        # Display configuration
        st.subheader("Current Configuration")
        st.json(config, expanded=True)
        
        st.divider()
        
        # Export options
        st.subheader("📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Full State (JSON)"):
                state_json = json.dumps(state, indent=2)
                st.download_button(
                    "Download State",
                    data=state_json,
                    file_name=f"orchestrator_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export Checkpoints (CSV)"):
                checkpoints = state.get('checkpoints', [])
                if checkpoints:
                    df = pd.DataFrame(checkpoints)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Checkpoints",
                        data=csv,
                        file_name=f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("Export LLM Calls (JSON)"):
                llm_calls = state.get('llm_calls', [])
                if llm_calls:
                    calls_json = json.dumps(llm_calls, indent=2)
                    st.download_button(
                        "Download LLM Calls",
                        data=calls_json,
                        file_name=f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    def run(self):
        """Main dashboard entry point"""
        # Load state
        state = self.load_state()
        
        # Determine view mode
        orchestrator = state.get('orchestrator', {})
        orch_status = orchestrator.get('status', 'unknown')
        is_live = orch_status in ['monitoring', 'initializing']
        
        # Header
        if is_live:
            st.title("ArcRL Dashboard")
        else:
            st.title("ArcRL Dashboard")
        
        # Render sidebar
        self.render_sidebar(state)
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Live Overview",
            "📈 Performance",
            "🤖 LLM Decisions",
            "📜 Task History",
            "🏥 System Health",
            "⚙️ Configuration"
        ])
        
        with tab1:
            self.render_live_overview(state)
        
        with tab2:
            self.render_performance_metrics(state)
        
        with tab3:
            self.render_llm_decision_log(state)
        
        with tab4:
            self.render_task_history(state)
        
        with tab5:
            self.render_system_health(state)
        
        with tab6:
            self.render_configuration(state)
        
        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()


if __name__ == "__main__":
    import sys
    import subprocess
    
    # Check if running via streamlit or directly
    if 'streamlit' in sys.modules:
        # Running via streamlit - normal execution
        dashboard = Dashboard()
        dashboard.run()
    else:
        # Running directly with Python - launch streamlit
        print("🚀 Launching ArcRL Dashboard...")
        print("📊 You can browse past experiments and view live training data")
        print("🔄 Close the browser or press Ctrl+C to exit\n")
        
        # Get the path to this file
        dashboard_path = Path(__file__).resolve()
        
        # Launch streamlit
        try:
            subprocess.run(["streamlit", "run", str(dashboard_path)])
        except KeyboardInterrupt:
            print("\n👋 Dashboard closed")
        except FileNotFoundError:
            print("❌ Error: streamlit not found. Please install it:")
            print("   pip install streamlit")
            sys.exit(1)
