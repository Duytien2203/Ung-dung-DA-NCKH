"""
SmartResearch Studio - Streamlit Cloud Entry Point
Executes app.py
"""
import runpy
import sys
import os

# Ensure current directory is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

runpy.run_path(os.path.join(current_dir, "app.py"), run_name="__main__")
