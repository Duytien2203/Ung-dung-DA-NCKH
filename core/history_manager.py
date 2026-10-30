"""
Analysis History & Audit Manager for SmartResearch Studio.
Tracks user actions, tools used, step data summaries, and provides step rollback.
"""

import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


STEP_BADGE_COLORS = {
    1: "#3b82f6",  # Blue
    2: "#6366f1",  # Indigo
    3: "#0284c7",  # Sky
    4: "#10b981",  # Emerald
    5: "#14b8a6",  # Teal
    6: "#f59e0b",  # Amber
    7: "#8b5cf6",  # Purple
    8: "#ec4899",  # Pink
    9: "#ef4444",  # Rose
    10: "#059669"  # Green
}

STEP_ICONS = {i: "" for i in range(1, 11)}


class AnalysisHistoryManager:
    """
    Manages analysis action history and audit log across 10 workflow steps.
    Provides snapshot storage, concise data summary, and rollback functionality.
    """
    def __init__(self, max_entries: int = 50):
        self.entries: List[Dict[str, Any]] = []
        self.max_entries = max_entries

    def log(
        self,
        step_num: int,
        step_name: str,
        tool_name: str,
        data_summary: str = "",
        df: Optional[pd.DataFrame] = None,
        extra_info: Optional[str] = None
    ) -> str:
        """Add an action entry to history if not an immediate duplicate."""
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%d/%m/%Y")
        entry_id = f"hist_{now.strftime('%Y%m%d_%H%M%S_%f')}"

        # Avoid exact duplicate of the most recent action
        if self.entries:
            last = self.entries[-1]
            if (last["step_num"] == step_num and 
                last["tool_name"] == tool_name and 
                last["data_summary"] == data_summary and 
                last.get("date_str") == date_str):
                return last["id"]

        shape_str = f"N = {len(df):,} quan sát, {len(df.columns)} biến" if df is not None else ""
        if not data_summary and shape_str:
            data_summary = shape_str
        
        # Save snapshot only if df provided (e.g. data load or modification)
        df_snapshot = df.copy() if df is not None else None

        entry = {
            "id": entry_id,
            "time_str": time_str,
            "date_str": date_str,
            "step_num": step_num,
            "step_name": step_name,
            "tool_name": tool_name,
            "data_summary": data_summary,
            "shape_str": shape_str,
            "df_snapshot": df_snapshot,
            "extra_info": extra_info or "",
            "badge_color": STEP_BADGE_COLORS.get(step_num, "#3b82f6"),
            "icon": ""
        }
        self.entries.append(entry)
        
        # Keep maximum entries to prevent excessive memory usage
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        return entry_id

    def get_entries(self, reverse: bool = True) -> List[Dict[str, Any]]:
        """Return history entries (default: newest first)."""
        if reverse:
            return list(reversed(self.entries))
        return list(self.entries)

    def get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None

    def clear(self):
        self.entries = []

    def count(self) -> int:
        return len(self.entries)
