"""summary.py stub for ri_analysis"""
from typing import Any, Dict

IS_STUB = True

def summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "summary": "Stub summary: model status is healthy.",
        "highlights": ["No severe drift detected", "CARA-T indicates KEEP"],
    }

def validate_summary(text: str, report: Dict[str, Any]) -> bool:
    return True

def template_summary(report: Dict[str, Any]) -> str:
    return "Template summary: all metrics within bounds."

