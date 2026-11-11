"""decision.py stub for ri_decision"""
from __future__ import annotations

from typing import Any, Dict

IS_STUB = True

def decide(drift: Dict[str, Any], cara: Dict[str, Any], quality: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    basis = cara.get("basis", "measured")
    if basis == "estimated":
        zc = cara.get("staleness_est_lo", 0.05) or 0.05
        # Cap zone at HUMAN: cannot be RETRAIN, can be KEEP or HUMAN
        zone = "HUMAN"
        reason = "estimated from model confidence, labels pending"
        return {
            "P": 0.50,
            "zone": zone,
            "zc": zc,
            "zm": 0.05,
            "d": 0.02,
            "reason": reason,
        }

    return {
        "P": 0.25,
        "zone": "KEEP",
        "zc": 0.1,
        "zm": 0.05,
        "d": 0.02,
        "reason": "Stub decision: drift and staleness are low.",
    }

def what_if(drift: Dict[str, Any], cara: Dict[str, Any], quality: Dict[str, Any], cfg_overrides: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "P": 0.30,
        "zone": "KEEP",
        "zc": 0.1,
        "zm": 0.05,
        "d": 0.02,
        "reason": "Stub what_if scenario.",
    }
