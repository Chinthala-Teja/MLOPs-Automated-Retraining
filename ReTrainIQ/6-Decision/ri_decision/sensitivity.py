"""sensitivity.py - how the outcome moves with the automatic-retrain limit and with kappa. SIMULATION."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ri_cara import replay_baselines as rb
from ri_cara.cara_t import tau_for_kappa
from ri_decision.calibrate_score import load_project
from ri_init.config import load_config

IS_STUB = False


def run_sensitivity(project_id: str = "demo", limits: Sequence[float] = (0.55, 0.65, 0.75),
                    kappa_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
                    batches_dir: Optional[str | Path] = None, limit: Optional[int] = None,
                    bank: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Rows for each retrain limit (score policy) and each kappa multiplier (CARA-T and score policy)."""
    data = load_project(project_id, batches_dir, limit)
    profile, calib = data["profile"], data["calib"]
    base_cfg = load_config(None)
    bank = bank or rb.build_bank(data["batches"], calib, profile, base_cfg)
    rows: List[Dict[str, Any]] = []
    for mult in kappa_multipliers:
        scaled = {**calib, **tau_for_kappa(calib, mult)}
        kappa, tau = float(scaled["kappa"]), float(scaled["tau_star"])
        res = rb.simulate(bank, kappa, lambda t, cur, st, since: st >= tau)
        rows.append({"policy": "CARA-T", "kappa_multiplier": mult, "retrain_limit": None, "tau_star": tau,
                     "human_zone_batches": None, **res})
        for lim in limits:
            if mult != 1.0 and lim != 0.65:
                continue
            cfg = {**base_cfg, "retrain_limit": float(lim), "keep_limit": round(1.0 - float(lim), 4)}
            # keep the safety guard (drift alone must stay below the limit) at low limits
            cfg["a3"] = min(float(cfg["a3"]), round(0.9 * math.log(float(lim) / (1.0 - float(lim))), 4))
            log: List[Dict[str, Any]] = []
            res = rb.simulate(bank, kappa, rb.score_rule(bank, scaled, profile, cfg, log))
            rows.append({"policy": "ReTrainIQ score", "kappa_multiplier": mult, "retrain_limit": lim,
                         "tau_star": tau, "a3_used": cfg["a3"],
                         "human_zone_batches": sum(1 for r in log if r["zone"] == "HUMAN"), **res})
    out = pd.DataFrame(rows)
    out["retrain_at"] = out["retrain_at"].apply(lambda v: " ".join(str(i) for i in v))
    out.attrs["simulated"] = True
    return out


def analyze_sensitivity(report: Dict[str, Any], param_ranges: Dict[str, Any]) -> Dict[str, Any]:
    """What-if table for ONE stored report: zone under each retrain limit in param_ranges['retrain_limit']."""
    from ri_decision.decision import what_if

    table = []
    for lim in param_ranges.get("retrain_limit", (0.55, 0.65, 0.75)):
        res = what_if(report["drift"], report["cara"], report["quality"],
                      {"retrain_limit": float(lim), "keep_limit": round(1.0 - float(lim), 4)})
        table.append({"retrain_limit": lim, "P": res["P"], "zone": res["zone"]})
    return {"status": "OK", "sensitivity_table": table}
