"""demo.py - the guided demo. Every step replays stored demo data; shifts are SIMULATED and labelled so."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import ri_decision.orchestrator as orch
from ri_drift.scenarios import inject_covariate_shift
from ri_init.config import WORKSPACE

DEMO_STEPS: List[Dict[str, Any]] = [
    {"key": "register", "title": "1. Register the reference model",
     "what": "Reset the demo project: load model v1, compute the holdout baseline, calibrate CARA-T "
             "(tau*, kappa) and store the drift reference.", "batches": [], "inject": None},
    {"key": "stable", "title": "2. Replay stable batches",
     "what": "Three batches drawn from the same distribution as the training data. Expected: no drift, "
             "staleness below tau*, zone KEEP.", "batches": [0, 1, 2], "inject": None},
    {"key": "covariate", "title": "3. Inject covariate drift (simulated)",
     "what": "Two batches whose rows are resampled so the first numeric feature shifts by about 1.5 standard "
             "deviations. Labels stay attached to their rows, so the input distribution changes but the "
             "relationship does not. Expected: Module 2 flags drift; drift alone cannot trigger retraining.",
     "batches": [3, 4], "inject": "covariate"},
    {"key": "concept", "title": "4. Concept shift (simulated)",
     "what": "A batch generated with a different relationship between features and target. Expected: "
             "staleness jumps above tau*, the score crosses the retrain line, a candidate is trained, "
             "verified against the champion and promoted.", "batches": [26], "inject": None},
    {"key": "after", "title": "5. Monitor the new version",
     "what": "Two more batches from the new regime, served by the promoted model. Expected: zone KEEP again, "
             "with a shadow comparison against the previous version.", "batches": [27, 28], "inject": None},
]


def stream_files(project_id: str) -> List[Path]:
    return sorted((WORKSPACE / project_id / "data" / "stream").glob("batch_*.csv"))


def run_demo_step(project_id: str, index: int, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run one demo step through the real pipeline and return one row per processed batch."""
    step = DEMO_STEPS[index]
    if step["key"] == "register":
        res = orch.reset_project(project_id)
        return [{"batch": "setup", "zone": "", "score": None, "action": res["status"], "drift_share": None,
                 "staleness": None, "issues": "; ".join(res.get("issues", []))}]
    files = stream_files(project_id)
    tmp = WORKSPACE / project_id / "data" / "uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for number in step["batches"]:
        if number >= len(files):
            continue
        frame = pd.read_csv(files[number])
        if step["inject"] == "covariate":
            numeric = profile.get("features", {}).get("numeric", [])
            if numeric:
                frame = inject_covariate_shift(frame, {**profile, "seed": 42 + number}, numeric[0], 1.5, 0)
        send = tmp / f"demo_{step['key']}_{files[number].name}"
        frame.to_csv(send, index=False)
        result = orch.on_new_data(project_id, send)
        for rep in result.get("reports", []):
            rows.append({"batch": files[number].name, "check": rep["report_id"], "zone": rep["zone"],
                         "score": rep["decision"].get("P"), "action": rep["action"],
                         "drift_share": rep["drift"].get("drift_share"),
                         "staleness": rep["cara"].get("staleness")})
    return rows
