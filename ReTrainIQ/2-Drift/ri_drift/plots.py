"""plots.py - plotly figures for Module 2 (drift)."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go

from ri_drift.drift import PSI_FLAG

IS_STUB = False

_FLAG_COLOR = "#c0392b"
_OK_COLOR = "#2e7d6b"
_REF_COLOR = "#7f8c8d"


def top_drifted(result: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """The k features with the strongest drift (flagged first, then by PSI)."""
    feats = list(result.get("per_feature", []))
    feats.sort(key=lambda f: (bool(f.get("flag")), float(f.get("psi") or 0.0)), reverse=True)
    return feats[:k]


def drift_bar(result: Dict[str, Any]) -> go.Figure:
    """PSI per feature; flagged features are highlighted; the PSI threshold is a dashed line."""
    feats = sorted(result.get("per_feature", []), key=lambda f: float(f.get("psi") or 0.0))
    fig = go.Figure(go.Bar(
        x=[float(f.get("psi") or 0.0) for f in feats],
        y=[f["feature"] for f in feats],
        orientation="h",
        marker_color=[_FLAG_COLOR if f.get("flag") else _OK_COLOR for f in feats],
        hovertext=[
            f"KS={f['ks_stat']:.3f}, p={f['p_value']:.3g}" if f.get("ks_stat") is not None else "categorical"
            for f in feats
        ],
    ))
    fig.add_vline(x=PSI_FLAG, line_dash="dash", line_color=_FLAG_COLOR,
                  annotation_text=f"PSI {PSI_FLAG}")
    fig.update_layout(title="Drift per feature (PSI)", xaxis_title="PSI", height=260 + 22 * len(feats),
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def distribution_overlay(reference_df: pd.DataFrame, new_df: pd.DataFrame, feature: str, ftype: str) -> go.Figure:
    """Reference versus new distribution: overlaid histograms (numeric) or grouped bars (categorical)."""
    fig = go.Figure()
    if ftype == "numeric":
        ref = pd.to_numeric(reference_df[feature], errors="coerce").dropna()
        new = pd.to_numeric(new_df[feature], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=ref, name="reference", histnorm="probability",
                                   opacity=0.55, marker_color=_REF_COLOR, nbinsx=30))
        fig.add_trace(go.Histogram(x=new, name="new batch", histnorm="probability",
                                   opacity=0.55, marker_color=_FLAG_COLOR, nbinsx=30))
        fig.update_layout(barmode="overlay")
    else:
        ref_sh = reference_df[feature].astype(str).value_counts(normalize=True)
        new_sh = new_df[feature].astype(str).value_counts(normalize=True)
        cats = sorted(set(ref_sh.index) | set(new_sh.index))
        fig.add_trace(go.Bar(x=cats, y=[float(ref_sh.get(c, 0.0)) for c in cats],
                             name="reference", marker_color=_REF_COLOR))
        fig.add_trace(go.Bar(x=cats, y=[float(new_sh.get(c, 0.0)) for c in cats],
                             name="new batch", marker_color=_FLAG_COLOR))
        fig.update_layout(barmode="group")
    fig.update_layout(title=f"{feature}: reference vs new", yaxis_title="share", height=300,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig
