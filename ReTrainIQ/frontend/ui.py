"""ui.py - shared look and feel: theme CSS, cards, status pills and Plotly charts.

Everything here only DISPLAYS values that the backend computed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

INK, MUTED, LINE, ACCENT = "#111827", "#6B7280", "#E5E7EB", "#4F46E5"
TONE = {"good": "#059669", "warn": "#D97706", "bad": "#DC2626", "info": "#4F46E5", "idle": "#6B7280"}
ZONE_TONE = {"KEEP": "good", "HUMAN": "warn", "RETRAIN": "bad", "BLOCKED": "idle"}
ZONE_LABEL = {"KEEP": "KEEP", "HUMAN": "HUMAN REVIEW", "RETRAIN": "RETRAIN", "BLOCKED": "INVALID DATA"}
EVENT_TONE = {"PROMOTED": "good", "VERIFY_FAILED": "warn", "ROLLED_BACK": "bad"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em; color: #111827;}
section[data-testid="stSidebar"] {background: #0F172A;}
section[data-testid="stSidebar"] label p, section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #E5E7EB;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #94A3B8;}
.rq-brand {font-size: 22px; font-weight: 700; color: #FFFFFF; margin-bottom: 0;}
.rq-brand span {color: #818CF8;}
.rq-sub {font-size: 12px; color: #94A3B8; margin-top: 2px;}
.rq-card {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px 18px;
          box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%;}
.rq-label {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #6B7280; font-weight: 600;}
.rq-value {font-size: 26px; font-weight: 700; color: #111827; line-height: 1.25;}
.rq-note {font-size: 13px; color: #6B7280;}
.rq-pill {display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12px; font-weight: 700;
          letter-spacing: .04em;}
.rq-hero {background: linear-gradient(120deg, #312E81 0%, #4F46E5 60%, #6366F1 100%); border-radius: 16px;
          padding: 26px 30px; color: #FFFFFF; margin-bottom: 18px;}
.rq-hero h1 {color: #FFFFFF; font-size: 28px; margin: 0 0 6px 0;}
.rq-hero p {color: #E0E7FF; margin: 0; font-size: 15px;}
.rq-step {border-left: 3px solid #E5E7EB; padding: 2px 0 10px 14px; margin-left: 6px;}
.rq-step.done {border-color: #059669;} .rq-step.fail {border-color: #DC2626;} .rq-step.skip {border-color: #D1D5DB;}
.rq-step b {color: #111827;} .rq-step span {color: #6B7280; font-size: 13px;}
.rq-sim {background: #FFFBEB; border: 1px solid #FCD34D; color: #92400E; border-radius: 8px; padding: 8px 12px;
         font-size: 13px; font-weight: 600;}
.rq-flow {display: flex; gap: 8px; flex-wrap: wrap; align-items: stretch;}
.rq-flow div {flex: 1; min-width: 150px; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
              padding: 12px 14px;}
.rq-flow div b {display: block; color: #4F46E5; font-size: 12px; letter-spacing: .05em;}
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, tone: str = "info") -> str:
    color = TONE.get(tone, TONE["info"])
    return f"<span class='rq-pill' style='background:{color}1A;color:{color}'>{text}</span>"


def zone_pill(zone: Optional[str]) -> str:
    return pill(ZONE_LABEL.get(zone or "", zone or "n/a"), ZONE_TONE.get(zone or "", "idle"))


def card(label: str, value: str, note: str = "", tone: Optional[str] = None) -> None:
    color = TONE.get(tone, INK) if tone else INK
    st.markdown(f"<div class='rq-card'><div class='rq-label'>{label}</div>"
                f"<div class='rq-value' style='color:{color}'>{value}</div>"
                f"<div class='rq-note'>{note}</div></div>", unsafe_allow_html=True)


def hero(title: str, text: str) -> None:
    st.markdown(f"<div class='rq-hero'><h1>{title}</h1><p>{text}</p></div>", unsafe_allow_html=True)


def page_title(title: str, text: str) -> None:
    st.markdown(f"## {title}")
    st.caption(text)


def steps(items: List[Dict[str, str]]) -> None:
    """Vertical pipeline. Each item: {'title', 'detail', 'state'} with state done | fail | skip | todo."""
    icon = {"done": "&#10003;", "fail": "&#10007;", "skip": "&#8212;", "todo": "&#9675;"}
    html = "".join(
        f"<div class='rq-step {i.get('state', 'todo')}'><b>{icon.get(i.get('state', 'todo'), '')} {i['title']}</b>"
        f"<br><span>{i.get('detail', '')}</span></div>" for i in items)
    st.markdown(html, unsafe_allow_html=True)


def simulated_banner(text: str = "SIMULATION: replayed demo data with injected shifts. Not real production monitoring.") -> None:
    st.markdown(f"<div class='rq-sim'>{text}</div>", unsafe_allow_html=True)


def flow_strip() -> None:
    st.markdown(
        "<div class='rq-flow'>"
        "<div><b>MODULE 1</b>Understand the ML project<br><span class='rq-note'>once, at setup</span></div>"
        "<div><b>MODULE 2</b>Data drift<br><span class='rq-note'>every batch; warning only</span></div>"
        "<div><b>MODULE 3</b>CARA-T staleness<br><span class='rq-note'>measured, or estimated while labels are late</span></div>"
        "<div><b>MODULE 4</b>Model quality<br><span class='rq-note'>needs labels</span></div>"
        "<div><b>DECISION</b>Score and zone<br><span class='rq-note'>KEEP / HUMAN / RETRAIN</span></div>"
        "<div><b>ACTION</b>Train, verify, promote<br><span class='rq-note'>rollback always possible</span></div>"
        "</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- charts
def _layout(fig: go.Figure, height: int, title: str = "") -> go.Figure:
    fig.update_layout(height=height, title=title, margin=dict(l=10, r=10, t=44 if title else 16, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font=dict(color=INK, size=13),
                      legend=dict(orientation="h", y=-0.3))
    fig.update_xaxes(gridcolor="#F3F4F6", linecolor=LINE)
    fig.update_yaxes(gridcolor="#F3F4F6", linecolor=LINE)
    return fig


def gauge(p: float, cfg: Dict[str, Any]) -> go.Figure:
    keep, retrain = 100 * cfg["keep_limit"], 100 * cfg["retrain_limit"]
    fig = go.Figure(go.Indicator(
        mode="gauge", value=100 * p,
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": INK, "thickness": 0.25},
               "steps": [{"range": [0, keep], "color": "#D1FAE5"}, {"range": [keep, retrain], "color": "#FEF3C7"},
                         {"range": [retrain, 100], "color": "#FEE2E2"}]}))
    fig = _layout(fig, 250, f"Decision score: {100 * p:.1f} / 100")
    fig.update_layout(margin=dict(l=36, r=44, t=56, b=10))
    return fig


def _events(fig: go.Figure, hist: pd.DataFrame, row: Optional[int] = None) -> None:
    for _, r in hist[hist["event"] != ""].iterrows():
        color = TONE[EVENT_TONE.get(r["event"], "idle")]
        kwargs = {"row": row, "col": 1} if row else {}
        fig.add_vline(x=r["check"], line_color=color, line_width=2, opacity=0.55, **kwargs)


def timeline(hist: pd.DataFrame, cfg: Dict[str, Any]) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Decision score", "Staleness vs tau*", "Primary metric", "Drift share"))
    colors = [TONE[ZONE_TONE.get(z, "idle")] for z in hist["zone"]]
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["P"], mode="lines+markers", name="score",
                             marker=dict(color=colors, size=10), line=dict(color="#9CA3AF")), row=1, col=1)
    fig.add_hline(y=cfg["retrain_limit"], line_dash="dash", line_color=TONE["bad"], row=1, col=1)
    fig.add_hline(y=cfg["keep_limit"], line_dash="dash", line_color=TONE["good"], row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["staleness"], mode="lines+markers", name="staleness",
                             line=dict(color=INK)), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["tau_star"], mode="lines", name="tau*",
                             line=dict(color=TONE["bad"], dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["primary_metric"], mode="lines+markers", name="metric",
                             line=dict(color=ACCENT)), row=3, col=1)
    fig.add_trace(go.Bar(x=hist["check"], y=hist["drift_share"], name="drift share",
                         marker_color=TONE["warn"]), row=4, col=1)
    for row in (1, 2, 3, 4):
        _events(fig, hist, row)
    for _, r in hist[hist["event"] != ""].iterrows():
        fig.add_annotation(x=r["check"], y=1.0, yref="paper", yshift=14, showarrow=False,
                           text=r["event"].replace("_", " ").title(),
                           font=dict(size=11, color=TONE[EVENT_TONE.get(r["event"], "idle")]))
    fig.update_xaxes(title_text="check id", row=4, col=1)
    fig = _layout(fig, 720)
    fig.update_layout(showlegend=False, margin=dict(t=56))
    return fig


def staleness_chart(hist: pd.DataFrame, current_check: Optional[int] = None) -> go.Figure:
    fig = go.Figure()
    measured = hist[hist["basis"] == "measured"]
    estimated = hist[hist["basis"] == "estimated"]
    fig.add_trace(go.Scatter(x=measured["check"], y=measured["staleness"], mode="lines+markers",
                             name="measured staleness", line=dict(color=INK)))
    if len(estimated):
        fig.add_trace(go.Scatter(x=estimated["check"], y=estimated["staleness"], mode="markers",
                                 name="estimated (labels pending)",
                                 marker=dict(symbol="diamond-open", size=10, color=TONE["warn"])))
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["tau_star"], mode="lines", name="threshold tau*",
                             line=dict(color=TONE["bad"], dash="dash")))
    triggers = hist[hist["event"].isin(["PROMOTED", "VERIFY_FAILED"])]
    if len(triggers):
        fig.add_trace(go.Scatter(x=triggers["check"], y=triggers["staleness"], mode="markers",
                                 name="retraining attempted",
                                 marker=dict(symbol="star", size=16, color=TONE["bad"])))
    if current_check is not None and current_check in set(hist["check"]):
        cur = hist[hist["check"] == current_check].iloc[0]
        fig.add_trace(go.Scatter(x=[cur["check"]], y=[cur["staleness"]], mode="markers", name="this check",
                                 marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=ACCENT, width=3))))
    fig.update_xaxes(title_text="check id")
    fig.update_yaxes(title_text="relative staleness")
    return _layout(fig, 360, "Staleness history against the calibrated threshold")


def metric_bars(table: List[Dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(table)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frame["metric"], y=frame["reference"], name="baseline (holdout)", marker_color="#9CA3AF"))
    fig.add_trace(go.Bar(x=frame["metric"], y=frame["current"], name="current batch", marker_color=ACCENT))
    fig.update_layout(barmode="group")
    return _layout(fig, 330, "Baseline versus current")


def confusion_heatmap(matrix: List[List[int]], labels: List[Any]) -> go.Figure:
    names = [str(v) for v in labels]
    fig = go.Figure(go.Heatmap(z=matrix, x=names, y=names, colorscale="Blues", showscale=False,
                               text=matrix, texttemplate="%{text}"))
    fig.update_xaxes(title_text="predicted", type="category")
    fig.update_yaxes(title_text="actual", type="category", autorange="reversed")
    return _layout(fig, 330, "Confusion matrix")


def curve_chart(curve: Dict[str, List[float]], title: str, x_title: str, y_title: str, diagonal: bool) -> go.Figure:
    fig = go.Figure(go.Scatter(x=curve["x"], y=curve["y"], mode="lines", line=dict(color=ACCENT, width=3)))
    if diagonal:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#D1D5DB", dash="dash")))
    fig.update_xaxes(title_text=x_title, range=[0, 1])
    fig.update_yaxes(title_text=y_title, range=[0, 1.02])
    fig = _layout(fig, 330, title)
    fig.update_layout(showlegend=False)
    return fig


def contribution_bars(terms: Dict[str, float]) -> go.Figure:
    names = {"b": "bias b", "a1_zc": "staleness a1*zc", "a2_zm": "quality a2*zm", "a3_d": "drift a3*d"}
    keys = [k for k in names if k in terms]
    values = [float(terms[k]) for k in keys]
    fig = go.Figure(go.Bar(x=values, y=[names[k] for k in keys], orientation="h",
                           marker_color=[TONE["bad"] if v > 0 else TONE["good"] for v in values]))
    fig.update_xaxes(title_text="contribution to the logit (right = towards RETRAIN)")
    return _layout(fig, 250, "What moved the score")
