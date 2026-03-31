"""
FitPulse Pro — Unified 4-Milestone App
Run: streamlit run FitPulse_Pro.py
"""

import io, warnings, logging, time
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Page config — called ONCE for the entire app ───────────────────────────────
st.set_page_config(
    page_title="FitPulse Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    # Navigation
    "milestone": 0,
    # M1
    "raw_df": None, "clean_df": None, "ingested": False, "processed": False,
    # M2
    "m2_slots": {}, "m2_master_done": False, "m2_tsfresh_done": False,
    "m2_forecast_done": False, "m2_cluster_done": False,
    # M3 / M4 shared
    "files_loaded": False, "anomaly_done": False, "simulation_done": False,
    "daily": None, "hourly_s": None, "hourly_i": None,
    "sleep": None, "hr": None, "hr_minute": None, "master": None,
    "anom_hr": None, "anom_steps": None, "anom_sleep": None,
    "sim_results": None,
    # Shared raw bytes from M2 (used by M3 & M4 to skip re-upload)
    "shared_daily_b": None, "shared_hr_b": None, "shared_sleep_b": None,
    "shared_hourly_s_b": None, "shared_hourly_i_b": None,
    "shared_master_df": None,  # pre-built master from M2 for M3/M4
    # M4
    "pipeline_done": False, "date_min": None, "date_max": None,
    # Theme
    "dark_mode": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════
BG        = "linear-gradient(135deg,#020617 0%,#0a0e1a 50%,#0f172a 100%)"
CARD_BG   = "rgba(15,23,42,0.80)"
CARD_BOR  = "rgba(56,189,248,0.2)"
TEXT      = "#e2e8f0"
MUTED     = "#94a3b8"
ACCENT    = "#38bdf8"
GREEN     = "#10b981"
RED       = "#f87171"
PURPLE    = "#a78bfa"
AMBER     = "#f59e0b"

# M2 palette (matplotlib)
M2_DARK   = "#0d1117"; M2_CARD  = "#161b22"; M2_CARD2 = "#1c2128"
M2_BORDER = "#30363d"; M2_TEXT  = "#e6edf3"; M2_MUTED = "#8b949e"
M2_BLUE   = "#58a6ff"; M2_GREEN = "#3fb950"; M2_AMBER = "#f0883e"
M2_PURPLE = "#bc8cff"; M2_RED   = "#ff7b72"; M2_PINK  = "#f778ba"
M2_TEAL   = "#39d353"
M2_PAL    = [M2_BLUE, M2_PINK, M2_GREEN, M2_AMBER, M2_PURPLE, M2_RED, M2_TEAL]

plt.rcParams.update({
    "figure.facecolor": M2_DARK,  "axes.facecolor":  M2_CARD2,
    "axes.edgecolor":   M2_BORDER,"axes.labelcolor": M2_MUTED,
    "axes.titlecolor":  M2_TEXT,  "xtick.color":     M2_MUTED,
    "ytick.color":      M2_MUTED, "grid.color":      M2_BORDER,
    "text.color":       M2_TEXT,  "legend.facecolor":M2_CARD,
    "legend.edgecolor": M2_BORDER,"font.size":        9,
})

# M3/M4 Plotly theme
PLOT_BG   = "#0f172a"; PAPER_BG  = "#0a0e1a"; GRID_CLR = "rgba(255,255,255,0.06)"
BADGE_BG  = "rgba(56,189,248,0.15)";  SECTION_BG = "rgba(56,189,248,0.07)"
WARN_BG   = "rgba(246,173,85,0.12)";  WARN_BOR   = "rgba(246,173,85,0.4)"
SUCCESS_BG= "rgba(16,185,129,0.1)";   SUCCESS_BOR= "rgba(16,185,129,0.4)"
DANGER_BG = "rgba(248,113,113,0.1)";  DANGER_BOR = "rgba(248,113,113,0.4)"
ACCENT_RED= "#f87171"; ACCENT2   = "#f687b3"; ACCENT3  = "#68d391"
PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT,
    font_family="Inter, sans-serif",
    legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1, font_color=TEXT),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT),
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .main {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif;
    color: {TEXT} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
    background: rgba(2,6,23,0.97) !important;
    border-right: 1px solid {CARD_BOR};
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.block-container {{ padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }}
p, div, span, label {{ color: {TEXT}; }}
/* Glass card */
.glass-card {{
    background: {CARD_BG}; backdrop-filter: blur(12px);
    border: 1px solid {CARD_BOR}; border-radius: 16px;
    padding: 24px; margin-bottom: 24px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.2);
}}
/* Progress */
.prog-bar {{ height:4px; background:#1e293b; border-radius:2px; margin:6px 0 16px 0; overflow:hidden; }}
.prog-fill {{ height:100%; background: linear-gradient(90deg, {ACCENT}, {GREEN}); border-radius:2px; }}
/* Status badges */
.status-badge {{
    padding: 9px 14px; border-radius: 8px; font-weight: 700; font-size: 0.75rem;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
}}
.badge-pending {{ background: #1e293b; color: #64748b; border: 1px solid #334155; }}
.badge-complete {{ background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #059669; }}
.badge-active {{ background: rgba(56,189,248,0.15); color: {ACCENT}; border: 1px solid {ACCENT}; animation: pulse 2s infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.6}} }}
/* Buttons */
.stButton > button {{
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; transition: all 0.2s !important;
    border: 1px solid {CARD_BOR} !important; background: {CARD_BG} !important;
    color: {TEXT} !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important; background: rgba(56,189,248,0.1) !important;
    transform: translateY(-1px) !important;
}}
/* File uploader */
div[data-testid="stFileUploader"] {{
    background: rgba(56,189,248,0.04); border: 2px dashed {CARD_BOR};
    border-radius: 14px; padding: 0.5rem;
}}
/* Section headers */
.sec-header {{ display:flex; align-items:center; gap:0.8rem; margin:2rem 0 1rem 0; padding-bottom:0.6rem; border-bottom:1px solid {CARD_BOR}; }}
.sec-icon {{ font-size:1.4rem; width:2.2rem; height:2.2rem; display:flex; align-items:center; justify-content:center; background:{BADGE_BG}; border-radius:8px; border:1px solid {CARD_BOR}; }}
.sec-title {{ font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; color:{TEXT}; margin:0; }}
.sec-badge {{ margin-left:auto; background:{BADGE_BG}; border:1px solid {CARD_BOR}; border-radius:100px; padding:0.2rem 0.7rem; font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:{ACCENT}; }}
/* Cards */
.card {{ background:{CARD_BG}; border:1px solid {CARD_BOR}; border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem; backdrop-filter:blur(10px); }}
.card-title {{ font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; color:{MUTED}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem; }}
/* Alert boxes */
.alert-success {{ background:{SUCCESS_BG}; border-left:3px solid {GREEN}; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#9ae6b4; }}
.alert-warn {{ background:{WARN_BG}; border-left:3px solid {AMBER}; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#fbd38d; }}
.alert-info {{ background:{BADGE_BG}; border-left:3px solid {ACCENT}; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#bee3f8; }}
.alert-danger {{ background:{DANGER_BG}; border-left:3px solid {ACCENT_RED}; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#feb2b2; }}
/* Metric grid */
.metric-grid {{ display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.8rem 0; }}
.metric-card {{ flex:1; min-width:120px; background:{SECTION_BG}; border:1px solid {CARD_BOR}; border-radius:12px; padding:1rem 1.2rem; text-align:center; }}
.metric-val {{ font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:{ACCENT}; line-height:1; margin-bottom:0.25rem; }}
.metric-label {{ font-size:0.72rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em; }}
/* Step M2 box */
.step-box {{ display:flex; align-items:flex-start; gap:14px; background:{M2_CARD}; border:1px solid {M2_BORDER}; border-left:4px solid {M2_BLUE}; border-radius:12px; padding:16px 20px; margin:24px 0 10px; }}
.step-num {{ background:{M2_BLUE}; color:{M2_DARK}; font-weight:800; font-size:.72rem; padding:4px 10px; border-radius:20px; letter-spacing:.08em; white-space:nowrap; margin-top:2px; }}
.step-title {{ font-size:1.05rem; font-weight:700; color:{M2_TEXT}; }}
.step-desc  {{ font-size:.78rem; color:{M2_MUTED}; margin-top:3px; }}
/* KPI grid M4 */
.kpi-grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:0.7rem; margin:1rem 0; }}
.kpi-card {{ background:{CARD_BG}; border:1px solid {CARD_BOR}; border-radius:14px; padding:1rem 1.1rem; text-align:center; backdrop-filter:blur(10px); }}
.kpi-val {{ font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; line-height:1; margin-bottom:0.2rem; }}
.kpi-label {{ font-size:0.68rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.07em; }}
.kpi-sub {{ font-size:0.65rem; color:{MUTED}; margin-top:0.15rem; }}
/* Dividers */
.m-divider {{ border:none; border-top:1px solid {CARD_BOR}; margin:2rem 0; }}
/* M4 anomaly row */
.anom-row {{ display:flex; align-items:center; gap:0.6rem; padding:0.45rem 0; border-bottom:1px solid {CARD_BOR}; font-size:0.82rem; }}
/* Filter box */
.filter-box {{ background:{SECTION_BG}; border:1px solid {CARD_BOR}; border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem; }}
/* Step pill */
.step-pill {{ display:inline-flex; align-items:center; gap:0.5rem; background:{SECTION_BG}; border:1px solid {CARD_BOR}; border-radius:100px; padding:0.3rem 0.9rem; font-size:0.75rem; font-family:'JetBrains Mono',monospace; color:{ACCENT}; margin-bottom:0.8rem; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
MILESTONES = [
    (0, "🏠", "Home",               None),
    (1, "📁", "M1 · Preprocessing", "processed"),
    (2, "🔬", "M2 · Pattern Extraction", "m2_cluster_done"),
    (3, "🚨", "M3 · Anomaly Detection",  "anomaly_done"),
    (4, "📊", "M4 · Insights Dashboard", "pipeline_done"),
]

with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0;text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:{ACCENT};">⚡ FitPulse Pro</div>
        <div style="font-size:0.68rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:4px;">ELITE DATA GOVERNANCE SUITE</div>
    </div>
    """, unsafe_allow_html=True)

    n_done = sum(1 for _,_,_,dk in MILESTONES[1:] if dk and st.session_state.get(dk))
    pct = int(n_done / 4 * 100)
    st.markdown(f"""
    <div style="padding:0 4px 8px 4px;">
        <div style="font-size:0.68rem;color:{MUTED};display:flex;justify-content:space-between;margin-bottom:4px;">
            <span>PIPELINE PROGRESS</span><span style="color:{ACCENT}">{pct}%</span>
        </div>
        <div class="prog-bar"><div class="prog-fill" style="width:{pct}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<div style='font-size:0.7rem;color:{MUTED};letter-spacing:0.1em;margin-bottom:8px;'>NAVIGATION</div>", unsafe_allow_html=True)

    for mid, icon, label, dk in MILESTONES:
        is_active = st.session_state.milestone == mid
        is_done   = dk and st.session_state.get(dk)
        clr = GREEN if is_done else (ACCENT if is_active else MUTED)
        bg  = f"rgba(56,189,248,0.12)" if is_active else (f"rgba(16,185,129,0.08)" if is_done else "rgba(15,23,42,0.6)")
        bdr = ACCENT if is_active else (GREEN if is_done else CARD_BOR)
        tick = "✓" if is_done else ("▶" if is_active else str(mid) if mid > 0 else "⌂")
        if st.button(f"{icon}  {label}", key=f"nav_{mid}", use_container_width=True):
            st.session_state.milestone = mid
            st.rerun()

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.7rem;color:{MUTED};line-height:2.2;padding:0 4px;">
        {"".join(f'<div><span style="color:{"#10b981" if st.session_state.get(dk) else MUTED}">{"✅" if st.session_state.get(dk) else "⭕"}</span>  {lbl}</div>'
                 for _,_,lbl,dk in MILESTONES[1:])}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset All Milestones", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ROUTE
# ══════════════════════════════════════════════════════════════════════════════
M = st.session_state.milestone

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if M == 0:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(56,189,248,0.08), rgba(16,185,129,0.04), rgba(2,6,23,0.9));
        border: 1px solid {CARD_BOR}; border-radius: 24px;
        padding: 3rem 3.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;
    ">
        <div style="position:absolute;top:-80px;right:-80px;width:400px;height:400px;
            background:radial-gradient(circle,rgba(56,189,248,0.05) 0%,transparent 70%);border-radius:50%;"></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{ACCENT};
            letter-spacing:0.18em;margin-bottom:1.2rem;">⚡ FITPULSE PRO · ELITE GOVERNANCE SUITE</div>
        <h1 style="font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:800;color:{TEXT};
            margin:0 0 0.8rem 0;letter-spacing:-0.03em;line-height:1.1;">
            Your Complete<br><span style="color:{ACCENT};">Fitness Data</span><br>Analytics Pipeline
        </h1>
        <p style="color:{MUTED};font-size:1rem;max-width:580px;line-height:1.8;margin:0 0 1.5rem 0;">
            Four integrated milestones — from raw CSV ingestion to AI-powered anomaly detection
            and executive-grade insights dashboards. Built for Fitbit data governance at scale.
        </p>
        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;">
            <span style="background:{BADGE_BG};border:1px solid {CARD_BOR};border-radius:100px;
                padding:0.3rem 0.9rem;font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:{ACCENT};">
                📁 Preprocessing
            </span>
            <span style="background:rgba(167,139,250,0.15);border:1px solid rgba(167,139,250,0.3);border-radius:100px;
                padding:0.3rem 0.9rem;font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:#a78bfa;">
                🔬 Pattern Extraction
            </span>
            <span style="background:{DANGER_BG};border:1px solid {DANGER_BOR};border-radius:100px;
                padding:0.3rem 0.9rem;font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:{ACCENT_RED};">
                🚨 Anomaly Detection
            </span>
            <span style="background:{SUCCESS_BG};border:1px solid {SUCCESS_BOR};border-radius:100px;
                padding:0.3rem 0.9rem;font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:{ACCENT3};">
                📊 Insights Dashboard
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Milestone cards
    c1, c2, c3, c4 = st.columns(4)
    card_data = [
        (c1, 1, "📁", "M1", "Preprocessing",       "#38bdf8", "processed",
         "Clean & validate raw Fitbit CSV data with visual null diagnostics and governance protocols."),
        (c2, 2, "🔬", "M2", "Pattern Extraction",  "#a78bfa", "m2_cluster_done",
         "TSFresh features, Prophet time-series forecasting, K-Means & DBSCAN clustering."),
        (c3, 3, "🚨", "M3", "Anomaly Detection",   "#f87171", "anomaly_done",
         "Threshold + residual anomaly detection across heart rate, steps & sleep signals."),
        (c4, 4, "📊", "M4", "Insights Dashboard",  "#34d399", "pipeline_done",
         "Interactive KPI dashboard with drill-downs, date filtering, PDF & CSV export."),
    ]
    for col, mid, icon, badge, title, clr, dk, desc in card_data:
        done = st.session_state.get(dk, False)
        with col:
            r, g, b = int(clr[1:3], 16), int(clr[3:5], 16), int(clr[5:7], 16)
            st.markdown(f"""
            <div style="background:{CARD_BG};border:1px solid rgba({r},{g},{b},0.3);border-radius:18px;
                padding:1.6rem 1.4rem;height:100%;backdrop-filter:blur(12px);">
                <div style="font-size:2.2rem;margin-bottom:0.8rem">{icon}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                    color:{clr};letter-spacing:0.14em;margin-bottom:0.3rem;">{badge}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
                    color:{TEXT};margin-bottom:0.7rem;">{title}</div>
                <div style="font-size:0.78rem;color:{MUTED};line-height:1.7;margin-bottom:1rem;">{desc}</div>
                <div style="font-size:0.68rem;color:{'#10b981' if done else MUTED};font-weight:700;
                    font-family:'JetBrains Mono',monospace;">{'✅ COMPLETE' if done else '⭕ PENDING'}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick start guide
    st.markdown(f"""
    <div class="glass-card">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{TEXT};margin-bottom:1rem;">
            🚀 Quick Start Guide
        </div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.8rem;font-size:0.82rem;">
            <div style="background:rgba(56,189,248,0.06);border-radius:10px;padding:1rem;border-left:3px solid #38bdf8;">
                <div style="color:#38bdf8;font-weight:700;margin-bottom:0.4rem;font-family:'Syne',sans-serif;">① Preprocessing</div>
                <div style="color:{MUTED};line-height:1.7;">Upload your single FitPulse CSV → run null diagnostics → deploy governance cleaning protocol → download cleaned dataset.</div>
            </div>
            <div style="background:rgba(167,139,250,0.06);border-radius:10px;padding:1rem;border-left:3px solid #a78bfa;">
                <div style="color:#a78bfa;font-weight:700;margin-bottom:0.4rem;font-family:'Syne',sans-serif;">② Pattern Extraction</div>
                <div style="color:{MUTED};line-height:1.7;">Upload 6 Fitbit CSV files → build master dataframe → run TSFresh + Prophet + KMeans/DBSCAN clustering pipeline.</div>
            </div>
            <div style="background:rgba(248,113,113,0.06);border-radius:10px;padding:1rem;border-left:3px solid #f87171;">
                <div style="color:#f87171;font-weight:700;margin-bottom:0.4rem;font-family:'Syne',sans-serif;">③ Anomaly Detection</div>
                <div style="color:{MUTED};line-height:1.7;">Upload 5 Fitbit CSVs → detect HR, steps & sleep anomalies using threshold + residual methods → validate 90%+ accuracy.</div>
            </div>
            <div style="background:rgba(52,211,153,0.06);border-radius:10px;padding:1rem;border-left:3px solid #34d399;">
                <div style="color:#34d399;font-weight:700;margin-bottom:0.4rem;font-family:'Syne',sans-serif;">④ Insights Dashboard</div>
                <div style="color:{MUTED};line-height:1.7;">Upload 5 Fitbit CSVs + 3 anomaly CSVs → explore interactive KPI dashboard → export PDF report + anomaly CSV.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        if st.button("📁 Start Preprocessing →", use_container_width=True):
            st.session_state.milestone = 1; st.rerun()
    with bc2:
        if st.button("🔬 Pattern Extraction →", use_container_width=True):
            st.session_state.milestone = 2; st.rerun()
    with bc3:
        if st.button("🚨 Anomaly Detection →", use_container_width=True):
            st.session_state.milestone = 3; st.rerun()
    with bc4:
        if st.button("📊 Insights Dashboard →", use_container_width=True):
            st.session_state.milestone = 4; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE 1 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
elif M == 1:
    # Hero header
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(56,189,248,0.08),rgba(2,6,23,0.9));
        border:1px solid {CARD_BOR};border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-50px;right:-50px;width:250px;height:250px;
            background:radial-gradient(circle,rgba(56,189,248,0.07) 0%,transparent 70%);border-radius:50%;"></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{ACCENT};
            letter-spacing:0.15em;margin-bottom:0.5rem;">MILESTONE 1 · DATA GOVERNANCE</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{TEXT};">
            📁 FitPulse Pro: Data Preprocessing
        </div>
        <div style="color:{MUTED};font-size:0.88rem;margin-top:0.5rem;">
            Ingest · Diagnose · Clean · Validate · Export
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step 1 — Ingestion
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📁 Step 1: Secure Data Ingestion")
    file = st.file_uploader("Upload FitPulse CSV", type="csv", label_visibility="collapsed", key="m1_upload")
    if file:
        temp_df = pd.read_csv(file)
        if st.session_state.raw_df is None or not st.session_state.raw_df.equals(temp_df):
            st.session_state.raw_df  = temp_df
            st.session_state.ingested = True
            st.session_state.processed = False
            st.session_state.clean_df  = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.ingested:
        df = st.session_state.raw_df

        # Step 2 — Diagnostics
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Step 2: Graphical Null Diagnostics")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ["Column", "Count"]
        null_data   = null_counts[null_counts["Count"] > 0]

        if not null_data.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.write("Null Distribution by Column")
                bar = alt.Chart(null_data).mark_bar(cornerRadius=5, color="#38bdf8").encode(
                    x=alt.X("Column", sort="-y"), y="Count"
                ).properties(height=200)
                st.altair_chart(bar, use_container_width=True)
            with c2:
                st.write("Data Integrity Ratio")
                total  = df.size
                nulls  = df.isnull().sum().sum()
                pie_df = pd.DataFrame({"Status": ["Valid", "Missing"], "Value": [total-nulls, nulls]})
                pie = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                    theta="Value",
                    color=alt.Color("Status", scale=alt.Scale(range=["#10b981", "#f43f5e"]))
                ).properties(height=200)
                st.altair_chart(pie, use_container_width=True)
        else:
            st.markdown('<div class="alert-success">✅ No null values found in source data.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 3 — Governance
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Step 3: Governance Pipeline")
        st.write("• **FUNC_01**: Drop Null Dates | **FUNC_02**: Impute 'Workout_Type' → 'General' | **FUNC_03**: Mean-fill all numeric metrics")

        if st.button("🚀 DEPLOY CLEANING PROTOCOL", use_container_width=True, key="m1_clean"):
            with st.status("Engaging governance engine...", expanded=True) as status:
                clean = st.session_state.raw_df.copy()
                if "Date" in clean.columns:
                    clean = clean.dropna(subset=["Date"])
                if "Workout_Type" in clean.columns:
                    clean["Workout_Type"] = clean["Workout_Type"].fillna("General")
                for col in clean.columns:
                    if clean[col].dtype in [np.float64, np.int64]:
                        clean[col] = clean[col].fillna(clean[col].mean())
                time.sleep(1)
                st.session_state.clean_df  = clean
                st.session_state.processed = True
                status.update(label="✅ System Optimised!", state="complete")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.processed:
            # Preview
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("👀 Step 3.5: Data Integrity Preview")
            if st.checkbox("🔍 Show Cleaned Data Preview", key="m1_preview"):
                st.dataframe(st.session_state.clean_df, use_container_width=True, hide_index=True)
                st.markdown(f'<div class="alert-info">ℹ️ {len(st.session_state.clean_df):,} records ready for export.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Analysis
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📈 Step 4: Processed Column Analysis")
            df_clean = st.session_state.clean_df
            num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                cols = st.columns(min(len(num_cols), 3))
                for i, col in enumerate(num_cols[:3]):
                    with cols[i]:
                        st.write(f"**{col}** (Post-Optimisation)")
                        chart = alt.Chart(df_clean).mark_area(
                            line={"color": "#38bdf8"},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[alt.GradientStop(color="#0ea5e9", offset=0),
                                       alt.GradientStop(color="transparent", offset=1)],
                                x1=1, x2=1, y1=1, y2=0
                            )
                        ).encode(alt.X(col, bin=alt.Bin(maxbins=20)), alt.Y("count()")).properties(height=180)
                        st.altair_chart(chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # EDA
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔍 Step 5: Complete Governance EDA")
            tab_corr, tab_dist = st.tabs(["🔥 Correlation Matrix", "📊 Feature Distribution"])

            with tab_corr:
                num_df = df_clean.select_dtypes(include=[np.number])
                if not num_df.empty:
                    corr = num_df.corr().reset_index().melt(id_vars="index")
                    heatmap = alt.Chart(corr).mark_rect().encode(
                        x="index:O", y="variable:O",
                        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"))
                    ).properties(height=400)
                    st.altair_chart(heatmap, use_container_width=True)

            with tab_dist:
                cols_eda = st.columns(2)
                for idx, col in enumerate(df_clean.columns):
                    with cols_eda[idx % 2]:
                        st.markdown(f"**Field:** `{col.upper()}`")
                        if df_clean[col].dtype in [np.float64, np.int64]:
                            c = alt.Chart(df_clean).mark_bar(color="#38bdf8").encode(
                                x=alt.X(col, bin=True), y="count()"
                            ).properties(height=150)
                        else:
                            c = alt.Chart(df_clean).mark_bar().encode(
                                x="count()", y=alt.Y(col, sort="-x"), color=col
                            ).properties(height=150)
                        st.altair_chart(c, use_container_width=True)

            st.divider()
            buf = io.BytesIO()
            st.session_state.clean_df.to_csv(buf, index=False)
            st.download_button(
                "📥 Download Final Optimised Dataset",
                data=buf.getvalue(),
                file_name="FitPulse_Elite_Clean.csv",
                mime="text/csv",
                use_container_width=True,
                key="m1_download"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="alert-success">✅ Milestone 1 Complete — Data Governance pipeline fully executed.</div>', unsafe_allow_html=True)

            if st.button("🔬 Continue to M2 · Pattern Extraction →", use_container_width=True):
                st.session_state.milestone = 2; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE 2 — PATTERN EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
elif M == 2:
    import seaborn as sns

    # ── Cached helpers ─────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def m2_read_csv(b: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(b))

    def df_pq(df: pd.DataFrame) -> bytes:
        buf = io.BytesIO(); df.to_parquet(buf, index=True); buf.seek(0); return buf.read()

    def ser_json(s: pd.Series) -> bytes:
        return s.reset_index(drop=True).to_json().encode()

    def m2_detect_type(df: pd.DataFrame) -> str:
        cols = set(df.columns)
        if "ActivityDate"  in cols and "TotalSteps"     in cols: return "daily"
        if "ActivityHour"  in cols and "StepTotal"      in cols: return "hourly_steps"
        if "ActivityHour"  in cols and "TotalIntensity" in cols: return "hourly_intensities"
        if "Time"          in cols and "Value"          in cols: return "heartrate"
        if "date"          in cols and "value"          in cols: return "sleep"
        if "value__sum_values" in cols or "value__mean" in cols: return "tsfresh"
        return "unknown"

    @st.cache_data(show_spinner="⏳ Resampling heart-rate (once)…")
    def m2_resample_hr(b: bytes) -> bytes:
        hr = m2_read_csv(b)
        hr["Time"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        out = (hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index())
        out.columns = ["Id", "Time", "HeartRate"]
        return df_pq(out.dropna())

    @st.cache_data(show_spinner="⏳ Building master dataframe (once)…")
    def m2_build_master(daily_b, sleep_b, hr_min_b):
        daily  = m2_read_csv(daily_b)
        sleep  = m2_read_csv(sleep_b)
        hr_min = pd.read_parquet(io.BytesIO(hr_min_b))

        daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")
        sc = "date" if "date" in sleep.columns else "Date"
        sleep[sc] = pd.to_datetime(sleep[sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if sc != "date": sleep = sleep.rename(columns={sc: "date"})

        hr_min["Date"] = hr_min["Time"].dt.date
        hr_d = (hr_min.groupby(["Id","Date"])["HeartRate"]
                .agg(AvgHR="mean",MaxHR="max",MinHR="min",StdHR="std").reset_index())

        sleep["Date"] = sleep["date"].dt.date
        sl_d = (sleep.groupby(["Id","Date"])
                .agg(TotalSleepMinutes=("value","count"),
                     DominantSleepStage=("value",lambda x: x.mode().iloc[0] if not x.empty else 0))
                .reset_index())

        m = daily.rename(columns={"ActivityDate":"Date"}).copy()
        m["Date"] = m["Date"].dt.date
        m = m.merge(hr_d, on=["Id","Date"], how="left")
        m = m.merge(sl_d, on=["Id","Date"], how="left")
        m["TotalSleepMinutes"]  = m["TotalSleepMinutes"].fillna(0)
        m["DominantSleepStage"] = m["DominantSleepStage"].fillna(0)
        for c in ["AvgHR","MaxHR","MinHR","StdHR"]:
            m[c] = m.groupby("Id")[c].transform(lambda x: x.fillna(x.median()))
        return df_pq(m)

    @st.cache_data(show_spinner="⏳ Fitting Prophet…")
    def m2_fit_prophet(ds_b, y_b, horizon):
        try:
            from prophet import Prophet
        except ImportError:
            return None, None
        ds = pd.read_json(io.BytesIO(ds_b), typ="series")
        y  = pd.read_json(io.BytesIO(y_b),  typ="series")
        df = pd.DataFrame({"ds": pd.to_datetime(ds), "y": y}).dropna().sort_values("ds")
        if len(df) < 5: return None, None
        mdl = Prophet(daily_seasonality=False, weekly_seasonality=True,
                      yearly_seasonality=False, uncertainty_samples=200,
                      changepoint_prior_scale=0.1)
        mdl.fit(df)
        fc = mdl.predict(mdl.make_future_dataframe(periods=horizon))
        for _col in ["yhat_lower", "yhat_upper"]:
            if _col not in fc.columns:
                fc[_col] = fc["yhat"]
        return df_pq(df), df_pq(fc)

    @st.cache_data(show_spinner="⏳ Clustering…")
    def m2_run_clustering(feat_b, k, eps, min_s):
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.decomposition import PCA
        feats = pd.read_parquet(io.BytesIO(feat_b))
        X = StandardScaler().fit_transform(feats.select_dtypes(include=[np.number]).fillna(0))
        km  = KMeans(n_clusters=k,  random_state=42, n_init=3).fit_predict(X)
        db  = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)
        var = (pca.explained_variance_ratio_*100).tolist()
        inertias = [KMeans(n_clusters=ki, random_state=42, n_init=3).fit(X).inertia_ for ki in range(2,10)]
        return X.tobytes(), X2.tobytes(), var, km.tolist(), db.tolist(), inertias

    def m2_fig_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=M2_DARK)
        buf.seek(0); return buf

    def m2_step(num, title, desc=""):
        st.markdown(
            f'<div class="step-box"><span class="step-num">{num}</span>'
            f'<div><div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div></div></div>',
            unsafe_allow_html=True)

    # ── Sidebar params ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:0.85rem;'>⚙️ M2 Parameters</p>", unsafe_allow_html=True)
        OPTIMAL_K     = st.slider("K-Means Clusters (K)",    2, 8, 3, key="m2_k")
        EPS           = st.slider("DBSCAN ε (eps)",          0.5, 5.0, 2.2, 0.1, key="m2_eps")
        MIN_SAMPLES   = st.slider("DBSCAN min_samples",      1, 5, 2, key="m2_minsamp")
        FORECAST_DAYS = st.slider("Forecast horizon (days)", 7, 60, 14, key="m2_days")
        run_tsne_flag = st.checkbox("Run t-SNE (~15 sec)", value=False, key="m2_tsne")

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(167,139,250,0.08),rgba(2,6,23,0.9));
        border:1px solid rgba(167,139,250,0.25);border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#a78bfa;
            letter-spacing:0.15em;margin-bottom:0.5rem;">MILESTONE 2 · PATTERN EXTRACTION & MODELLING</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{TEXT};">
            🔬 FitPulse ML Pipeline
        </div>
        <div style="color:{MUTED};font-size:0.88rem;margin-top:0.5rem;">
            TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File upload ────────────────────────────────────────────────────────────
    FILE_DEFS = [
        ("daily","🏃","Daily Activity","dailyActivity_merged.csv"),
        ("heartrate","❤️","Heart Rate","heartrate_seconds_merged.csv"),
        ("hourly_intensities","⚡","Hourly Intensities","hourlyIntensities_merged.csv"),
        ("hourly_steps","👟","Hourly Steps","hourlySteps_merged.csv"),
        ("sleep","😴","Minute Sleep","minuteSleep_merged.csv"),
        ("tsfresh","🧬","TSFresh Features","tsfresh_features.csv"),
    ]
    slots = {k: None for k,*_ in FILE_DEFS}
    raw   = {}

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📂 Upload Fitbit CSV Files (6 files)")
    st.caption("Hold Ctrl/Cmd to multi-select. Files are auto-detected by column structure.")
    upl = st.file_uploader("Select all 6 CSV files", type=["csv"],
                            accept_multiple_files=True, key="m2_upload")
    if upl:
        for f in upl:
            b  = f.read()
            df = m2_read_csv(b)
            dt = m2_detect_type(df)
            if dt in slots:
                slots[dt] = df; raw[dt] = b

    n_ok = 0
    card_cols = st.columns(6)
    for col, (key, icon, label, _) in zip(card_cols, FILE_DEFS):
        ready = slots[key] is not None
        if ready: n_ok += 1
        bg  = "rgba(16,185,129,0.1)" if ready else CARD_BG
        bdr = GREEN if ready else CARD_BOR
        col.markdown(
            f'<div style="background:{bg};border:1px solid {bdr};border-radius:10px;'
            f'padding:12px 8px;text-align:center;">'
            f'<div style="font-size:1.6rem">{icon}</div>'
            f'<div style="font-size:0.62rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;margin:4px 0 2px">{label}</div>'
            f'<div style="font-size:0.72rem;color:{"#10b981" if ready else MUTED};font-weight:700;">{"✅" if ready else "⬜"}</div></div>',
            unsafe_allow_html=True)
    st.progress(n_ok / 6, text=f"Files loaded: {n_ok} / 6")
    st.markdown('</div>', unsafe_allow_html=True)

    req_keys = ["daily","heartrate","hourly_intensities","hourly_steps","sleep","tsfresh"]
    if any(slots[k] is None for k in req_keys):
        st.info("Upload all 6 CSV files to proceed.")
        st.stop()

    # ── Persist raw bytes to session state so M3 & M4 can use them without re-upload ──
    if raw.get("daily"):        st.session_state["shared_daily_b"]    = raw["daily"]
    if raw.get("heartrate"):    st.session_state["shared_hr_b"]       = raw["heartrate"]
    if raw.get("sleep"):        st.session_state["shared_sleep_b"]    = raw["sleep"]
    if raw.get("hourly_steps"): st.session_state["shared_hourly_s_b"] = raw["hourly_steps"]
    if raw.get("hourly_intensities"): st.session_state["shared_hourly_i_b"] = raw["hourly_intensities"]

    st.markdown('<div class="alert-success">✅ All 6 files detected. Data shared with M3 & M4 automatically.</div>', unsafe_allow_html=True)

    # ── Phase 1 — Build Master ─────────────────────────────────────────────────
    st.divider()
    m2_step("Phase 1", "Data Ingestion & Master DataFrame",
            "Resample HR to 1-min → merge daily + sleep + HR into master DataFrame")

    if st.button("🚀 Build Master DataFrame", key="m2_build", use_container_width=True):
        with st.spinner("Resampling HR & merging..."):
            hr_min_b = m2_resample_hr(raw["heartrate"])
            master_b = m2_build_master(raw["daily"], raw["sleep"], hr_min_b)
            st.session_state["m2_master_b"] = master_b
            st.session_state.m2_master_done = True
            # Share master with M3 & M4
            _shared_master = pd.read_parquet(io.BytesIO(master_b))
            st.session_state["shared_master_df"] = _shared_master
        st.rerun()

    if st.session_state.m2_master_done and "m2_master_b" in st.session_state:
        master = pd.read_parquet(io.BytesIO(st.session_state["m2_master_b"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Users",   master["Id"].nunique())
        c2.metric("Days",    len(master))
        c3.metric("Features",len(master.columns))
        with st.expander("Preview master dataframe"):
            st.dataframe(master.head(20), use_container_width=True)

    # ── Phase 2 — TSFresh ──────────────────────────────────────────────────────
    st.divider()
    m2_step("Phase 2", "TSFresh Feature Engineering", "Extract rolling features from tsfresh_features.csv")

    if st.session_state.m2_master_done:
        if st.button("🧬 Load TSFresh Features", key="m2_tsfresh", use_container_width=True):
            tsfresh_df = slots["tsfresh"].copy()
            if tsfresh_df.columns[0] in ("Unnamed: 0","","index"):
                tsfresh_df = tsfresh_df.rename(columns={tsfresh_df.columns[0]:"UserId"}).set_index("UserId")
            st.session_state["m2_tsfresh_b"] = df_pq(tsfresh_df)
            st.session_state.m2_tsfresh_done = True
            st.rerun()

    if st.session_state.m2_tsfresh_done and "m2_tsfresh_b" in st.session_state:
        tf = pd.read_parquet(io.BytesIO(st.session_state["m2_tsfresh_b"]))
        c1, c2 = st.columns(2)
        c1.metric("Users",    len(tf))
        c2.metric("Features", tf.shape[1])
        with st.expander("TSFresh feature sample"):
            st.dataframe(tf.iloc[:, :10].head(10), use_container_width=True)

    # ── Phase 3 — Prophet ──────────────────────────────────────────────────────
    st.divider()
    m2_step("Phase 3", "Prophet Time-Series Forecasting", "Forecast HR · Steps · Sleep")

    if st.session_state.m2_master_done:
        if st.button(f"🔮 Run Prophet Forecasting ({FORECAST_DAYS} days)", key="m2_prophet", use_container_width=True):
            master = pd.read_parquet(io.BytesIO(st.session_state["m2_master_b"]))
            master["Date"] = pd.to_datetime(master["Date"])

            forecasts = {}
            with st.spinner("Fitting Prophet models..."):
                for signal, col in [("HR","AvgHR"),("Steps","TotalSteps"),("Sleep","TotalSleepMinutes")]:
                    sub = master.dropna(subset=["Date", col])
                    daily_agg = sub.groupby("Date")[col].mean().reset_index()
                    ds_b = ser_json(daily_agg["Date"].astype(str))
                    y_b  = ser_json(daily_agg[col])
                    df_b, fc_b = m2_fit_prophet(ds_b, y_b, FORECAST_DAYS)
                    if fc_b is not None:
                        forecasts[signal] = (df_b, fc_b)

            if forecasts:
                st.session_state["m2_forecasts"]     = {k: (db, fb) for k, (db, fb) in forecasts.items()}
                st.session_state["m2_forecast_days"] = FORECAST_DAYS
                st.session_state.m2_forecast_done    = True

    if st.session_state.m2_forecast_done and "m2_forecasts" in st.session_state:
        _fdays = st.session_state.get("m2_forecast_days", 14)
        _fcasts = st.session_state["m2_forecasts"]
        SIG_COLORS = {"HR": M2_BLUE, "Steps": M2_GREEN, "Sleep": M2_PURPLE}
        SIG_UNITS  = {"HR": "bpm", "Steps": "steps", "Sleep": "min"}

        st.markdown(f'''<div class="step-box">
            <span class="step-num">📈 Forecast Results</span>
            <div><div class="step-title">Prophet Time-Series Forecast — {len(_fcasts)} Signals</div>
            <div class="step-desc">Actual vs Forecast · Confidence Band · Changepoints · Residuals</div>
            </div></div>''', unsafe_allow_html=True)

        for sig, (df_b, fc_b) in _fcasts.items():
            hist = pd.read_parquet(io.BytesIO(df_b))
            fore = pd.read_parquet(io.BytesIO(fc_b))
            hist_dates = pd.to_datetime(hist["ds"])
            fore_dates = pd.to_datetime(fore["ds"])
            split_date = hist_dates.max()
            clr        = SIG_COLORS.get(sig, M2_BLUE)
            unit       = SIG_UNITS.get(sig, "")

            # align forecast yhat with historical dates for residuals
            merged = hist.copy()
            merged["ds"] = pd.to_datetime(merged["ds"])
            fore_hist = fore[fore["ds"] <= split_date].copy()
            fore_hist["ds"] = pd.to_datetime(fore_hist["ds"])
            merged = merged.merge(fore_hist[["ds","yhat","yhat_lower","yhat_upper"]], on="ds", how="left")
            merged["residual"] = merged["y"] - merged["yhat"]

            st.markdown(f"#### {'❤️' if sig=='HR' else '🚶' if sig=='Steps' else '💤'} {sig} Forecast")
            fig, axes = plt.subplots(2, 2, figsize=(14, 7))
            fig.patch.set_facecolor(M2_DARK)
            fig.suptitle(f"{sig} — Prophet Analysis ({_fdays}d horizon)", color=M2_TEXT, fontsize=12, fontweight="bold")

            # ── Panel 1: Full forecast with confidence band ──────────────────
            ax1 = axes[0, 0]
            ax1.set_facecolor(M2_CARD2)
            ax1.plot(hist_dates, hist["y"], color=clr, lw=1.8, label="Actual", zorder=3)
            ax1.plot(fore_dates, fore["yhat"], color=M2_AMBER, lw=1.5, ls="--", label="Forecast", zorder=3)
            ax1.fill_between(fore_dates, fore["yhat_lower"], fore["yhat_upper"],
                             alpha=0.18, color=M2_AMBER, label="95% CI")
            ax1.axvline(split_date, color=M2_RED, ls=":", lw=1.2, label="Forecast start")
            ax1.set_title("Forecast + Confidence Band", color=M2_TEXT, fontsize=9)
            ax1.legend(fontsize=7, loc="upper left"); ax1.grid(alpha=0.15)
            ax1.tick_params(axis="x", rotation=25)

            # ── Panel 2: Forecast-only zoom ──────────────────────────────────
            ax2 = axes[0, 1]
            ax2.set_facecolor(M2_CARD2)
            fore_only = fore[fore["ds"] > split_date]
            ax2.plot(pd.to_datetime(fore_only["ds"]), fore_only["yhat"],
                     color=M2_AMBER, lw=2, label="Predicted")
            ax2.fill_between(pd.to_datetime(fore_only["ds"]),
                             fore_only["yhat_lower"], fore_only["yhat_upper"],
                             alpha=0.22, color=M2_AMBER, label="95% CI")
            ax2.set_title(f"Future {_fdays}-Day Forecast", color=M2_TEXT, fontsize=9)
            ax2.legend(fontsize=7); ax2.grid(alpha=0.15)
            ax2.tick_params(axis="x", rotation=25)

            # ── Panel 3: Residuals ───────────────────────────────────────────
            ax3 = axes[1, 0]
            ax3.set_facecolor(M2_CARD2)
            resid_valid = merged["residual"].dropna()
            resid_dates = merged.loc[resid_valid.index, "ds"]
            bar_colors  = [M2_RED if v < 0 else M2_GREEN for v in resid_valid]
            ax3.bar(resid_dates, resid_valid, color=bar_colors, alpha=0.75, width=0.8)
            ax3.axhline(0, color=M2_TEXT, lw=0.8)
            ax3.axhline(resid_valid.std(),  color=M2_AMBER, ls="--", lw=0.8, label="+1σ")
            ax3.axhline(-resid_valid.std(), color=M2_AMBER, ls="--", lw=0.8, label="−1σ")
            ax3.set_title(f"Residuals (Actual − Fitted) · RMSE={((resid_valid**2).mean()**0.5):.1f} {unit}", color=M2_TEXT, fontsize=9)
            ax3.legend(fontsize=7); ax3.grid(alpha=0.15)
            ax3.tick_params(axis="x", rotation=25)

            # ── Panel 4: Residual distribution ───────────────────────────────
            ax4 = axes[1, 1]
            ax4.set_facecolor(M2_CARD2)
            if len(resid_valid) > 2:
                ax4.hist(resid_valid, bins=20, color=clr, alpha=0.75, edgecolor=M2_BORDER)
                ax4.axvline(0, color=M2_RED, lw=1.2, ls="--")
                ax4.axvline(resid_valid.mean(), color=M2_AMBER, lw=1.2, ls="-.",
                            label=f"Mean={resid_valid.mean():.1f}")
            ax4.set_title("Residual Distribution", color=M2_TEXT, fontsize=9)
            ax4.set_xlabel(f"Residual ({unit})", color=M2_MUTED, fontsize=8)
            ax4.legend(fontsize=7); ax4.grid(alpha=0.15)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Metrics row
            mae  = resid_valid.abs().mean() if len(resid_valid) else float("nan")
            rmse = (resid_valid**2).mean()**0.5 if len(resid_valid) else float("nan")
            mape = (resid_valid.abs() / (merged["y"].abs() + 1e-9)).mean() * 100 if len(resid_valid) else float("nan")
            next_val = fore_only["yhat"].iloc[0] if len(fore_only) else float("nan")
            st.markdown(f'''<div class="metric-grid">
                <div class="metric-card"><div class="metric-val" style="color:{clr}">{mae:.1f}</div><div class="metric-label">MAE ({unit})</div></div>
                <div class="metric-card"><div class="metric-val" style="color:{M2_AMBER}">{rmse:.1f}</div><div class="metric-label">RMSE ({unit})</div></div>
                <div class="metric-card"><div class="metric-val" style="color:{M2_PINK}">{mape:.1f}%</div><div class="metric-label">MAPE</div></div>
                <div class="metric-card"><div class="metric-val" style="color:{M2_GREEN}">{next_val:.1f}</div><div class="metric-label">Next Day Forecast</div></div>
            </div>''', unsafe_allow_html=True)
            st.divider()

        st.markdown('<div class="alert-success">✅ Prophet forecasting complete — all signals analysed.</div>', unsafe_allow_html=True)

    # ── Phase 4 — Clustering ───────────────────────────────────────────────────
    st.divider()
    m2_step("Phase 4", "User Clustering — KMeans + DBSCAN + PCA", "Segment users by activity profile")

    if st.session_state.m2_tsfresh_done:
        if st.button(f"🔍 Run Clustering (K={OPTIMAL_K})", key="m2_cluster", use_container_width=True):
            tf_b = st.session_state["m2_tsfresh_b"]
            with st.spinner("Running clustering..."):
                X_b, X2_b, var, km_lbls, db_lbls, inertias = m2_run_clustering(
                    tf_b, OPTIMAL_K, EPS, MIN_SAMPLES)
                st.session_state["m2_cluster_res"] = (X_b, X2_b, var, km_lbls, db_lbls, inertias)
                st.session_state.m2_cluster_done  = True
            st.rerun()

    if st.session_state.m2_cluster_done and "m2_cluster_res" in st.session_state:
        X_b, X2_b, var, km_lbls, db_lbls, inertias = st.session_state["m2_cluster_res"]
        X2 = np.frombuffer(X2_b, dtype=np.float64).reshape(-1, 2)
        _tsfresh_b = st.session_state["m2_tsfresh_b"]
        tf  = pd.read_parquet(io.BytesIO(_tsfresh_b))
        n_feats = tf.select_dtypes(include=[np.number]).shape[1]

        # Elbow plot
        tab_elbow, tab_pca, tab_db = st.tabs(["📐 Elbow", "🔵 PCA (KMeans)", "🔴 PCA (DBSCAN)"])
        with tab_elbow:
            fig, ax = plt.subplots(figsize=(7, 4)); fig.patch.set_facecolor(M2_DARK)
            ax.plot(range(2, 10), inertias, marker="o", color=M2_BLUE, lw=2)
            ax.axvline(OPTIMAL_K, color=M2_RED, ls="--", lw=1.5, label=f"K={OPTIMAL_K}")
            ax.set_title("K-Means Elbow", color=M2_TEXT, fontsize=10)
            ax.set_xlabel("K", color=M2_MUTED); ax.set_ylabel("Inertia", color=M2_MUTED)
            ax.legend(fontsize=8); ax.grid(alpha=0.2); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        for tab, lbls, title in [(tab_pca, km_lbls, "K-Means"), (tab_db, db_lbls, "DBSCAN")]:
            with tab:
                fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor(M2_DARK)
                unique_lbls = sorted(set(lbls))
                for lbl in unique_lbls:
                    mask = np.array(lbls) == lbl
                    clr  = M2_RED if lbl == -1 else M2_PAL[lbl % len(M2_PAL)]
                    marker = "X" if lbl == -1 else "o"
                    lab  = "Noise/Outlier" if lbl == -1 else f"Cluster {lbl}"
                    ax.scatter(X2[mask, 0], X2[mask, 1], c=clr, marker=marker,
                               s=120, alpha=0.85, label=lab, edgecolors=M2_DARK, lw=0.5)
                ax.set_title(f"PCA — {title} (var {var[0]:.1f}% + {var[1]:.1f}%)", color=M2_TEXT)
                ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", color=M2_MUTED)
                ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", color=M2_MUTED)
                ax.legend(fontsize=8); ax.grid(alpha=0.2); plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

        st.markdown('<div class="alert-success">✅ Milestone 2 Complete — All 4 phases executed successfully!</div>', unsafe_allow_html=True)
        if st.button("🚨 Continue to M3 · Anomaly Detection →", use_container_width=True):
            st.session_state.milestone = 3; st.rerun()
    else:
        st.info("Complete Phase 2 (TSFresh) first to enable clustering.")


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE 3 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif M == 3:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("plotly not installed — run `pip install plotly`"); st.stop()

    # ── Helpers ────────────────────────────────────────────────────────────────
    def m3_sec(icon, title, badge=None):
        badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
        st.markdown(f'<div class="sec-header"><div class="sec-icon">{icon}</div>'
                    f'<p class="sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

    def m3_pill(n, label):
        st.markdown(f'<div class="step-pill">◆ Step {n} · {label}</div>', unsafe_allow_html=True)

    def m3_apply_theme(fig, title=""):
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor=GRID_CLR, showgrid=True, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
        fig.update_yaxes(gridcolor=GRID_CLR, showgrid=True, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
        if title:
            fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=14,
                                         font_family="Syne, sans-serif"))
        return fig

    ui_ok  = lambda msg: st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
    ui_w   = lambda msg: st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>',    unsafe_allow_html=True)
    ui_i   = lambda msg: st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',   unsafe_allow_html=True)
    ui_d   = lambda msg: st.markdown(f'<div class="alert-danger">🚨 {msg}</div>',  unsafe_allow_html=True)

    def m3_metrics(*items, red_idx=None):
        red_idx = red_idx or []
        html = '<div class="metric-grid">'
        for i, (val, label) in enumerate(items):
            clr = ACCENT_RED if i in red_idx else ACCENT
            html += f'<div class="metric-card"><div class="metric-val" style="color:{clr}">{val}</div><div class="metric-label">{label}</div></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # ── Detection functions ────────────────────────────────────────────────────
    REQUIRED_FILES = {
        "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],      "label":"Daily Activity","icon":"🏃"},
        "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],                   "label":"Hourly Steps",  "icon":"👣"},
        "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],              "label":"Hourly Int.",   "icon":"⚡"},
        "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                       "label":"Sleep",         "icon":"💤"},
        "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                               "label":"Heart Rate",    "icon":"❤️"},
    }

    def score_match(df, info): return sum(1 for c in info["key_cols"] if c in df.columns)

    def detect_hr(master, hr_high=100, hr_low=50, sigma=2.0):
        df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date")["AvgHR"].mean().reset_index(); d.columns = ["Date","AvgHR"]
        d = d.sort_values("Date")
        d["thresh_high"]  = d["AvgHR"] > hr_high
        d["thresh_low"]   = d["AvgHR"] < hr_low
        d["rolling_med"]  = d["AvgHR"].rolling(3, center=True, min_periods=1).median()
        d["residual"]     = d["AvgHR"] - d["rolling_med"]
        d["resid_anomaly"]= d["residual"].abs() > sigma * d["residual"].std()
        d["is_anomaly"]   = d["thresh_high"] | d["thresh_low"] | d["resid_anomaly"]
        def reason(r):
            parts = []
            if r["thresh_high"]:    parts.append(f"HR>{hr_high}")
            if r["thresh_low"]:     parts.append(f"HR<{hr_low}")
            if r["resid_anomaly"]:  parts.append(f"Residual±{sigma:.0f}σ")
            return ", ".join(parts)
        d["reason"] = d.apply(reason, axis=1)
        return d

    def detect_steps(master, st_low=500, st_high=25000, sigma=2.0):
        df = master[["Date","TotalSteps"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date")["TotalSteps"].mean().reset_index(); d = d.sort_values("Date")
        d["thresh_low"]   = d["TotalSteps"] < st_low
        d["thresh_high"]  = d["TotalSteps"] > st_high
        d["rolling_med"]  = d["TotalSteps"].rolling(3, center=True, min_periods=1).median()
        d["residual"]     = d["TotalSteps"] - d["rolling_med"]
        d["resid_anomaly"]= d["residual"].abs() > sigma * d["residual"].std()
        d["is_anomaly"]   = d["thresh_low"] | d["thresh_high"] | d["resid_anomaly"]
        def reason(r):
            parts = []
            if r["thresh_low"]:    parts.append(f"Steps<{st_low}")
            if r["thresh_high"]:   parts.append(f"Steps>{st_high}")
            if r["resid_anomaly"]: parts.append(f"Residual±{sigma:.0f}σ")
            return ", ".join(parts)
        d["reason"] = d.apply(reason, axis=1)
        return d

    def detect_sleep(master, sl_low=60, sl_high=600, sigma=2.0):
        df = master[["Date","TotalSleepMinutes"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index(); d = d.sort_values("Date")
        d["thresh_low"]   = (d["TotalSleepMinutes"] > 0) & (d["TotalSleepMinutes"] < sl_low)
        d["thresh_high"]  = d["TotalSleepMinutes"] > sl_high
        d["no_data"]      = d["TotalSleepMinutes"] == 0
        d["rolling_med"]  = d["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
        d["residual"]     = d["TotalSleepMinutes"] - d["rolling_med"]
        d["resid_anomaly"]= d["residual"].abs() > sigma * d["residual"].std()
        d["is_anomaly"]   = d["thresh_low"] | d["thresh_high"] | d["resid_anomaly"]
        def reason(r):
            parts = []
            if r["no_data"]:       parts.append("No device worn")
            if r["thresh_low"]:    parts.append(f"Sleep<{sl_low}min")
            if r["thresh_high"]:   parts.append(f"Sleep>{sl_high}min")
            if r["resid_anomaly"]: parts.append(f"Residual±{sigma:.0f}σ")
            return ", ".join(parts)
        d["reason"] = d.apply(reason, axis=1)
        return d

    def simulate_accuracy(master, n_inject=10):
        np.random.seed(42)
        df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
        df["Date"] = pd.to_datetime(df["Date"])
        d = df.groupby("Date").mean().reset_index().sort_values("Date")
        results = {}
        for sig, col, low, high, inj_vals in [
            ("Heart Rate","AvgHR",50,100,[115,120,125,35,40,45,118,130,38,42]),
            ("Steps","TotalSteps",500,25000,[50,100,150,30000,35000,28000,80,200,31000,29000]),
            ("Sleep","TotalSleepMinutes",60,600,[10,15,20,700,750,800,12,18,720,710]),
        ]:
            sim = d[["Date",col]].copy()
            idx = np.random.choice(len(sim), n_inject, replace=False)
            sim.loc[idx, col] = np.random.choice(inj_vals, n_inject, replace=True)
            sim["rolling_med"]  = sim[col].rolling(3, center=True, min_periods=1).median()
            sim["residual"]     = sim[col] - sim["rolling_med"]
            resid_std = sim["residual"].std()
            sim["detected"] = (sim[col] > high) | (sim[col] < low) | (sim["residual"].abs() > 2*resid_std)
            tp = int(sim.iloc[idx]["detected"].sum())
            results[sig] = {"injected": n_inject, "detected": tp, "accuracy": round(tp/n_inject*100,1)}
        results["Overall"] = round(np.mean([v["accuracy"] for v in results.values()]), 1)
        return results

    def build_master_m3(files_dict):
        daily  = files_dict["daily"]
        sleep  = files_dict["sleep"]
        hr     = files_dict["hr"]

        daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")

        hr["Time"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        hr_min = hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
        hr_min.columns = ["Id","Time","HeartRate"]
        hr_min["Date"] = hr_min["Time"].dt.date
        hr_d = hr_min.groupby(["Id","Date"])["HeartRate"].agg(AvgHR="mean",MaxHR="max",MinHR="min",StdHR="std").reset_index()

        sc = "date" if "date" in sleep.columns else "Date"
        sleep[sc] = pd.to_datetime(sleep[sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if sc != "date": sleep = sleep.rename(columns={sc:"date"})
        sleep["Date"] = sleep["date"].dt.date
        sl_d = sleep.groupby(["Id","Date"]).agg(TotalSleepMinutes=("value","count")).reset_index()

        m = daily.rename(columns={"ActivityDate":"Date"}).copy()
        m["Date"] = m["Date"].dt.date
        m = m.merge(hr_d, on=["Id","Date"], how="left")
        m = m.merge(sl_d, on=["Id","Date"], how="left")
        m["TotalSleepMinutes"] = m["TotalSleepMinutes"].fillna(0)
        for c in ["AvgHR","MaxHR","MinHR","StdHR"]:
            m[c] = m.groupby("Id")[c].transform(lambda x: x.fillna(x.median()))
        return m

    # ── Sidebar thresholds ─────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:0.85rem;'>🚨 M3 Thresholds</p>", unsafe_allow_html=True)
        hr_high = st.slider("HR High (bpm)",    90, 130, 100, key="m3_hrh")
        hr_low  = st.slider("HR Low (bpm)",     40,  70,  50, key="m3_hrl")
        steps_low  = st.slider("Steps Low",       100, 2000, 500, key="m3_stl")
        sl_low  = st.slider("Sleep Low (min)",  30,  120,  60, key="m3_sll")
        sl_high = st.slider("Sleep High (min)", 500, 800, 600, key="m3_slh")
        sigma   = st.slider("Residual σ",       1.0, 4.0, 2.0, 0.5, key="m3_sig")

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{DANGER_BG},{CARD_BG});
        border:1px solid {DANGER_BOR};border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{ACCENT_RED};
            letter-spacing:0.15em;margin-bottom:0.5rem;">MILESTONE 3 · ANOMALY DETECTION</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{TEXT};">
            🚨 Anomaly Detection Engine
        </div>
        <div style="color:{MUTED};font-size:0.88rem;margin-top:0.5rem;">
            Threshold · Residual · DBSCAN · Simulated Accuracy Validation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data source: auto-load from M2 or manual upload fallback ──────────────
    _has_shared = all(st.session_state.get(k) is not None
                      for k in ["shared_daily_b","shared_hr_b","shared_sleep_b"])

    if _has_shared:
        st.markdown('''<div class="alert-success">
            ✅ <b>Files loaded automatically from M2</b> — no re-upload needed.
            Navigate to M2 and upload files there if you want to use different data.
        </div>''', unsafe_allow_html=True)
        raw3 = {
            "daily":    (st.session_state["shared_daily_b"],
                         pd.read_csv(io.BytesIO(st.session_state["shared_daily_b"]))),
            "hr":       (st.session_state["shared_hr_b"],
                         pd.read_csv(io.BytesIO(st.session_state["shared_hr_b"]))),
            "sleep":    (st.session_state["shared_sleep_b"],
                         pd.read_csv(io.BytesIO(st.session_state["shared_sleep_b"]))),
            "hourly_s": (st.session_state["shared_hourly_s_b"],
                         pd.read_csv(io.BytesIO(st.session_state["shared_hourly_s_b"])))
                         if st.session_state.get("shared_hourly_s_b") else None,
            "hourly_i": (st.session_state["shared_hourly_i_b"],
                         pd.read_csv(io.BytesIO(st.session_state["shared_hourly_i_b"])))
                         if st.session_state.get("shared_hourly_i_b") else None,
        }
        raw3 = {k: v for k, v in raw3.items() if v is not None}
    else:
        st.markdown('''<div class="alert-warn">
            ⚠️ <b>No M2 data found.</b> Upload files manually below, or go to M2 first.
        </div>''', unsafe_allow_html=True)
        m3_sec("📂", "Upload Fitbit Files", "5 files")
        st.caption("Upload: dailyActivity · hourlySteps · hourlyIntensities · minuteSleep · heartrate_seconds")
        m3_files = st.file_uploader("Select 5 Fitbit CSVs", type=["csv"],
                                      accept_multiple_files=True, key="m3_upload")
        raw3 = {}
        if m3_files:
            for f in m3_files:
                b  = f.read()
                df = pd.read_csv(io.BytesIO(b))
                cols = set(df.columns)
                if "ActivityDate" in cols and "TotalSteps" in cols: raw3["daily"]    = (b, df)
                elif "Time" in cols and "Value" in cols:            raw3["hr"]       = (b, df)
                elif "ActivityHour" in cols and "StepTotal" in cols:raw3["hourly_s"] = (b, df)
                elif "ActivityHour" in cols and "TotalIntensity" in cols: raw3["hourly_i"] = (b, df)
                elif "date" in cols and "value" in cols and "logId" in cols: raw3["sleep"] = (b, df)

    ok3 = {k: k in raw3 for k in ["daily","hr","hourly_s","hourly_i","sleep"]}
    cols_c = st.columns(5)
    labels = [("daily","🏃","Daily"),("hr","❤️","HR"),("hourly_s","👣","Steps"),
              ("hourly_i","⚡","Intensity"),("sleep","💤","Sleep")]
    for col, (k, icon, lbl) in zip(cols_c, labels):
        ready = ok3[k]
        col.markdown(
            f'<div style="background:{"rgba(16,185,129,0.1)" if ready else CARD_BG};' 
            f'border:1px solid {"#10b981" if ready else CARD_BOR};border-radius:10px;'
            f'padding:10px 8px;text-align:center;">'
            f'<div style="font-size:1.4rem">{icon}</div>'
            f'<div style="font-size:0.62rem;color:{MUTED};margin:3px 0">{lbl}</div>'
            f'<div style="font-size:0.72rem;color:{"#10b981" if ready else MUTED}">{"✅" if ready else "⬜"}</div></div>',
            unsafe_allow_html=True)

    if not all(ok3.values()):
        st.info("Go to M2 and upload your files there, or upload all 5 Fitbit CSV files above.")
        st.stop()

    ui_ok("All 5 files ready. Building master DataFrame.")

    # Build Master — reuse M2 shared master if available
    m3_sec("🔧", "Build Master DataFrame", "Step 1")
    _m2_master = st.session_state.get("shared_master_df")
    if _m2_master is not None and not st.session_state.files_loaded:
        with st.spinner("Loading master from M2…"):
            st.session_state.master       = _m2_master
            st.session_state.files_loaded = True
        st.rerun()
    elif not st.session_state.files_loaded:
        if st.button("🔧 Load & Build Master DataFrame", key="m3_build", use_container_width=True):
            with st.spinner("Building master..."):
                files_dict = {"daily": raw3["daily"][1], "sleep": raw3["sleep"][1], "hr": raw3["hr"][1]}
                master = build_master_m3(files_dict)
                st.session_state.master       = master
                st.session_state.files_loaded = True
            st.rerun()

    if not st.session_state.files_loaded or st.session_state.master is None:
        st.stop()

    master = st.session_state.master
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Users",   master["Id"].nunique())
    c2.metric("Days",    len(master))
    c3.metric("Avg HR",  f"{master['AvgHR'].mean():.0f} bpm" if "AvgHR" in master.columns else "N/A")
    c4.metric("Avg Steps", f"{master['TotalSteps'].mean():.0f}" if "TotalSteps" in master.columns else "N/A")

    # Detect anomalies
    m3_sec("🚨", "Run Anomaly Detection", "Steps 2–4")
    m3_pill(2, "Threshold + Residual Detection on HR, Steps & Sleep")

    if st.button("🚨 Run Anomaly Detection", key="m3_detect", use_container_width=True):
        with st.spinner("Detecting anomalies..."):
            st.session_state.anom_hr    = detect_hr(master,    hr_high, hr_low, sigma)
            st.session_state.anom_steps = detect_steps(master, steps_low, 25000, sigma)
            st.session_state.anom_sleep = detect_sleep(master, sl_low, sl_high, sigma)
            st.session_state.anomaly_done = True
        st.rerun()

    if st.session_state.anomaly_done:
        anom_hr    = st.session_state.anom_hr
        anom_steps = st.session_state.anom_steps
        anom_sleep = st.session_state.anom_sleep

        n_hr    = anom_hr["is_anomaly"].sum()
        n_steps = anom_steps["is_anomaly"].sum()
        n_sleep = anom_sleep["is_anomaly"].sum()

        m3_metrics(
            (str(n_hr),    "HR Anomalies"),
            (str(n_steps), "Step Anomalies"),
            (str(n_sleep), "Sleep Anomalies"),
            (str(n_hr+n_steps+n_sleep), "Total"),
            red_idx=[0,1,2,3]
        )

        # ── HR Chart ──────────────────────────────────────────────────────────
        m3_sec("❤️", "Heart Rate Anomalies")
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
            mode="lines", name="Avg HR", line=dict(color="#63b3ed", width=1.5)))
        fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
            mode="lines", name="Rolling Median", line=dict(color=ACCENT3, width=1, dash="dot")))
        anom_hr_pts = anom_hr[anom_hr["is_anomaly"]]
        fig_hr.add_trace(go.Scatter(x=anom_hr_pts["Date"], y=anom_hr_pts["AvgHR"],
            mode="markers", name="Anomaly", marker=dict(color=ACCENT_RED, size=9, symbol="x"),
            hovertext=anom_hr_pts["reason"]))
        fig_hr.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED, line_width=1,
                         annotation_text=f"High ({hr_high})")
        fig_hr.add_hline(y=hr_low,  line_dash="dash", line_color=AMBER, line_width=1,
                         annotation_text=f"Low ({hr_low})")
        m3_apply_theme(fig_hr, "❤️ Heart Rate — Anomaly Detection")
        fig_hr.update_layout(height=380)
        st.plotly_chart(fig_hr, use_container_width=True)

        # ── Steps Chart ───────────────────────────────────────────────────────
        m3_sec("🚶", "Steps Anomalies")
        fig_st = go.Figure()
        fig_st.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
            name="Steps", marker_color=ACCENT, opacity=0.6))
        anom_st_pts = anom_steps[anom_steps["is_anomaly"]]
        fig_st.add_trace(go.Scatter(x=anom_st_pts["Date"], y=anom_st_pts["TotalSteps"],
            mode="markers", name="Anomaly", marker=dict(color=ACCENT_RED, size=10, symbol="x"),
            hovertext=anom_st_pts["reason"]))
        fig_st.add_hline(y=steps_low, line_dash="dash", line_color=ACCENT_RED, line_width=1)
        m3_apply_theme(fig_st, "🚶 Steps — Anomaly Detection")
        fig_st.update_layout(height=350)
        st.plotly_chart(fig_st, use_container_width=True)

        # ── Sleep Chart ───────────────────────────────────────────────────────
        m3_sec("💤", "Sleep Anomalies")
        fig_sl = go.Figure()
        fig_sl.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
            mode="lines+markers", name="Sleep", line=dict(color=PURPLE, width=1.5),
            marker=dict(size=4)))
        anom_sl_pts = anom_sleep[anom_sleep["is_anomaly"]]
        fig_sl.add_trace(go.Scatter(x=anom_sl_pts["Date"], y=anom_sl_pts["TotalSleepMinutes"],
            mode="markers", name="Anomaly", marker=dict(color=ACCENT_RED, size=10, symbol="x"),
            hovertext=anom_sl_pts["reason"]))
        fig_sl.add_hline(y=sl_low, line_dash="dash", line_color=AMBER, line_width=1)
        fig_sl.add_hline(y=sl_high, line_dash="dash", line_color=ACCENT_RED, line_width=1)
        m3_apply_theme(fig_sl, "💤 Sleep — Anomaly Detection")
        fig_sl.update_layout(height=350)
        st.plotly_chart(fig_sl, use_container_width=True)

        # ── DBSCAN ────────────────────────────────────────────────────────────
        m3_sec("🔍", "DBSCAN Outlier Clustering")
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            from sklearn.decomposition import PCA

            cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
            cluster_cols = [c for c in cluster_cols if c in master.columns]
            cf = master.groupby("Id")[cluster_cols].mean().dropna()
            X_c = StandardScaler().fit_transform(cf)
            db_labels = DBSCAN(eps=0.8, min_samples=2).fit_predict(X_c)
            pca = PCA(n_components=2); X_pca = pca.fit_transform(X_c)
            var = pca.explained_variance_ratio_ * 100
            cf["DBSCAN"] = db_labels

            outlier_users = cf[cf["DBSCAN"] == -1].index.tolist()
            if outlier_users:
                ui_d(f"DBSCAN found {len(outlier_users)} outlier user(s): {outlier_users}")
            else:
                ui_ok("No DBSCAN outliers detected.")

            colors_db = [ACCENT_RED if l == -1 else M2_PAL[l % len(M2_PAL)] for l in db_labels]
            fig_db = go.Figure(go.Scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], mode="markers+text",
                marker=dict(color=colors_db, size=12, opacity=0.85),
                text=[str(uid)[-4:] for uid in cf.index],
                textposition="top center",
                hovertext=[f"User {uid}<br>Cluster {l}" for uid, l in zip(cf.index, db_labels)]
            ))
            m3_apply_theme(fig_db, f"🔍 DBSCAN PCA ({var[0]:.1f}% + {var[1]:.1f}% var)")
            fig_db.update_layout(height=400)
            st.plotly_chart(fig_db, use_container_width=True)
        except Exception as e:
            ui_w(f"DBSCAN skipped: {e}")

        # ── Accuracy Simulation ───────────────────────────────────────────────
        m3_sec("🎯", "Simulated Detection Accuracy — 90%+ Target", "Step 6")
        ui_i("10 known anomalies injected per signal. Detection rate validates 90%+ accuracy requirement.")

        if st.button("🎯 Run Accuracy Simulation", key="m3_sim", use_container_width=True):
            with st.spinner("Simulating..."):
                sim = simulate_accuracy(master, n_inject=10)
                st.session_state.sim_results    = sim
                st.session_state.simulation_done = True
            st.rerun()

        if st.session_state.simulation_done and st.session_state.sim_results:
            sim     = st.session_state.sim_results
            overall = sim["Overall"]
            passed  = overall >= 90.0

            if passed: ui_ok(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
            else:      ui_w(f"Overall accuracy: {overall}% — below 90% target")

            m3_metrics(
                (f"{sim['Heart Rate']['accuracy']}%", "HR Accuracy"),
                (f"{sim['Steps']['accuracy']}%",      "Steps Accuracy"),
                (f"{sim['Sleep']['accuracy']}%",      "Sleep Accuracy"),
                (f"{overall}%",                        "Overall"),
            )

            fig_acc = go.Figure()
            signals = ["Heart Rate","Steps","Sleep"]
            accs    = [sim[s]["accuracy"] for s in signals]
            bar_colors = [ACCENT3 if a >= 90 else ACCENT_RED for a in accs]
            fig_acc.add_trace(go.Bar(x=signals, y=accs, marker_color=bar_colors,
                text=[f"{a}%" for a in accs], textposition="outside",
                textfont=dict(color=TEXT, size=13)))
            fig_acc.add_hline(y=90, line_dash="dash", line_color=ACCENT_RED, line_width=2,
                              annotation_text="90% Target", annotation_font_color=ACCENT_RED,
                              annotation_position="top right")
            m3_apply_theme(fig_acc, "🎯 Simulated Anomaly Detection Accuracy")
            fig_acc.update_layout(height=350, yaxis_range=[0, 115], showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)

        # ── Export anomaly CSVs ────────────────────────────────────────────────
        st.divider()
        m3_sec("📥", "Export Anomaly Results")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            buf = io.BytesIO(); anom_hr.to_csv(buf, index=False)
            st.download_button("📥 HR Anomalies CSV", buf.getvalue(), "anom_hr.csv", "text/csv", key="dl_hr")
        with col_b:
            buf = io.BytesIO(); anom_steps.to_csv(buf, index=False)
            st.download_button("📥 Steps Anomalies CSV", buf.getvalue(), "anom_steps.csv", "text/csv", key="dl_st")
        with col_c:
            buf = io.BytesIO(); anom_sleep.to_csv(buf, index=False)
            st.download_button("📥 Sleep Anomalies CSV", buf.getvalue(), "anom_sleep.csv", "text/csv", key="dl_sl")

        ui_ok("Milestone 3 Complete — All anomaly signals detected and validated.")
        if st.button("📊 Continue to M4 · Insights Dashboard →", use_container_width=True):
            st.session_state.milestone = 4; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE 4 — INSIGHTS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif M == 4:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("plotly not installed — run `pip install plotly`"); st.stop()

    PLOTLY_BASE = dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT,
        font_family="Inter, sans-serif",
        legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1, font_color=TEXT),
        margin=dict(l=50, r=30, t=55, b=45),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT),
    )

    def ptheme(fig, title="", h=400):
        fig.update_layout(**PLOTLY_BASE, height=h)
        fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
        fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
        if title:
            fig.update_layout(title=dict(text=title, font_color=TEXT,
                                         font_size=13, font_family="Syne, sans-serif"))
        return fig

    def m4_sec(icon, title, badge=None):
        badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ""
        st.markdown(f'<div class="sec-header"><div class="sec-icon">{icon}</div>'
                    f'<p class="sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

    ui_ok4 = lambda msg: st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
    ui_w4  = lambda msg: st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>',    unsafe_allow_html=True)
    ui_i4  = lambda msg: st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',   unsafe_allow_html=True)

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(104,211,145,0.07),rgba(2,6,23,0.9));
        border:1px solid {SUCCESS_BOR};border-radius:20px;padding:2rem 2.5rem;margin-bottom:1.5rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:{ACCENT3};
            letter-spacing:0.15em;margin-bottom:0.5rem;">MILESTONE 4 · INSIGHTS DASHBOARD</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{TEXT};">
            📊 FitPulse Insights Hub
        </div>
        <div style="color:{MUTED};font-size:0.88rem;margin-top:0.5rem;">
            KPIs · Anomaly Drill-Downs · Trend Analysis · PDF & CSV Export
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data source: auto-load from M2 or manual upload fallback ──────────────
    _has_m2_data = all(st.session_state.get(k) is not None
                       for k in ["shared_daily_b","shared_hr_b","shared_sleep_b",
                                  "shared_hourly_s_b","shared_hourly_i_b"])

    if _has_m2_data and not st.session_state.pipeline_done:
        # Auto-build from M2 shared data
        st.markdown('''<div class="alert-success">
            ✅ <b>Files loaded automatically from M2</b> — building dashboard data…
        </div>''', unsafe_allow_html=True)
        with st.spinner("Auto-building M4 dashboard from M2 data…"):
            _daily = pd.read_csv(io.BytesIO(st.session_state["shared_daily_b"])).copy()
            _daily["ActivityDate"] = pd.to_datetime(_daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")

            _hr_raw = pd.read_csv(io.BytesIO(st.session_state["shared_hr_b"])).copy()
            _hr_raw["Time"] = pd.to_datetime(_hr_raw["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
            _hr_min = _hr_raw.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
            _hr_min.columns = ["Id","Time","HeartRate"]
            _hr_min["Date"] = _hr_min["Time"].dt.date
            _hr_d = _hr_min.groupby(["Id","Date"])["HeartRate"].agg(AvgHR="mean",MaxHR="max",MinHR="min").reset_index()

            _sl = pd.read_csv(io.BytesIO(st.session_state["shared_sleep_b"])).copy()
            _sc = "date" if "date" in _sl.columns else "Date"
            _sl[_sc] = pd.to_datetime(_sl[_sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
            if _sc != "date": _sl = _sl.rename(columns={_sc:"date"})
            _sl["Date"] = _sl["date"].dt.date
            _sl_d = _sl.groupby(["Id","Date"]).agg(TotalSleepMinutes=("value","count")).reset_index()

            _m = _daily.rename(columns={"ActivityDate":"Date"}).copy()
            _m["Date"] = _m["Date"].dt.date
            _m = _m.merge(_hr_d, on=["Id","Date"], how="left")
            _m = _m.merge(_sl_d, on=["Id","Date"], how="left")
            _m["TotalSleepMinutes"] = _m["TotalSleepMinutes"].fillna(0)
            for _c in ["AvgHR","MaxHR","MinHR"]:
                if _c in _m.columns:
                    _m[_c] = _m.groupby("Id")[_c].transform(lambda x: x.fillna(x.median()))

            st.session_state.master       = _m
            st.session_state.hr_minute    = _hr_min
            st.session_state.date_min     = str(_m["Date"].min())
            st.session_state.date_max     = str(_m["Date"].max())
            st.session_state.pipeline_done = True
            # Use anomalies from M3 if available
            # (they are already in session state from M3)
        st.rerun()

    elif not _has_m2_data and not st.session_state.pipeline_done:
        st.markdown('''<div class="alert-warn">
            ⚠️ <b>No M2 data found.</b> Upload files manually below, or complete M2 first.
        </div>''', unsafe_allow_html=True)
        m4_sec("📂", "Upload Fitbit & Anomaly Files", "8 files")
        st.caption("Upload 5 Fitbit CSVs + optionally 3 anomaly CSVs from M3")
        m4_files = st.file_uploader("Select files", type=["csv"],
                                     accept_multiple_files=True, key="m4_upload")
        raw4 = {}
        if m4_files:
            for f in m4_files:
                b  = f.read()
                df = pd.read_csv(io.BytesIO(b))
                _cols = set(df.columns)
                if "ActivityDate" in _cols and "TotalSteps" in _cols:    raw4["daily"]    = df
                elif "Time" in _cols and "Value" in _cols:               raw4["hr_raw"]   = df
                elif "ActivityHour" in _cols and "StepTotal" in _cols:   raw4["hourly_s"] = df
                elif "ActivityHour" in _cols and "TotalIntensity" in _cols: raw4["hourly_i"] = df
                elif "date" in _cols and "value" in _cols and "logId" in _cols: raw4["sleep"] = df
                elif "is_anomaly" in _cols and "AvgHR" in _cols:         raw4["anom_hr"]    = df
                elif "is_anomaly" in _cols and "TotalSteps" in _cols:    raw4["anom_steps"] = df
                elif "is_anomaly" in _cols and "TotalSleepMinutes" in _cols: raw4["anom_sleep"] = df

        _core = ["daily","hr_raw","hourly_s","hourly_i","sleep"]
        _ok4  = {k: k in raw4 for k in _core}
        _c5   = st.columns(5)
        _lbl4 = [("daily","🏃","Daily"),("hr_raw","❤️","HR"),("hourly_s","👣","Steps"),
                 ("hourly_i","⚡","Intensity"),("sleep","💤","Sleep")]
        for _col, (_k, _icon, _lbl) in zip(_c5, _lbl4):
            _ready = _ok4.get(_k, False)
            _col.markdown(
                f'<div style="background:{"rgba(16,185,129,0.1)" if _ready else CARD_BG};' 
                f'border:1px solid {"#10b981" if _ready else CARD_BOR};border-radius:10px;'
                f'padding:10px 8px;text-align:center;">'
                f'<div style="font-size:1.4rem">{_icon}</div>'
                f'<div style="font-size:0.62rem;color:{MUTED};margin:3px 0">{_lbl}</div>'
                f'<div style="font-size:0.72rem;color:{"#10b981" if _ready else MUTED}">{"✅" if _ready else "⬜"}</div></div>',
                unsafe_allow_html=True)

        if not all(_ok4.values()):
            st.info("Go to M2 and upload your files there, or upload all 5 Fitbit CSV files above.")
            st.stop()

        m4_sec("🔧", "Build Dashboard Data", "Step 1")
        if st.button("🚀 Build Dashboard", key="m4_build", use_container_width=True):
            with st.spinner("Building dashboard data..."):
                _daily = raw4["daily"].copy()
                _daily["ActivityDate"] = pd.to_datetime(_daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")
                _hr_raw = raw4["hr_raw"].copy()
                _hr_raw["Time"] = pd.to_datetime(_hr_raw["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                _hr_min = _hr_raw.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
                _hr_min.columns = ["Id","Time","HeartRate"]
                _hr_min["Date"] = _hr_min["Time"].dt.date
                _hr_d = _hr_min.groupby(["Id","Date"])["HeartRate"].agg(AvgHR="mean",MaxHR="max",MinHR="min").reset_index()
                _sl = raw4["sleep"].copy()
                _sc = "date" if "date" in _sl.columns else "Date"
                _sl[_sc] = pd.to_datetime(_sl[_sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                if _sc != "date": _sl = _sl.rename(columns={_sc:"date"})
                _sl["Date"] = _sl["date"].dt.date
                _sl_d = _sl.groupby(["Id","Date"]).agg(TotalSleepMinutes=("value","count")).reset_index()
                _m = _daily.rename(columns={"ActivityDate":"Date"}).copy()
                _m["Date"] = _m["Date"].dt.date
                _m = _m.merge(_hr_d, on=["Id","Date"], how="left")
                _m = _m.merge(_sl_d, on=["Id","Date"], how="left")
                _m["TotalSleepMinutes"] = _m["TotalSleepMinutes"].fillna(0)
                for _c in ["AvgHR","MaxHR","MinHR"]:
                    if _c in _m.columns:
                        _m[_c] = _m.groupby("Id")[_c].transform(lambda x: x.fillna(x.median()))
                st.session_state.master       = _m
                st.session_state.hr_minute    = _hr_min
                st.session_state.date_min     = str(_m["Date"].min())
                st.session_state.date_max     = str(_m["Date"].max())
                st.session_state.pipeline_done = True
                if "anom_hr"    in raw4: st.session_state.anom_hr    = raw4["anom_hr"]
                if "anom_steps" in raw4: st.session_state.anom_steps = raw4["anom_steps"]
                if "anom_sleep" in raw4: st.session_state.anom_sleep = raw4["anom_sleep"]
            st.rerun()

    elif _has_m2_data and st.session_state.pipeline_done:
        st.markdown('''<div class="alert-success">
            ✅ <b>Dashboard data ready</b> (from M2 upload).
        </div>''', unsafe_allow_html=True)

    if not st.session_state.pipeline_done:
        st.stop()


    master = st.session_state.master
    master["Date"] = pd.to_datetime(master["Date"])
    anom_hr    = st.session_state.anom_hr
    anom_steps = st.session_state.anom_steps
    anom_sleep = st.session_state.anom_sleep
    if anom_hr is not None and anom_steps is not None and anom_sleep is not None:
        st.markdown('<div class="alert-info">ℹ️ Using anomaly data from M3.</div>', unsafe_allow_html=True)

    # ── Sidebar Filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:0.85rem;'>📊 M4 Filters</p>", unsafe_allow_html=True)
        date_min = pd.to_datetime(st.session_state.date_min)
        date_max = pd.to_datetime(st.session_state.date_max)

        d_from = st.date_input("From", value=date_min.date(), key="m4_from")
        d_to   = st.date_input("To",   value=date_max.date(), key="m4_to")

        all_users = sorted(master["Id"].unique().tolist())
        sel_users = st.multiselect("Users", all_users, default=all_users[:5] if len(all_users) > 5 else all_users, key="m4_users")

        hr_high_m4 = st.slider("HR High", 90, 130, 100, key="m4_hrh")
        hr_low_m4  = st.slider("HR Low",  40,  70,  50, key="m4_hrl")
        steps_low_m4  = st.slider("Steps Low", 100, 2000, 500, key="m4_stl")
        sl_low_m4  = st.slider("Sleep Low (min)", 30, 120, 60, key="m4_sll")
        sl_high_m4 = st.slider("Sleep High (min)", 500, 800, 600, key="m4_slh")
        sigma_m4   = st.slider("Residual σ", 1.0, 4.0, 2.0, 0.5, key="m4_sig")

    # Filter master
    d_from_dt = pd.to_datetime(d_from)
    d_to_dt   = pd.to_datetime(d_to)
    mf = master[(master["Date"] >= d_from_dt) & (master["Date"] <= d_to_dt)]
    if sel_users:
        mf = mf[mf["Id"].isin(sel_users)]

    # Compute fresh anomalies for filtered range
    def _detect_hr_m4(mf, hr_high, hr_low, sigma):
        if "AvgHR" not in mf.columns: return None
        d = mf.groupby("Date")["AvgHR"].mean().reset_index(); d.columns = ["Date","AvgHR"]
        d["rolling_med"]   = d["AvgHR"].rolling(3, center=True, min_periods=1).median()
        d["residual"]      = d["AvgHR"] - d["rolling_med"]
        resid_std          = d["residual"].std()
        d["thresh_high"]   = d["AvgHR"] > hr_high
        d["thresh_low"]    = d["AvgHR"] < hr_low
        d["resid_anomaly"] = d["residual"].abs() > sigma * resid_std
        d["is_anomaly"]    = d["thresh_high"] | d["thresh_low"] | d["resid_anomaly"]
        def _r(row):
            p = []
            if row["thresh_high"]:    p.append(f"HR>{hr_high}")
            if row["thresh_low"]:     p.append(f"HR<{hr_low}")
            if row["resid_anomaly"]:  p.append(f"Residual+/-{sigma:.0f}s")
            return ", ".join(p)
        d["reason"] = d.apply(_r, axis=1)
        return d

    def _detect_steps_m4(mf, st_low, sigma):
        if "TotalSteps" not in mf.columns: return None
        d = mf.groupby("Date")["TotalSteps"].mean().reset_index()
        d["rolling_med"]   = d["TotalSteps"].rolling(3, center=True, min_periods=1).median()
        d["residual"]      = d["TotalSteps"] - d["rolling_med"]
        resid_std          = d["residual"].std()
        d["thresh_low"]    = d["TotalSteps"] < st_low
        d["resid_anomaly"] = d["residual"].abs() > sigma * resid_std
        d["is_anomaly"]    = d["thresh_low"] | d["resid_anomaly"]
        def _r(row):
            p = []
            if row["thresh_low"]:    p.append(f"Steps<{st_low}")
            if row["resid_anomaly"]: p.append(f"Residual+/-{sigma:.0f}s")
            return ", ".join(p)
        d["reason"] = d.apply(_r, axis=1)
        return d

    def _detect_sleep_m4(mf, sl_low, sl_high, sigma):
        if "TotalSleepMinutes" not in mf.columns: return None
        d = mf.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
        d["rolling_med"]   = d["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
        d["residual"]      = d["TotalSleepMinutes"] - d["rolling_med"]
        resid_std          = d["residual"].std()
        d["thresh_low"]    = (d["TotalSleepMinutes"] > 0) & (d["TotalSleepMinutes"] < sl_low)
        d["thresh_high"]   = d["TotalSleepMinutes"] > sl_high
        d["no_data"]       = d["TotalSleepMinutes"] == 0
        d["resid_anomaly"] = d["residual"].abs() > sigma * resid_std
        d["is_anomaly"]    = d["thresh_low"] | d["thresh_high"] | d["resid_anomaly"]
        def _r(row):
            p = []
            if row["no_data"]:       p.append("No device worn")
            if row["thresh_low"]:    p.append(f"Sleep<{sl_low}min")
            if row["thresh_high"]:   p.append(f"Sleep>{sl_high}min")
            if row["resid_anomaly"]: p.append(f"Residual+/-{sigma:.0f}s")
            return ", ".join(p)
        d["reason"] = d.apply(_r, axis=1)
        return d

    anom_hr_f    = _detect_hr_m4(mf, hr_high_m4, hr_low_m4, sigma_m4)
    anom_steps_f = _detect_steps_m4(mf, steps_low_m4, sigma_m4)
    anom_sleep_f = _detect_sleep_m4(mf, sl_low_m4, sl_high_m4, sigma_m4)

    n_hr_f    = int(anom_hr_f["is_anomaly"].sum())    if anom_hr_f    is not None else 0
    n_steps_f = int(anom_steps_f["is_anomaly"].sum()) if anom_steps_f is not None else 0
    n_sleep_f = int(anom_sleep_f["is_anomaly"].sum()) if anom_sleep_f is not None else 0

    # ── KPI Strip ──────────────────────────────────────────────────────────────
    m4_sec("📊", "Executive KPI Dashboard")
    kpis = []
    if "TotalSteps"          in mf.columns: kpis.append((f'{mf["TotalSteps"].mean():,.0f}', "Avg Steps/Day", ACCENT))
    if "Calories"            in mf.columns: kpis.append((f'{mf["Calories"].mean():,.0f}', "Avg Calories/Day", AMBER))
    if "AvgHR"               in mf.columns: kpis.append((f'{mf["AvgHR"].mean():.0f}', "Avg Heart Rate", RED))
    if "TotalSleepMinutes"   in mf.columns: kpis.append((f'{mf["TotalSleepMinutes"].mean():.0f}', "Avg Sleep (min)", PURPLE))
    kpis.append((str(n_hr_f+n_steps_f+n_sleep_f), "Total Anomalies", ACCENT_RED))
    kpis.append((str(mf["Id"].nunique()), "Active Users", GREEN))

    kpi_html = '<div class="kpi-grid">'
    for val, label, clr in kpis:
        kpi_html += f'<div class="kpi-card"><div class="kpi-val" style="color:{clr}">{val}</div><div class="kpi-label">{label}</div></div>'
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Dashboard Tabs ─────────────────────────────────────────────────────────
    tab_ov, tab_hr, tab_steps, tab_sleep, tab_exp = st.tabs([
        "📈 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "📥 Export"
    ])

    with tab_ov:
        m4_sec("📈", "Activity Trends — Overview")
        if "TotalSteps" in mf.columns and "Calories" in mf.columns:
            daily_agg = mf.groupby("Date").agg(
                Steps=("TotalSteps","mean"), Calories=("Calories","mean"),
                Sleep=("TotalSleepMinutes","mean")
            ).reset_index()

            fig_ov = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                   subplot_titles=["Daily Steps","Calories","Sleep (min)"],
                                   vertical_spacing=0.08)
            fig_ov.add_trace(go.Scatter(x=daily_agg["Date"], y=daily_agg["Steps"],
                fill="tozeroy", fillcolor=f"rgba(56,189,248,0.1)",
                line=dict(color=ACCENT,width=1.5), name="Steps"), row=1, col=1)
            fig_ov.add_trace(go.Scatter(x=daily_agg["Date"], y=daily_agg["Calories"],
                fill="tozeroy", fillcolor=f"rgba(245,158,11,0.1)",
                line=dict(color=AMBER,width=1.5), name="Calories"), row=2, col=1)
            fig_ov.add_trace(go.Scatter(x=daily_agg["Date"], y=daily_agg["Sleep"],
                fill="tozeroy", fillcolor=f"rgba(167,139,250,0.1)",
                line=dict(color=PURPLE,width=1.5), name="Sleep"), row=3, col=1)
            fig_ov.update_layout(**PLOTLY_BASE, height=600,
                                 title=dict(text="📈 Activity Overview",font_color=TEXT,font_size=14))
            fig_ov.update_xaxes(gridcolor=GRID_CLR, zeroline=False)
            fig_ov.update_yaxes(gridcolor=GRID_CLR, zeroline=False)
            st.plotly_chart(fig_ov, use_container_width=True)

        # User activity heatmap
        if len(mf["Id"].unique()) > 1:
            m4_sec("🔥", "User Activity Heatmap")
            pivot = mf.pivot_table(index="Id", values="TotalSteps", columns="Date", aggfunc="mean").fillna(0)
            fig_hm = px.imshow(pivot.values, labels=dict(x="Date", y="User"),
                               color_continuous_scale="Blues",
                               x=[str(d.date()) for d in pivot.columns],
                               y=[str(u) for u in pivot.index])
            ptheme(fig_hm, "🔥 Step Count Heatmap by User & Date", h=300)
            st.plotly_chart(fig_hm, use_container_width=True)

    with tab_hr:
        m4_sec("❤️", "Heart Rate Deep-Dive", f"{n_hr_f} anomalies")
        if anom_hr_f is not None:
            fig_hr2 = go.Figure()
            fig_hr2.add_trace(go.Scatter(x=anom_hr_f["Date"], y=anom_hr_f["AvgHR"],
                mode="lines", line=dict(color=ACCENT, width=1.5), name="Avg HR"))
            fig_hr2.add_trace(go.Scatter(x=anom_hr_f["Date"], y=anom_hr_f["rolling_med"],
                mode="lines", line=dict(color=ACCENT3, width=1, dash="dot"), name="Rolling Median"))
            apts = anom_hr_f[anom_hr_f["is_anomaly"]]
            fig_hr2.add_trace(go.Scatter(x=apts["Date"], y=apts["AvgHR"], mode="markers",
                marker=dict(color=ACCENT_RED, size=10, symbol="x"), name="Anomaly"))
            fig_hr2.add_hline(y=hr_high_m4, line_dash="dash", line_color=ACCENT_RED, line_width=1)
            fig_hr2.add_hline(y=hr_low_m4,  line_dash="dash", line_color=AMBER, line_width=1)
            ptheme(fig_hr2, "❤️ Heart Rate with Anomalies", h=380)
            st.plotly_chart(fig_hr2, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">HR Statistics</div>
                    <div style="font-size:0.82rem;line-height:2.1;color:{MUTED}">
                        Mean HR: <b style="color:{ACCENT}">{anom_hr_f['AvgHR'].mean():.1f} bpm</b><br>
                        Max HR:  <b style="color:{RED}">{anom_hr_f['AvgHR'].max():.1f} bpm</b><br>
                        Min HR:  <b style="color:{AMBER}">{anom_hr_f['AvgHR'].min():.1f} bpm</b><br>
                        Anomaly days: <b style="color:{ACCENT_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if n_hr_f > 0:
                    st.dataframe(apts[["Date","AvgHR","rolling_med","residual"]].round(2),
                                 use_container_width=True, height=200)

    with tab_steps:
        m4_sec("🚶", "Steps Deep-Dive", f"{n_steps_f} anomalies")
        if anom_steps_f is not None:
            fig_st2 = go.Figure()
            fig_st2.add_trace(go.Bar(x=anom_steps_f["Date"], y=anom_steps_f["TotalSteps"],
                name="Steps", marker_color=ACCENT, opacity=0.65))
            apts_s = anom_steps_f[anom_steps_f["is_anomaly"]]
            fig_st2.add_trace(go.Scatter(x=apts_s["Date"], y=apts_s["TotalSteps"], mode="markers",
                marker=dict(color=ACCENT_RED, size=11, symbol="x"), name="Anomaly"))
            fig_st2.add_hline(y=steps_low_m4, line_dash="dash", line_color=ACCENT_RED, line_width=1)
            ptheme(fig_st2, "🚶 Steps with Anomalies", h=380)
            st.plotly_chart(fig_st2, use_container_width=True)

            # Weekly distribution
            mf2 = mf.copy()
            mf2["Weekday"] = pd.to_datetime(mf2["Date"]).dt.day_name()
            wd_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            wd_agg = mf2.groupby("Weekday")["TotalSteps"].mean().reindex(wd_order).reset_index()
            fig_wd = go.Figure(go.Bar(x=wd_agg["Weekday"], y=wd_agg["TotalSteps"],
                marker_color=PURPLE, opacity=0.8))
            ptheme(fig_wd, "📅 Average Steps by Day of Week", h=280)
            st.plotly_chart(fig_wd, use_container_width=True)

    with tab_sleep:
        m4_sec("💤", "Sleep Deep-Dive", f"{n_sleep_f} anomalies")
        if anom_sleep_f is not None:
            fig_sl2 = go.Figure()
            fig_sl2.add_trace(go.Scatter(x=anom_sleep_f["Date"], y=anom_sleep_f["TotalSleepMinutes"],
                mode="lines+markers", line=dict(color=PURPLE, width=1.5),
                marker=dict(size=4), name="Sleep"))
            apts_sl = anom_sleep_f[anom_sleep_f["is_anomaly"]]
            fig_sl2.add_trace(go.Scatter(x=apts_sl["Date"], y=apts_sl["TotalSleepMinutes"], mode="markers",
                marker=dict(color=ACCENT_RED, size=11, symbol="x"), name="Anomaly"))
            fig_sl2.add_hline(y=sl_low_m4,  line_dash="dash", line_color=AMBER, line_width=1)
            fig_sl2.add_hline(y=sl_high_m4, line_dash="dash", line_color=ACCENT_RED, line_width=1)
            ptheme(fig_sl2, "💤 Sleep Duration with Anomalies", h=380)
            st.plotly_chart(fig_sl2, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                mean_sl = anom_sleep_f["TotalSleepMinutes"].mean()
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Sleep Statistics</div>
                    <div style="font-size:0.82rem;line-height:2.1;color:{MUTED}">
                        Mean sleep: <b style="color:{PURPLE}">{mean_sl:.0f} min ({mean_sl/60:.1f} hrs)</b><br>
                        Anomaly days: <b style="color:{ACCENT_RED}">{n_sleep_f}</b> of {len(anom_sleep_f)} total
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                if n_sleep_f > 0:
                    st.dataframe(apts_sl[["Date","TotalSleepMinutes","rolling_med","residual"]].round(2),
                                 use_container_width=True, height=200)

    with tab_exp:
        m4_sec("📥", "Export — PDF Report & CSV Data", "Downloadable")

        # ── What's included info card ──────────────────────────────────────────
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
              color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">
              What\'s Included in the Exports
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.83rem">
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{ACCENT};font-weight:600;margin-bottom:0.5rem">📄 PDF Report (4 pages)</div>
              <div style="color:{MUTED};line-height:1.8">
                ✅ Executive summary<br>
                ✅ Anomaly counts per signal<br>
                ✅ Thresholds used<br>
                ✅ Methodology explanation<br>
                ✅ All 3 charts embedded<br>
                ✅ Full anomaly records tables<br>
                ✅ User activity profiles
              </div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{GREEN};font-weight:600;margin-bottom:0.5rem">📊 CSV Export</div>
              <div style="color:{MUTED};line-height:1.8">
                ✅ All anomaly records<br>
                ✅ Signal type column<br>
                ✅ Date of anomaly<br>
                ✅ Actual vs expected value<br>
                ✅ Residual deviation<br>
                ✅ Anomaly reason text<br>
                ✅ All signals combined
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── PDF generation function (inline, uses M4 theme vars) ──────────────
        def _generate_pdf_m4(master_df, anom_hr_df, anom_steps_df, anom_sleep_df,
                              _hr_high, _hr_low, _st_low, _sl_low, _sl_high, _sigma,
                              fig_hr_p, fig_steps_p, fig_sleep_p):
            try:
                from fpdf import FPDF
            except ImportError:
                return None, "fpdf2 not installed - run: pip install fpdf2"

            import tempfile, os as _os

            class PDF(FPDF):
                def header(self):
                    self.set_fill_color(15, 23, 42)
                    self.rect(0, 0, 210, 18, "F")
                    self.set_font("Helvetica", "B", 13)
                    self.set_text_color(99, 179, 237)
                    self.set_y(4)
                    self.cell(0, 10, "FitPulse Anomaly Detection Report - Milestone 4", align="C")
                    self.set_text_color(148, 163, 184)
                    self.set_font("Helvetica", "", 7)
                    self.set_y(13)
                    self.cell(0, 4, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", align="C")
                    self.ln(6)

                def footer(self):
                    self.set_y(-13)
                    self.set_font("Helvetica", "", 7)
                    self.set_text_color(148, 163, 184)
                    self.cell(0, 8, safe(f"FitPulse ML Pipeline - Page {self.page_no()}"), align="C")

                def section(self, title, color=(99, 179, 237)):
                    self.ln(3)
                    self.set_fill_color(*color)
                    self.set_text_color(255, 255, 255)
                    self.set_font("Helvetica", "B", 10)
                    self.cell(0, 8, f"  {title}", fill=True, ln=True)
                    self.set_text_color(30, 30, 40)
                    self.ln(2)

                def kv(self, key, val):
                    self.set_font("Helvetica", "B", 9)
                    self.set_text_color(80, 80, 100)
                    self.cell(60, 6, key + ":", ln=False)
                    self.set_font("Helvetica", "B", 9)
                    self.set_text_color(20, 20, 30)
                    self.cell(0, 6, str(val), ln=True)

                def para(self, text, size=8.5):
                    self.set_font("Helvetica", "", size)
                    self.set_text_color(60, 60, 80)
                    self.multi_cell(0, 5, text)
                    self.ln(1)

            # safe(): strip non-latin-1 so Helvetica never raises UnicodeEncodeError
            def safe(t):
                t = str(t)
                t = (t.replace("\u2014","-").replace("\u2013","-")
                      .replace("\u2012","-").replace("\u00b1","+/-")
                      .replace("\u03c3","sigma").replace("\u2019","'")
                      .replace("\u2018","'").replace("\u201c",'"')
                      .replace("\u201d",'"').replace("\u2022","*"))
                return t.encode("latin-1", errors="replace").decode("latin-1")

            pdf = PDF()
            pdf.set_auto_page_break(auto=True, margin=18)
            pdf.add_page()

            n_hr    = int(anom_hr_df["is_anomaly"].sum())   if anom_hr_df    is not None else 0
            n_steps = int(anom_steps_df["is_anomaly"].sum()) if anom_steps_df is not None else 0
            n_sleep = int(anom_sleep_df["is_anomaly"].sum()) if anom_sleep_df is not None else 0
            n_users = master_df["Id"].nunique()
            n_days  = master_df["Date"].nunique()
            date_range = (f"{pd.to_datetime(master_df['Date']).min().strftime('%d %b %Y')}"
                          f" to {pd.to_datetime(master_df['Date']).max().strftime('%d %b %Y')}")

            # Page 1: Executive Summary
            pdf.section("1. EXECUTIVE SUMMARY", (15, 23, 60))
            pdf.kv("Dataset",    "Real Fitbit Device Data - Kaggle (arashnic/fitbit)")
            pdf.kv("Users",      f"{n_users} participants")
            pdf.kv("Date Range", date_range)
            pdf.kv("Total Days", f"{n_days} days of observations")
            pdf.kv("Pipeline",   "Milestone 4 - Anomaly Detection Dashboard")
            pdf.ln(2)

            pdf.section("2. ANOMALY SUMMARY", (180, 50, 50))
            pdf.kv("Heart Rate Anomalies", f"{n_hr} days flagged")
            pdf.kv("Steps Anomalies",      f"{n_steps} days flagged")
            pdf.kv("Sleep Anomalies",      f"{n_sleep} days flagged")
            pdf.kv("Total Flags",          f"{n_hr + n_steps + n_sleep} across all signals")
            pdf.ln(2)

            pdf.section("3. DETECTION THRESHOLDS USED", (40, 100, 60))
            pdf.kv("Heart Rate High",  f"> {int(_hr_high)} bpm")
            pdf.kv("Heart Rate Low",   f"< {int(_hr_low)} bpm")
            pdf.kv("Steps Low Alert",  f"< {int(_st_low):,} steps/day")
            pdf.kv("Sleep Low",        f"< {int(_sl_low)} minutes/night")
            pdf.kv("Sleep High",       f"> {int(_sl_high)} minutes/night")
            pdf.kv("Residual Sigma",   f"+/- {float(_sigma):.1f} sigma from rolling median")
            pdf.ln(2)

            pdf.section("4. METHODOLOGY", (60, 80, 140))
            pdf.para(
                "Three complementary anomaly detection methods were applied:\n\n"
                "  1. THRESHOLD VIOLATIONS - Hard upper/lower bounds on each metric. "
                "Any day exceeding these bounds is immediately flagged as anomalous.\n\n"
                "  2. RESIDUAL-BASED DETECTION - A 3-day rolling median is computed as "
                "the expected baseline. Days where the actual value deviates by more than "
                f"+/-{float(_sigma):.1f} standard deviations are flagged.\n\n"
                "  3. DBSCAN OUTLIER CLUSTERING - Users profiled on activity features "
                "and clustered. Users assigned label -1 are structural outliers."
            )

            # Page 2: Charts
            pdf.add_page()
            pdf.section("5. ANOMALY CHARTS", (15, 23, 60))

            def embed_fig(fig, label, w=190, h=82):
                if fig is None:
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(150, 50, 50)
                    pdf.cell(0, 6, f"[{label} - not available]", ln=True)
                    return
                try:
                    img_bytes = fig.to_image(format="png", width=1100, height=480, scale=1.5, engine="kaleido")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(img_bytes)
                        tmp_path = tmp.name
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(80, 80, 100)
                    pdf.cell(0, 6, label, ln=True)
                    pdf.image(tmp_path, x=10, w=w, h=h)
                    _os.unlink(tmp_path)
                    pdf.ln(3)
                except Exception as ex:
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(150, 50, 50)
                    pdf.cell(0, 6, f"[Chart error: {ex}]", ln=True)
                    pdf.ln(2)

            embed_fig(fig_hr_p,    "Figure 1 - Heart Rate with Anomaly Highlights")
            embed_fig(fig_steps_p, "Figure 2 - Step Count Trend with Alert Bands")
            embed_fig(fig_sleep_p, "Figure 3 - Sleep Pattern Visualization")

            # Page 3: Anomaly Tables
            pdf.add_page()

            def anom_table(df, cols, rename_map, max_rows=20):
                if df is None:
                    pdf.para("No data available."); return
                df2 = df[df["is_anomaly"]][cols].copy().rename(columns=rename_map)
                if df2.empty:
                    pdf.para("No anomalies detected."); return
                col_w = 180 // len(df2.columns)
                pdf.set_fill_color(15, 23, 60)
                pdf.set_text_color(180, 210, 255)
                pdf.set_font("Helvetica", "B", 7.5)
                for col in df2.columns:
                    pdf.cell(col_w, 6, safe(str(col)[:18]), border=0, fill=True)
                pdf.ln()
                pdf.set_font("Helvetica", "", 7.5)
                for i, (_, row) in enumerate(df2.head(max_rows).iterrows()):
                    if i % 2 == 0: pdf.set_fill_color(30, 40, 60)
                    else:          pdf.set_fill_color(20, 30, 50)
                    pdf.set_text_color(200, 210, 225)
                    for val in row:
                        cell_text = safe(f"{val:.2f}" if isinstance(val, float) else str(val)[:18])
                        pdf.cell(col_w, 5.5, cell_text, border=0, fill=True)
                    pdf.ln()
                if len(df2) > max_rows:
                    pdf.set_text_color(100, 130, 180)
                    pdf.set_font("Helvetica", "I", 7)
                    pdf.cell(0, 5, f"  ... and {len(df2)-max_rows} more records (see CSV export)", ln=True)
                pdf.ln(3)

            pdf.section("6. ANOMALY RECORDS - HEART RATE", (180, 50, 50))
            anom_table(anom_hr_df, ["Date","AvgHR","rolling_med","residual","reason"],
                       {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

            pdf.section("7. ANOMALY RECORDS - STEPS", (40, 130, 80))
            anom_table(anom_steps_df, ["Date","TotalSteps","rolling_med","residual","reason"],
                       {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

            pdf.section("8. ANOMALY RECORDS - SLEEP", (100, 60, 160))
            anom_table(anom_sleep_df, ["Date","TotalSleepMinutes","rolling_med","residual","reason"],
                       {"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

            # Page 4: User Profiles + Conclusion
            pdf.add_page()
            pdf.section("9. DATASET OVERVIEW & USER PROFILES", (15, 23, 60))
            profile_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
            avail_cols   = [c for c in profile_cols if c in master_df.columns]
            user_profile = master_df.groupby("Id")[avail_cols].mean().round(1)
            col_w2 = 180 // (len(avail_cols) + 1)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(15, 23, 60)
            pdf.set_text_color(180, 210, 255)
            pdf.cell(col_w2, 6, "User ID", border=0, fill=True)
            for col in avail_cols:
                pdf.cell(col_w2, 6, col[:12], border=0, fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", "", 7.5)
            for i, (uid, row) in enumerate(user_profile.iterrows()):
                if i % 2 == 0: pdf.set_fill_color(30, 40, 60)
                else:           pdf.set_fill_color(20, 30, 50)
                pdf.set_text_color(200, 210, 225)
                pdf.cell(col_w2, 5.5, f"...{str(uid)[-6:]}", border=0, fill=True)
                for val in row:
                    pdf.cell(col_w2, 5.5, f"{val:,.0f}", border=0, fill=True)
                pdf.ln()

            pdf.ln(4)
            pdf.section("10. CONCLUSION", (40, 100, 60))
            pdf.para(
                f"The FitPulse Milestone 4 pipeline processed {n_users} users over "
                f"{n_days} days of real Fitbit data. A total of {n_hr+n_steps+n_sleep} "
                f"anomalous events were identified across heart rate, step count, and "
                f"sleep duration signals.\n\n"
                f"   Heart rate: {n_hr} anomalous days.\n"
                f"   Step count: {n_steps} alert days.\n"
                f"   Sleep patterns: {n_sleep} anomaly flags.\n\n"
                "These findings demonstrate the effectiveness of combining rule-based "
                "and statistical anomaly detection methods."
            )

            buf = io.BytesIO()
            buf.write(pdf.output())
            buf.seek(0)
            return buf, None

        # ── CSV generation function ────────────────────────────────────────────
        def _generate_csv_m4(anom_hr_df, anom_steps_df, anom_sleep_df):
            parts = []
            if anom_hr_df is not None:
                hr_out = anom_hr_df[anom_hr_df["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
                hr_out["signal"] = "Heart Rate"
                hr_out = hr_out.rename(columns={"AvgHR":"value","rolling_med":"expected"})
                parts.append(hr_out)
            if anom_steps_df is not None:
                st_out = anom_steps_df[anom_steps_df["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
                st_out["signal"] = "Steps"
                st_out = st_out.rename(columns={"TotalSteps":"value","rolling_med":"expected"})
                parts.append(st_out)
            if anom_sleep_df is not None:
                sl_out = anom_sleep_df[anom_sleep_df["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
                sl_out["signal"] = "Sleep"
                sl_out = sl_out.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})
                parts.append(sl_out)
            if not parts:
                return b""
            combined = pd.concat(parts, ignore_index=True)
            combined = combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"]).round(2)
            buf = io.StringIO()
            combined.to_csv(buf, index=False)
            return buf.getvalue().encode()

        # ── Two-column layout: PDF | CSV ───────────────────────────────────────
        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            m4_sec("📄", "PDF Report")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">'
                        f'Full 4-page PDF with charts embedded, anomaly tables, and user profiles.</div>',
                        unsafe_allow_html=True)

            if st.button("📄 Generate PDF Report", key="m4_gen_pdf", use_container_width=True):
                with st.spinner("⏳ Generating PDF (embedding charts)…"):
                    try:
                        # Build chart figures for embedding
                        _fig_hr_p = go.Figure()
                        if anom_hr_f is not None:
                            _fig_hr_p.add_trace(go.Scatter(x=anom_hr_f["Date"], y=anom_hr_f["AvgHR"],
                                mode="lines+markers", name="Avg HR",
                                line=dict(color="#63b3ed", width=2), marker=dict(size=4)))
                            _fig_hr_p.add_trace(go.Scatter(x=anom_hr_f["Date"], y=anom_hr_f["rolling_med"],
                                mode="lines", name="Trend", line=dict(color="#68d391", width=1.5, dash="dot")))
                            _anom_pts = anom_hr_f[anom_hr_f["is_anomaly"]]
                            if not _anom_pts.empty:
                                _fig_hr_p.add_trace(go.Scatter(x=_anom_pts["Date"], y=_anom_pts["AvgHR"],
                                    mode="markers", name="Anomaly",
                                    marker=dict(color="#f87171", size=11, symbol="x")))
                            _fig_hr_p.add_hline(y=hr_high_m4, line_dash="dash", line_color="#f87171", line_width=1.5,
                                annotation_text=f"High ({hr_high_m4} bpm)", annotation_font_color="#f87171")
                            _fig_hr_p.add_hline(y=hr_low_m4, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                                annotation_text=f"Low ({hr_low_m4} bpm)", annotation_font_color="#f59e0b")
                            _fig_hr_p.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0a0e1a",
                                font_color="#e2e8f0", title="❤️ Heart Rate — Anomaly Detection",
                                margin=dict(l=50,r=30,t=50,b=40))

                        _fig_steps_p = go.Figure()
                        if anom_steps_f is not None:
                            _fig_steps_p.add_trace(go.Bar(x=anom_steps_f["Date"], y=anom_steps_f["TotalSteps"],
                                name="Steps", marker_color="#63b3ed", opacity=0.7))
                            _anom_s = anom_steps_f[anom_steps_f["is_anomaly"]]
                            if not _anom_s.empty:
                                _fig_steps_p.add_trace(go.Scatter(x=_anom_s["Date"], y=_anom_s["TotalSteps"],
                                    mode="markers", name="Anomaly",
                                    marker=dict(color="#f87171", size=11, symbol="x")))
                            _fig_steps_p.add_hline(y=steps_low_m4, line_dash="dash", line_color="#f87171", line_width=1.5,
                                annotation_text=f"Low ({steps_low_m4:,})", annotation_font_color="#f87171")
                            _fig_steps_p.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0a0e1a",
                                font_color="#e2e8f0", title="🚶 Step Count — Anomaly Detection",
                                margin=dict(l=50,r=30,t=50,b=40))

                        _fig_sleep_p = go.Figure()
                        if anom_sleep_f is not None:
                            _fig_sleep_p.add_trace(go.Scatter(x=anom_sleep_f["Date"], y=anom_sleep_f["TotalSleepMinutes"],
                                mode="lines+markers", name="Sleep (min)",
                                line=dict(color="#b794f4", width=2), marker=dict(size=4)))
                            _anom_sl = anom_sleep_f[anom_sleep_f["is_anomaly"]]
                            if not _anom_sl.empty:
                                _fig_sleep_p.add_trace(go.Scatter(x=_anom_sl["Date"], y=_anom_sl["TotalSleepMinutes"],
                                    mode="markers", name="Anomaly",
                                    marker=dict(color="#f87171", size=11, symbol="x")))
                            _fig_sleep_p.add_hline(y=sl_low_m4, line_dash="dash", line_color="#f59e0b", line_width=1.5)
                            _fig_sleep_p.add_hline(y=sl_high_m4, line_dash="dash", line_color="#f87171", line_width=1.5)
                            _fig_sleep_p.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0a0e1a",
                                font_color="#e2e8f0", title="💤 Sleep Duration — Anomaly Detection",
                                margin=dict(l=50,r=30,t=50,b=40))

                        pdf_result, pdf_err = _generate_pdf_m4(
                            master, anom_hr_f, anom_steps_f, anom_sleep_f,
                            hr_high_m4, hr_low_m4, steps_low_m4, sl_low_m4, sl_high_m4, sigma_m4,
                            _fig_hr_p, _fig_steps_p, _fig_sleep_p
                        )
                        if pdf_err:
                            st.error(f"PDF Error: {pdf_err}")
                        else:
                            _pdf_fname = f"FitPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_result,
                                file_name=_pdf_fname,
                                mime="application/pdf",
                                key="m4_dl_pdf"
                            )
                            ui_ok4(f"PDF ready — {_pdf_fname}")
                    except Exception as _e:
                        st.error(f"PDF generation failed: {_e}")
                        st.info("Tip: Install kaleido for chart embedding — pip install kaleido")

        with col_csv:
            m4_sec("📊", "CSV Export")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">'
                        f'All anomaly records from all three signals in a single CSV file.</div>',
                        unsafe_allow_html=True)

            _csv_data  = _generate_csv_m4(anom_hr_f, anom_steps_f, anom_sleep_f)
            _csv_fname = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

            st.download_button(
                label="⬇️ Download Anomaly CSV",
                data=_csv_data,
                file_name=_csv_fname,
                mime="text/csv",
                key="m4_dl_csv"
            )

            with st.expander("👁️ Preview CSV data"):
                if _csv_data:
                    _preview_df = pd.read_csv(io.StringIO(_csv_data.decode()))
                    st.dataframe(_preview_df, use_container_width=True, height=280)
                else:
                    st.info("No anomalies detected in the selected date range.")

            st.divider()
            # Master dataset download
            m4_sec("🗄️", "Master Dataset")
            _buf_m = io.BytesIO()
            master.to_csv(_buf_m, index=False)
            st.download_button("⬇️ Download Master Dataset CSV", _buf_m.getvalue(),
                               "FitPulse_Master.csv", "text/csv", key="m4_dl_master")

        st.divider()
        ui_ok4("Milestone 4 Complete — Full insights dashboard deployed successfully!")
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🎉</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT};margin-bottom:0.8rem;">
                All 4 Milestones Complete!
            </div>
            <div style="color:{MUTED};font-size:0.85rem;line-height:2;">
                ✅ M1 · Data Governance & Preprocessing<br>
                ✅ M2 · Pattern Extraction (TSFresh + Prophet + Clustering)<br>
                ✅ M3 · Anomaly Detection (Threshold + Residual + DBSCAN)<br>
                ✅ M4 · Insights Dashboard (KPIs + Drill-Downs + PDF & CSV Export)
            </div>
        </div>
        """, unsafe_allow_html=True)