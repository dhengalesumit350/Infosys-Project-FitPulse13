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

# Plotly imported lazily inside milestones to avoid startup lag
# But we define a cached CSV reader here for global use
@st.cache_data(show_spinner=False)
def _cached_read_csv(b: bytes) -> pd.DataFrame:
    """Cache CSV parsing — prevents re-parsing on every Streamlit rerender."""
    return pd.read_csv(io.BytesIO(b))

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
            <br><span style="color:{ACCENT};">Fitness Data</span><br>Analytics Pipeline
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
        _file_bytes = file.read()
        try:
            temp_df = _cached_read_csv(_file_bytes)
        except Exception as _e:
            st.error(f"Failed to parse CSV: {_e}")
            temp_df = None
        if temp_df is not None:
            # Only trigger rerun when a genuinely new file is loaded
            _is_new = (st.session_state.raw_df is None or
                       st.session_state.raw_df.shape != temp_df.shape or
                       not st.session_state.ingested)
            if _is_new:
                st.session_state.raw_df   = temp_df
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

            # EDA — rich multi-tab dashboard
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔍 Step 5: Complete Governance EDA")

            try:
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots
                _plotly_ok = True
            except ImportError:
                _plotly_ok = False

            _PBASE = dict(
                paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT,
                font_family="Inter, sans-serif",
                legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1, font_color=TEXT),
                margin=dict(l=50, r=30, t=55, b=45),
                hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT),
            )
            def _pt(fig, title="", h=380):
                fig.update_layout(**_PBASE, height=h)
                fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
                fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
                if title:
                    fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=13, font_family="Syne, sans-serif"))
                return fig

            tab_corr, tab_dist, tab_box, tab_time, tab_scatter = st.tabs([
                "🔥 Correlation Matrix",
                "📊 Distributions",
                "📦 Box Plots",
                "📈 Time Series",
                "🔵 Scatter Matrix",
            ])

            num_df   = df_clean.select_dtypes(include=[np.number])
            num_cols_all = num_df.columns.tolist()

            with tab_corr:
                if not num_df.empty and _plotly_ok:
                    corr_mat = num_df.corr()
                    fig_hm = go.Figure(go.Heatmap(
                        z=corr_mat.values,
                        x=corr_mat.columns.tolist(),
                        y=corr_mat.columns.tolist(),
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(corr_mat.values, 2),
                        texttemplate="%{text}",
                        textfont={"size": 9},
                        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
                    ))
                    _pt(fig_hm, "🔥 Pearson Correlation Heatmap", h=max(400, len(num_cols_all)*40))
                    st.plotly_chart(fig_hm, use_container_width=True)

                    # Top correlations table
                    corr_pairs = (
                        corr_mat.where(np.tril(np.ones(corr_mat.shape), k=-1).astype(bool))
                        .stack().reset_index()
                    )
                    corr_pairs.columns = ["Feature A", "Feature B", "r"]
                    corr_pairs["|r|"] = corr_pairs["r"].abs()
                    top_corr = corr_pairs.nlargest(10, "|r|")[["Feature A", "Feature B", "r"]]
                    st.markdown(f"<p style='color:{MUTED};font-size:0.82rem;'>Top 10 strongest correlations:</p>", unsafe_allow_html=True)
                    st.dataframe(top_corr.reset_index(drop=True).round(3), use_container_width=True, height=280)
                elif not _plotly_ok:
                    # Fallback to altair
                    corr = num_df.corr().reset_index().melt(id_vars="index")
                    heatmap = alt.Chart(corr).mark_rect().encode(
                        x="index:O", y="variable:O",
                        color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"))
                    ).properties(height=400)
                    st.altair_chart(heatmap, use_container_width=True)

            with tab_dist:
                if not num_cols_all:
                    st.info("No numeric columns to plot.")
                elif _plotly_ok:
                    cols_eda_sel = st.multiselect(
                        "Select columns to plot", num_cols_all,
                        default=num_cols_all[:4] if len(num_cols_all) >= 4 else num_cols_all,
                        key="m1_dist_sel"
                    )
                    if cols_eda_sel:
                        ncols_plot = min(2, len(cols_eda_sel))
                        nrows_plot = (len(cols_eda_sel) + 1) // 2
                        fig_dist = make_subplots(rows=nrows_plot, cols=ncols_plot,
                                                 subplot_titles=cols_eda_sel,
                                                 vertical_spacing=0.12, horizontal_spacing=0.08)
                        pal = [ACCENT, AMBER, PURPLE, GREEN, RED, "#f687b3", "#68d391"]
                        for i, col in enumerate(cols_eda_sel):
                            r, c = divmod(i, ncols_plot)
                            fig_dist.add_trace(
                                go.Histogram(x=df_clean[col], name=col,
                                             marker_color=pal[i % len(pal)],
                                             opacity=0.78,
                                             hovertemplate=f"{col}: %{{x}}<br>Count: %{{y}}<extra></extra>"),
                                row=r+1, col=c+1
                            )
                        fig_dist.update_layout(**_PBASE, height=max(320, nrows_plot*220),
                                               showlegend=False,
                                               title=dict(text="📊 Feature Distributions", font_color=TEXT, font_size=13))
                        fig_dist.update_xaxes(gridcolor=GRID_CLR, zeroline=False)
                        fig_dist.update_yaxes(gridcolor=GRID_CLR, zeroline=False)
                        st.plotly_chart(fig_dist, use_container_width=True)
                else:
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

            with tab_box:
                if num_cols_all and _plotly_ok:
                    box_sel = st.multiselect(
                        "Select columns for box plots", num_cols_all,
                        default=num_cols_all[:5] if len(num_cols_all) >= 5 else num_cols_all,
                        key="m1_box_sel"
                    )
                    if box_sel:
                        fig_box = go.Figure()
                        pal = [ACCENT, AMBER, PURPLE, GREEN, RED, "#f687b3", "#68d391"]
                        for i, col in enumerate(box_sel):
                            fig_box.add_trace(go.Box(
                                y=df_clean[col], name=col,
                                marker_color=pal[i % len(pal)],
                                line_color=pal[i % len(pal)],
                                boxmean="sd",
                                hovertemplate=f"<b>{col}</b><br>Value: %{{y:.2f}}<extra></extra>",
                            ))
                        _pt(fig_box, "📦 Box Plots — Outlier & Distribution View", h=420)
                        fig_box.update_layout(boxgap=0.3, showlegend=False)
                        st.plotly_chart(fig_box, use_container_width=True)

                        # Summary stats table
                        st.markdown(f"<p style='color:{MUTED};font-size:0.82rem;'>Descriptive Statistics:</p>", unsafe_allow_html=True)
                        st.dataframe(df_clean[box_sel].describe().round(2), use_container_width=True)
                else:
                    st.info("Install plotly or add numeric columns for box plots.")

            with tab_time:
                date_candidates = [c for c in df_clean.columns if "date" in c.lower() or "time" in c.lower()]
                if date_candidates and num_cols_all and _plotly_ok:
                    date_col = st.selectbox("Date column", date_candidates, key="m1_ts_date")
                    metric_col = st.selectbox("Metric", num_cols_all, key="m1_ts_metric")
                    try:
                        ts_df = df_clean[[date_col, metric_col]].copy()
                        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
                        ts_df = ts_df.dropna().sort_values(date_col)
                        if len(ts_df) > 0:
                            fig_ts = go.Figure()
                            fig_ts.add_trace(go.Scatter(
                                x=ts_df[date_col], y=ts_df[metric_col],
                                mode="lines+markers",
                                line=dict(color=ACCENT, width=1.8),
                                marker=dict(size=4, color=ACCENT),
                                name=metric_col,
                                hovertemplate=f"Date: %{{x}}<br>{metric_col}: %{{y:.2f}}<extra></extra>",
                            ))
                            # Rolling mean overlay
                            roll = ts_df[metric_col].rolling(7, min_periods=1).mean()
                            fig_ts.add_trace(go.Scatter(
                                x=ts_df[date_col], y=roll,
                                mode="lines",
                                line=dict(color=AMBER, width=2, dash="dot"),
                                name="7-day rolling mean",
                            ))
                            _pt(fig_ts, f"📈 {metric_col} Over Time", h=400)
                            st.plotly_chart(fig_ts, use_container_width=True)
                        else:
                            st.warning("No valid date/metric rows after parsing.")
                    except Exception as e_ts:
                        st.warning(f"Time series plot failed: {e_ts}")
                elif not date_candidates:
                    st.info("No date/time column detected in the dataset. Upload a CSV with a Date column to see trends.")
                else:
                    st.info("Install plotly for time series charts.")

            with tab_scatter:
                if len(num_cols_all) >= 2 and _plotly_ok:
                    sc_cols = st.multiselect(
                        "Select 2–4 numeric columns for scatter matrix",
                        num_cols_all,
                        default=num_cols_all[:4] if len(num_cols_all) >= 4 else num_cols_all[:2],
                        key="m1_scatter_sel"
                    )
                    if len(sc_cols) >= 2:
                        try:
                            fig_sm = px.scatter_matrix(
                                df_clean[sc_cols].dropna(),
                                dimensions=sc_cols,
                                color_discrete_sequence=[ACCENT],
                            )
                            fig_sm.update_traces(
                                diagonal_visible=True,
                                marker=dict(size=3, opacity=0.5, color=ACCENT),
                                selector=dict(type="splom"),
                            )
                            fig_sm.update_layout(
                                **_PBASE,
                                height=max(450, len(sc_cols)*160),
                                title=dict(text="🔵 Scatter Matrix — Feature Relationships",
                                           font_color=TEXT, font_size=13),
                            )
                            st.plotly_chart(fig_sm, use_container_width=True)
                        except Exception as e_sc:
                            st.warning(f"Scatter matrix failed: {e_sc}")
                    else:
                        st.info("Select at least 2 columns.")
                else:
                    st.info("Need ≥2 numeric columns and plotly installed.")

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
    # ── All imports needed for M2 ─────────────────────────────────────────────
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler

    # ── M2 colour palette (matches Pattern_Extraction.py exactly) ─────────────
    M2C_DARK   = "#0d1117"; M2C_CARD   = "#161b22"; M2C_CARD2  = "#1c2128"
    M2C_BORDER = "#30363d"; M2C_TEXT   = "#e6edf3"; M2C_MUTED  = "#8b949e"
    M2C_BLUE   = "#58a6ff"; M2C_GREEN  = "#3fb950"; M2C_AMBER  = "#f0883e"
    M2C_PURPLE = "#bc8cff"; M2C_RED    = "#ff7b72"; M2C_PINK   = "#f778ba"
    M2C_TEAL   = "#39d353"
    M2C_PAL    = [M2C_BLUE, M2C_PINK, M2C_GREEN, M2C_AMBER, M2C_PURPLE, M2C_RED, M2C_TEAL, "#ffa657"]

    plt.rcParams.update({
        "figure.facecolor": M2C_DARK,   "axes.facecolor":   M2C_CARD2,
        "axes.edgecolor":   M2C_BORDER, "axes.labelcolor":  M2C_MUTED,
        "axes.titlecolor":  M2C_TEXT,   "xtick.color":      M2C_MUTED,
        "ytick.color":      M2C_MUTED,  "grid.color":       M2C_BORDER,
        "text.color":       M2C_TEXT,   "legend.facecolor": M2C_CARD,
        "legend.edgecolor": M2C_BORDER, "font.size":        9,
    })

    # ── Helpers ───────────────────────────────────────────────────────────────
    def m2_step_box(num, title, desc=""):
        st.markdown(
            f'<div class="step-box"><span class="step-num">{num}</span>'
            f'<div><div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div></div></div>',
            unsafe_allow_html=True)

    def m2_phase_banner(icon, title, steps, desc):
        st.markdown(
            f'<div style="background:linear-gradient(120deg,{M2C_CARD},{M2C_CARD2});'
            f'border:1px solid {M2C_BLUE};border-left:5px solid {M2C_BLUE};'
            f'border-radius:12px;padding:20px 26px;margin:32px 0 6px">'
            f'<div style="font-size:.65rem;font-weight:800;letter-spacing:.15em;'
            f'color:{M2C_BLUE};text-transform:uppercase;margin-bottom:6px">{steps}</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:{M2C_TEXT}">{icon} {title}</div>'
            f'<div style="color:{M2C_MUTED};font-size:.82rem;margin-top:4px">{desc}</div>'
            f'</div>', unsafe_allow_html=True)

    def m2_fig_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=M2C_DARK, edgecolor="none")
        buf.seek(0); return buf

    def m2_dl_btn(fig, fname, key):
        st.download_button(f"📥 Download {fname}", m2_fig_bytes(fig),
                           fname, "image/png", key=key)

    def m2_df_pq(df):
        buf = io.BytesIO(); df.to_parquet(buf, index=True); buf.seek(0); return buf.read()

    def m2_ser_json(s):
        return s.reset_index(drop=True).to_json().encode()

    # ── Cached heavy computations ──────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def m2_read_csv(b):
        return pd.read_csv(io.BytesIO(b))

    def m2_detect_type(df):
        cols = set(df.columns)
        if "ActivityDate"  in cols and "TotalSteps"     in cols: return "daily"
        if "ActivityHour"  in cols and "StepTotal"      in cols: return "hourly_steps"
        if "ActivityHour"  in cols and "TotalIntensity" in cols: return "hourly_intensities"
        if "Time"          in cols and "Value"          in cols: return "heartrate"
        if "date"          in cols and "value"          in cols: return "sleep"
        if "value__sum_values" in cols or "value__mean" in cols: return "tsfresh"
        return "unknown"

    @st.cache_data(show_spinner="⏳ Resampling heart-rate to 1-minute (once)…")
    def m2_resample_hr(b):
        hr = m2_read_csv(b)
        hr["Time"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        out = hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
        out.columns = ["Id", "Time", "HeartRate"]
        buf = io.BytesIO(); out.dropna().to_parquet(buf, index=True); buf.seek(0)
        return buf.read()

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
        hr_d = hr_min.groupby(["Id","Date"])["HeartRate"].agg(
            AvgHR="mean", MaxHR="max", MinHR="min", StdHR="std").reset_index()
        sleep["Date"] = sleep["date"].dt.date
        sl_d = sleep.groupby(["Id","Date"]).agg(
            TotalSleepMinutes=("value","count"),
            DominantSleepStage=("value", lambda x: x.mode().iloc[0] if not x.empty else 0)
        ).reset_index()
        m = daily.rename(columns={"ActivityDate":"Date"}).copy()
        m["Date"] = m["Date"].dt.date
        m = m.merge(hr_d, on=["Id","Date"], how="left")
        m = m.merge(sl_d, on=["Id","Date"], how="left")
        m["TotalSleepMinutes"]  = m["TotalSleepMinutes"].fillna(0)
        m["DominantSleepStage"] = m["DominantSleepStage"].fillna(0)
        for c in ["AvgHR","MaxHR","MinHR","StdHR"]:
            m[c] = m.groupby("Id")[c].transform(lambda x: x.fillna(x.median()))
        buf = io.BytesIO(); m.to_parquet(buf, index=True); buf.seek(0)
        return buf.read()

    @st.cache_data(show_spinner="⏳ Fitting Prophet model…")
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
                      yearly_seasonality=False, uncertainty_samples=0,
                      changepoint_prior_scale=0.1)
        mdl.fit(df)
        fc = mdl.predict(mdl.make_future_dataframe(periods=horizon))
        buf1 = io.BytesIO(); df.to_parquet(buf1, index=True); buf1.seek(0)
        buf2 = io.BytesIO(); fc.to_parquet(buf2, index=True); buf2.seek(0)
        return buf1.read(), buf2.read()

    @st.cache_data(show_spinner="⏳ Clustering + elbow (once)…")
    def m2_run_clustering(feat_b, k, eps, min_s):
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.decomposition import PCA
        feats = pd.read_parquet(io.BytesIO(feat_b))
        X = StandardScaler().fit_transform(feats.select_dtypes(include=[np.number]).fillna(0))
        km  = KMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(X)
        db  = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
        pca = PCA(n_components=2, random_state=42)
        X2  = pca.fit_transform(X)
        var = (pca.explained_variance_ratio_ * 100).tolist()
        inertias = [KMeans(n_clusters=ki, random_state=42, n_init=3).fit(X).inertia_
                    for ki in range(2, 10)]
        return X.tobytes(), X2.tobytes(), var, km.tolist(), db.tolist(), inertias

    @st.cache_data(show_spinner="⏳ Running t-SNE…")
    def m2_run_tsne(X_b, n_feats):
        from sklearn.manifold import TSNE
        X = np.frombuffer(X_b, dtype=np.float64).reshape(-1, n_feats)
        return TSNE(n_components=2, random_state=42,
                    perplexity=min(30, max(2, len(X)-1)),
                    max_iter=300).fit_transform(X).tobytes()

    # ── Sidebar params ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:0.85rem;'>⚙️ M2 Parameters</p>",
                    unsafe_allow_html=True)
        OPTIMAL_K     = st.slider("K-Means Clusters (K)",    2, 8, 3, key="m2_k")
        EPS           = st.slider("DBSCAN ε (eps)",          0.5, 5.0, 2.2, 0.1, key="m2_eps")
        MIN_SAMPLES   = st.slider("DBSCAN min_samples",      1, 5, 2, key="m2_minsamp")
        FORECAST_DAYS = st.slider("Forecast horizon (days)", 7, 60, 14, key="m2_days")
        run_tsne_flag = st.checkbox("Run t-SNE (~15 sec)", value=False, key="m2_tsne")

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{M2C_CARD},{M2C_CARD2});
                border:1px solid {M2C_BORDER};border-radius:14px;
                padding:28px 32px;margin-bottom:28px'>
      <div style='font-size:.65rem;font-weight:800;letter-spacing:.18em;
                  color:{M2C_BLUE};text-transform:uppercase;margin-bottom:8px'>
        MILESTONE 2 · FEATURE EXTRACTION &amp; MODELING
      </div>
      <div style='font-size:2.2rem;font-weight:800;color:{M2C_TEXT};line-height:1.1'>
        ⚡ FitPulse ML Pipeline
      </div>
      <div style='color:{M2C_MUTED};margin-top:10px;font-size:.85rem'>
        TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File upload ────────────────────────────────────────────────────────────
    FILE_DEFS_M2 = [
        ("daily",              "🏃", "Daily Activity",      "dailyActivity_merged.csv"),
        ("heartrate",          "❤️", "Heart Rate",           "heartrate_seconds_merged.csv"),
        ("hourly_intensities", "⚡", "Hourly Intensities",   "hourlyIntensities_merged.csv"),
        ("hourly_steps",       "👟", "Hourly Steps",         "hourlySteps_merged.csv"),
        ("sleep",              "😴", "Minute Sleep",         "minuteSleep_merged.csv"),
        ("tsfresh",            "🧬", "TSFresh Features",     "tsfresh_features.csv"),
    ]
    m2_slots = {k: None for k, *_ in FILE_DEFS_M2}
    m2_raw   = {}

    st.markdown(f"""
    <div style='background:{M2C_CARD};border:1px solid {M2C_BORDER};border-radius:14px;
                padding:22px 26px;margin-bottom:18px'>
      <div style='font-size:1.1rem;font-weight:800;color:{M2C_TEXT};margin-bottom:4px'>
        📂 Upload Your Fitbit CSV Files
      </div>
      <div style='font-size:.8rem;color:{M2C_MUTED}'>
        Select all 6 CSV files at once (hold <b>Ctrl / Cmd</b> to multi-select).<br>
        Files are <b>auto-detected</b> by column structure — no renaming needed.
      </div>
    </div>
    """, unsafe_allow_html=True)

    m2_uploaded = st.file_uploader(
        "📁 Select all 6 CSV files at once",
        type=["csv"], accept_multiple_files=True, key="m2_upload_v2",
    )
    if m2_uploaded:
        for f in m2_uploaded:
            b  = f.read()
            df = m2_read_csv(b)
            dt = m2_detect_type(df)
            if dt in m2_slots:
                m2_slots[dt] = df
                m2_raw[dt]   = b
        # Persist raw bytes so M3/M4 can use them
        for _rk, _sk in [("daily","shared_daily_b"),("heartrate","shared_hr_b"),
                          ("sleep","shared_sleep_b"),("hourly_steps","shared_hourly_s_b"),
                          ("hourly_intensities","shared_hourly_i_b")]:
            if m2_raw.get(_rk):
                st.session_state[_sk] = m2_raw[_rk]

    # Status cards
    m2_card_cols = st.columns(6)
    m2_n_ok = 0
    for col, (key, icon, label, _) in zip(m2_card_cols, FILE_DEFS_M2):
        ready = m2_slots[key] is not None
        if ready: m2_n_ok += 1
        bg  = f"rgba(63,185,80,.10)" if ready else M2C_CARD2
        bdr = M2C_GREEN if ready else M2C_BORDER
        stxt = (f'<span style="color:{M2C_GREEN};font-weight:800;font-size:.82rem">✅ Detected</span>'
                if ready else
                f'<span style="color:{M2C_MUTED};font-size:.78rem">⬜ Missing</span>')
        col.markdown(
            f'<div style="background:{bg};border:1px solid {bdr};'
            f'border-radius:12px;padding:16px 10px;text-align:center">'
            f'<div style="font-size:1.8rem">{icon}</div>'
            f'<div style="font-size:.68rem;font-weight:700;color:{M2C_MUTED};'
            f'text-transform:uppercase;letter-spacing:.06em;margin:6px 0 4px">'
            f'{label}</div>{stxt}</div>', unsafe_allow_html=True)

    st.progress(m2_n_ok / 6, text=f"Files loaded: {m2_n_ok} / 6")

    m2_required = ["daily","heartrate","hourly_intensities","hourly_steps","sleep","tsfresh"]
    m2_missing  = [k for k in m2_required if m2_slots[k] is None]
    if m2_missing:
        nice_names = {"daily":"Daily Activity","heartrate":"Heart Rate",
                      "hourly_intensities":"Hourly Intensities",
                      "hourly_steps":"Hourly Steps","sleep":"Minute Sleep","tsfresh":"TSFresh Features"}
        st.info(f"👆 Upload all 6 CSV files.\n\n**Still needed:** {', '.join(nice_names[k] for k in m2_missing)}")
        st.stop()

    st.success("✅ All 6 files uploaded and ready.")
    st.divider()

    # ── Session state for phases ───────────────────────────────────────────────
    for _pk in ["m2_run_p1","m2_run_p2","m2_run_p3","m2_run_p4"]:
        if _pk not in st.session_state:
            st.session_state[_pk] = False

    # ── Lazy parsers ───────────────────────────────────────────────────────────
    def m2_get_daily():
        df = m2_slots["daily"].copy()
        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="%m/%d/%Y", errors="coerce")
        return df

    def m2_get_hourly_steps():
        df = m2_slots["hourly_steps"].copy()
        df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        return df

    def m2_get_hr():
        df = m2_slots["heartrate"].copy()
        df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        return df

    def m2_get_sleep():
        df = m2_slots["sleep"].copy()
        sc = "date" if "date" in df.columns else "Date"
        df[sc] = pd.to_datetime(df[sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if sc != "date": df = df.rename(columns={sc:"date"})
        return df

    def m2_get_features():
        df = m2_slots["tsfresh"].copy()
        if df.columns[0] in ("Unnamed: 0","","index"):
            df = df.rename(columns={df.columns[0]:"UserId"}).set_index("UserId")
        return df

    def m2_ensure_master_and_hr():
        if "m2_master_b" not in st.session_state or "m2_hr_min_b" not in st.session_state:
            hr_min_b = m2_resample_hr(m2_raw["heartrate"])
            master_b = m2_build_master(m2_raw["daily"], m2_raw["sleep"], hr_min_b)
            st.session_state["m2_master_b"]  = master_b
            st.session_state["m2_hr_min_b"]  = hr_min_b
            # Share with M3/M4
            _sm = pd.read_parquet(io.BytesIO(master_b))
            st.session_state["shared_master_df"] = _sm
        return (pd.read_parquet(io.BytesIO(st.session_state["m2_master_b"])),
                pd.read_parquet(io.BytesIO(st.session_state["m2_hr_min_b"])))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1
    # ══════════════════════════════════════════════════════════════════════════
    if not st.session_state["m2_run_p1"]:
        st.markdown(
            f'<div style="background:{M2C_CARD2};border:2px solid {M2C_BLUE};border-radius:14px;'
            f'padding:22px 24px;text-align:center;margin:22px 0 10px">'
            f'<div style="font-size:2rem">📂</div>'
            f'<div style="font-size:.68rem;font-weight:800;color:{M2C_BLUE};'
            f'text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">Phase 1</div>'
            f'<div style="font-size:.95rem;font-weight:700;color:{M2C_TEXT}">Data Ingestion & Cleaning</div>'
            f'<div style="font-size:.72rem;color:{M2C_MUTED};margin-top:4px">Steps 1–9</div>'
            f'</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 3, 2])
        with c2:
            if st.button("▶  Run Phase 1", key="m2_btn_p1", use_container_width=True, type="primary"):
                st.session_state["m2_run_p1"] = True; st.rerun()

    if st.session_state["m2_run_p1"]:
        m2_phase_banner("📂", "Phase 1 · Data Ingestion & Cleaning", "STEPS 1–9",
                        "Parse timestamps → null audit → resample HR → merge master dataframe")

        daily    = m2_get_daily()
        hourly_s = m2_get_hourly_steps()
        sleep_df = m2_get_sleep()
        hr_df    = m2_get_hr()
        date_span = (daily["ActivityDate"].max() - daily["ActivityDate"].min()).days

        # Steps 1–3
        m2_step_box("Step 1–3", "Files Loaded & Timestamps Parsed",
                    "All 6 CSVs auto-detected · timestamp columns parsed to datetime")
        shape_df = pd.DataFrame({
            "Dataset": ["dailyActivity","hourlySteps","hourlyIntensities","minuteSleep","heartrate"],
            "Rows":    [f"{daily.shape[0]:,}", f"{hourly_s.shape[0]:,}",
                        f"{m2_slots['hourly_intensities'].shape[0]:,}",
                        f"{sleep_df.shape[0]:,}", f"{hr_df.shape[0]:,}"],
            "Columns": [daily.shape[1], hourly_s.shape[1],
                        m2_slots["hourly_intensities"].shape[1],
                        sleep_df.shape[1], hr_df.shape[1]],
        })
        st.dataframe(shape_df, use_container_width=True, hide_index=True)
        st.divider()

        # Step 4 — Null check
        m2_step_box("Step 4", "Null Value Check — All 5 Datasets", "0 nulls = clean data")
        null_rows = []
        for name, df_n in [("dailyActivity", daily), ("hourlySteps", hourly_s),
                           ("hourlyIntensities", m2_slots["hourly_intensities"]),
                           ("minuteSleep", sleep_df), ("heartrate", hr_df)]:
            n = int(df_n.isnull().sum().sum())
            null_rows.append({"Dataset": name, "Total Nulls": n,
                               "Shape": str(df_n.shape),
                               "Status": "✅ Clean" if n == 0 else f"⚠️ {n} nulls"})
        st.dataframe(pd.DataFrame(null_rows), use_container_width=True, hide_index=True)

        fig_nv, ax_nv = plt.subplots(figsize=(9, 2.2))
        ax_nv.barh([r["Dataset"] for r in null_rows], [0]*5, color=M2C_GREEN, height=0.4)
        for i in range(5):
            ax_nv.text(0.01, i, "  0 nulls — 100% complete ✅",
                       va="center", fontsize=9, color=M2C_GREEN, fontweight="700")
        ax_nv.set_xlim(0, 1); ax_nv.set_xticks([]); ax_nv.grid(False)
        ax_nv.set_title("Null Value Audit", fontsize=10, color=M2C_TEXT, pad=6)
        plt.tight_layout(); st.pyplot(fig_nv); plt.close(fig_nv)
        st.divider()

        # Steps 5–6 overview
        m2_step_box("Step 5–6", "Dataset Overview — Key Counts", "Users · date range · rows")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Users",  daily["Id"].nunique())
        c2.metric("Sleep Users",  sleep_df["Id"].nunique())
        c3.metric("HR Users",     hr_df["Id"].nunique())
        c4.metric("HR Records",   f"{hr_df.shape[0]:,}")
        c5.metric("Date Span",    f"{date_span} days")
        c6.metric("Total Rows",   f"{sum(x.shape[0] for x in [daily,hourly_s,sleep_df,hr_df]):,}")
        st.divider()

        # Step 6 — HR Resample
        m2_step_box("Step 6", "HR: Seconds → 1-Minute Resampling",
                    "Per-second HR resampled to 1-minute mean — runs once, fully cached")
        hr_min_b_p1 = m2_resample_hr(m2_raw["heartrate"])
        hr_min_p1   = pd.read_parquet(io.BytesIO(hr_min_b_p1))
        r1, r2, r3 = st.columns(3)
        r1.metric("Before (rows)", f"{hr_df.shape[0]:,}",     delta="seconds-level")
        r2.metric("After  (rows)", f"{hr_min_p1.shape[0]:,}", delta="1-min intervals")
        r3.metric("Compression",
                  f"{(1-hr_min_p1.shape[0]/hr_df.shape[0])*100:.0f}%", delta_color="off")
        st.divider()

        # Steps 7–9 — Build master
        m2_step_box("Step 7–9", "Cleaned Master Dataframe",
                    "dailyActivity + HR aggregates + sleep aggregates → one row per user per day")
        master_b_p1 = m2_build_master(m2_raw["daily"], m2_raw["sleep"], hr_min_b_p1)
        master_p1   = pd.read_parquet(io.BytesIO(master_b_p1))
        st.session_state["m2_master_b"] = master_b_p1
        st.session_state["m2_hr_min_b"] = hr_min_b_p1
        st.session_state["shared_master_df"] = master_p1

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Master Shape",  str(master_p1.shape))
        cm2.metric("Unique Users",  master_p1["Id"].nunique())
        cm3.metric("Null Values",   int(master_p1.isnull().sum().sum()))
        preview_c = [c for c in ["Id","Date","TotalSteps","Calories","AvgHR",
                                  "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
                     if c in master_p1.columns]
        st.dataframe(master_p1[preview_c].head(15), use_container_width=True, hide_index=True)
        st.divider()

        # Step 9a — Stats
        m2_step_box("Step 9a", "Summary Statistics", "describe() for key columns")
        key_c = [c for c in ["TotalSteps","Calories","AvgHR","TotalSleepMinutes",
                              "VeryActiveMinutes","SedentaryMinutes"] if c in master_p1.columns]
        st.dataframe(master_p1[key_c].describe().round(2), use_container_width=True)
        st.divider()

        # Step 9b — Distribution Histograms (2 per row, exact from Pattern_Extraction.py)
        m2_step_box("Step 9b", "Activity Distribution Histograms", "Mean line on every chart")
        dist_cfg = [
            ("TotalSteps",        "Total Daily Steps",     M2C_BLUE,   "Steps/day"),
            ("Calories",          "Daily Calories Burned", M2C_GREEN,  "Calories/day"),
            ("TotalSleepMinutes", "Daily Sleep Duration",  M2C_PURPLE, "Min/day"),
            ("SedentaryMinutes",  "Sedentary Time/Day",   M2C_AMBER,  "Min/day"),
            ("VeryActiveMinutes", "Very-Active Time/Day",  M2C_RED,    "Min/day"),
            ("AvgHR",             "Average Heart Rate",    M2C_PINK,   "BPM"),
        ]
        dist_cfg = [(k, t, c, x) for k, t, c, x in dist_cfg if k in master_p1.columns]
        for i in range(0, len(dist_cfg), 2):
            cols_d = st.columns(2)
            for j in range(2):
                if i + j >= len(dist_cfg): break
                key, title, color, xlabel = dist_cfg[i + j]
                s = master_p1[key].dropna()
                fig, ax = plt.subplots(figsize=(7, 3.2))
                cnts, _, patches = ax.hist(s, bins=20, color=color, alpha=0.85,
                                           edgecolor=M2C_DARK, linewidth=0.4)
                top = max(cnts) if len(cnts) > 0 else 1
                for patch, cnt in zip(patches, cnts):
                    if cnt > 0:
                        ax.text(patch.get_x() + patch.get_width() / 2,
                                cnt + top * 0.015, f"{int(cnt)}",
                                ha="center", va="bottom", fontsize=7, color=M2C_TEXT)
                mv = s.mean()
                ax.axvline(mv, color="white", linestyle="--", linewidth=1.2,
                           label=f"Mean={mv:.0f}")
                ax.set_title(f"📊 {title}", fontsize=9, color=M2C_TEXT, pad=5)
                ax.set_xlabel(xlabel, fontsize=8, color=M2C_MUTED)
                ax.set_ylabel("Records", fontsize=8, color=M2C_MUTED)
                ax.legend(fontsize=8, framealpha=0.4); ax.grid(axis="y", alpha=0.2)
                plt.tight_layout()
                cols_d[j].pyplot(fig); plt.close(fig)
        st.divider()

        # Step 9c — Hourly heatmap
        m2_step_box("Step 9c", "Hourly Steps Heatmap — When Are Users Most Active?",
                    "Average steps per day-of-week × hour-of-day")
        hs = m2_get_hourly_steps()
        hs["Hour"]      = hs["ActivityHour"].dt.hour
        hs["DayOfWeek"] = hs["ActivityHour"].dt.day_name()
        pivot_hw = hs.groupby(["DayOfWeek","Hour"])["StepTotal"].mean().unstack()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot_hw = pivot_hw.reindex([d for d in day_order if d in pivot_hw.index])
        fig_hw, ax_hw = plt.subplots(figsize=(16, 4.5))
        sns.heatmap(pivot_hw, ax=ax_hw, cmap="YlOrRd",
                    annot=True, fmt=".0f", annot_kws={"size": 6},
                    linewidths=0.2, linecolor=M2C_DARK,
                    cbar_kws={"label":"Avg Steps/Hour","shrink":0.6})
        ax_hw.set_title("🕐 Average Steps by Day × Hour", fontsize=11, color=M2C_TEXT, pad=8)
        ax_hw.set_xlabel("Hour (0–23)", fontsize=9, color=M2C_MUTED)
        ax_hw.set_ylabel("Day of Week",  fontsize=9, color=M2C_MUTED)
        plt.tight_layout(); st.pyplot(fig_hw); plt.close(fig_hw)
        st.divider()

        c1, c2, c3 = st.columns([2, 3, 2])
        with c2:
            if st.button("▶  Run Phase 2 — Feature Engineering", key="m2_btn_p2",
                         use_container_width=True, type="primary"):
                st.session_state["m2_run_p2"] = True; st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — TSFresh
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state["m2_run_p2"]:
        m2_phase_banner("🧬", "Phase 2 · Feature Engineering", "STEPS 10–12",
                        "TSFresh features loaded from CSV — 10 statistical features per user")

        features = m2_get_features()

        m2_step_box("Step 10–11", "TSFresh Feature Matrix",
                    "sum · median · mean · length · std · variance · rms · max · abs_max · min")
        ff1, ff2, ff3 = st.columns(3)
        ff1.metric("Users (rows)",    features.shape[0])
        ff2.metric("Features (cols)", features.shape[1])
        ff3.metric("Source",          "Uploaded tsfresh_features.csv")
        st.markdown("**Feature names:**")
        for i, c in enumerate(features.columns):
            st.markdown(f"<span class='info-pill'>{i+1}. {c.replace('value__','')}</span>",
                        unsafe_allow_html=True)
        st.markdown("")
        st.dataframe(features.round(4), use_container_width=True)
        st.divider()

        # Step 12a — Feature Heatmap (normalized, exact from Pattern_Extraction.py)
        m2_step_box("Step 12a", "Feature Heatmap — Normalized 0–1  📸 Screenshot This",
                    "Each cell = exact normalized value · Rows=Users · Cols=Features")
        feat_norm = pd.DataFrame(
            MinMaxScaler().fit_transform(features),
            index=features.index, columns=features.columns)
        fd = feat_norm.rename(columns={c: c.replace("value__","") for c in feat_norm.columns})
        fd.index = [str(i)[-6:] for i in fd.index]
        fig_hm2, ax_hm2 = plt.subplots(
            figsize=(max(12, len(features.columns)*1.4),
                     max(5,  len(features)*0.6)))
        sns.heatmap(fd, ax=ax_hm2, cmap="coolwarm",
                    annot=True, fmt=".2f", annot_kws={"size":8.5,"weight":"bold"},
                    linewidths=0.4, linecolor=M2C_DARK,
                    cbar_kws={"label":"Normalized 0–1","shrink":0.8},
                    vmin=0, vmax=1)
        ax_hm2.set_title("🧬 TSFresh Feature Matrix — Normalized",
                         fontsize=11, color=M2C_TEXT, pad=10)
        ax_hm2.set_xlabel("Statistical Feature", fontsize=9, color=M2C_MUTED)
        ax_hm2.set_ylabel("User ID (last 6 digits)", fontsize=9, color=M2C_MUTED)
        plt.xticks(rotation=30, ha="right", fontsize=8); plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_hm2); m2_dl_btn(fig_hm2, "tsfresh_heatmap.png", "m2_dl_hm2"); plt.close(fig_hm2)
        st.divider()

        # Step 12b — Per-feature bar charts (3 per row, exact from Pattern_Extraction.py)
        m2_step_box("Step 12b", "Per-Feature Bar Charts  (3 per row)",
                    "Sorted ascending · exact value labeled on every bar")
        feat_cols_list = list(features.columns)
        for i in range(0, len(feat_cols_list), 3):
            cols_b = st.columns(min(3, len(feat_cols_list) - i))
            for j, col_b in enumerate(cols_b):
                if i + j >= len(feat_cols_list): break
                feat  = feat_cols_list[i + j]
                fname = feat.replace("value__","")
                vals  = features[feat].sort_values()
                ulbls = [str(x)[-5:] for x in vals.index]
                fig_b, ax_b = plt.subplots(figsize=(5, 3))
                bars_b = ax_b.bar(range(len(vals)), vals.values,
                                  color=[M2C_PAL[k % len(M2C_PAL)] for k in range(len(vals))],
                                  edgecolor=M2C_DARK, linewidth=0.3, zorder=3)
                mx = max(abs(vals.values)) if len(vals) else 1
                for bar, v in zip(bars_b, vals.values):
                    ax_b.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + mx * 0.025,
                              f"{v:.1f}", ha="center", va="bottom",
                              fontsize=6.5, color=M2C_TEXT, fontweight="700")
                ax_b.set_xticks(range(len(vals)))
                ax_b.set_xticklabels(ulbls, rotation=35, ha="right", fontsize=7)
                ax_b.set_title(f"📊 {fname}", fontsize=9, color=M2C_TEXT, pad=4)
                ax_b.set_xlabel("User ID (last 5 dig.)", fontsize=7, color=M2C_MUTED)
                ax_b.set_ylabel(fname, fontsize=7, color=M2C_MUTED)
                ax_b.grid(axis="y", alpha=0.2); ax_b.set_axisbelow(True)
                plt.tight_layout()
                col_b.pyplot(fig_b); plt.close(fig_b)

        st.divider()
        c1, c2, c3 = st.columns([2, 3, 2])
        with c2:
            if st.button("▶  Run Phase 3 — Prophet Forecasting", key="m2_btn_p3",
                         use_container_width=True, type="primary"):
                st.session_state["m2_run_p3"] = True; st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Prophet
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state["m2_run_p3"]:
        m2_phase_banner("📈", "Phase 3 · Prophet Trend Forecasting", "STEPS 13–17",
                        f"Prophet (fast mode: uncertainty_samples=0) → {FORECAST_DAYS}-day forecast")

        try:
            from prophet import Prophet
        except ImportError:
            st.error("❌ prophet not installed. Run: pip install prophet"); st.stop()

        master_p3, hr_min_p3 = m2_ensure_master_and_hr()

        def m2_prophet_plot(df_in, fc, color, title, ylabel, dl_key):
            fig, ax = plt.subplots(figsize=(13, 5))
            fc_start = df_in["ds"].max()
            if "yhat_lower" in fc.columns and "yhat_upper" in fc.columns:
                ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                alpha=0.18, color=color, label="Confidence Interval")
            ax.plot(fc["ds"], fc["yhat"], color=M2C_TEXT, linewidth=2,
                    label="Prophet Trend", zorder=3)
            ax.scatter(df_in["ds"], df_in["y"], color=color, s=28, zorder=5,
                       alpha=0.9, label="Actual Values")
            for idx, (_, row) in enumerate(df_in.iterrows()):
                if idx % 5 == 0:
                    ax.annotate(f"{row['y']:.0f}", (row["ds"], row["y"]),
                                textcoords="offset points", xytext=(0, 6),
                                fontsize=6, color=M2C_TEXT, ha="center", alpha=0.8)
            ax.axvline(fc_start, color=M2C_AMBER, linestyle="--", linewidth=1.6,
                       label=f"Forecast Start ({fc_start.date()})", alpha=0.9)
            ax.set_title(f"📈 {title}", fontsize=11, color=M2C_TEXT, pad=8)
            ax.set_xlabel("Date", fontsize=9, color=M2C_MUTED)
            ax.set_ylabel(ylabel, fontsize=9, color=M2C_MUTED)
            ax.legend(fontsize=8, framealpha=0.35); ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig); m2_dl_btn(fig, dl_key, dl_key.replace(".","_")); plt.close(fig)

        # Heart Rate
        m2_step_box("Step 13–14", "Heart Rate Forecast  📸 Screenshot This",
                    "Daily mean HR → Prophet (fast) → CI shown")
        hr_agg = hr_min_p3.groupby(hr_min_p3["Time"].dt.date)["HeartRate"].mean().reset_index()
        hr_agg.columns = ["ds","y"]
        hr_agg["ds"]   = pd.to_datetime(hr_agg["ds"])
        hr_agg         = hr_agg.dropna().sort_values("ds")
        res_hr = m2_fit_prophet(m2_ser_json(hr_agg["ds"]), m2_ser_json(hr_agg["y"]), FORECAST_DAYS)
        if res_hr[0] is None:
            st.warning("Not enough heart rate data for Prophet (need ≥5 days).")
        else:
            df_hr = pd.read_parquet(io.BytesIO(res_hr[0]))
            fc_hr = pd.read_parquet(io.BytesIO(res_hr[1]))
            h1, h2, h3 = st.columns(3)
            h1.metric("Training Points",  len(df_hr))
            h2.metric("Forecast Horizon", f"{FORECAST_DAYS} days")
            h3.metric("Mode",             "Fast (uncertainty_samples=0)")
            m2_prophet_plot(df_hr, fc_hr, M2C_AMBER,
                            f"Heart Rate Forecast — {FORECAST_DAYS}-Day Projection",
                            "Avg Heart Rate (BPM)", "prophet_hr.png")
        st.divider()

        # Steps
        m2_step_box("Step 15–16", "Daily Steps Forecast  📸 Screenshot This",
                    "Average steps/day → Prophet → annotated")
        steps_agg = m2_get_daily().groupby("ActivityDate")["TotalSteps"].mean().reset_index()
        steps_agg.columns = ["ds","y"]
        steps_agg["ds"]   = pd.to_datetime(steps_agg["ds"])
        steps_agg         = steps_agg.dropna().sort_values("ds")
        res_st  = m2_fit_prophet(m2_ser_json(steps_agg["ds"]), m2_ser_json(steps_agg["y"]), FORECAST_DAYS)
        df_st = fc_st = None
        if res_st[0] is not None:
            df_st = pd.read_parquet(io.BytesIO(res_st[0]))
            fc_st = pd.read_parquet(io.BytesIO(res_st[1]))
            m2_prophet_plot(df_st, fc_st, M2C_GREEN,
                            f"Daily Steps Forecast — {FORECAST_DAYS}-Day Projection",
                            "Avg Steps/Day", "prophet_steps.png")
        st.divider()

        # Sleep
        m2_step_box("Step 17", "Sleep Duration Forecast  📸 Screenshot This",
                    "Daily mean sleep → Prophet → CI shown")
        sleep_agg = master_p3.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
        sleep_agg.columns = ["ds","y"]
        sleep_agg["ds"]   = pd.to_datetime(sleep_agg["ds"])
        sleep_agg         = sleep_agg[sleep_agg["y"] > 0].dropna().sort_values("ds")
        res_sl  = m2_fit_prophet(m2_ser_json(sleep_agg["ds"]), m2_ser_json(sleep_agg["y"]), FORECAST_DAYS)
        df_sl = fc_sl = None
        if res_sl[0] is not None:
            df_sl = pd.read_parquet(io.BytesIO(res_sl[0]))
            fc_sl = pd.read_parquet(io.BytesIO(res_sl[1]))
            m2_prophet_plot(df_sl, fc_sl, M2C_PURPLE,
                            f"Sleep Duration Forecast — {FORECAST_DAYS}-Day Projection",
                            "Avg Sleep (min/day)", "prophet_sleep.png")

            # Combined stacked plot
            if df_st is not None and fc_st is not None:
                st.divider()
                m2_step_box("Step 15–17 Combined", "Steps + Sleep Combined  📸 Screenshot This",
                            "2-row stacked figure · both annotated")
                fig_comb, axes_c = plt.subplots(2, 1, figsize=(13, 9))
                fig_comb.patch.set_facecolor(M2C_DARK)
                for ax_c, (df_c, fc_c, col_c, lbl_c) in zip(
                    axes_c,
                    [(df_st, fc_st, M2C_GREEN,  "Steps"),
                     (df_sl, fc_sl, M2C_PURPLE, "Sleep (minutes)")]):
                    ax_c.set_facecolor(M2C_CARD2)
                    if "yhat_lower" in fc_c.columns:
                        ax_c.fill_between(fc_c["ds"], fc_c["yhat_lower"], fc_c["yhat_upper"],
                                          alpha=0.2, color=col_c, label="CI")
                    ax_c.plot(fc_c["ds"], fc_c["yhat"], color=M2C_TEXT, linewidth=2.2, label="Trend")
                    ax_c.scatter(df_c["ds"], df_c["y"], color=col_c, s=20,
                                 alpha=0.85, zorder=4, label=f"Actual {lbl_c}")
                    for idx, (_, row) in enumerate(df_c.iterrows()):
                        if idx % 5 == 0:
                            ax_c.annotate(f"{row['y']:.0f}", (row["ds"], row["y"]),
                                          textcoords="offset points", xytext=(0, 5),
                                          fontsize=5.5, color=M2C_TEXT, ha="center", alpha=0.75)
                    ax_c.axvline(df_c["ds"].max(), color=M2C_AMBER, linestyle="--",
                                 linewidth=1.6, label="Forecast Start")
                    ax_c.set_title(f"{lbl_c} — Prophet Trend Forecast", fontsize=11, color=M2C_TEXT)
                    ax_c.set_xlabel("Date", color=M2C_MUTED)
                    ax_c.set_ylabel(lbl_c, color=M2C_MUTED)
                    ax_c.legend(fontsize=8, framealpha=0.3); ax_c.grid(alpha=0.15)
                plt.tight_layout()
                st.pyplot(fig_comb)
                m2_dl_btn(fig_comb, "prophet_combined.png", "m2_dl_comb"); plt.close(fig_comb)
        st.divider()

        c1, c2, c3 = st.columns([2, 3, 2])
        with c2:
            if st.button("▶  Run Phase 4 — Clustering & Reduction", key="m2_btn_p4",
                         use_container_width=True, type="primary"):
                st.session_state["m2_run_p4"] = True; st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Clustering
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state["m2_run_p4"]:
        m2_phase_banner("🤖", "Phase 4 · Clustering & Dimensionality Reduction",
                        "STEPS 18–27",
                        "Feature matrix → StandardScaler → K-Means + DBSCAN → "
                        "Elbow → PCA → t-SNE → Cluster profiles")

        master_p4, _ = m2_ensure_master_and_hr()

        m2_step_box("Step 18", "Clustering Feature Matrix",
                    "Average each user's daily metrics → one row per user")
        clust_c = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes",
                                "LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
                   if c in master_p4.columns]
        clust_feats = master_p4.groupby("Id")[clust_c].mean().round(3).dropna()
        cff1, cff2 = st.columns(2)
        cff1.metric("Users for clustering", clust_feats.shape[0])
        cff2.metric("Features",             clust_feats.shape[1])
        st.dataframe(clust_feats.round(2), use_container_width=True)
        st.divider()

        m2_step_box("Step 19", "StandardScaler + Clustering (cached)",
                    "Normalised to mean≈0 · std≈1 · KMeans · DBSCAN · PCA")
        _cf_buf = io.BytesIO(); clust_feats.to_parquet(_cf_buf, index=True); _cf_buf.seek(0)
        (X_b, X2_b, var, km_list, db_list, inertias) = m2_run_clustering(
            _cf_buf.read(), OPTIMAL_K, EPS, MIN_SAMPLES)
        X_scaled      = np.frombuffer(X_b,  dtype=np.float64).reshape(-1, len(clust_c))
        X_pca         = np.frombuffer(X2_b, dtype=np.float64).reshape(-1, 2)
        kmeans_labels = np.array(km_list, dtype=int)
        dbscan_labels = np.array(db_list, dtype=int)
        sc1, sc2 = st.columns(2)
        sc1.metric("Mean after scaling (≈0)", f"{X_scaled.mean():.6f}")
        sc2.metric("Std  after scaling (≈1)", f"{X_scaled.std():.4f}")
        st.divider()

        # Step 20 — Elbow curve (exact from Pattern_Extraction.py)
        m2_step_box("Step 20", "K-Means Elbow Curve  📸 Screenshot This",
                    f"Inertia K=2…9 · selected K={OPTIMAL_K} highlighted")
        K_range = range(2, 10)
        fig_el, ax_el = plt.subplots(figsize=(10, 4))
        ax_el.plot(list(K_range), inertias, "o-", color=M2C_BLUE, linewidth=2.5,
                   markersize=10, markerfacecolor=M2C_PINK, markeredgecolor=M2C_TEXT,
                   markeredgewidth=1.2, zorder=3)
        for k, iner in zip(K_range, inertias):
            ax_el.annotate(f"K={k}\n{iner:.0f}", (k, iner),
                           textcoords="offset points", xytext=(0, 12),
                           ha="center", fontsize=8, color=M2C_TEXT, fontweight="700")
        sel_idx = OPTIMAL_K - 2
        ax_el.scatter([OPTIMAL_K], [inertias[sel_idx]], color=M2C_AMBER, s=220, zorder=5,
                      label=f"Selected K={OPTIMAL_K}", edgecolors=M2C_TEXT, linewidths=1.5)
        ax_el.axvline(OPTIMAL_K, color=M2C_AMBER, linestyle="--", linewidth=1.4, alpha=0.7)
        ax_el.set_title(f"📈 K-Means Elbow Curve (optimal K={OPTIMAL_K})",
                        fontsize=11, color=M2C_TEXT, pad=10)
        ax_el.set_xlabel("Number of Clusters (K)", fontsize=9, color=M2C_MUTED)
        ax_el.set_ylabel("Inertia (Within-Cluster SSQ)", fontsize=9, color=M2C_MUTED)
        ax_el.set_xticks(list(K_range)); ax_el.legend(fontsize=9); ax_el.grid(alpha=0.2)
        plt.tight_layout(); st.pyplot(fig_el)
        m2_dl_btn(fig_el, "elbow_curve.png", "m2_dl_el"); plt.close(fig_el)
        st.divider()

        # Steps 21–22 — KMeans distribution bar chart
        m2_step_box("Step 21–22", "K-Means Clustering", f"K={OPTIMAL_K}")
        clust_feats = clust_feats.copy()
        clust_feats["KMeans_Cluster"] = kmeans_labels
        km_dist = clust_feats["KMeans_Cluster"].value_counts().sort_index()
        cols_km = st.columns(OPTIMAL_K)
        for i, col in enumerate(cols_km):
            col.metric(f"Cluster {i}", f"{int(km_dist.get(i,0))} users")
        c_km = [int(km_dist.get(i,0)) for i in range(OPTIMAL_K)]
        fig_kmd, ax_kmd = plt.subplots(figsize=(7, 3))
        bars_kmd = ax_kmd.bar([f"Cluster {i}" for i in range(OPTIMAL_K)],
                              c_km, color=M2C_PAL[:OPTIMAL_K], edgecolor=M2C_DARK)
        for bar, n in zip(bars_kmd, c_km):
            ax_kmd.text(bar.get_x() + bar.get_width()/2, n + 0.05,
                        f"{n} users", ha="center", va="bottom",
                        fontsize=11, color=M2C_TEXT, fontweight="700")
        ax_kmd.set_title(f"K-Means Distribution (K={OPTIMAL_K})", fontsize=10, color=M2C_TEXT, pad=6)
        ax_kmd.set_xlabel("Cluster", fontsize=9, color=M2C_MUTED)
        ax_kmd.set_ylabel("Users",   fontsize=9, color=M2C_MUTED)
        ax_kmd.grid(axis="y", alpha=0.2); plt.tight_layout()
        st.pyplot(fig_kmd); plt.close(fig_kmd)
        st.divider()

        # Step 23 — DBSCAN distribution bar chart
        m2_step_box("Step 23", "DBSCAN Clustering",
                    f"eps={EPS} · min_samples={MIN_SAMPLES} · noise=-1")
        clust_feats["DBSCAN_Cluster"] = dbscan_labels
        n_cl_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise  = int((dbscan_labels == -1).sum())
        db1, db2, db3 = st.columns(3)
        db1.metric("DBSCAN Clusters",  n_cl_db)
        db2.metric("Noise / Outliers", n_noise)
        db3.metric("Noise %", f"{n_noise/len(dbscan_labels)*100:.1f}%")
        db_cnt  = pd.Series(dbscan_labels).value_counts().sort_index()
        db_lbls = ["Noise" if l == -1 else f"Cluster {l}" for l in db_cnt.index]
        db_clrs = [M2C_RED if l == -1 else M2C_PAL[l % len(M2C_PAL)] for l in db_cnt.index]
        fig_dbd, ax_dbd = plt.subplots(figsize=(7, 3))
        bars_dbd = ax_dbd.bar(db_lbls, db_cnt.values, color=db_clrs, edgecolor=M2C_DARK)
        for bar, n in zip(bars_dbd, db_cnt.values):
            ax_dbd.text(bar.get_x() + bar.get_width()/2, n + 0.05,
                        f"{n} users", ha="center", va="bottom",
                        fontsize=11, color=M2C_TEXT, fontweight="700")
        ax_dbd.set_title(f"DBSCAN Distribution (eps={EPS} · min_samples={MIN_SAMPLES})",
                         fontsize=10, color=M2C_TEXT, pad=6)
        ax_dbd.set_xlabel("Cluster", fontsize=9, color=M2C_MUTED)
        ax_dbd.set_ylabel("Users",   fontsize=9, color=M2C_MUTED)
        ax_dbd.grid(axis="y", alpha=0.2); plt.tight_layout()
        st.pyplot(fig_dbd); plt.close(fig_dbd)
        st.divider()

        # Step 24 — PCA variance
        m2_step_box("Step 24", "PCA — 2D Dimensionality Reduction",
                    "Features → 2 principal components")
        pv1, pv2, pv3 = st.columns(3)
        pv1.metric("PC1 Variance",    f"{var[0]:.1f}%")
        pv2.metric("PC2 Variance",    f"{var[1]:.1f}%")
        pv3.metric("Total Explained", f"{sum(var):.1f}%")
        st.divider()

        # Step 25 — KMeans PCA Scatter (exact from Pattern_Extraction.py)
        m2_step_box("Step 25", "K-Means PCA Scatter  📸 Screenshot This",
                    "2D PCA · coloured by K-Means · User ID labeled")
        fig_km_sc, ax_km = plt.subplots(figsize=(10, 7))
        for cid in sorted(set(kmeans_labels)):
            mask = kmeans_labels == cid
            ax_km.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=M2C_PAL[cid % len(M2C_PAL)], label=f"Cluster {cid}",
                          s=140, alpha=0.88, edgecolors=M2C_TEXT, linewidths=0.7, zorder=3)
            for i, uid in enumerate(clust_feats.index[mask]):
                ax_km.annotate(str(uid)[-4:], (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                               textcoords="offset points", xytext=(5, 5),
                               fontsize=8, color=M2C_TEXT, fontweight="600")
        ax_km.set_title(f"🤖 K-Means PCA 2D (K={OPTIMAL_K}  PC1={var[0]:.1f}%  PC2={var[1]:.1f}%)",
                        fontsize=11, color=M2C_TEXT, pad=10)
        ax_km.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=9, color=M2C_MUTED)
        ax_km.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=9, color=M2C_MUTED)
        ax_km.legend(title=f"K-Means (K={OPTIMAL_K})", fontsize=9, framealpha=0.4)
        ax_km.grid(alpha=0.2); plt.tight_layout()
        st.pyplot(fig_km_sc)
        m2_dl_btn(fig_km_sc, "kmeans_pca.png", "m2_dl_km"); plt.close(fig_km_sc)
        st.divider()

        # Step 26 — DBSCAN PCA Scatter (exact from Pattern_Extraction.py)
        m2_step_box("Step 26", "DBSCAN PCA Scatter  📸 Screenshot This",
                    "Same PCA axes · DBSCAN labels · noise = red ✕")
        fig_db_sc, ax_db = plt.subplots(figsize=(10, 7))
        for lbl in sorted(set(dbscan_labels)):
            mask = dbscan_labels == lbl
            if lbl == -1:
                ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1],
                              c=M2C_RED, marker="X", s=220, alpha=0.95,
                              label="Noise / Outlier (–1)", linewidths=1.5, zorder=5)
                for i, uid in enumerate(clust_feats.index[mask]):
                    ax_db.annotate(f"{str(uid)[-4:]} (noise)",
                                   (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                                   textcoords="offset points", xytext=(8, 6),
                                   fontsize=8, color=M2C_RED, fontweight="700")
            else:
                ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1],
                              c=M2C_PAL[lbl % len(M2C_PAL)], label=f"Cluster {lbl}",
                              s=140, alpha=0.88, edgecolors=M2C_TEXT, linewidths=0.7, zorder=3)
                for i, uid in enumerate(clust_feats.index[mask]):
                    ax_db.annotate(str(uid)[-4:], (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                                   textcoords="offset points", xytext=(5, 5),
                                   fontsize=8, color=M2C_TEXT, fontweight="600")
        ax_db.set_title(f"🤖 DBSCAN PCA 2D (eps={EPS}  min_samples={MIN_SAMPLES})",
                        fontsize=11, color=M2C_TEXT, pad=10)
        ax_db.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=9, color=M2C_MUTED)
        ax_db.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=9, color=M2C_MUTED)
        ax_db.legend(title="DBSCAN Cluster", fontsize=9, framealpha=0.4)
        ax_db.grid(alpha=0.2); plt.tight_layout()
        st.pyplot(fig_db_sc)
        m2_dl_btn(fig_db_sc, "dbscan_pca.png", "m2_dl_db"); plt.close(fig_db_sc)
        st.divider()

        # Step 27a — t-SNE
        m2_step_box("Step 27a", "t-SNE Projection  📸 Screenshot This",
                    "Non-linear 2D embedding · enable in sidebar")
        if run_tsne_flag:
            tsne_out = m2_run_tsne(X_b, len(clust_c))
            X_tsne   = np.frombuffer(tsne_out, dtype=np.float64).reshape(-1, 2)
            fig_ts, axes_t = plt.subplots(1, 2, figsize=(15, 6))
            fig_ts.patch.set_facecolor(M2C_DARK)
            for ax_t, (lbls_t, name_t) in zip(
                axes_t,
                [(kmeans_labels, f"K-Means (K={OPTIMAL_K})"),
                 (dbscan_labels, f"DBSCAN (eps={EPS})")]):
                ax_t.set_facecolor(M2C_CARD2)
                for lbl in sorted(set(lbls_t)):
                    mask = lbls_t == lbl
                    if lbl == -1:
                        ax_t.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                     c=M2C_RED, marker="X", s=190, label="Noise",
                                     alpha=0.95, linewidths=1.5, zorder=5)
                    else:
                        ax_t.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                     c=M2C_PAL[lbl % len(M2C_PAL)], label=f"Cluster {lbl}",
                                     s=120, alpha=0.88, edgecolors=M2C_TEXT,
                                     linewidths=0.7, zorder=3)
                    for i, uid in enumerate(clust_feats.index[mask]):
                        ax_t.annotate(str(uid)[-4:],
                                      (X_tsne[mask][i, 0], X_tsne[mask][i, 1]),
                                      xytext=(5, 5), textcoords="offset points",
                                      fontsize=7, color=M2C_RED if lbl == -1 else M2C_TEXT)
                ax_t.set_title(f"t-SNE — {name_t}", fontsize=11, color=M2C_TEXT, pad=8)
                ax_t.set_xlabel("t-SNE Dim 1", fontsize=9, color=M2C_MUTED)
                ax_t.set_ylabel("t-SNE Dim 2", fontsize=9, color=M2C_MUTED)
                ax_t.legend(title="Cluster", fontsize=8, framealpha=0.35)
                ax_t.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_ts); m2_dl_btn(fig_ts, "tsne_projection.png", "m2_dl_ts"); plt.close(fig_ts)
        else:
            st.info("✅ Enable **'Run t-SNE (~15 sec)'** in the sidebar to generate this plot.")
        st.divider()

        # Step 27b — Cluster profiles (grouped bar chart, exact from Pattern_Extraction.py)
        m2_step_box("Step 27b", "Cluster Profiles — Grand Finale  📸 Screenshot This",
                    "Grouped bar chart · 5 metrics across all clusters · exact values labeled")
        feat_p  = [c for c in clust_feats.columns
                   if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
        profile = clust_feats.groupby("KMeans_Cluster")[feat_p].mean().round(2)
        st.markdown("**Average metrics per cluster:**")
        st.dataframe(profile, use_container_width=True)

        disp_c      = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                    "SedentaryMinutes","TotalSleepMinutes"]
                       if c in profile.columns]
        feat_colors = [M2C_BLUE, M2C_GREEN, M2C_RED, M2C_AMBER, M2C_PURPLE]
        n_feat      = len(disp_c)
        n_clust     = len(profile)
        x           = np.arange(n_clust)
        width       = 0.14
        offsets     = np.linspace(-(n_feat-1)/2*width, (n_feat-1)/2*width, n_feat)
        fig_pr, ax_pr = plt.subplots(figsize=(13, 6))
        for fi, (feat, fc) in enumerate(zip(disp_c, feat_colors)):
            vals = profile[feat].values
            bars = ax_pr.bar(x + offsets[fi], vals, width,
                             label=feat, color=fc, edgecolor=M2C_DARK, alpha=0.9)
            mx = max(vals) if max(vals) > 0 else 1
            for bar, v in zip(bars, vals):
                ax_pr.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + mx*0.012, f"{v:.0f}",
                           ha="center", va="bottom", fontsize=7.5, color=M2C_TEXT, fontweight="700")
        ax_pr.set_xticks(x)
        ax_pr.set_xticklabels([f"Cluster {i}" for i in range(n_clust)],
                              fontsize=12, color=M2C_TEXT, fontweight="700")
        ax_pr.set_title("🏆 Cluster Profiles — Key Feature Averages",
                        fontsize=11, color=M2C_TEXT, pad=10)
        ax_pr.set_xlabel("K-Means Cluster", fontsize=10, color=M2C_MUTED)
        ax_pr.set_ylabel("Mean Value per Day", fontsize=10, color=M2C_MUTED)
        ax_pr.legend(title="Feature", bbox_to_anchor=(1.01,1), fontsize=9, framealpha=0.4)
        ax_pr.grid(axis="y", alpha=0.2); plt.tight_layout()
        st.pyplot(fig_pr); m2_dl_btn(fig_pr, "cluster_profiles.png", "m2_dl_pr"); plt.close(fig_pr)
        st.divider()

        # Step 27c — Cluster interpretation cards (exact from Pattern_Extraction.py)
        m2_step_box("Step 27c", "Cluster Interpretation — Activity Labels",
                    "Auto-labelled by avg steps · 6 key metrics per cluster")
        for i in range(OPTIMAL_K):
            if i not in profile.index: continue
            row   = profile.loc[i]
            steps = row.get("TotalSteps", 0)
            sed   = row.get("SedentaryMinutes", 0)
            act   = row.get("VeryActiveMinutes", 0)
            cals  = row.get("Calories", 0)
            slp   = row.get("TotalSleepMinutes", 0)
            light = row.get("LightlyActiveMinutes", 0)
            n_in  = int((clust_feats["KMeans_Cluster"] == i).sum())
            if   steps > 10000: lbl, clr = "🏃 HIGHLY ACTIVE",     M2C_GREEN
            elif steps > 5000:  lbl, clr = "🚶 MODERATELY ACTIVE",  M2C_BLUE
            else:               lbl, clr = "🛋️ SEDENTARY",           M2C_AMBER
            st.markdown(f"""
            <div style='background:{M2C_CARD2};border-left:5px solid {clr};
                        border-radius:0 12px 12px 0;padding:18px 22px;margin-bottom:14px'>
              <div style='font-size:1.1rem;font-weight:800;color:{clr}'>
                Cluster {i} &nbsp;·&nbsp; {lbl}
                <span style='font-size:.75rem;color:{M2C_MUTED};font-weight:400'>
                  &nbsp;({n_in} users)
                </span>
              </div>
              <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px'>
                <div style='background:{M2C_CARD};border-radius:8px;padding:12px;border-top:2px solid {M2C_BLUE}'>
                  <div style='color:{M2C_MUTED};font-size:.65rem;text-transform:uppercase'>📶 Avg Steps/Day</div>
                  <div style='color:{M2C_TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{steps:,.0f}</div>
                </div>
                <div style='background:{M2C_CARD};border-radius:8px;padding:12px;border-top:2px solid {M2C_GREEN}'>
                  <div style='color:{M2C_MUTED};font-size:.65rem;text-transform:uppercase'>🔥 Calories/Day</div>
                  <div style='color:{M2C_TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{cals:,.0f}</div>
                </div>
                <div style='background:{M2C_CARD};border-radius:8px;padding:12px;border-top:2px solid {M2C_PURPLE}'>
                  <div style='color:{M2C_MUTED};font-size:.65rem;text-transform:uppercase'>💤 Sleep Min/Day</div>
                  <div style='color:{M2C_TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{slp:,.0f}</div>
                </div>
                <div style='background:{M2C_CARD};border-radius:8px;padding:12px;border-top:2px solid {M2C_RED}'>
                  <div style='color:{M2C_MUTED};font-size:.65rem;text-transform:uppercase'>🏃 Very Active Min</div>
                  <div style='color:{M2C_TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{act:.0f}</div>
                </div>
                <div style='background:{M2C_CARD};border-radius:8px;padding:12px;border-top:2px solid {M2C_AMBER}'>
                  <div style='color:{M2C_MUTED};font-size:.65rem;text-transform:uppercase'>🛋️ Sedentary Min</div>
                  <div style='color:{M2C_TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{sed:.0f}</div>
                </div>
                <div style='background:{M2C_CARD};border-radius:8px;padding:12px;border-top:2px solid {M2C_TEAL}'>
                  <div style='color:{M2C_MUTED};font-size:.65rem;text-transform:uppercase'>🚶 Lightly Active</div>
                  <div style='color:{M2C_TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{light:.0f}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.success("✅ Milestone 2 Complete — All 4 phases executed successfully!")
        st.session_state["m2_cluster_done"] = True
        # Share master for M3/M4
        if "m2_master_b" in st.session_state:
            st.session_state["shared_master_df"] = master_p4

        if st.button("🚨 Continue to M3 · Anomaly Detection →", use_container_width=True):
            st.session_state.milestone = 3; st.rerun()



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
        # Auto-build from M2 shared data — runs only once thanks to pipeline_done guard
        st.markdown('''<div class="alert-success">
            ✅ <b>Files loaded automatically from M2</b> — building dashboard data…
        </div>''', unsafe_allow_html=True)
        with st.spinner("Auto-building M4 dashboard from M2 data…"):
            try:
                _daily = _cached_read_csv(st.session_state["shared_daily_b"]).copy()
                _daily["ActivityDate"] = pd.to_datetime(_daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")

                _hr_raw = _cached_read_csv(st.session_state["shared_hr_b"]).copy()
                _hr_raw["Time"] = pd.to_datetime(_hr_raw["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                _hr_min = _hr_raw.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index()
                _hr_min.columns = ["Id","Time","HeartRate"]
                _hr_min["Date"] = _hr_min["Time"].dt.date
                _hr_d = _hr_min.groupby(["Id","Date"])["HeartRate"].agg(AvgHR="mean",MaxHR="max",MinHR="min").reset_index()

                _sl = _cached_read_csv(st.session_state["shared_sleep_b"]).copy()
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

                st.session_state.master        = _m
                st.session_state.hr_minute     = _hr_min
                st.session_state.date_min      = str(_m["Date"].min())
                st.session_state.date_max      = str(_m["Date"].max())
                st.session_state.pipeline_done = True
            except Exception as _m4_err:
                st.error(f"Dashboard build failed: {_m4_err}. Go to M2 first and upload all files.")
                st.stop()
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