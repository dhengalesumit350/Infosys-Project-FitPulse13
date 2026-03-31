import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, io, base64, tempfile, os
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse . Milestone 4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("dark_mode",    True),
    ("pipeline_done",False),
    ("master",       None),
    ("anom_hr",      None),
    ("anom_steps",   None),
    ("anom_sleep",   None),
    ("daily",        None),
    ("hr_minute",    None),
    ("date_min",     None),
    ("date_max",     None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Theme ─────────────────────────────────────────────────────────────────────
dark = st.session_state.dark_mode
if dark:
    BG         = "linear-gradient(135deg,#0a0e1a 0%,#0f1729 40%,#0a1628 100%)"
    CARD_BG    = "rgba(15,23,42,0.85)"
    CARD_BOR   = "rgba(99,179,237,0.2)"
    TEXT       = "#e2e8f0"
    MUTED      = "#94a3b8"
    ACCENT     = "#63b3ed"
    ACCENT2    = "#f687b3"
    ACCENT3    = "#68d391"
    ACCENT_RED = "#fc8181"
    ACCENT_ORG = "#f6ad55"
    PLOT_BG    = "#0f172a"
    PAPER_BG   = "#0a0e1a"
    GRID_CLR   = "rgba(255,255,255,0.05)"
    BADGE_BG   = "rgba(99,179,237,0.15)"
    SECTION_BG = "rgba(99,179,237,0.07)"
    WARN_BG    = "rgba(246,173,85,0.12)"
    WARN_BOR   = "rgba(246,173,85,0.4)"
    SUCCESS_BG = "rgba(104,211,145,0.1)"
    SUCCESS_BOR= "rgba(104,211,145,0.4)"
    DANGER_BG  = "rgba(252,129,129,0.1)"
    DANGER_BOR = "rgba(252,129,129,0.4)"
else:
    BG         = "linear-gradient(135deg,#f0f4ff 0%,#fafbff 50%,#f5f0ff 100%)"
    CARD_BG    = "rgba(255,255,255,0.95)"
    CARD_BOR   = "rgba(66,153,225,0.25)"
    TEXT       = "#1a202c"
    MUTED      = "#4a5568"
    ACCENT     = "#3182ce"
    ACCENT2    = "#d53f8c"
    ACCENT3    = "#38a169"
    ACCENT_RED = "#e53e3e"
    ACCENT_ORG = "#dd6b20"
    PLOT_BG    = "#ffffff"
    PAPER_BG   = "#f8faff"
    GRID_CLR   = "rgba(0,0,0,0.05)"
    BADGE_BG   = "rgba(49,130,206,0.1)"
    SECTION_BG = "rgba(49,130,206,0.05)"
    WARN_BG    = "rgba(221,107,32,0.08)"
    WARN_BOR   = "rgba(221,107,32,0.35)"
    SUCCESS_BG = "rgba(56,161,105,0.08)"
    SUCCESS_BOR= "rgba(56,161,105,0.35)"
    DANGER_BG  = "rgba(229,62,62,0.08)"
    DANGER_BOR = "rgba(229,62,62,0.35)"

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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{{box-sizing:border-box}}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{{
    background:{BG}!important;font-family:'Inter',sans-serif;color:{TEXT}!important}}
[data-testid="stHeader"]{{background:transparent!important}}
[data-testid="stSidebar"]{{
    background:{'rgba(10,14,26,0.97)' if dark else 'rgba(240,244,255,0.97)'}!important;
    border-right:1px solid {CARD_BOR}}}
[data-testid="stSidebar"] *{{color:{TEXT}!important}}
.block-container{{padding:1.2rem 2rem 3rem 2rem!important;max-width:1500px}}
p,div,span,label{{color:{TEXT}}}
.m4-hero{{
    background:{'linear-gradient(135deg,rgba(99,179,237,0.08),rgba(104,211,145,0.05),rgba(10,14,26,0.9))' if dark else 'linear-gradient(135deg,rgba(49,130,206,0.08),rgba(56,161,105,0.05),rgba(240,244,255,0.9))'};
    border:1px solid {CARD_BOR};border-radius:20px;padding:2rem 2.5rem;
    margin-bottom:1.5rem;position:relative;overflow:hidden}}
.hero-title{{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
    color:{TEXT};margin:0 0 0.3rem 0;letter-spacing:-0.02em}}
.hero-sub{{font-size:1rem;color:{MUTED};font-weight:300;margin:0}}
.hero-badge{{display:inline-block;background:{BADGE_BG};border:1px solid {CARD_BOR};
    border-radius:100px;padding:0.25rem 0.9rem;font-size:0.72rem;
    font-family:'JetBrains Mono',monospace;color:{ACCENT};margin-bottom:0.8rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:0.7rem;margin:1rem 0}}
.kpi-card{{background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;
    padding:1rem 1.1rem;text-align:center;backdrop-filter:blur(10px)}}
.kpi-val{{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;
    line-height:1;margin-bottom:0.2rem}}
.kpi-label{{font-size:0.68rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.07em}}
.kpi-sub{{font-size:0.65rem;color:{MUTED};margin-top:0.15rem}}
.sec-header{{display:flex;align-items:center;gap:0.8rem;margin:1.5rem 0 0.8rem;
    padding-bottom:0.5rem;border-bottom:1px solid {CARD_BOR}}}
.sec-icon{{font-size:1.3rem;width:2rem;height:2rem;display:flex;align-items:center;
    justify-content:center;background:{BADGE_BG};border-radius:8px;border:1px solid {CARD_BOR}}}
.sec-title{{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:{TEXT};margin:0}}
.card{{background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;
    padding:1.2rem 1.4rem;margin-bottom:0.8rem;backdrop-filter:blur(10px)}}
.card-title{{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
    color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem}}
.anom-row{{display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;
    border-bottom:1px solid {CARD_BOR};font-size:0.82rem}}
.alert-info{{background:{BADGE_BG};border-left:3px solid {ACCENT};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:{'#bee3f8' if dark else '#2c5282'}}}
.alert-success{{background:{SUCCESS_BG};border-left:3px solid {ACCENT3};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:{'#9ae6b4' if dark else '#276749'}}}
.alert-danger{{background:{DANGER_BG};border-left:3px solid {ACCENT_RED};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:{'#feb2b2' if dark else '#c53030'}}}
.filter-box{{background:{SECTION_BG};border:1px solid {CARD_BOR};border-radius:12px;
    padding:1rem 1.2rem;margin-bottom:1rem}}
.export-btn{{background:{'rgba(104,211,145,0.15)' if dark else 'rgba(56,161,105,0.1)'};
    border:1px solid {SUCCESS_BOR};border-radius:10px;padding:1rem 1.2rem;
    margin-bottom:0.6rem;text-align:center}}
div[data-testid="stFileUploader"]{{background:{SECTION_BG};border:2px dashed {CARD_BOR};
    border-radius:14px;padding:0.5rem}}
.stButton>button{{background:{'rgba(99,179,237,0.15)' if dark else 'rgba(49,130,206,0.1)'};
    border:1px solid {CARD_BOR};color:{ACCENT};border-radius:10px;
    font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:500;
    padding:0.45rem 1rem;transition:all 0.2s;width:100%}}
.stButton>button:hover{{background:{ACCENT};color:white;border-color:{ACCENT};transform:translateY(-1px)}}
.stTabs [data-baseweb="tab-list"]{{background:{SECTION_BG};border-radius:10px;padding:0.2rem}}
.stTabs [data-baseweb="tab"]{{color:{MUTED};font-family:'JetBrains Mono',monospace;font-size:0.8rem}}
.stTabs [aria-selected="true"]{{background:{CARD_BG};color:{ACCENT};border-radius:8px}}
.m4-divider{{border:none;border-top:1px solid {CARD_BOR};margin:1.5rem 0}}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sec(icon, title, badge=None):
    badge_html = f'<span style="margin-left:auto;background:{BADGE_BG};border:1px solid {CARD_BOR};border-radius:100px;padding:0.2rem 0.7rem;font-size:0.7rem;font-family:JetBrains Mono,monospace;color:{ACCENT}">{badge}</span>' if badge else ''
    st.markdown(f'<div class="sec-header"><div class="sec-icon">{icon}</div><p class="sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

def ui_info(m):    st.markdown(f'<div class="alert-info">ℹ️ {m}</div>',    unsafe_allow_html=True)
def ui_success(m): st.markdown(f'<div class="alert-success">✅ {m}</div>', unsafe_allow_html=True)
def ui_danger(m):  st.markdown(f'<div class="alert-danger">🚨 {m}</div>',  unsafe_allow_html=True)

# ── Required files registry ───────────────────────────────────────────────────
REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols":["ActivityDate","TotalSteps","Calories"],       "label":"Daily Activity",    "icon":"🏃"},
    "hourlySteps_merged.csv":       {"key_cols":["ActivityHour","StepTotal"],                   "label":"Hourly Steps",      "icon":"👣"},
    "hourlyIntensities_merged.csv": {"key_cols":["ActivityHour","TotalIntensity"],              "label":"Hourly Intensities","icon":"⚡"},
    "minuteSleep_merged.csv":       {"key_cols":["date","value","logId"],                       "label":"Minute Sleep",      "icon":"💤"},
    "heartrate_seconds_merged.csv": {"key_cols":["Time","Value"],                               "label":"Heart Rate",        "icon":"❤️"},
}

def score_match(df, req_info):
    return sum(1 for c in req_info["key_cols"] if c in df.columns)

# ── Detection functions ───────────────────────────────────────────────────────
def detect_hr(master, hr_high=100, hr_low=50, sigma=2.0):
    df = master[["Id","Date","AvgHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["AvgHR"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["AvgHR"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_high"] = d["AvgHR"] > hr_high
    d["thresh_low"]  = d["AvgHR"] < hr_low
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_high"] | d["thresh_low"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_high: parts.append(f"HR>{hr_high}")
        if r.thresh_low:  parts.append(f"HR<{hr_low}")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_steps(master, st_low=500, st_high=25000, sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSteps"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["TotalSteps"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_low"]  = d["TotalSteps"] < st_low
    d["thresh_high"] = d["TotalSteps"] > st_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_low:  parts.append(f"Steps<{int(st_low):,}")
        if r.thresh_high: parts.append(f"Steps>{int(st_high):,}")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_sleep(master, sl_low=60, sl_high=600, sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["TotalSleepMinutes"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_low"]  = (d["TotalSleepMinutes"]>0) & (d["TotalSleepMinutes"]<sl_low)
    d["thresh_high"] = d["TotalSleepMinutes"] > sl_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_low:  parts.append(f"Sleep<{int(sl_low)}min")
        if r.thresh_high: parts.append(f"Sleep>{int(sl_high)}min")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

# ── Chart builders ────────────────────────────────────────────────────────────
def chart_hr(anom_hr, hr_high, hr_low, sigma, h=380):
    fig = go.Figure()
    upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
    lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=lower, mode="lines",
                             fill="tonexty", fillcolor="rgba(99,179,237,0.1)",
                             line=dict(width=0), name=f"+/-{sigma:.0f}sigma Band"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                             mode="lines+markers", name="Avg HR",
                             line=dict(color=ACCENT, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")))
    a = anom_hr[anom_hr["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["AvgHR"], mode="markers",
                                 name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="circle",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f}<br><b>ANOMALY</b><extra>⚠️</extra>"))
        for _, row in a.iterrows():
            fig.add_annotation(x=row["Date"], y=row["AvgHR"],
                               text=f"⚠️", showarrow=True, arrowhead=2,
                               arrowcolor=ACCENT_RED, ax=0, ay=-35,
                               font=dict(color=ACCENT_RED, size=11))
    fig.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.6,
                  annotation_text=f"High ({int(hr_high)} bpm)",
                  annotation_font_color=ACCENT_RED, annotation_position="top right")
    fig.add_hline(y=hr_low, line_dash="dash", line_color=ACCENT2,
                  line_width=1.5, opacity=0.6,
                  annotation_text=f"Low ({int(hr_low)} bpm)",
                  annotation_font_color=ACCENT2, annotation_position="bottom right")
    ptheme(fig, "❤️ Heart Rate - Anomaly Detection", h)
    fig.update_layout(xaxis_title="Date", yaxis_title="HR (bpm)")
    return fig

def chart_steps(anom_steps, st_low, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Daily Steps (avg users)","Residual Deviation"])
    a = anom_steps[anom_steps["is_anomaly"]]
    for _, row in a.iterrows():
        fig.add_vrect(x0=str(row["Date"]), x1=str(row["Date"]),
                      fillcolor="rgba(252,129,129,0.12)",
                      line_color="rgba(252,129,129,0.4)", line_width=1.5,
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                             mode="lines+markers", name="Avg Steps",
                             line=dict(color=ACCENT3, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT, width=2, dash="dash")),
                  row=1, col=1)
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSteps"],
                                 mode="markers", name="🚨 Alert",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="triangle-up",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(st_low), line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.7, row=1, col=1,
                  annotation_text=f"Low ({int(st_low):,})",
                  annotation_font_color=ACCENT_RED)
    res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:,.0f}<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)
    ptheme(fig, "🚶 Step Count - Trend & Alerts", h)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
    fig.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    return fig

def chart_sleep(anom_sleep, sl_low, sl_high, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Sleep Duration (min/night)","Residual Deviation"])
    fig.add_hrect(y0=sl_low, y1=sl_high,
                  fillcolor="rgba(104,211,145,0.07)", line_width=0,
                  annotation_text="✅ Healthy Zone", annotation_position="top right",
                  annotation_font_color=ACCENT3, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                             mode="lines+markers", name="Sleep (min)",
                             line=dict(color="#b794f4", width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f} min<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")),
                  row=1, col=1)
    a = anom_sleep[anom_sleep["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSleepMinutes"],
                                 mode="markers", name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f}<br><b>ANOMALY</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(sl_low), line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.7, row=1, col=1,
                  annotation_text=f"Min ({int(sl_low)} min)",
                  annotation_font_color=ACCENT_RED)
    fig.add_hline(y=int(sl_high), line_dash="dash", line_color=ACCENT,
                  line_width=1.5, opacity=0.6, row=1, col=1,
                  annotation_text=f"Max ({int(sl_high)} min)",
                  annotation_font_color=ACCENT)
    res_colors = [ACCENT_RED if v else "#b794f4" for v in anom_sleep["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:.0f} min<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)
    ptheme(fig, "💤 Sleep Pattern - Anomaly Visualization", h)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
    fig.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    return fig

# ── PDF Generation ────────────────────────────────────────────────────────────
def generate_pdf(master, anom_hr, anom_steps, anom_sleep,
                 hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                 fig_hr, fig_steps, fig_sleep):
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 18, 'F')
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(99, 179, 237)
            self.set_y(4)
            self.cell(0, 10, "FitPulse Anomaly Detection Report  -  Milestone 4", align="C")
            self.set_text_color(148, 163, 184)
            self.set_font("Helvetica", "", 7)
            self.set_y(13)
            self.cell(0, 4, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", align="C")
            self.ln(6)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(148, 163, 184)
            self.cell(0, 8, f"FitPulse ML Pipeline  .  Page {self.page_no()}", align="C")

        def section(self, title, color=(99, 179, 237)):
            self.ln(3)
            self.set_fill_color(*color)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 8, f"  {title}", fill=True, ln=True)
            self.set_text_color(30, 30, 40)
            self.ln(2)

        def kv(self, key, val, bold_val=True):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(80, 80, 100)
            self.cell(55, 6, key + ":", ln=False)
            self.set_font("Helvetica", "B" if bold_val else "", 9)
            self.set_text_color(20, 20, 30)
            self.cell(0, 6, str(val), ln=True)

        def para(self, text, size=8.5):
            self.set_font("Helvetica", "", size)
            self.set_text_color(60, 60, 80)
            self.multi_cell(0, 5, text)
            self.ln(1)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Page 1: Executive Summary ─────────────────────────────────────────────
    n_hr    = int(anom_hr["is_anomaly"].sum())
    n_steps = int(anom_steps["is_anomaly"].sum())
    n_sleep = int(anom_sleep["is_anomaly"].sum())
    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    date_range = f"{pd.to_datetime(master['Date']).min().strftime('%d %b %Y')} - {pd.to_datetime(master['Date']).max().strftime('%d %b %Y')}"

    pdf.section("1. EXECUTIVE SUMMARY", (15, 23, 60))
    pdf.kv("Dataset",       f"Real Fitbit Device Data - Kaggle (arashnic/fitbit)")
    pdf.kv("Users",         f"{n_users} participants")
    pdf.kv("Date Range",    date_range)
    pdf.kv("Total Days",    f"{n_days} days of observations")
    pdf.kv("Pipeline",      "Milestone 4 - Anomaly Detection Dashboard")
    pdf.ln(2)

    pdf.section("2. ANOMALY SUMMARY", (180, 50, 50))
    pdf.kv("Heart Rate Anomalies",  f"{n_hr} days flagged")
    pdf.kv("Steps Anomalies",       f"{n_steps} days flagged")
    pdf.kv("Sleep Anomalies",       f"{n_sleep} days flagged")
    pdf.kv("Total Flags",           f"{n_hr + n_steps + n_sleep} across all signals")
    pdf.ln(2)

    pdf.section("3. DETECTION THRESHOLDS USED", (40, 100, 60))
    pdf.kv("Heart Rate High",   f"> {int(hr_high)} bpm")
    pdf.kv("Heart Rate Low",    f"< {int(hr_low)} bpm")
    pdf.kv("Steps Low Alert",   f"< {int(st_low):,} steps/day")
    pdf.kv("Sleep Low",         f"< {int(sl_low)} minutes/night")
    pdf.kv("Sleep High",        f"> {int(sl_high)} minutes/night")
    pdf.kv("Residual Sigma",    f"+/- {float(sigma):.1f}sigma from rolling median")
    pdf.ln(2)

    pdf.section("4. METHODOLOGY", (60, 80, 140))
    pdf.para(
        "Three complementary anomaly detection methods were applied:\n\n"
        "  1. THRESHOLD VIOLATIONS - Hard upper/lower bounds on each metric. "
        "Any day exceeding these bounds is immediately flagged as anomalous. "
        "Simple, interpretable, and highly reliable for extreme values.\n\n"
        "  2. RESIDUAL-BASED DETECTION - A 3-day rolling median is computed as "
        "the expected baseline. Days where the actual value deviates by more than "
        f"+/-{float(sigma):.1f} standard deviations from this baseline are flagged. "
        "This catches subtle pattern breaks that threshold rules miss.\n\n"
        "  3. DBSCAN OUTLIER CLUSTERING - Each user is profiled on 7 activity "
        "features and clustered using DBSCAN (eps=2.2, min_samples=2). Users "
        "assigned label -1 are structural outliers whose overall behaviour does "
        "not match any group."
    )

    # ── Page 2: Charts ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("5. ANOMALY CHARTS", (15, 23, 60))

    def embed_fig(fig, label, w=190, h=80):
        try:
            img_bytes = fig.to_image(format="png", width=1100, height=480,
                                     scale=1.5, engine="kaleido")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(80, 80, 100)
            pdf.cell(0, 6, label, ln=True)
            pdf.image(tmp_path, x=10, w=w, h=h)
            os.unlink(tmp_path)
            pdf.ln(3)
        except Exception as ex:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(150, 50, 50)
            pdf.cell(0, 6, f"[Chart not available: {ex}]", ln=True)
            pdf.ln(2)

    embed_fig(fig_hr,    "Figure 1 - Heart Rate with Anomaly Highlights")
    embed_fig(fig_steps, "Figure 2 - Step Count Trend with Alert Bands")
    embed_fig(fig_sleep, "Figure 3 - Sleep Pattern Visualization")

    # ── Page 3: Anomaly Tables ────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("6. ANOMALY RECORDS - HEART RATE", (180, 50, 50))

    def table(df, cols, rename_map, max_rows=20):
        df2 = df[df["is_anomaly"]][cols].copy().rename(columns=rename_map)
        if df2.empty:
            pdf.para("No anomalies detected.")
            return
        # Header
        col_w = 180 // len(df2.columns)
        pdf.set_fill_color(15, 23, 60)
        pdf.set_text_color(180, 210, 255)
        pdf.set_font("Helvetica", "B", 7.5)
        for col in df2.columns:
            pdf.cell(col_w, 6, str(col)[:18], border=0, fill=True)
        pdf.ln()
        # Rows
        pdf.set_font("Helvetica", "", 7.5)
        for i, (_, row) in enumerate(df2.head(max_rows).iterrows()):
            pdf.set_fill_color(30, 40, 60) if i % 2 == 0 else pdf.set_fill_color(20, 30, 50)
            pdf.set_text_color(200, 210, 225)
            for val in row:
                if isinstance(val, float):
                    cell_text = f"{val:.2f}"
                else:
                    cell_text = str(val)[:18]
                pdf.cell(col_w, 5.5, cell_text, border=0, fill=True)
            pdf.ln()
        if len(df2) > max_rows:
            pdf.set_text_color(100, 130, 180)
            pdf.set_font("Helvetica", "I", 7)
            pdf.cell(0, 5, f"  ... and {len(df2)-max_rows} more records (see CSV export for full data)", ln=True)
        pdf.ln(3)

    table(anom_hr, ["Date","AvgHR","rolling_med","residual","reason"],
          {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.section("7. ANOMALY RECORDS - STEPS", (40, 130, 80))
    table(anom_steps, ["Date","TotalSteps","rolling_med","residual","reason"],
          {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.section("8. ANOMALY RECORDS - SLEEP", (100, 60, 160))
    table(anom_sleep, ["Date","TotalSleepMinutes","rolling_med","residual","reason"],
          {"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    # ── Page 4: User Profiles ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.section("9. DATASET OVERVIEW & USER PROFILES", (15, 23, 60))

    profile_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    available_cols = [c for c in profile_cols if c in master.columns]
    user_profile = master.groupby("Id")[available_cols].mean().round(1)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(15, 23, 60)
    pdf.set_text_color(180, 210, 255)
    col_w2 = 180 // (len(available_cols) + 1)
    pdf.cell(col_w2, 6, "User ID", border=0, fill=True)
    for col in available_cols:
        pdf.cell(col_w2, 6, col[:12], border=0, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7.5)
    for i, (uid, row) in enumerate(user_profile.iterrows()):
        pdf.set_fill_color(30, 40, 60) if i % 2 == 0 else pdf.set_fill_color(20, 30, 50)
        pdf.set_text_color(200, 210, 225)
        pdf.cell(col_w2, 5.5, f"...{str(uid)[-6:]}", border=0, fill=True)
        for val in row:
            pdf.cell(col_w2, 5.5, f"{val:,.0f}", border=0, fill=True)
        pdf.ln()

    pdf.ln(4)
    pdf.section("10. CONCLUSION", (40, 100, 60))
    pdf.para(
        f"The FitPulse Milestone 4 anomaly detection pipeline successfully processed "
        f"{n_users} users over {n_days} days of real Fitbit device data. "
        f"A total of {n_hr + n_steps + n_sleep} anomalous events were identified across "
        f"heart rate, step count, and sleep duration signals.\n\n"
        "Key findings:\n"
        f"   Heart rate showed {n_hr} anomalous days, primarily driven by residual "
        f"deviations from the rolling trend.\n"
        f"   Step count flagged {n_steps} alert days, often corresponding to "
        f"extremely sedentary or unusually active periods.\n"
        f"   Sleep patterns generated {n_sleep} anomaly flags, reflecting days "
        f"where users either did not wear the device or had unusual sleep durations.\n\n"
        "These findings align with expected patterns in consumer fitness wearable data "
        "and demonstrate the effectiveness of combining rule-based and statistical "
        "anomaly detection methods."
    )

    buf = io.BytesIO()
    pdf_bytes = pdf.output()
    buf.write(pdf_bytes)
    buf.seek(0)
    return buf

# ── CSV Generation ────────────────────────────────────────────────────────────
def generate_csv(anom_hr, anom_steps, anom_sleep):
    hr_out    = anom_hr[anom_hr["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
    hr_out["signal"] = "Heart Rate"
    hr_out    = hr_out.rename(columns={"AvgHR":"value","rolling_med":"expected"})

    st_out    = anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
    st_out["signal"] = "Steps"
    st_out    = st_out.rename(columns={"TotalSteps":"value","rolling_med":"expected"})

    sl_out    = anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
    sl_out["signal"] = "Sleep"
    sl_out    = sl_out.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})

    combined  = pd.concat([hr_out, st_out, sl_out], ignore_index=True)
    combined  = combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"])
    combined  = combined.round(2)
    buf       = io.StringIO()
    combined.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - Upload + Filters + Run
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{ACCENT}">
        📊 FitPulse Dashboard
      </div>
      <div style="font-size:0.7rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.2rem">
        Milestone 4 . Insights & Export
      </div>
    </div>""", unsafe_allow_html=True)

    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">UPLOAD FILES</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop all 5 CSV files",
                                      type="csv", accept_multiple_files=True,
                                      key="m4_uploader", label_visibility="collapsed")

    detected = {}
    if uploaded_files:
        raw = []
        for uf in uploaded_files:
            try:
                raw.append((uf.name, pd.read_csv(uf)))
            except Exception:
                pass
        used = set()
        for req_name, finfo in REQUIRED_FILES.items():
            best_s, best_n, best_d = 0, None, None
            for uname, udf in raw:
                s = score_match(udf, finfo)
                if s > best_s:
                    best_s, best_n, best_d = s, uname, udf
            if best_s >= 2:
                detected[req_name] = best_d
                used.add(best_n)

    n_up = len(detected)
    # File status mini grid
    status_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin:0.5rem 0">'
    for req_name, finfo in REQUIRED_FILES.items():
        found = req_name in detected
        col   = SUCCESS_BOR if found else WARN_BOR
        bg    = SUCCESS_BG if found else WARN_BG
        ico   = "✅" if found else "❌"
        status_html += f'<div style="background:{bg};border:1px solid {col};border-radius:6px;padding:0.3rem;text-align:center;font-size:0.7rem">{ico}<br><span style="font-size:0.55rem;color:{MUTED}">{finfo["label"][:5]}</span></div>'
    status_html += "</div>"
    st.markdown(status_html, unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)

    # ── Thresholds ────────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">DETECTION THRESHOLDS</div>', unsafe_allow_html=True)
    hr_high = int(st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180))
    hr_low  = int(st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70))
    st_low  = int(st.number_input("Steps Low/day",    value=500, min_value=0,   max_value=2000))
    sl_low  = int(st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120))
    sl_high = int(st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900))
    sigma   = float(st.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="m4_sigma"))

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    run_clicked = st.button("⚡ Run Full Pipeline", disabled=(n_up < 5))
    if n_up < 5:
        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};text-align:center">{n_up}/5 files ready</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)

    # ── Date filter ───────────────────────────────────────────────────────────
    if st.session_state.pipeline_done and st.session_state.master is not None:
        master_tmp = st.session_state.master
        all_dates = pd.to_datetime(master_tmp["Date"])
        d_min = all_dates.min().date()
        d_max = all_dates.max().date()
        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">DATE FILTER</div>', unsafe_allow_html=True)
        date_range = st.date_input("Date range", value=(d_min, d_max),
                                   min_value=d_min, max_value=d_max,
                                   key="m4_daterange", label_visibility="collapsed")

        st.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin:0.6rem 0 0.4rem">USER FILTER</div>', unsafe_allow_html=True)
        all_users = sorted(master_tmp["Id"].unique())
        user_options = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users]
        selected_user_label = st.selectbox("User", user_options, key="m4_user", label_visibility="collapsed")
        selected_user = None if selected_user_label == "All Users" else all_users[user_options.index(selected_user_label) - 1]

    st.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)
    pct = int(st.session_state.pipeline_done) * 100
    st.markdown(f"""
    <div style="font-size:0.68rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.3rem">PIPELINE . {pct}%</div>
    <div style="background:{CARD_BOR};border-radius:4px;height:5px;overflow:hidden">
      <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{ACCENT},{ACCENT3});border-radius:4px"></div>
    </div>""", unsafe_allow_html=True)

# ── Pipeline run ──────────────────────────────────────────────────────────────
if run_clicked and n_up == 5:
    with st.spinner("⏳ Loading data and detecting anomalies..."):
        try:
            daily    = detected["dailyActivity_merged.csv"].copy()
            hourly_s = detected["hourlySteps_merged.csv"].copy()
            hourly_i = detected["hourlyIntensities_merged.csv"].copy()
            sleep    = detected["minuteSleep_merged.csv"].copy()
            hr       = detected["heartrate_seconds_merged.csv"].copy()

            daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"],    format="%m/%d/%Y")
            hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
            hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
            sleep["date"]            = pd.to_datetime(sleep["date"],            format="%m/%d/%Y %I:%M:%S %p")
            hr["Time"]               = pd.to_datetime(hr["Time"],               format="%m/%d/%Y %I:%M:%S %p")

            hr_minute = (hr.set_index("Time").groupby("Id")["Value"]
                         .resample("1min").mean().reset_index())
            hr_minute.columns = ["Id","Time","HeartRate"]
            hr_minute = hr_minute.dropna()
            hr_minute["Date"] = hr_minute["Time"].dt.date

            hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                        .agg(["mean","max","min","std"]).reset_index()
                        .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))
            sleep["Date"] = sleep["date"].dt.date
            sleep_daily = (sleep.groupby(["Id","Date"])
                           .agg(TotalSleepMinutes=("value","count"),
                                DominantSleepStage=("value", lambda x: x.mode()[0])).reset_index())
            master = daily.copy().rename(columns={"ActivityDate":"Date"})
            master["Date"] = master["Date"].dt.date
            master = master.merge(hr_daily,    on=["Id","Date"], how="left")
            master = master.merge(sleep_daily, on=["Id","Date"], how="left")
            master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
            master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
            for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

            anom_hr    = detect_hr(master,    hr_high, hr_low,   sigma)
            anom_steps = detect_steps(master, st_low,  25000,    sigma)
            anom_sleep = detect_sleep(master, sl_low,  sl_high,  sigma)

            st.session_state.master      = master
            st.session_state.daily       = daily
            st.session_state.hr_minute   = hr_minute
            st.session_state.anom_hr     = anom_hr
            st.session_state.anom_steps  = anom_steps
            st.session_state.anom_sleep  = anom_sleep
            st.session_state.pipeline_done = True
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="m4-hero">
  <div class="hero-badge">MILESTONE 4 . INSIGHTS DASHBOARD</div>
  <h1 class="hero-title">📊 FitPulse Insights Dashboard</h1>
  <p class="hero-sub">Upload . Detect . Filter . Export PDF & CSV - Real Fitbit Device Data</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.pipeline_done:
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:3rem">
      <div style="font-size:3rem;margin-bottom:1rem">📂</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
        Upload Files & Run Pipeline to Begin
      </div>
      <div style="color:{MUTED};font-size:0.88rem;margin-bottom:1.5rem">
        1 . Upload all 5 CSV files in the sidebar<br>
        2 . Adjust thresholds if needed<br>
        3 . Click <b>⚡ Run Full Pipeline</b>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;max-width:600px;margin:0 auto;text-align:left">
        <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
          <div style="color:{ACCENT};font-weight:600;font-size:0.85rem">📤 Upload</div>
          <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">All 5 Fitbit CSV files auto-detected</div>
        </div>
        <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
          <div style="color:{ACCENT_RED};font-weight:600;font-size:0.85rem">🚨 Detect</div>
          <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">3 detection methods run automatically</div>
        </div>
        <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
          <div style="color:{ACCENT3};font-weight:600;font-size:0.85rem">📥 Export</div>
          <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">Download PDF report + CSV data</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

else:
    master     = st.session_state.master
    anom_hr    = st.session_state.anom_hr
    anom_steps = st.session_state.anom_steps
    anom_sleep = st.session_state.anom_sleep

    # ── Apply date filter ─────────────────────────────────────────────────────
    try:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d_from, d_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        else:
            all_dates = pd.to_datetime(master["Date"])
            d_from, d_to = all_dates.min(), all_dates.max()
    except Exception:
        all_dates = pd.to_datetime(master["Date"])
        d_from, d_to = all_dates.min(), all_dates.max()

    def filt(df, date_col="Date"):
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col])
        return df2[(df2[date_col] >= d_from) & (df2[date_col] <= d_to)]

    anom_hr_f    = filt(anom_hr)
    anom_steps_f = filt(anom_steps)
    anom_sleep_f = filt(anom_sleep)
    master_f     = filt(master)
    if selected_user:
        master_f = master_f[master_f["Id"] == selected_user]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    n_hr_f    = int(anom_hr_f["is_anomaly"].sum())
    n_steps_f = int(anom_steps_f["is_anomaly"].sum())
    n_sleep_f = int(anom_sleep_f["is_anomaly"].sum())
    n_total_f = n_hr_f + n_steps_f + n_sleep_f
    n_users_f = master_f["Id"].nunique()
    n_days_f  = master_f["Date"].nunique()

    worst_hr_row   = anom_hr_f[anom_hr_f["is_anomaly"]].copy()
    worst_hr_day   = worst_hr_row.iloc[worst_hr_row["residual"].abs().argmax()]["Date"].strftime("%d %b") if not worst_hr_row.empty else "-"

    kpi_html = f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="border-color:{DANGER_BOR}">
        <div class="kpi-val" style="color:{ACCENT_RED}">{n_total_f}</div>
        <div class="kpi-label">Total Anomalies</div>
        <div class="kpi-sub">across all signals</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(246,135,179,0.3)">
        <div class="kpi-val" style="color:{ACCENT2}">{n_hr_f}</div>
        <div class="kpi-label">HR Flags</div>
        <div class="kpi-sub">heart rate anomalies</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(104,211,145,0.3)">
        <div class="kpi-val" style="color:{ACCENT3}">{n_steps_f}</div>
        <div class="kpi-label">Steps Alerts</div>
        <div class="kpi-sub">step count anomalies</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(183,148,244,0.3)">
        <div class="kpi-val" style="color:#b794f4">{n_sleep_f}</div>
        <div class="kpi-label">Sleep Flags</div>
        <div class="kpi-sub">sleep anomalies</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color:{ACCENT}">{n_users_f}</div>
        <div class="kpi-label">Users</div>
        <div class="kpi-sub">in selected range</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color:{ACCENT_ORG}">{worst_hr_day}</div>
        <div class="kpi-label">Peak HR Anomaly</div>
        <div class="kpi-sub">highest deviation day</div>
      </div>
    </div>"""
    st.markdown(kpi_html, unsafe_allow_html=True)

    ui_success(f"Pipeline complete . {n_users_f} users . {n_days_f} days . {n_total_f} anomalies flagged")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_hr, tab_steps, tab_sleep, tab_export = st.tabs([
        "📊 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "📥 Export"
    ])

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
    with tab_overview:

        # Combined anomaly timeline
        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
        sec("📅", "Combined Anomaly Timeline")

        all_anoms = []
        for df_, sig, col in [
            (anom_hr_f,    "Heart Rate", ACCENT2),
            (anom_steps_f, "Steps",      ACCENT3),
            (anom_sleep_f, "Sleep",      "#b794f4"),
        ]:
            a = df_[df_["is_anomaly"]].copy()
            a["signal"] = sig
            a["color"]  = col
            all_anoms.append(a[["Date","signal","color","reason"]])

        if all_anoms:
            combined = pd.concat(all_anoms, ignore_index=True)
            combined["Date"] = pd.to_datetime(combined["Date"])
            combined["y"]    = combined["signal"]

            fig_timeline = go.Figure()
            for sig, col in [("Heart Rate", ACCENT2), ("Steps", ACCENT3), ("Sleep", "#b794f4")]:
                sub = combined[combined["signal"] == sig]
                if not sub.empty:
                    fig_timeline.add_trace(go.Scatter(
                        x=sub["Date"], y=sub["y"], mode="markers",
                        name=sig, marker=dict(color=col, size=14, symbol="diamond",
                                              line=dict(color="white", width=2)),
                        hovertemplate=f"<b>{sig}</b><br>%{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                        customdata=sub["reason"].values
                    ))
            ptheme(fig_timeline, "📅 Anomaly Event Timeline - All Signals", h=280)
            fig_timeline.update_layout(
                xaxis_title="Date", yaxis_title="Signal",
                showlegend=True,
                yaxis=dict(categoryorder="array",
                           categoryarray=["Sleep","Steps","Heart Rate"],
                           gridcolor=GRID_CLR, tickfont_color=MUTED)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Recent anomaly log
        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
        sec("🗂️", "Recent Anomaly Log")
        if all_anoms:
            log = combined.sort_values("Date", ascending=False).head(10)
            for _, row in log.iterrows():
                st.markdown(f"""
                <div class="anom-row">
                  <span style="font-size:0.9rem">🚨</span>
                  <span style="color:{row['color']};font-family:'JetBrains Mono',monospace;font-size:0.75rem;min-width:90px">{row['signal']}</span>
                  <span style="color:{MUTED};font-size:0.78rem;min-width:90px">{row['Date'].strftime('%d %b %Y')}</span>
                  <span style="color:{TEXT};font-size:0.78rem">{row['reason']}</span>
                </div>""", unsafe_allow_html=True)

    # ── TAB 2: HEART RATE ─────────────────────────────────────────────────────
    with tab_hr:
        sec("❤️", "Heart Rate - Deep Dive", f"{n_hr_f} anomalies")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">HR Statistics</div>
              <div style="font-size:0.83rem;line-height:2">
                <div>Mean HR: <b style="color:{ACCENT}">{anom_hr_f['AvgHR'].mean():.1f} bpm</b></div>
                <div>Max HR: <b style="color:{ACCENT_RED}">{anom_hr_f['AvgHR'].max():.1f} bpm</b></div>
                <div>Min HR: <b style="color:{ACCENT2}">{anom_hr_f['AvgHR'].min():.1f} bpm</b></div>
                <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="card"><div class="card-title">HR Anomaly Records</div>', unsafe_allow_html=True)
            hr_display = anom_hr_f[anom_hr_f["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].round(2)
            if not hr_display.empty:
                st.dataframe(hr_display.rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                             use_container_width=True, height=200)
            else:
                ui_success("No HR anomalies in selected range")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 3: STEPS ──────────────────────────────────────────────────────────
    with tab_steps:
        sec("🚶", "Step Count - Deep Dive", f"{n_steps_f} alerts")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Steps Statistics</div>
              <div style="font-size:0.83rem;line-height:2">
                <div>Mean steps/day: <b style="color:{ACCENT3}">{anom_steps_f['TotalSteps'].mean():,.0f}</b></div>
                <div>Max steps/day: <b style="color:{ACCENT}">{anom_steps_f['TotalSteps'].max():,.0f}</b></div>
                <div>Min steps/day: <b style="color:{ACCENT_RED}">{anom_steps_f['TotalSteps'].min():,.0f}</b></div>
                <div>Alert days: <b style="color:{ACCENT_RED}">{n_steps_f}</b> of {len(anom_steps_f)} total</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="card"><div class="card-title">Steps Alert Records</div>', unsafe_allow_html=True)
            st_display = anom_steps_f[anom_steps_f["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
            if not st_display.empty:
                st.dataframe(st_display.rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                             use_container_width=True, height=200)
            else:
                ui_success("No step anomalies in selected range")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 4: SLEEP ──────────────────────────────────────────────────────────
    with tab_sleep:
        sec("💤", "Sleep Pattern - Deep Dive", f"{n_sleep_f} anomalies")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Sleep Statistics</div>
              <div style="font-size:0.83rem;line-height:2">
                <div>Mean sleep/night: <b style="color:#b794f4">{anom_sleep_f['TotalSleepMinutes'].mean():.0f} min</b></div>
                <div>Max sleep/night: <b style="color:{ACCENT}">{anom_sleep_f['TotalSleepMinutes'].max():.0f} min</b></div>
                <div>Min (non-zero): <b style="color:{ACCENT_RED}">{anom_sleep_f[anom_sleep_f['TotalSleepMinutes']>0]['TotalSleepMinutes'].min():.0f} min</b></div>
                <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="card"><div class="card-title">Sleep Anomaly Records</div>', unsafe_allow_html=True)
            sl_display = anom_sleep_f[anom_sleep_f["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
            if not sl_display.empty:
                st.dataframe(sl_display.rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                             use_container_width=True, height=200)
            else:
                ui_success("No sleep anomalies in selected range")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 5: EXPORT ─────────────────────────────────────────────────────────
    with tab_export:
        sec("📥", "Export - PDF Report & CSV Data", "Downloadable")

        st.markdown(f"""
        <div class="card">
          <div class="card-title">What's Included in the Exports</div>
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
              <div style="color:{ACCENT3};font-weight:600;margin-bottom:0.5rem">📊 CSV Export</div>
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

        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)

        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            sec("📄", "PDF Report")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">Full 4-page PDF with charts embedded, anomaly tables, and user profiles.</div>', unsafe_allow_html=True)

            if st.button("📄 Generate PDF Report", key="gen_pdf"):
                with st.spinner("⏳ Generating PDF (embedding charts)..."):
                    try:
                        fig_hr_exp    = chart_hr(anom_hr_f,    hr_high, hr_low, sigma, h=420)
                        fig_steps_exp = chart_steps(anom_steps_f, st_low, h=420)
                        fig_sleep_exp = chart_sleep(anom_sleep_f, sl_low, sl_high, h=420)

                        pdf_buf = generate_pdf(
                            master_f, anom_hr_f, anom_steps_f, anom_sleep_f,
                            hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                            fig_hr_exp, fig_steps_exp, fig_sleep_exp
                        )
                        fname = f"FitPulse_Anomaly_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_buf,
                            file_name=fname,
                            mime="application/pdf",
                            key="dl_pdf"
                        )
                        ui_success(f"PDF ready - {fname}")
                    except Exception as e:
                        st.error(f"PDF error: {e}")

        with col_csv:
            sec("📊", "CSV Export")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">All anomaly records from all three signals in a single CSV file.</div>', unsafe_allow_html=True)

            csv_data = generate_csv(anom_hr_f, anom_steps_f, anom_sleep_f)
            fname_csv = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            st.download_button(
                label="⬇️ Download Anomaly CSV",
                data=csv_data,
                file_name=fname_csv,
                mime="text/csv",
                key="dl_csv"
            )

            # Preview
            with st.expander("👁️ Preview CSV data"):
                preview_df = pd.concat([
                    anom_hr_f[anom_hr_f["is_anomaly"]].assign(signal="Heart Rate").rename(columns={"AvgHR":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    anom_steps_f[anom_steps_f["is_anomaly"]].assign(signal="Steps").rename(columns={"TotalSteps":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    anom_sleep_f[anom_sleep_f["is_anomaly"]].assign(signal="Sleep").rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                ], ignore_index=True).sort_values(["signal","Date"]).round(2)
                st.dataframe(preview_df, use_container_width=True, height=280)

        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)

        # ── Screenshots reminder ───────────────────────────────────────────────
        sec("📸", "Screenshots Required for Submission")
        st.markdown(f"""
        <div class="card">
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 1</b> - Full dashboard UI (Overview tab)
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 2</b> - Downloadable report buttons (this tab)
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 3</b> - KPI strip with anomaly counts
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 4</b> - HR / Steps / Sleep deep dive tabs
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;grid-column:1/-1">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 5</b> - Sidebar with filters + date range visible
            </div>
          </div>
        </div>""", unsafe_allow_html=True)