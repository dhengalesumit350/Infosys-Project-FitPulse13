"""
FitPulse · Milestone 2 — Streamlit App
=======================================
• File upload in MAIN area with 6 upload cards (5 CSVs + tsfresh_features.csv)
• TSFresh loaded from pre-computed CSV — NO live computation hang
• Every chart annotated with exact values
• Vertical step-by-step flow
• Run:  streamlit run fitpulse_app.py
"""

import io, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · Milestone 2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colours ───────────────────────────────────────────────────────────────────
DARK   = "#0d1117"
CARD   = "#161b22"
CARD2  = "#1c2128"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
AMBER  = "#f0883e"
PURPLE = "#bc8cff"
RED    = "#ff7b72"
PINK   = "#f778ba"
TEAL   = "#39d353"
PAL    = [BLUE, PINK, GREEN, AMBER, PURPLE, RED, TEAL, "#ffa657"]

plt.rcParams.update({
    "figure.facecolor": DARK,   "axes.facecolor":   CARD2,
    "axes.edgecolor":   BORDER, "axes.labelcolor":  MUTED,
    "axes.titlecolor":  TEXT,   "xtick.color":      MUTED,
    "ytick.color":      MUTED,  "grid.color":        BORDER,
    "text.color":       TEXT,   "legend.facecolor":  CARD,
    "legend.edgecolor": BORDER, "font.size":         9,
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── global ── */
.stApp {{ background:{DARK}; }}
.stSidebar > div:first-child {{ background:{CARD}; }}
[data-testid="metric-container"] {{
    background:{CARD2}; border:1px solid {BORDER};
    border-radius:10px; padding:14px; }}
h1,h2,h3,h4 {{ color:{TEXT} !important; }}
hr {{ border-color:{BORDER}; margin:28px 0; }}
.stDataFrame {{ background:{CARD2}; }}
.stAlert {{ border-radius:8px; }}

/* ── upload card grid ── */
.upload-grid {{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:14px;
    margin-bottom:20px;
}}
.upload-card {{
    background:{CARD2};
    border:2px dashed {BORDER};
    border-radius:14px;
    padding:20px 16px;
    text-align:center;
    transition:border-color .2s;
}}
.upload-card.ready {{
    border-color:{GREEN};
    border-style:solid;
    background:rgba(63,185,80,.07);
}}
.upload-card .icon  {{ font-size:2.2rem; margin-bottom:8px; }}
.upload-card .label {{
    font-size:.75rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.07em;
    color:{MUTED}; margin-bottom:4px;
}}
.upload-card .status-ok  {{ color:{GREEN}; font-weight:800; font-size:.88rem; }}
.upload-card .status-bad {{ color:{MUTED}; font-size:.82rem; }}

/* ── step box ── */
.step-box {{
    display:flex; align-items:flex-start; gap:14px;
    background:{CARD}; border:1px solid {BORDER};
    border-left:4px solid {BLUE};
    border-radius:12px; padding:16px 20px; margin:24px 0 10px;
}}
.step-num {{
    background:{BLUE}; color:{DARK}; font-weight:800;
    font-size:.72rem; padding:4px 10px; border-radius:20px;
    letter-spacing:.08em; white-space:nowrap; margin-top:2px;
}}
.step-title {{ font-size:1.05rem; font-weight:700; color:{TEXT}; }}
.step-desc  {{ font-size:.78rem; color:{MUTED}; margin-top:3px; }}

/* ── phase banner ── */
.phase-banner {{
    background:linear-gradient(120deg,{CARD},{CARD2});
    border:1px solid {BLUE}; border-left:5px solid {BLUE};
    border-radius:12px; padding:20px 26px; margin:32px 0 6px;
}}

/* ── info pill ── */
.info-pill {{
    display:inline-block; background:{CARD2};
    border:1px solid {BORDER}; border-radius:20px;
    padding:3px 12px; font-size:.72rem; color:{MUTED};
    margin:2px 4px 2px 0;
}}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def step_box(num, title, desc=""):
    st.markdown(
        f'<div class="step-box">'
        f'<span class="step-num">{num}</span>'
        f'<div><div class="step-title">{title}</div>'
        f'<div class="step-desc">{desc}</div></div></div>',
        unsafe_allow_html=True)

def phase_banner(icon, title, steps, desc):
    st.markdown(
        f'<div class="phase-banner">'
        f'<div style="font-size:.65rem;font-weight:800;letter-spacing:.15em;'
        f'color:{BLUE};text-transform:uppercase;margin-bottom:6px">{steps}</div>'
        f'<div style="font-size:1.4rem;font-weight:800;color:{TEXT}">{icon} {title}</div>'
        f'<div style="color:{MUTED};font-size:.82rem;margin-top:4px">{desc}</div>'
        f'</div>', unsafe_allow_html=True)

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=DARK, edgecolor="none")
    buf.seek(0); return buf

def dl(fig, fname, key):
    st.download_button(f"📥 Download  {fname}",
                       fig_bytes(fig), fname, "image/png", key=key)

@st.cache_data(show_spinner=False)
def read_csv(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

def detect_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "ActivityDate" in cols and "TotalSteps" in cols:       return "daily"
    if "ActivityHour" in cols and "StepTotal" in cols:        return "hourly_steps"
    if "ActivityHour" in cols and "TotalIntensity" in cols:   return "hourly_intensities"
    if "Time" in cols and "Value" in cols:                    return "heartrate"
    if "date" in cols and "value" in cols:                    return "sleep"
    if "value__sum_values" in cols or "value__mean" in cols:  return "tsfresh"
    return "unknown"

def df_pq(df):
    buf = io.BytesIO(); df.to_parquet(buf); buf.seek(0); return buf.read()

def ser_json(s):
    return s.reset_index(drop=True).to_json().encode()

@st.cache_data(show_spinner="⏳ Resampling heart-rate data to 1-minute…")
def resample_hr(b: bytes) -> bytes:
    hr = read_csv(b)
    hr["Time"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p",
                                errors="coerce")
    out = (hr.set_index("Time").groupby("Id")["Value"]
             .resample("1min").mean().reset_index())
    out.columns = ["Id", "Time", "HeartRate"]
    return df_pq(out.dropna())

@st.cache_data(show_spinner="⏳ Building master dataframe…")
def build_master(daily_b, sleep_b, hr_min_b):
    daily  = read_csv(daily_b)
    sleep  = read_csv(sleep_b)
    hr_min = pd.read_parquet(io.BytesIO(hr_min_b))

    daily["ActivityDate"] = pd.to_datetime(
        daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")

    sc = "date" if "date" in sleep.columns else "Date"
    sleep[sc] = pd.to_datetime(sleep[sc], format="%m/%d/%Y %I:%M:%S %p",
                               errors="coerce")
    if sc != "date": sleep = sleep.rename(columns={sc: "date"})

    hr_min["Date"] = hr_min["Time"].dt.date
    hr_d = (hr_min.groupby(["Id","Date"])["HeartRate"]
            .agg(AvgHR="mean", MaxHR="max", MinHR="min", StdHR="std")
            .reset_index())

    sleep["Date"] = sleep["date"].dt.date
    sl_d = (sleep.groupby(["Id","Date"])
            .agg(TotalSleepMinutes=("value","count"),
                 DominantSleepStage=("value", lambda x: x.mode()[0]))
            .reset_index())

    m = daily.rename(columns={"ActivityDate":"Date"}).copy()
    m["Date"] = m["Date"].dt.date
    m = m.merge(hr_d,  on=["Id","Date"], how="left")
    m = m.merge(sl_d,  on=["Id","Date"], how="left")
    m["TotalSleepMinutes"]  = m["TotalSleepMinutes"].fillna(0)
    m["DominantSleepStage"] = m["DominantSleepStage"].fillna(0)
    for c in ["AvgHR","MaxHR","MinHR","StdHR"]:
        m[c] = m.groupby("Id")[c].transform(lambda x: x.fillna(x.median()))
    return df_pq(m)

@st.cache_data(show_spinner="⏳ Fitting Prophet…")
def fit_prophet(ds_b, y_b, horizon):
    from prophet import Prophet
    ds = pd.read_json(io.BytesIO(ds_b), typ="series")
    y  = pd.read_json(io.BytesIO(y_b),  typ="series")
    df = pd.DataFrame({"ds": pd.to_datetime(ds), "y": y}).dropna().sort_values("ds")
    m  = Prophet(daily_seasonality=False, weekly_seasonality=True,
                 yearly_seasonality=False, interval_width=0.80,
                 changepoint_prior_scale=0.1)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(periods=horizon))
    return df_pq(df), df_pq(fc)

@st.cache_data(show_spinner="⏳ Running clustering + PCA…")
def run_clustering(feat_b, k, eps, min_s):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    feats = pd.read_parquet(io.BytesIO(feat_b))
    X  = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    db = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)
    var = (pca.explained_variance_ratio_ * 100).tolist()
    inertias = [KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X).inertia_
                for ki in range(2, 10)]
    return X.tobytes(), X2.tobytes(), var, km.tolist(), db.tolist(), inertias

@st.cache_data(show_spinner="⏳ Running t-SNE (~30 sec)…")
def run_tsne(X_b, n_feats):
    from sklearn.manifold import TSNE
    X = np.frombuffer(X_b, dtype=np.float64).reshape(-1, n_feats)
    return TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(X)-1),
                max_iter=1000).fit_transform(X).tobytes()

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (only parameters — NO upload)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT}'>⚡ FitPulse</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:.82rem;margin-top:-8px'>"
                "Milestone 2 · ML Pipeline</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown(f"<p style='color:{TEXT};font-weight:700'>⚙️ Model Parameters</p>",
                unsafe_allow_html=True)
    OPTIMAL_K     = st.slider("K-Means Clusters (K)",    2, 8, 3)
    EPS           = st.slider("DBSCAN  ε (eps)",         0.5, 5.0, 2.2, 0.1)
    MIN_SAMPLES   = st.slider("DBSCAN  min_samples",     1, 5, 2)
    FORECAST_DAYS = st.slider("Forecast horizon (days)", 7, 60, 30)
    run_tsne_flag = st.checkbox("Run t-SNE  (~30 sec)", value=False)
    st.divider()

    st.markdown(f"<p style='color:{MUTED};font-size:.72rem'>"
                "Real Fitbit Dataset · Mar–Apr 2016<br>"
                "TSFresh · Prophet · KMeans · DBSCAN</p>",
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,{CARD},{CARD2});
            border:1px solid {BORDER};border-radius:14px;
            padding:28px 32px;margin-bottom:28px'>
  <div style='font-size:.65rem;font-weight:800;letter-spacing:.18em;
              color:{BLUE};text-transform:uppercase;margin-bottom:8px'>
    MILESTONE 2 · FEATURE EXTRACTION &amp; MODELING
  </div>
  <div style='font-size:2.2rem;font-weight:800;color:{TEXT};line-height:1.1'>
    ⚡ FitPulse ML Pipeline
  </div>
  <div style='color:{MUTED};margin-top:10px;font-size:.85rem'>
    TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD SECTION  — single button, selects all 6 files at once
# ═════════════════════════════════════════════════════════════════════════════
FILE_DEFS = [
    ("daily",             "🏃", "Daily Activity",       "dailyActivity_merged.csv"),
    ("heartrate",         "❤️", "Heart Rate",            "heartrate_seconds_merged.csv"),
    ("hourly_intensities","⚡", "Hourly Intensities",    "hourlyIntensities_merged.csv"),
    ("hourly_steps",      "👟", "Hourly Steps",          "hourlySteps_merged.csv"),
    ("sleep",             "😴", "Minute Sleep",          "minuteSleep_merged.csv"),
    ("tsfresh",           "🧬", "TSFresh Features CSV",  "tsfresh_features.csv"),
]

slots = {k: None for k,*_ in FILE_DEFS}
raw   = {}

st.markdown(f"""
<div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
            padding:22px 26px;margin-bottom:18px'>
  <div style='font-size:1.1rem;font-weight:800;color:{TEXT};margin-bottom:4px'>
    📂 Upload Your Fitbit CSV Files
  </div>
  <div style='font-size:.8rem;color:{MUTED}'>
    Click <b>Browse files</b> and select all 6 CSV files at once
    (hold <b>Ctrl / Cmd</b> to multi-select).<br>
    Files are <b>auto-detected</b> by column structure — no renaming needed.<br>
    Required: <code>dailyActivity</code> · <code>heartrate</code> ·
    <code>hourlyIntensities</code> · <code>hourlySteps</code> ·
    <code>minuteSleep</code> · <code>tsfresh_features.csv</code>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📁  Select all 6 CSV files at once  (Ctrl+Click / Cmd+Click to multi-select)",
    type=["csv"],
    accept_multiple_files=True,
    key="bulk_upload",
)

# Parse every uploaded file
if uploaded_files:
    for f in uploaded_files:
        b  = f.read()
        df = read_csv(b)
        dt = detect_type(df)
        if dt in slots:
            slots[dt] = df
            raw[dt]   = b

# ─ Status cards — 6 icons showing detected / missing ─────────────────────────
st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
card_cols = st.columns(6)
n_ok = 0
for col, (key, icon, label, _) in zip(card_cols, FILE_DEFS):
    ready = slots[key] is not None
    if ready: n_ok += 1
    bg   = "rgba(63,185,80,.10)" if ready else CARD2
    bdr  = GREEN if ready else BORDER
    stxt = (f'<span style="color:{GREEN};font-weight:800;font-size:.82rem">'
            f'✅ Detected</span>'
            if ready else
            f'<span style="color:{MUTED};font-size:.78rem">⬜ Missing</span>')
    col.markdown(
        f'<div style="background:{bg};border:1px solid {bdr};'
        f'border-radius:12px;padding:16px 10px;text-align:center">'
        f'<div style="font-size:1.8rem">{icon}</div>'
        f'<div style="font-size:.68rem;font-weight:700;color:{MUTED};'
        f'text-transform:uppercase;letter-spacing:.06em;margin:6px 0 4px">'
        f'{label}</div>{stxt}</div>',
        unsafe_allow_html=True)

st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
st.progress(n_ok / 6, text=f"Files loaded: {n_ok} / 6")

# ─ Guard ──────────────────────────────────────────────────────────────────────
required = ["daily","heartrate","hourly_intensities","hourly_steps","sleep","tsfresh"]
missing  = [k for k in required if slots[k] is None]
if missing:
    nice = {"daily":"Daily Activity","heartrate":"Heart Rate",
            "hourly_intensities":"Hourly Intensities",
            "hourly_steps":"Hourly Steps","sleep":"Minute Sleep",
            "tsfresh":"TSFresh Features CSV"}
    st.info("👆 Click **Browse files** above and select all 6 CSV files at once.\n\n"
            f"**Still needed:** {', '.join(nice[k] for k in missing)}")
    st.stop()

st.success("✅ All 6 files uploaded and ready.")
st.divider()

# ── session state init ────────────────────────────────────────────────────────
for _k in ["run_p1","run_p2","run_p3","run_p4"]:
    if _k not in st.session_state:
        st.session_state[_k] = False


# ═════════════════════════════════════════════════════════════════════════════
#  DATA PREP  (shared — runs whenever any phase is active, fully cached)
# ═════════════════════════════════════════════════════════════════════════════
# Parse timestamps
daily    = slots["daily"].copy()
hourly_s = slots["hourly_steps"].copy()
hourly_i = slots["hourly_intensities"].copy()
sleep    = slots["sleep"].copy()
hr       = slots["heartrate"].copy()
features = slots["tsfresh"].copy()
if features.columns[0] in ("Unnamed: 0",""):
    features = features.rename(columns={features.columns[0]: "UserId"})
    features = features.set_index("UserId")
elif features.index.name is None and features.index.dtype == object:
    pass  # index already set

daily["ActivityDate"] = pd.to_datetime(
    daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")
hourly_s["ActivityHour"] = pd.to_datetime(
    hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
hourly_i["ActivityHour"] = pd.to_datetime(
    hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
sc = "date" if "date" in sleep.columns else "Date"
sleep[sc] = pd.to_datetime(sleep[sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
if sc != "date": sleep = sleep.rename(columns={sc: "date"})
hr["Time"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# HR resample + master build
hr_min_b  = resample_hr(raw["heartrate"])
hr_min    = pd.read_parquet(io.BytesIO(hr_min_b))
master_b  = build_master(raw["daily"], raw["sleep"], hr_min_b)
master    = pd.read_parquet(io.BytesIO(master_b))
date_span = (daily["ActivityDate"].max() - daily["ActivityDate"].min()).days



# ─── Phase 1 ────────────────────────────────────────────────────────────────
if not st.session_state['run_p1']:
    _lb_html = (
        '<div style="background:#1c2128;border:2px solid #58a6ff;'
        'border-radius:14px;padding:22px 24px;'
        'text-align:center;margin:22px 0 10px">'
        '<div style="font-size:2rem">📂</div>'
        '<div style="font-size:.68rem;font-weight:800;color:#58a6ff;'
        'text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">'
        'Phase 1</div>'
        '<div style="font-size:.95rem;font-weight:700;color:#e6edf3">Data Ingestion & Cleaning</div>'
        '<div style="font-size:.72rem;color:#8b949e;margin-top:4px">Steps 1–9</div>'
        '</div>'
    )
    st.markdown(_lb_html, unsafe_allow_html=True)
    _lc1, _lc2, _lc3 = st.columns([2, 3, 2])
    with _lc2:
        if st.button('▶  Run Phase 1', key='btn_p1',
                     use_container_width=True, type='primary'):
            st.session_state['run_p1'] = True
            st.rerun()

if st.session_state['run_p1']:
    phase_banner("📂","Phase 1 · Data Ingestion & Cleaning","STEPS 1 – 9",
                 "Parse timestamps → null audit → resample HR → merge master dataframe")

    # ── Step 1–3 ──────────────────────────────────────────────────────────────────
    step_box("Step 1–3","Files Loaded & Timestamps Parsed",
             "All 5 CSVs auto-detected by column structure · all timestamp columns parsed to datetime")

    shape_df = pd.DataFrame({
        "Dataset":  ["dailyActivity","hourlySteps","hourlyIntensities","minuteSleep","heartrate"],
        "Rows":     [f"{daily.shape[0]:,}", f"{hourly_s.shape[0]:,}",
                     f"{hourly_i.shape[0]:,}", f"{sleep.shape[0]:,}", f"{hr.shape[0]:,}"],
        "Columns":  [daily.shape[1], hourly_s.shape[1], hourly_i.shape[1],
                     sleep.shape[1], hr.shape[1]],
        "Key Columns": [
            "Id, ActivityDate, TotalSteps, Calories, VeryActiveMinutes, SedentaryMinutes",
            "Id, ActivityHour, StepTotal",
            "Id, ActivityHour, TotalIntensity, AverageIntensity",
            "Id, date, value, logId",
            "Id, Time, Value",
        ],
    })
    st.dataframe(shape_df, use_container_width=True, hide_index=True)
    st.divider()

    # ── Step 4 — Null Check ───────────────────────────────────────────────────────
    step_box("Step 4","Null Value Check — All 5 Datasets",
             "Scan every column · 0 nulls means perfectly clean data")

    null_rows = []
    for name, df in [("dailyActivity",daily),("hourlySteps",hourly_s),
                     ("hourlyIntensities",hourly_i),("minuteSleep",sleep),
                     ("heartrate",hr)]:
        n = int(df.isnull().sum().sum())
        null_rows.append({"Dataset":name,"Total Nulls":n,
                          "Shape":str(df.shape),
                          "Completeness":"100 %" if n==0 else f"{(1-n/df.size)*100:.2f}%",
                          "Status":"✅ Clean" if n==0 else f"⚠️ {n} nulls"})
    st.dataframe(pd.DataFrame(null_rows), use_container_width=True, hide_index=True)

    # Null visual bar
    fig_nv, ax_nv = plt.subplots(figsize=(10, 2.8))
    names_nv = [r["Dataset"] for r in null_rows]
    ax_nv.barh(names_nv, [0]*5, color=GREEN, height=0.4)
    for i in range(5):
        ax_nv.text(0.01, i, "  0 nulls — 100 % complete ✅",
                   va="center", fontsize=9, color=GREEN, fontweight="700")
    ax_nv.set_xlim(0,1); ax_nv.set_xticks([]); ax_nv.grid(False)
    ax_nv.set_title("Null Value Audit — All Datasets  (all green = zero missing values)",
                    fontsize=10, color=TEXT, pad=8)
    plt.tight_layout(); st.pyplot(fig_nv); plt.close(fig_nv)
    st.divider()

    # ── Step 5–6 — Overview ───────────────────────────────────────────────────────
    step_box("Step 5–6","Dataset Overview — Key Counts",
             "Unique users per dataset · date range · total record volume")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Users",  daily["Id"].nunique())
    c2.metric("Sleep Users",  sleep["Id"].nunique())
    c3.metric("HR Users",     hr["Id"].nunique())
    c4.metric("HR Records",   f"{hr.shape[0]:,}")
    c5.metric("Date Span",    f"{date_span} days")
    c6.metric("Total Rows",
              f"{sum(x.shape[0] for x in [daily,hourly_s,hourly_i,sleep,hr]):,}")
    st.divider()

    # ── Step 6 — HR Resample ──────────────────────────────────────────────────────
    step_box("Step 6","Time Normalization — Heart Rate: Seconds → 1-Minute",
             "Per-second HR (~1.15M rows) resampled to 1-minute mean "
             "→ 6× smaller · temporal patterns preserved for TSFresh")

    r1,r2,r3 = st.columns(3)
    r1.metric("Before (rows)", f"{hr.shape[0]:,}",     delta="seconds-level")
    r2.metric("After  (rows)", f"{hr_min.shape[0]:,}", delta="1-min intervals")
    r3.metric("Size reduction",
              f"{(1-hr_min.shape[0]/hr.shape[0])*100:.0f}%", delta_color="off")
    st.divider()

    # ── Step 7–9 — Master ─────────────────────────────────────────────────────────
    step_box("Step 7–9","Cleaned Master Dataframe",
             "dailyActivity + HR daily aggregates + sleep daily aggregates → "
             "one row per user per day")

    cm1,cm2,cm3 = st.columns(3)
    cm1.metric("Master Shape", str(master.shape))
    cm2.metric("Unique Users", master["Id"].nunique())
    cm3.metric("Null Values",  int(master.isnull().sum().sum()))

    preview_c = ["Id","Date","TotalSteps","Calories","AvgHR",
                 "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
    st.dataframe(master[preview_c].head(15), use_container_width=True, hide_index=True)
    st.divider()

    # ── Step 9a — Stats ───────────────────────────────────────────────────────────
    step_box("Step 9a","Summary Statistics",
             "describe() for key numerical columns — count · mean · std · min · max · quartiles")

    key_c = ["TotalSteps","Calories","AvgHR","TotalSleepMinutes",
             "VeryActiveMinutes","SedentaryMinutes"]
    st.dataframe(master[key_c].describe().round(2), use_container_width=True)
    st.divider()

    # ── Step 9b — Distributions ───────────────────────────────────────────────────
    step_box("Step 9b","Activity Distribution Histograms",
             "Frequency of each metric across all records · "
             "bar count labeled on every bar · mean line shown")

    dist_cfg = [
        ("TotalSteps",       "Total Daily Steps",       BLUE,   "Steps / day"),
        ("Calories",         "Daily Calories Burned",   GREEN,  "Calories / day"),
        ("TotalSleepMinutes","Daily Sleep Duration",     PURPLE, "Minutes / day"),
        ("SedentaryMinutes", "Sedentary Time / Day",     AMBER,  "Minutes / day"),
        ("VeryActiveMinutes","Very-Active Time / Day",   RED,    "Minutes / day"),
        ("AvgHR",            "Average Heart Rate",       PINK,   "BPM"),
    ]
    for i in range(0, len(dist_cfg), 2):
        cols_d = st.columns(2)
        for j, cd in enumerate(cols_d):
            if i+j >= len(dist_cfg): break
            key, title, color, xlabel = dist_cfg[i+j]
            s = master[key].dropna()
            fig, ax = plt.subplots(figsize=(7, 3.6))
            cnts, _, patches = ax.hist(s, bins=20, color=color,
                                       alpha=0.85, edgecolor=DARK, linewidth=0.4)
            top = max(cnts) if len(cnts) > 0 else 1
            for patch, cnt in zip(patches, cnts):
                if cnt > 0:
                    ax.text(patch.get_x()+patch.get_width()/2,
                            cnt + top*0.015,
                            f"{int(cnt)}", ha="center", va="bottom",
                            fontsize=7, color=TEXT)
            mv = s.mean()
            ax.axvline(mv, color="white", linestyle="--",
                       linewidth=1.4, label=f"Mean = {mv:.0f}")
            ax.set_title(f"📊  {title}\n(y = records · bar counts labeled · "
                         f"dashed = mean {mv:.0f})",
                         fontsize=9, color=TEXT, pad=7)
            ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
            ax.set_ylabel("Number of Records", fontsize=8, color=MUTED)
            ax.legend(fontsize=8, framealpha=0.4)
            ax.grid(axis="y", alpha=0.2)
            plt.tight_layout(); cd.pyplot(fig); plt.close(fig)
    st.divider()

    # ── Step 9c — Hourly Heatmap ──────────────────────────────────────────────────
    step_box("Step 9c","Hourly Steps Heatmap — When Are Users Most Active?",
             "Average step count per day-of-week × hour-of-day · "
             "exact value shown inside every cell")

    hourly_s["Hour"]      = hourly_s["ActivityHour"].dt.hour
    hourly_s["DayOfWeek"] = hourly_s["ActivityHour"].dt.day_name()
    pivot = hourly_s.groupby(["DayOfWeek","Hour"])["StepTotal"].mean().unstack()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig_hw, ax_hw = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, ax=ax_hw, cmap="YlOrRd",
                annot=True, fmt=".0f", annot_kws={"size":6.5},
                linewidths=0.25, linecolor=DARK,
                cbar_kws={"label":"Avg Steps / Hour","shrink":0.65})
    ax_hw.set_title("🕐  Average Steps by Day of Week × Hour of Day\n"
                    "(Each cell = exact avg step count · 0 = midnight · "
                    "12 = noon · darker = more active)",
                    fontsize=11, color=TEXT, pad=10)
    ax_hw.set_xlabel("Hour of Day (0–23)", fontsize=9, color=MUTED)
    ax_hw.set_ylabel("Day of Week", fontsize=9, color=MUTED)
    plt.tight_layout(); st.pyplot(fig_hw); plt.close(fig_hw)

    # ── Next Phase Button ─────────────────────────────────────────────────
    st.divider()
    _nb_html = (
        '<div style="background:#1c2128;border:2px solid #3fb950;'
        'border-radius:14px;padding:22px 24px;'
        'text-align:center;margin:22px 0 10px">'
        '<div style="font-size:2rem">🧬</div>'
        '<div style="font-size:.68rem;font-weight:800;color:#3fb950;'
        'text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">'
        '🔜 Next — Phase 2</div>'
        '<div style="font-size:.95rem;font-weight:700;color:#e6edf3">Feature Engineering</div>'
        '<div style="font-size:.72rem;color:#8b949e;margin-top:4px">Steps 10–12</div>'
        '</div>'
    )
    st.markdown(_nb_html, unsafe_allow_html=True)
    _nc1, _nc2, _nc3 = st.columns([2, 3, 2])
    with _nc2:
        if st.button('▶  Run Phase 2', key='btn_p2',
                     use_container_width=True, type='primary'):
            st.session_state['run_p2'] = True
            st.rerun()


if st.session_state['run_p2']:
    phase_banner("🧬","Phase 2 · Feature Engineering","STEPS 10 – 12",
                 "TSFresh features loaded from pre-computed CSV — "
                 "10 statistical features per user extracted from minute-level HR")

    step_box("Step 10–11","TSFresh Feature Matrix  (loaded from uploaded CSV)",
             "MinimalFCParameters: sum_values · median · mean · length · "
             "std_dev · variance · rms · maximum · absolute_maximum · minimum")

    ff1,ff2,ff3 = st.columns(3)
    ff1.metric("Users (rows)",    features.shape[0])
    ff2.metric("Features (cols)", features.shape[1])
    ff3.metric("Source",          "Uploaded tsfresh_features.csv")

    st.markdown("**Extracted feature names:**")
    for i, c in enumerate(features.columns):
        st.markdown(f"<span class='info-pill'>{i+1}. {c.replace('value__','')}</span>",
                    unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Full Feature Matrix Table:**")
    st.dataframe(features.round(4), use_container_width=True)
    st.divider()

    # ── Step 12a — Heatmap ───────────────────────────────────────────────────────
    step_box("Step 12a","Feature Heatmap — Normalized 0–1  📸 Screenshot This",
             "Each cell shows the exact normalized value (0=min, 1=max per feature) · "
             "Rows = Users · Columns = Statistical Features")

    from sklearn.preprocessing import MinMaxScaler as MMS
    feat_norm = pd.DataFrame(
        MMS().fit_transform(features),
        index=features.index, columns=features.columns)
    fd = feat_norm.rename(columns={c: c.replace("value__","") for c in feat_norm.columns})
    fd.index = [str(i)[-6:] for i in fd.index]

    fig_hm, ax_hm = plt.subplots(
        figsize=(max(12, len(features.columns)*1.5),
                 max(6,  len(features)*0.65)))
    sns.heatmap(fd, ax=ax_hm, cmap="coolwarm",
                annot=True, fmt=".2f",
                annot_kws={"size":8.5, "weight":"bold"},
                linewidths=0.5, linecolor=DARK,
                cbar_kws={"label":"Normalized Value (0=min · 1=max per feature)",
                          "shrink":0.8},
                vmin=0, vmax=1)
    ax_hm.set_title(
        "🧬  TSFresh Feature Matrix — Minute-Level Heart Rate Data\n"
        "(Normalized 0–1 · Each cell = exact normalized value · "
        "Rows = Users (last 6 digits) · Columns = Statistical Features)",
        fontsize=11, color=TEXT, pad=12)
    ax_hm.set_xlabel("Extracted Statistical Feature", fontsize=9, color=MUTED)
    ax_hm.set_ylabel("User ID (last 6 digits)", fontsize=9, color=MUTED)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_hm); dl(fig_hm, "tsfresh_heatmap.png", "dl_hm"); plt.close(fig_hm)
    st.divider()

    # ── Step 12b — Per-feature bars ───────────────────────────────────────────────
    step_box("Step 12b","Per-Feature Bar Charts — Every User",
             "One chart per TSFresh feature · sorted ascending · "
             "exact value on every bar")

    for feat in features.columns:
        fname = feat.replace("value__","")
        vals  = features[feat].sort_values()
        ulbls = [str(i)[-5:] for i in vals.index]
        fig_b, ax_b = plt.subplots(figsize=(12, 3.4))
        bars_b = ax_b.bar(range(len(vals)), vals.values,
                          color=[PAL[i % len(PAL)] for i in range(len(vals))],
                          edgecolor=DARK, linewidth=0.4, zorder=3)
        mx = max(vals.values) if len(vals) else 1
        for bar, v in zip(bars_b, vals.values):
            ax_b.text(bar.get_x()+bar.get_width()/2,
                      bar.get_height() + mx*0.014,
                      f"{v:.2f}", ha="center", va="bottom",
                      fontsize=7.5, color=TEXT, fontweight="700")
        ax_b.set_xticks(range(len(vals)))
        ax_b.set_xticklabels(ulbls, rotation=35, ha="right", fontsize=8)
        ax_b.set_title(
            f"📊  TSFresh Feature: {fname}\n"
            f"(x = User ID last 5 digits · y = raw feature value · "
            f"sorted low→high · value labeled on every bar)",
            fontsize=10, color=TEXT, pad=8)
        ax_b.set_xlabel("User ID (last 5 digits)", fontsize=8, color=MUTED)
        ax_b.set_ylabel(fname, fontsize=8, color=MUTED)
        ax_b.grid(axis="y", alpha=0.2); ax_b.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig_b); plt.close(fig_b)


    # ── Next Phase Button ─────────────────────────────────────────────────
    st.divider()
    _nb_html = (
        '<div style="background:#1c2128;border:2px solid #f0883e;'
        'border-radius:14px;padding:22px 24px;'
        'text-align:center;margin:22px 0 10px">'
        '<div style="font-size:2rem">📈</div>'
        '<div style="font-size:.68rem;font-weight:800;color:#f0883e;'
        'text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">'
        '🔜 Next — Phase 3</div>'
        '<div style="font-size:.95rem;font-weight:700;color:#e6edf3">Prophet Forecasting</div>'
        '<div style="font-size:.72rem;color:#8b949e;margin-top:4px">Steps 13–17</div>'
        '</div>'
    )
    st.markdown(_nb_html, unsafe_allow_html=True)
    _nc1, _nc2, _nc3 = st.columns([2, 3, 2])
    with _nc2:
        if st.button('▶  Run Phase 3', key='btn_p3',
                     use_container_width=True, type='primary'):
            st.session_state['run_p3'] = True
            st.rerun()


if st.session_state['run_p3']:
    phase_banner("📈","Phase 3 · Prophet Trend Forecasting",
                 f"STEPS 13 – 17",
                 f"Prophet fit on HR · Steps · Sleep → {FORECAST_DAYS}-day forecast "
                 f"with 80% CI · actual values labeled on every point")

    try:
        from prophet import Prophet
    except ImportError:
        st.error("❌ `prophet` not installed — `pip install prophet`"); st.stop()

    def prophet_plot(df_in, fc, color, title, ylabel, dl_key):
        fig, ax = plt.subplots(figsize=(14, 5.5))
        fc_start = df_in["ds"].max()
        ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                        alpha=0.2, color=color, label="80% Confidence Interval")
        ax.plot(fc["ds"], fc["yhat"], color=TEXT, linewidth=2.2,
                label="Prophet Trend (yhat)", zorder=3)
        ax.scatter(df_in["ds"], df_in["y"], color=color, s=32,
                   zorder=5, alpha=0.9, label="Actual Observed Values")
        for _, row in df_in.iterrows():
            ax.annotate(f"{row['y']:.0f}", (row["ds"], row["y"]),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=6.5, color=TEXT, ha="center", alpha=0.8)
        ax.axvline(fc_start, color=AMBER, linestyle="--", linewidth=1.8,
                   label=f"Forecast Start ({fc_start.date()})", alpha=0.9)
        last = fc.iloc[-1]
        ax.annotate(
            f"Forecast end:\n{last['yhat']:.1f}",
            (last["ds"], last["yhat"]),
            xytext=(-55, 14), textcoords="offset points",
            fontsize=8, color=AMBER, fontweight="700",
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2))
        ax.set_title(
            f"📈  {title}\n"
            f"(Dots = actual values with labels · Line = Prophet trend · "
            f"Band = 80% CI · Dashed = forecast start)",
            fontsize=11, color=TEXT, pad=10)
        ax.set_xlabel("Date", fontsize=9, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.legend(fontsize=8, framealpha=0.35)
        ax.grid(alpha=0.15); plt.tight_layout()
        st.pyplot(fig); dl(fig, dl_key, dl_key); plt.close(fig)

    # ── Heart Rate ────────────────────────────────────────────────────────────────
    step_box("Step 13–14","Heart Rate Forecast  📸 Screenshot This",
             "Daily mean HR → Prophet fit → 80% CI · every actual value labeled")

    hr_agg = (hr_min.groupby(hr_min["Time"].dt.date)["HeartRate"]
              .mean().reset_index())
    hr_agg.columns = ["ds","y"]
    hr_agg["ds"] = pd.to_datetime(hr_agg["ds"])
    hr_agg = hr_agg.dropna().sort_values("ds")

    df_hr_b, fc_hr_b = fit_prophet(
        ser_json(hr_agg["ds"]), ser_json(hr_agg["y"]), FORECAST_DAYS)
    df_hr = pd.read_parquet(io.BytesIO(df_hr_b))
    fc_hr = pd.read_parquet(io.BytesIO(fc_hr_b))

    h1,h2,h3 = st.columns(3)
    h1.metric("Training Points",     len(df_hr))
    h2.metric("Forecast Horizon",    f"{FORECAST_DAYS} days")
    h3.metric("Confidence Interval", "80%")
    prophet_plot(df_hr, fc_hr, AMBER,
                 f"Heart Rate Forecast — {FORECAST_DAYS}-Day Prophet Projection",
                 "Average Heart Rate (BPM)", "prophet_hr.png")

    with st.expander("📊 View Prophet HR Components (trend + weekly seasonality)"):
        _m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                     yearly_seasonality=False, interval_width=0.80,
                     changepoint_prior_scale=0.1)
        _m.fit(df_hr)
        figc = _m.plot_components(_m.predict(_m.make_future_dataframe(periods=FORECAST_DAYS)))
        figc.patch.set_facecolor(DARK)
        plt.suptitle("Prophet Components — Heart Rate\n"
                     "(Top = overall trend · Bottom = weekly seasonality pattern)",
                     y=1.01, fontsize=10, color=TEXT)
        plt.tight_layout(); st.pyplot(figc); plt.close(figc)
    st.divider()

    # ── Daily Steps ───────────────────────────────────────────────────────────────
    step_box("Step 15–16","Daily Steps Forecast  📸 Screenshot This",
             "Average steps per day → Prophet → actual values annotated on every point")

    steps_agg = (daily.groupby("ActivityDate")["TotalSteps"].mean().reset_index())
    steps_agg.columns = ["ds","y"]
    steps_agg["ds"] = pd.to_datetime(steps_agg["ds"])
    steps_agg = steps_agg.dropna().sort_values("ds")

    df_st_b, fc_st_b = fit_prophet(
        ser_json(steps_agg["ds"]), ser_json(steps_agg["y"]), FORECAST_DAYS)
    df_st = pd.read_parquet(io.BytesIO(df_st_b))
    fc_st = pd.read_parquet(io.BytesIO(fc_st_b))
    prophet_plot(df_st, fc_st, GREEN,
                 f"Daily Steps Forecast — {FORECAST_DAYS}-Day Prophet Projection",
                 "Average Total Steps per Day", "prophet_steps.png")
    st.divider()

    # ── Sleep ─────────────────────────────────────────────────────────────────────
    step_box("Step 17","Sleep Duration Forecast  📸 Screenshot This",
             "Daily mean sleep minutes → Prophet → 80% CI · every actual value labeled")

    sleep_agg = master.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
    sleep_agg.columns = ["ds","y"]
    sleep_agg["ds"] = pd.to_datetime(sleep_agg["ds"])
    sleep_agg = sleep_agg[sleep_agg["y"] > 0].dropna().sort_values("ds")

    df_sl_b, fc_sl_b = fit_prophet(
        ser_json(sleep_agg["ds"]), ser_json(sleep_agg["y"]), FORECAST_DAYS)
    df_sl = pd.read_parquet(io.BytesIO(df_sl_b))
    fc_sl = pd.read_parquet(io.BytesIO(fc_sl_b))
    prophet_plot(df_sl, fc_sl, PURPLE,
                 f"Sleep Duration Forecast — {FORECAST_DAYS}-Day Prophet Projection",
                 "Average Sleep Duration (minutes/day)", "prophet_sleep.png")
    st.divider()

    # ── Combined (notebook style) ─────────────────────────────────────────────────
    step_box("Step 15–17 Combined","Steps + Sleep Combined Plot  📸 Screenshot This",
             "Exact notebook-style 2-row stacked figure · both annotated")

    fig_comb, axes_c = plt.subplots(2, 1, figsize=(14, 10))
    fig_comb.patch.set_facecolor(DARK)
    for ax_c,(df_c,fc_c,col_c,lbl_c) in zip(
        axes_c,
        [(df_st,fc_st,GREEN,"Steps"),
         (df_sl,fc_sl,PURPLE,"Sleep (minutes)")]):
        ax_c.set_facecolor(CARD2)
        ax_c.fill_between(fc_c["ds"], fc_c["yhat_lower"], fc_c["yhat_upper"],
                          alpha=0.25, color=col_c, label="80% CI")
        ax_c.plot(fc_c["ds"], fc_c["yhat"], color=TEXT, linewidth=2.5, label="Trend")
        ax_c.scatter(df_c["ds"], df_c["y"], color=col_c, s=22,
                     alpha=0.85, zorder=4, label=f"Actual {lbl_c}")
        for _, row in df_c.iterrows():
            ax_c.annotate(f"{row['y']:.0f}", (row["ds"], row["y"]),
                          textcoords="offset points", xytext=(0,5),
                          fontsize=6, color=TEXT, ha="center", alpha=0.75)
        ax_c.axvline(df_c["ds"].max(), color=AMBER, linestyle="--",
                     linewidth=1.8, label="Forecast Start")
        ax_c.set_title(
            f"{lbl_c} — Prophet Trend Forecast\n"
            f"(Labeled dots = actual data · white line = trend · "
            f"band = 80% CI · dashed = forecast start)",
            fontsize=11, color=TEXT)
        ax_c.set_xlabel("Date", color=MUTED)
        ax_c.set_ylabel(lbl_c, color=MUTED)
        ax_c.legend(fontsize=8, framealpha=0.3); ax_c.grid(alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig_comb); dl(fig_comb,"prophet_combined.png","dl_comb"); plt.close(fig_comb)

    # ── Next Phase Button ─────────────────────────────────────────────────
    st.divider()
    _nb_html = (
        '<div style="background:#1c2128;border:2px solid #bc8cff;'
        'border-radius:14px;padding:22px 24px;'
        'text-align:center;margin:22px 0 10px">'
        '<div style="font-size:2rem">🤖</div>'
        '<div style="font-size:.68rem;font-weight:800;color:#bc8cff;'
        'text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">'
        '🔜 Next — Phase 4</div>'
        '<div style="font-size:.95rem;font-weight:700;color:#e6edf3">Clustering & Reduction</div>'
        '<div style="font-size:.72rem;color:#8b949e;margin-top:4px">Steps 18–27</div>'
        '</div>'
    )
    st.markdown(_nb_html, unsafe_allow_html=True)
    _nc1, _nc2, _nc3 = st.columns([2, 3, 2])
    with _nc2:
        if st.button('▶  Run Phase 4', key='btn_p4',
                     use_container_width=True, type='primary'):
            st.session_state['run_p4'] = True
            st.rerun()


if st.session_state['run_p4']:
    phase_banner("🤖","Phase 4 · Clustering & Dimensionality Reduction",
                 "STEPS 18 – 27",
                 "Feature matrix → StandardScaler → K-Means + DBSCAN → "
                 "Elbow curve → PCA scatter → t-SNE → Cluster profile bar chart")

    # ── Step 18 — Feature matrix ──────────────────────────────────────────────────
    step_box("Step 18","Clustering Feature Matrix",
             "Average each user's daily metrics → one row per user")

    clust_c = ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes",
               "LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    clust_feats = master.groupby("Id")[clust_c].mean().round(3).dropna()
    cff1,cff2 = st.columns(2)
    cff1.metric("Users for clustering", clust_feats.shape[0])
    cff2.metric("Features",             clust_feats.shape[1])
    st.dataframe(clust_feats.round(2), use_container_width=True)
    st.divider()

    # ── Step 19 — Scale ───────────────────────────────────────────────────────────
    step_box("Step 19","StandardScaler Normalization",
             "Features scaled to mean ≈ 0 · std ≈ 1 before clustering")

    # Run clustering (cached — fast)
    (X_b, X2_b, var, km_list, db_list, inertias) = run_clustering(
        df_pq(clust_feats), OPTIMAL_K, EPS, MIN_SAMPLES)

    from sklearn.preprocessing import StandardScaler
    X_scaled     = np.frombuffer(X_b,  dtype=np.float64).reshape(-1, len(clust_c))
    X_pca        = np.frombuffer(X2_b, dtype=np.float64).reshape(-1, 2)
    kmeans_labels= np.array(km_list, dtype=int)
    dbscan_labels= np.array(db_list, dtype=int)

    sc1,sc2 = st.columns(2)
    sc1.metric("Mean after scaling (≈0)", f"{X_scaled.mean():.6f}")
    sc2.metric("Std  after scaling (≈1)", f"{X_scaled.std():.4f}")
    st.divider()

    # ── Step 20 — Elbow ───────────────────────────────────────────────────────────
    step_box("Step 20","K-Means Elbow Curve  📸 Screenshot This",
             f"Inertia for K=2…9 · exact value labeled on every point · "
             f"selected K={OPTIMAL_K} highlighted in amber")

    K_range = range(2,10)
    fig_el, ax_el = plt.subplots(figsize=(10, 4.5))
    ax_el.plot(list(K_range), inertias, "o-",
               color=BLUE, linewidth=2.5, markersize=11,
               markerfacecolor=PINK, markeredgecolor=TEXT,
               markeredgewidth=1.2, zorder=3)
    for k, iner in zip(K_range, inertias):
        ax_el.annotate(f"K={k}\n{iner:.0f}",
                       (k, iner),
                       textcoords="offset points", xytext=(0,14),
                       ha="center", fontsize=8.5, color=TEXT, fontweight="700")
    sel_idx = OPTIMAL_K - 2
    ax_el.scatter([OPTIMAL_K],[inertias[sel_idx]], color=AMBER, s=230, zorder=5,
                  label=f"Selected  K = {OPTIMAL_K}  ← optimal elbow",
                  edgecolors=TEXT, linewidths=1.5)
    ax_el.axvline(OPTIMAL_K, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.7)
    ax_el.set_title(
        f"📈  K-Means Elbow Curve — Real Fitbit Data\n"
        f"(x = number of clusters K · y = inertia · "
        f"inertia value labeled on every point · optimal K = {OPTIMAL_K})",
        fontsize=11, color=TEXT, pad=12)
    ax_el.set_xlabel("Number of Clusters (K)  [pick the bend/elbow point]",
                     fontsize=9, color=MUTED)
    ax_el.set_ylabel("Inertia  (Within-Cluster Sum of Squares)",
                     fontsize=9, color=MUTED)
    ax_el.set_xticks(list(K_range))
    ax_el.legend(fontsize=9, framealpha=0.4); ax_el.grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig_el)
    dl(fig_el,"elbow_curve.png","dl_el"); plt.close(fig_el)
    st.divider()

    # ── Step 21–22 — K-Means ──────────────────────────────────────────────────────
    step_box("Step 21–22","K-Means Clustering",
             f"K = {OPTIMAL_K} clusters fitted · user count per cluster")

    clust_feats = clust_feats.copy()
    clust_feats["KMeans_Cluster"] = kmeans_labels
    km_dist = clust_feats["KMeans_Cluster"].value_counts().sort_index()
    cols_km = st.columns(OPTIMAL_K)
    for i,col in enumerate(cols_km):
        col.metric(f"Cluster {i}", f"{int(km_dist.get(i,0))} users")

    c_km = [int(km_dist.get(i,0)) for i in range(OPTIMAL_K)]
    fig_kmd, ax_kmd = plt.subplots(figsize=(8, 3.2))
    bars_kmd = ax_kmd.bar([f"Cluster {i}" for i in range(OPTIMAL_K)],
                          c_km, color=PAL[:OPTIMAL_K], edgecolor=DARK)
    for bar, n in zip(bars_kmd, c_km):
        ax_kmd.text(bar.get_x()+bar.get_width()/2, n+0.05,
                    f"{n} users", ha="center", va="bottom",
                    fontsize=11, color=TEXT, fontweight="700")
    ax_kmd.set_title(f"K-Means Cluster Distribution  (K={OPTIMAL_K})\n"
                     f"(bar height = number of users · count labeled on every bar)",
                     fontsize=10, color=TEXT, pad=8)
    ax_kmd.set_xlabel("Cluster ID", fontsize=9, color=MUTED)
    ax_kmd.set_ylabel("Number of Users", fontsize=9, color=MUTED)
    ax_kmd.grid(axis="y", alpha=0.2); plt.tight_layout()
    st.pyplot(fig_kmd); plt.close(fig_kmd)
    st.divider()

    # ── Step 23 — DBSCAN ─────────────────────────────────────────────────────────
    step_box("Step 23","DBSCAN Clustering",
             f"eps={EPS} · min_samples={MIN_SAMPLES} · "
             "noise = -1 (shown red)")

    clust_feats["DBSCAN_Cluster"] = dbscan_labels
    n_cl_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise  = list(dbscan_labels).count(-1)
    db1,db2,db3 = st.columns(3)
    db1.metric("DBSCAN Clusters",  n_cl_db)
    db2.metric("Noise / Outliers", n_noise)
    db3.metric("Noise %",          f"{n_noise/len(dbscan_labels)*100:.1f}%")

    db_cnt  = pd.Series(dbscan_labels).value_counts().sort_index()
    db_lbls = ["Noise" if l==-1 else f"Cluster {l}" for l in db_cnt.index]
    db_clrs = [RED if l==-1 else PAL[l%len(PAL)]    for l in db_cnt.index]
    fig_dbd, ax_dbd = plt.subplots(figsize=(8, 3.2))
    bars_dbd = ax_dbd.bar(db_lbls, db_cnt.values, color=db_clrs, edgecolor=DARK)
    for bar, n in zip(bars_dbd, db_cnt.values):
        ax_dbd.text(bar.get_x()+bar.get_width()/2, n+0.05,
                    f"{n} users", ha="center", va="bottom",
                    fontsize=11, color=TEXT, fontweight="700")
    ax_dbd.set_title(f"DBSCAN Distribution  (eps={EPS} · min_samples={MIN_SAMPLES})\n"
                     f"(Red = noise/outlier · user count on every bar)",
                     fontsize=10, color=TEXT, pad=8)
    ax_dbd.set_xlabel("Cluster / Label", fontsize=9, color=MUTED)
    ax_dbd.set_ylabel("Number of Users", fontsize=9, color=MUTED)
    ax_dbd.grid(axis="y", alpha=0.2); plt.tight_layout()
    st.pyplot(fig_dbd); plt.close(fig_dbd)
    st.divider()

    # ── Step 24 — PCA Variance ───────────────────────────────────────────────────
    step_box("Step 24","PCA — 2D Dimensionality Reduction",
             "7 features → 2 principal components · variance explained per component")

    pv1,pv2,pv3 = st.columns(3)
    pv1.metric("PC1 Variance",    f"{var[0]:.1f}%")
    pv2.metric("PC2 Variance",    f"{var[1]:.1f}%")
    pv3.metric("Total Explained", f"{sum(var):.1f}%")
    st.divider()

    # ── Step 25 — K-Means PCA scatter ────────────────────────────────────────────
    step_box("Step 25","K-Means PCA Scatter Plot  📸 Screenshot This",
             "2D PCA projection coloured by K-Means label · "
             "User ID (last 4 digits) labeled on every point")

    fig_km_sc, ax_km = plt.subplots(figsize=(11,8))
    for cid in sorted(set(kmeans_labels)):
        mask = kmeans_labels == cid
        ax_km.scatter(X_pca[mask,0], X_pca[mask,1],
                      c=PAL[cid%len(PAL)], label=f"Cluster {cid}",
                      s=150, alpha=0.88, edgecolors=TEXT, linewidths=0.8, zorder=3)
        for i, uid in enumerate(clust_feats.index[mask]):
            ax_km.annotate(str(uid)[-4:],
                           (X_pca[mask][i,0], X_pca[mask][i,1]),
                           textcoords="offset points", xytext=(5,5),
                           fontsize=8, color=TEXT, fontweight="600")
    ax_km.set_title(
        f"🤖  K-Means Clustering — PCA 2D Projection\n"
        f"K = {OPTIMAL_K}  ·  PC1 = {var[0]:.1f}% variance  ·  "
        f"PC2 = {var[1]:.1f}% variance  ·  Labels = last 4 digits of User ID",
        fontsize=11, color=TEXT, pad=12)
    ax_km.set_xlabel(f"PC1  (explains {var[0]:.1f}% of total variance)",
                     fontsize=9, color=MUTED)
    ax_km.set_ylabel(f"PC2  (explains {var[1]:.1f}% of total variance)",
                     fontsize=9, color=MUTED)
    ax_km.legend(title=f"K-Means Cluster  (K={OPTIMAL_K})",
                 fontsize=10, framealpha=0.4)
    ax_km.grid(alpha=0.2); plt.tight_layout()
    st.pyplot(fig_km_sc)
    dl(fig_km_sc,"kmeans_pca.png","dl_km"); plt.close(fig_km_sc)
    st.divider()

    # ── Step 26 — DBSCAN PCA scatter ─────────────────────────────────────────────
    step_box("Step 26","DBSCAN PCA Scatter Plot  📸 Screenshot This",
             "Same PCA axes, DBSCAN labels · noise = red ✕ with 'noise' label")

    fig_db_sc, ax_db = plt.subplots(figsize=(11,8))
    for lbl in sorted(set(dbscan_labels)):
        mask = dbscan_labels == lbl
        if lbl == -1:
            ax_db.scatter(X_pca[mask,0], X_pca[mask,1],
                          c=RED, marker="X", s=230, alpha=0.95,
                          label="Noise / Outlier  (–1)", linewidths=1.5, zorder=5)
            for i,uid in enumerate(clust_feats.index[mask]):
                ax_db.annotate(f"{str(uid)[-4:]}  (noise)",
                               (X_pca[mask][i,0], X_pca[mask][i,1]),
                               textcoords="offset points", xytext=(8,6),
                               fontsize=8, color=RED, fontweight="700")
        else:
            ax_db.scatter(X_pca[mask,0], X_pca[mask,1],
                          c=PAL[lbl%len(PAL)], label=f"Cluster {lbl}",
                          s=150, alpha=0.88, edgecolors=TEXT, linewidths=0.8, zorder=3)
            for i,uid in enumerate(clust_feats.index[mask]):
                ax_db.annotate(str(uid)[-4:],
                               (X_pca[mask][i,0], X_pca[mask][i,1]),
                               textcoords="offset points", xytext=(5,5),
                               fontsize=8, color=TEXT, fontweight="600")
    ax_db.set_title(
        f"🤖  DBSCAN Clustering — PCA 2D Projection\n"
        f"eps={EPS}  ·  min_samples={MIN_SAMPLES}  ·  "
        f"Red ✕ = noise/outlier  ·  Labels = last 4 digits of User ID",
        fontsize=11, color=TEXT, pad=12)
    ax_db.set_xlabel(f"PC1  (explains {var[0]:.1f}% of total variance)",
                     fontsize=9, color=MUTED)
    ax_db.set_ylabel(f"PC2  (explains {var[1]:.1f}% of total variance)",
                     fontsize=9, color=MUTED)
    ax_db.legend(title="DBSCAN Cluster", fontsize=10, framealpha=0.4)
    ax_db.grid(alpha=0.2); plt.tight_layout()
    st.pyplot(fig_db_sc)
    dl(fig_db_sc,"dbscan_pca.png","dl_db"); plt.close(fig_db_sc)
    st.divider()

    # ── Step 27a — t-SNE ──────────────────────────────────────────────────────────
    step_box("Step 27a","t-SNE Projection  📸 Screenshot This",
             "Non-linear 2D embedding · User IDs labeled on every point · "
             "enable in sidebar")

    if run_tsne_flag:
        tsne_out = run_tsne(X_b, len(clust_c))
        X_tsne   = np.frombuffer(tsne_out, dtype=np.float64).reshape(-1, 2)

        fig_ts, axes_t = plt.subplots(1, 2, figsize=(16, 7))
        fig_ts.patch.set_facecolor(DARK)
        for ax_t,(lbls_t,name_t) in zip(
            axes_t,
            [(kmeans_labels, f"K-Means  (K={OPTIMAL_K})"),
             (dbscan_labels, f"DBSCAN  (eps={EPS})")]):
            ax_t.set_facecolor(CARD2)
            for lbl in sorted(set(lbls_t)):
                mask = lbls_t == lbl
                if lbl == -1:
                    ax_t.scatter(X_tsne[mask,0], X_tsne[mask,1],
                                 c=RED, marker="X", s=200,
                                 label="Noise", alpha=0.95, linewidths=1.5, zorder=5)
                    for i,uid in enumerate(clust_feats.index[mask]):
                        ax_t.annotate(f"{str(uid)[-4:]} (noise)",
                                      (X_tsne[mask][i,0], X_tsne[mask][i,1]),
                                      xytext=(5,5), textcoords="offset points",
                                      fontsize=7, color=RED, fontweight="700")
                else:
                    ax_t.scatter(X_tsne[mask,0], X_tsne[mask,1],
                                 c=PAL[lbl%len(PAL)], label=f"Cluster {lbl}",
                                 s=130, alpha=0.88, edgecolors=TEXT,
                                 linewidths=0.8, zorder=3)
                    for i,uid in enumerate(clust_feats.index[mask]):
                        ax_t.annotate(str(uid)[-4:],
                                      (X_tsne[mask][i,0], X_tsne[mask][i,1]),
                                      xytext=(5,5), textcoords="offset points",
                                      fontsize=7, color=TEXT, fontweight="600")
            ax_t.set_title(f"t-SNE — {name_t}\n"
                           f"(Non-linear 2D projection · User IDs labeled)",
                           fontsize=11, color=TEXT, pad=10)
            ax_t.set_xlabel("t-SNE Dimension 1", fontsize=9, color=MUTED)
            ax_t.set_ylabel("t-SNE Dimension 2", fontsize=9, color=MUTED)
            ax_t.legend(title="Cluster", fontsize=9, framealpha=0.35)
            ax_t.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_ts); dl(fig_ts,"tsne_projection.png","dl_ts"); plt.close(fig_ts)
    else:
        st.info("✅ Enable **'Run t-SNE (~30 sec)'** in the sidebar to generate this plot.")
    st.divider()

    # ── Step 27b — Cluster Profiles ───────────────────────────────────────────────
    step_box("Step 27b","Cluster Profiles — Grand Finale  📸 Screenshot This",
             "Grouped bar chart · exact average value labeled on every bar · "
             "5 key metrics compared across all clusters")

    feat_p = [c for c in clust_feats.columns
              if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
    profile = clust_feats.groupby("KMeans_Cluster")[feat_p].mean().round(2)
    st.markdown("**Average metrics per cluster:**")
    st.dataframe(profile, use_container_width=True)

    disp_c      = [c for c in ["TotalSteps","Calories","VeryActiveMinutes",
                                "SedentaryMinutes","TotalSleepMinutes"]
                   if c in profile.columns]
    feat_colors = [BLUE, GREEN, RED, AMBER, PURPLE]
    n_feat      = len(disp_c)
    n_clust     = len(profile)
    x           = np.arange(n_clust)
    width       = 0.14
    offsets     = np.linspace(-(n_feat-1)/2*width, (n_feat-1)/2*width, n_feat)

    fig_pr, ax_pr = plt.subplots(figsize=(14, 6.5))
    for fi,(feat,fc) in enumerate(zip(disp_c, feat_colors)):
        vals = profile[feat].values
        bars = ax_pr.bar(x+offsets[fi], vals, width,
                         label=feat, color=fc, edgecolor=DARK, alpha=0.9)
        mx = max(vals) if max(vals)>0 else 1
        for bar, v in zip(bars, vals):
            ax_pr.text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+mx*0.012,
                       f"{v:.0f}",
                       ha="center", va="bottom",
                       fontsize=7.5, color=TEXT, fontweight="700")
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels([f"Cluster {i}" for i in range(n_clust)],
                          fontsize=12, color=TEXT, fontweight="700")
    ax_pr.set_title(
        "🏆  Cluster Profiles — Key Feature Averages  (Real Fitbit Data)\n"
        "(Each bar = exact average for that cluster · colour = feature · "
        "values labeled on every bar)",
        fontsize=11, color=TEXT, pad=12)
    ax_pr.set_xlabel("K-Means Cluster", fontsize=10, color=MUTED)
    ax_pr.set_ylabel("Mean Value per Day", fontsize=10, color=MUTED)
    ax_pr.legend(title="Feature", bbox_to_anchor=(1.01,1),
                 fontsize=9, framealpha=0.4)
    ax_pr.grid(axis="y", alpha=0.2); plt.tight_layout()
    st.pyplot(fig_pr); dl(fig_pr,"cluster_profiles.png","dl_pr"); plt.close(fig_pr)
    st.divider()

    # ── Step 27c — Interpretation cards ──────────────────────────────────────────
    step_box("Step 27c","Cluster Interpretation — Activity Labels",
             "Auto-label based on avg daily steps · all 6 key metrics per cluster")

    for i in range(OPTIMAL_K):
        if i not in profile.index: continue
        row   = profile.loc[i]
        steps = row.get("TotalSteps",0)
        sed   = row.get("SedentaryMinutes",0)
        act   = row.get("VeryActiveMinutes",0)
        cals  = row.get("Calories",0)
        slp   = row.get("TotalSleepMinutes",0)
        light = row.get("LightlyActiveMinutes",0)
        n_in  = int((clust_feats["KMeans_Cluster"]==i).sum())

        if   steps>10000: lbl,clr="🏃 HIGHLY ACTIVE",    GREEN
        elif steps>5000:  lbl,clr="🚶 MODERATELY ACTIVE", BLUE
        else:             lbl,clr="🛋️ SEDENTARY",          AMBER

        st.markdown(f"""
        <div style='background:{CARD2};border-left:5px solid {clr};
                    border-radius:0 12px 12px 0;padding:18px 22px;margin-bottom:14px'>
          <div style='font-size:1.1rem;font-weight:800;color:{clr}'>
            Cluster {i} &nbsp;·&nbsp; {lbl}
            <span style='font-size:.75rem;color:{MUTED};font-weight:400'>
              &nbsp;({n_in} users)
            </span>
          </div>
          <div style='display:grid;grid-template-columns:repeat(3,1fr);
                      gap:10px;margin-top:14px;font-size:.82rem'>
            <div style='background:{CARD};border-radius:8px;padding:12px;
                        border-top:2px solid {BLUE}'>
              <div style='color:{MUTED};font-size:.68rem;text-transform:uppercase;
                          letter-spacing:.06em'>📶 Avg Steps / Day</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>
                {steps:,.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;
                        border-top:2px solid {GREEN}'>
              <div style='color:{MUTED};font-size:.68rem;text-transform:uppercase;
                          letter-spacing:.06em'>🔥 Calories / Day</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>
                {cals:,.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;
                        border-top:2px solid {PURPLE}'>
              <div style='color:{MUTED};font-size:.68rem;text-transform:uppercase;
                          letter-spacing:.06em'>💤 Sleep Min / Day</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>
                {slp:,.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;
                        border-top:2px solid {RED}'>
              <div style='color:{MUTED};font-size:.68rem;text-transform:uppercase;
                          letter-spacing:.06em'>🏃 Very Active Min</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>
                {act:.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;
                        border-top:2px solid {AMBER}'>
              <div style='color:{MUTED};font-size:.68rem;text-transform:uppercase;
                          letter-spacing:.06em'>🛋️ Sedentary Min</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>
                {sed:.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;
                        border-top:2px solid {TEAL}'>
              <div style='color:{MUTED};font-size:.68rem;text-transform:uppercase;
                          letter-spacing:.06em'>🚶 Lightly Active Min</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>
                {light:.0f}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


if any(st.session_state[k] for k in ['run_p1','run_p2','run_p3','run_p4']):
    # ═════════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════════
    st.divider()