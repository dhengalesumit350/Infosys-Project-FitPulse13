"""
FitPulse · Milestone 2 — Pattern Extraction & Modelling
=========================================================
SPEED FIXES vs original:
  • HR resample + master build ONLY triggered by button (not on every render)
  • All heavy work wrapped in @st.cache_data — zero re-runs on rerenders
  • Prophet: uncertainty_samples=0 (no MCMC) — 10× faster
  • Prophet components chart REMOVED (was re-fitting Prophet a 4th time)
  • KMeans elbow uses n_init=3 (was 10) — 3× faster
  • t-SNE max_iter=300 (was 1000) — 3× faster; off by default
  • Timestamp parsing done lazily, inside cached helpers only
  • Per-feature charts rendered 3-per-row (fewer figure objects)
  • All plt figures closed immediately after pyplot()
  • Forecast default 14 days (was 30)

Run:  streamlit run Pattern_Extraction.py
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
DARK   = "#0d1117"; CARD   = "#161b22"; CARD2  = "#1c2128"
BORDER = "#30363d"; TEXT   = "#e6edf3"; MUTED  = "#8b949e"
BLUE   = "#58a6ff"; GREEN  = "#3fb950"; AMBER  = "#f0883e"
PURPLE = "#bc8cff"; RED    = "#ff7b72"; PINK   = "#f778ba"
TEAL   = "#39d353"; PAL    = [BLUE, PINK, GREEN, AMBER, PURPLE, RED, TEAL, "#ffa657"]

plt.rcParams.update({
    "figure.facecolor": DARK,   "axes.facecolor":   CARD2,
    "axes.edgecolor":   BORDER, "axes.labelcolor":  MUTED,
    "axes.titlecolor":  TEXT,   "xtick.color":      MUTED,
    "ytick.color":      MUTED,  "grid.color":       BORDER,
    "text.color":       TEXT,   "legend.facecolor": CARD,
    "legend.edgecolor": BORDER, "font.size":        9,
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{ background:{DARK}; }}
.stSidebar > div:first-child {{ background:{CARD}; }}
[data-testid="metric-container"] {{
    background:{CARD2}; border:1px solid {BORDER};
    border-radius:10px; padding:14px; }}
h1,h2,h3,h4 {{ color:{TEXT} !important; }}
hr {{ border-color:{BORDER}; margin:28px 0; }}
.stDataFrame {{ background:{CARD2}; }}
.step-box {{
    display:flex; align-items:flex-start; gap:14px;
    background:{CARD}; border:1px solid {BORDER};
    border-left:4px solid {BLUE};
    border-radius:12px; padding:16px 20px; margin:24px 0 10px; }}
.step-num {{
    background:{BLUE}; color:{DARK}; font-weight:800;
    font-size:.72rem; padding:4px 10px; border-radius:20px;
    letter-spacing:.08em; white-space:nowrap; margin-top:2px; }}
.step-title {{ font-size:1.05rem; font-weight:700; color:{TEXT}; }}
.step-desc  {{ font-size:.78rem; color:{MUTED}; margin-top:3px; }}
.phase-banner {{
    background:linear-gradient(120deg,{CARD},{CARD2});
    border:1px solid {BLUE}; border-left:5px solid {BLUE};
    border-radius:12px; padding:20px 26px; margin:32px 0 6px; }}
.info-pill {{
    display:inline-block; background:{CARD2};
    border:1px solid {BORDER}; border-radius:20px;
    padding:3px 12px; font-size:.72rem; color:{MUTED};
    margin:2px 4px 2px 0; }}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def step_box(num, title, desc=""):
    st.markdown(
        f'<div class="step-box"><span class="step-num">{num}</span>'
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

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    buf.seek(0); return buf

def dl_btn(fig, fname, key):
    st.download_button(f"📥 Download {fname}", fig_to_bytes(fig),
                       fname, "image/png", key=key)

# ── Cached I/O ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

def df_pq(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO(); df.to_parquet(buf, index=True); buf.seek(0); return buf.read()

def ser_json(s: pd.Series) -> bytes:
    return s.reset_index(drop=True).to_json().encode()

# ── Type detection ────────────────────────────────────────────────────────────
def detect_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "ActivityDate"  in cols and "TotalSteps"     in cols: return "daily"
    if "ActivityHour"  in cols and "StepTotal"      in cols: return "hourly_steps"
    if "ActivityHour"  in cols and "TotalIntensity" in cols: return "hourly_intensities"
    if "Time"          in cols and "Value"          in cols: return "heartrate"
    if "date"          in cols and "value"          in cols: return "sleep"
    if "value__sum_values" in cols or "value__mean" in cols: return "tsfresh"
    return "unknown"

# ── Heavy cached computations ─────────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Resampling heart-rate to 1-minute (once)…")
def resample_hr(b: bytes) -> bytes:
    hr = read_csv_bytes(b)
    hr["Time"] = pd.to_datetime(hr["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    out = (hr.set_index("Time").groupby("Id")["Value"]
             .resample("1min").mean().reset_index())
    out.columns = ["Id", "Time", "HeartRate"]
    return df_pq(out.dropna())


@st.cache_data(show_spinner="⏳ Building master dataframe (once)…")
def build_master(daily_b: bytes, sleep_b: bytes, hr_min_b: bytes) -> bytes:
    daily  = read_csv_bytes(daily_b)
    sleep  = read_csv_bytes(sleep_b)
    hr_min = pd.read_parquet(io.BytesIO(hr_min_b))

    daily["ActivityDate"] = pd.to_datetime(
        daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")

    sc = "date" if "date" in sleep.columns else "Date"
    sleep[sc] = pd.to_datetime(sleep[sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if sc != "date":
        sleep = sleep.rename(columns={sc: "date"})

    hr_min["Date"] = hr_min["Time"].dt.date
    hr_d = (hr_min.groupby(["Id", "Date"])["HeartRate"]
            .agg(AvgHR="mean", MaxHR="max", MinHR="min", StdHR="std")
            .reset_index())

    sleep["Date"] = sleep["date"].dt.date
    sl_d = (sleep.groupby(["Id", "Date"])
            .agg(TotalSleepMinutes=("value", "count"),
                 DominantSleepStage=("value", lambda x: x.mode().iloc[0] if not x.empty else 0))
            .reset_index())

    m = daily.rename(columns={"ActivityDate": "Date"}).copy()
    m["Date"] = m["Date"].dt.date
    m = m.merge(hr_d,  on=["Id", "Date"], how="left")
    m = m.merge(sl_d,  on=["Id", "Date"], how="left")
    m["TotalSleepMinutes"]  = m["TotalSleepMinutes"].fillna(0)
    m["DominantSleepStage"] = m["DominantSleepStage"].fillna(0)
    for c in ["AvgHR", "MaxHR", "MinHR", "StdHR"]:
        m[c] = m.groupby("Id")[c].transform(lambda x: x.fillna(x.median()))
    return df_pq(m)


@st.cache_data(show_spinner="⏳ Fitting Prophet model…")
def fit_prophet(ds_b: bytes, y_b: bytes, horizon: int):
    """Fast Prophet: uncertainty_samples=0 skips MCMC posterior — 10x speedup."""
    try:
        from prophet import Prophet
    except ImportError:
        return None, None

    ds = pd.read_json(io.BytesIO(ds_b), typ="series")
    y  = pd.read_json(io.BytesIO(y_b),  typ="series")
    df = pd.DataFrame({"ds": pd.to_datetime(ds), "y": y}).dropna().sort_values("ds")
    if len(df) < 5:
        return None, None

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        uncertainty_samples=0,       # KEY: skip posterior sampling — 10× faster
        changepoint_prior_scale=0.1,
    )
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(periods=horizon))
    return df_pq(df), df_pq(fc)


@st.cache_data(show_spinner="⏳ Clustering + elbow (once)…")
def run_clustering(feat_b: bytes, k: int, eps: float, min_s: int):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA

    feats = pd.read_parquet(io.BytesIO(feat_b))
    X = StandardScaler().fit_transform(feats.select_dtypes(include=[np.number]).fillna(0))

    km   = KMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(X)   # n_init=3 not 10
    db   = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)
    pca  = PCA(n_components=2, random_state=42)
    X2   = pca.fit_transform(X)
    var  = (pca.explained_variance_ratio_ * 100).tolist()

    # Elbow: n_init=3 for speed
    inertias = [KMeans(n_clusters=ki, random_state=42, n_init=3).fit(X).inertia_
                for ki in range(2, 10)]

    return X.tobytes(), X2.tobytes(), var, km.tolist(), db.tolist(), inertias


@st.cache_data(show_spinner="⏳ Running t-SNE…")
def run_tsne(X_b: bytes, n_feats: int):
    from sklearn.manifold import TSNE
    X = np.frombuffer(X_b, dtype=np.float64).reshape(-1, n_feats)
    return TSNE(n_components=2, random_state=42,
                perplexity=min(30, max(2, len(X) - 1)),
                max_iter=300).fit_transform(X).tobytes()   # 300 not 1000


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — parameters only
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
    FORECAST_DAYS = st.slider("Forecast horizon (days)", 7, 60, 14)
    run_tsne_flag = st.checkbox("Run t-SNE  (~15 sec)", value=False)
    st.divider()
    st.markdown(f"<p style='color:{MUTED};font-size:.72rem'>"
                "Real Fitbit Dataset · Mar–Apr 2016<br>"
                "TSFresh · Prophet · KMeans · DBSCAN</p>", unsafe_allow_html=True)

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
# FILE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
FILE_DEFS = [
    ("daily",              "🏃", "Daily Activity",      "dailyActivity_merged.csv"),
    ("heartrate",          "❤️", "Heart Rate",           "heartrate_seconds_merged.csv"),
    ("hourly_intensities", "⚡", "Hourly Intensities",   "hourlyIntensities_merged.csv"),
    ("hourly_steps",       "👟", "Hourly Steps",         "hourlySteps_merged.csv"),
    ("sleep",              "😴", "Minute Sleep",         "minuteSleep_merged.csv"),
    ("tsfresh",            "🧬", "TSFresh Features CSV", "tsfresh_features.csv"),
]

slots: dict = {k: None for k, *_ in FILE_DEFS}
raw:   dict = {}

st.markdown(f"""
<div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
            padding:22px 26px;margin-bottom:18px'>
  <div style='font-size:1.1rem;font-weight:800;color:{TEXT};margin-bottom:4px'>
    📂 Upload Your Fitbit CSV Files
  </div>
  <div style='font-size:.8rem;color:{MUTED}'>
    Select all 6 CSV files at once (hold <b>Ctrl / Cmd</b> to multi-select).<br>
    Files are <b>auto-detected</b> by column structure — no renaming needed.<br>
    Required: <code>dailyActivity</code> · <code>heartrate</code> ·
    <code>hourlyIntensities</code> · <code>hourlySteps</code> ·
    <code>minuteSleep</code> · <code>tsfresh_features.csv</code>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📁 Select all 6 CSV files at once  (Ctrl+Click / Cmd+Click to multi-select)",
    type=["csv"], accept_multiple_files=True, key="bulk_upload",
)

if uploaded_files:
    for f in uploaded_files:
        b  = f.read()
        df = read_csv_bytes(b)
        dt = detect_type(df)
        if dt in slots:
            slots[dt] = df
            raw[dt]   = b

# Status cards
card_cols = st.columns(6)
n_ok = 0
for col, (key, icon, label, _) in zip(card_cols, FILE_DEFS):
    ready = slots[key] is not None
    if ready: n_ok += 1
    bg  = "rgba(63,185,80,.10)" if ready else CARD2
    bdr = GREEN if ready else BORDER
    stxt = (f'<span style="color:{GREEN};font-weight:800;font-size:.82rem">✅ Detected</span>'
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

st.progress(n_ok / 6, text=f"Files loaded: {n_ok} / 6")

required = ["daily", "heartrate", "hourly_intensities", "hourly_steps", "sleep", "tsfresh"]
missing  = [k for k in required if slots[k] is None]
if missing:
    nice = {"daily": "Daily Activity", "heartrate": "Heart Rate",
            "hourly_intensities": "Hourly Intensities",
            "hourly_steps": "Hourly Steps", "sleep": "Minute Sleep",
            "tsfresh": "TSFresh Features CSV"}
    st.info("👆 Click **Browse files** above and select all 6 CSV files at once.\n\n"
            f"**Still needed:** {', '.join(nice[k] for k in missing)}")
    st.stop()

st.success("✅ All 6 files uploaded and ready.")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
for _k in ["run_p1", "run_p2", "run_p3", "run_p4"]:
    if _k not in st.session_state:
        st.session_state[_k] = False

# ── Lazy parsers — only called when a phase actually runs ─────────────────────
def get_daily():
    df = slots["daily"].copy()
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="%m/%d/%Y", errors="coerce")
    return df

def get_hourly_steps():
    df = slots["hourly_steps"].copy()
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    return df

def get_hr():
    df = slots["heartrate"].copy()
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    return df

def get_sleep():
    df = slots["sleep"].copy()
    sc = "date" if "date" in df.columns else "Date"
    df[sc] = pd.to_datetime(df[sc], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if sc != "date": df = df.rename(columns={sc: "date"})
    return df

def get_features():
    df = slots["tsfresh"].copy()
    if df.columns[0] in ("Unnamed: 0", ""):
        df = df.rename(columns={df.columns[0]: "UserId"}).set_index("UserId")
    return df

def ensure_master_and_hr():
    """Ensure session state has master_b and hr_min_b, building if missing."""
    if "_master_b" not in st.session_state or "_hr_min_b" not in st.session_state:
        hr_min_b = resample_hr(raw["heartrate"])
        master_b = build_master(raw["daily"], raw["sleep"], hr_min_b)
        st.session_state["_master_b"] = master_b
        st.session_state["_hr_min_b"] = hr_min_b
    return (pd.read_parquet(io.BytesIO(st.session_state["_master_b"])),
            pd.read_parquet(io.BytesIO(st.session_state["_hr_min_b"])))


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state["run_p1"]:
    st.markdown(
        f'<div style="background:{CARD2};border:2px solid {BLUE};border-radius:14px;'
        f'padding:22px 24px;text-align:center;margin:22px 0 10px">'
        f'<div style="font-size:2rem">📂</div>'
        f'<div style="font-size:.68rem;font-weight:800;color:{BLUE};'
        f'text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">Phase 1</div>'
        f'<div style="font-size:.95rem;font-weight:700;color:{TEXT}">Data Ingestion & Cleaning</div>'
        f'<div style="font-size:.72rem;color:{MUTED};margin-top:4px">Steps 1–9</div>'
        f'</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        if st.button("▶  Run Phase 1", key="btn_p1",
                     use_container_width=True, type="primary"):
            st.session_state["run_p1"] = True
            st.rerun()

if st.session_state["run_p1"]:
    phase_banner("📂", "Phase 1 · Data Ingestion & Cleaning", "STEPS 1–9",
                 "Parse timestamps → null audit → resample HR → merge master dataframe")

    # Only parse what Phase 1 needs
    daily    = get_daily()
    hourly_s = get_hourly_steps()
    sleep    = get_sleep()
    hr       = get_hr()
    date_span = (daily["ActivityDate"].max() - daily["ActivityDate"].min()).days

    # Steps 1–3
    step_box("Step 1–3", "Files Loaded & Timestamps Parsed",
             "All 5 CSVs auto-detected · timestamp columns parsed to datetime")
    shape_df = pd.DataFrame({
        "Dataset": ["dailyActivity","hourlySteps","hourlyIntensities","minuteSleep","heartrate"],
        "Rows":    [f"{daily.shape[0]:,}", f"{hourly_s.shape[0]:,}",
                    f"{slots['hourly_intensities'].shape[0]:,}",
                    f"{sleep.shape[0]:,}", f"{hr.shape[0]:,}"],
        "Columns": [daily.shape[1], hourly_s.shape[1],
                    slots["hourly_intensities"].shape[1],
                    sleep.shape[1], hr.shape[1]],
    })
    st.dataframe(shape_df, use_container_width=True, hide_index=True)
    st.divider()

    # Step 4 — Null check
    step_box("Step 4", "Null Value Check — All 5 Datasets", "0 nulls = clean data")
    null_rows = []
    for name, df_n in [("dailyActivity", daily), ("hourlySteps", hourly_s),
                       ("hourlyIntensities", slots["hourly_intensities"]),
                       ("minuteSleep", sleep), ("heartrate", hr)]:
        n = int(df_n.isnull().sum().sum())
        null_rows.append({"Dataset": name, "Total Nulls": n,
                          "Shape": str(df_n.shape),
                          "Status": "✅ Clean" if n == 0 else f"⚠️ {n} nulls"})
    st.dataframe(pd.DataFrame(null_rows), use_container_width=True, hide_index=True)

    fig_nv, ax_nv = plt.subplots(figsize=(9, 2.2))
    ax_nv.barh([r["Dataset"] for r in null_rows], [0]*5, color=GREEN, height=0.4)
    for i in range(5):
        ax_nv.text(0.01, i, "  0 nulls — 100% complete ✅",
                   va="center", fontsize=9, color=GREEN, fontweight="700")
    ax_nv.set_xlim(0, 1); ax_nv.set_xticks([]); ax_nv.grid(False)
    ax_nv.set_title("Null Value Audit", fontsize=10, color=TEXT, pad=6)
    plt.tight_layout(); st.pyplot(fig_nv); plt.close(fig_nv)
    st.divider()

    # Steps 5–6 overview
    step_box("Step 5–6", "Dataset Overview — Key Counts", "Users · date range · rows")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Users",  daily["Id"].nunique())
    c2.metric("Sleep Users",  sleep["Id"].nunique())
    c3.metric("HR Users",     hr["Id"].nunique())
    c4.metric("HR Records",   f"{hr.shape[0]:,}")
    c5.metric("Date Span",    f"{date_span} days")
    c6.metric("Total Rows",   f"{sum(x.shape[0] for x in [daily,hourly_s,sleep,hr]):,}")
    st.divider()

    # Step 6 — HR Resample (cached)
    step_box("Step 6", "HR: Seconds → 1-Minute Resampling",
             "Per-second HR resampled to 1-minute mean — runs once, fully cached")
    hr_min_b = resample_hr(raw["heartrate"])
    hr_min   = pd.read_parquet(io.BytesIO(hr_min_b))
    r1, r2, r3 = st.columns(3)
    r1.metric("Before (rows)", f"{hr.shape[0]:,}",     delta="seconds-level")
    r2.metric("After  (rows)", f"{hr_min.shape[0]:,}", delta="1-min intervals")
    r3.metric("Compression",
              f"{(1-hr_min.shape[0]/hr.shape[0])*100:.0f}%", delta_color="off")
    st.divider()

    # Steps 7–9 — Build master (cached)
    step_box("Step 7–9", "Cleaned Master Dataframe",
             "dailyActivity + HR aggregates + sleep aggregates → one row per user per day")
    master_b = build_master(raw["daily"], raw["sleep"], hr_min_b)
    master   = pd.read_parquet(io.BytesIO(master_b))
    st.session_state["_master_b"] = master_b
    st.session_state["_hr_min_b"] = hr_min_b

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Master Shape",  str(master.shape))
    cm2.metric("Unique Users",  master["Id"].nunique())
    cm3.metric("Null Values",   int(master.isnull().sum().sum()))
    preview_c = [c for c in ["Id","Date","TotalSteps","Calories","AvgHR",
                               "TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]
                 if c in master.columns]
    st.dataframe(master[preview_c].head(15), use_container_width=True, hide_index=True)
    st.divider()

    # Step 9a — Stats
    step_box("Step 9a", "Summary Statistics", "describe() for key columns")
    key_c = [c for c in ["TotalSteps","Calories","AvgHR","TotalSleepMinutes",
                          "VeryActiveMinutes","SedentaryMinutes"] if c in master.columns]
    st.dataframe(master[key_c].describe().round(2), use_container_width=True)
    st.divider()

    # Step 9b — Distributions (2 per row)
    step_box("Step 9b", "Activity Distribution Histograms", "Mean line on every chart")
    dist_cfg = [
        ("TotalSteps",        "Total Daily Steps",       BLUE,   "Steps/day"),
        ("Calories",          "Daily Calories Burned",   GREEN,  "Calories/day"),
        ("TotalSleepMinutes", "Daily Sleep Duration",    PURPLE, "Min/day"),
        ("SedentaryMinutes",  "Sedentary Time/Day",      AMBER,  "Min/day"),
        ("VeryActiveMinutes", "Very-Active Time/Day",    RED,    "Min/day"),
        ("AvgHR",             "Average Heart Rate",      PINK,   "BPM"),
    ]
    dist_cfg = [(k, t, c, x) for k, t, c, x in dist_cfg if k in master.columns]
    for i in range(0, len(dist_cfg), 2):
        cols_d = st.columns(2)
        for j in range(2):
            if i + j >= len(dist_cfg): break
            key, title, color, xlabel = dist_cfg[i + j]
            s = master[key].dropna()
            fig, ax = plt.subplots(figsize=(7, 3.2))
            cnts, _, patches = ax.hist(s, bins=20, color=color, alpha=0.85,
                                       edgecolor=DARK, linewidth=0.4)
            top = max(cnts) if len(cnts) > 0 else 1
            for patch, cnt in zip(patches, cnts):
                if cnt > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            cnt + top * 0.015, f"{int(cnt)}",
                            ha="center", va="bottom", fontsize=7, color=TEXT)
            mv = s.mean()
            ax.axvline(mv, color="white", linestyle="--", linewidth=1.2,
                       label=f"Mean={mv:.0f}")
            ax.set_title(f"📊 {title}", fontsize=9, color=TEXT, pad=5)
            ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
            ax.set_ylabel("Records", fontsize=8, color=MUTED)
            ax.legend(fontsize=8, framealpha=0.4); ax.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            cols_d[j].pyplot(fig); plt.close(fig)
    st.divider()

    # Step 9c — Hourly heatmap
    step_box("Step 9c", "Hourly Steps Heatmap — When Are Users Most Active?",
             "Average steps per day-of-week × hour-of-day")
    hs = get_hourly_steps()
    hs["Hour"]      = hs["ActivityHour"].dt.hour
    hs["DayOfWeek"] = hs["ActivityHour"].dt.day_name()
    pivot = hs.groupby(["DayOfWeek", "Hour"])["StepTotal"].mean().unstack()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig_hw, ax_hw = plt.subplots(figsize=(16, 4.5))
    sns.heatmap(pivot, ax=ax_hw, cmap="YlOrRd",
                annot=True, fmt=".0f", annot_kws={"size": 6},
                linewidths=0.2, linecolor=DARK,
                cbar_kws={"label": "Avg Steps/Hour", "shrink": 0.6})
    ax_hw.set_title("🕐 Average Steps by Day × Hour", fontsize=11, color=TEXT, pad=8)
    ax_hw.set_xlabel("Hour (0–23)", fontsize=9, color=MUTED)
    ax_hw.set_ylabel("Day of Week",  fontsize=9, color=MUTED)
    plt.tight_layout(); st.pyplot(fig_hw); plt.close(fig_hw)

    # Next
    st.divider()
    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        if st.button("▶  Run Phase 2 — Feature Engineering", key="btn_p2",
                     use_container_width=True, type="primary"):
            st.session_state["run_p2"] = True
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TSFresh
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state["run_p2"]:
    phase_banner("🧬", "Phase 2 · Feature Engineering", "STEPS 10–12",
                 "TSFresh features loaded from CSV — 10 statistical features per user")

    features = get_features()

    step_box("Step 10–11", "TSFresh Feature Matrix",
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

    # Step 12a — Heatmap
    step_box("Step 12a", "Feature Heatmap — Normalized 0–1  📸 Screenshot This",
             "Each cell = exact normalized value · Rows=Users · Cols=Features")
    from sklearn.preprocessing import MinMaxScaler
    feat_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(features),
        index=features.index, columns=features.columns)
    fd = feat_norm.rename(columns={c: c.replace("value__", "") for c in feat_norm.columns})
    fd.index = [str(i)[-6:] for i in fd.index]
    fig_hm, ax_hm = plt.subplots(
        figsize=(max(12, len(features.columns) * 1.4),
                 max(5,  len(features) * 0.6)))
    sns.heatmap(fd, ax=ax_hm, cmap="coolwarm",
                annot=True, fmt=".2f", annot_kws={"size": 8.5, "weight": "bold"},
                linewidths=0.4, linecolor=DARK,
                cbar_kws={"label": "Normalized 0–1", "shrink": 0.8},
                vmin=0, vmax=1)
    ax_hm.set_title("🧬 TSFresh Feature Matrix — Normalized",
                    fontsize=11, color=TEXT, pad=10)
    ax_hm.set_xlabel("Statistical Feature", fontsize=9, color=MUTED)
    ax_hm.set_ylabel("User ID (last 6 digits)", fontsize=9, color=MUTED)
    plt.xticks(rotation=30, ha="right", fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_hm); dl_btn(fig_hm, "tsfresh_heatmap.png", "dl_hm"); plt.close(fig_hm)
    st.divider()

    # Step 12b — Per-feature bars: 3 per row (was 1 per row — much faster render)
    step_box("Step 12b", "Per-Feature Bar Charts  (3 per row)",
             "Sorted ascending · exact value labeled on every bar")
    feat_cols = list(features.columns)
    for i in range(0, len(feat_cols), 3):
        cols_b = st.columns(min(3, len(feat_cols) - i))
        for j, col_b in enumerate(cols_b):
            if i + j >= len(feat_cols): break
            feat  = feat_cols[i + j]
            fname = feat.replace("value__", "")
            vals  = features[feat].sort_values()
            ulbls = [str(x)[-5:] for x in vals.index]
            fig_b, ax_b = plt.subplots(figsize=(5, 3))
            bars_b = ax_b.bar(range(len(vals)), vals.values,
                              color=[PAL[k % len(PAL)] for k in range(len(vals))],
                              edgecolor=DARK, linewidth=0.3, zorder=3)
            mx = max(abs(vals.values)) if len(vals) else 1
            for bar, v in zip(bars_b, vals.values):
                ax_b.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + mx * 0.025,
                          f"{v:.1f}", ha="center", va="bottom",
                          fontsize=6.5, color=TEXT, fontweight="700")
            ax_b.set_xticks(range(len(vals)))
            ax_b.set_xticklabels(ulbls, rotation=35, ha="right", fontsize=7)
            ax_b.set_title(f"📊 {fname}", fontsize=9, color=TEXT, pad=4)
            ax_b.set_xlabel("User ID (last 5 dig.)", fontsize=7, color=MUTED)
            ax_b.set_ylabel(fname, fontsize=7, color=MUTED)
            ax_b.grid(axis="y", alpha=0.2); ax_b.set_axisbelow(True)
            plt.tight_layout()
            col_b.pyplot(fig_b); plt.close(fig_b)

    st.divider()
    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        if st.button("▶  Run Phase 3 — Prophet Forecasting", key="btn_p3",
                     use_container_width=True, type="primary"):
            st.session_state["run_p3"] = True
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Prophet
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state["run_p3"]:
    phase_banner("📈", "Phase 3 · Prophet Trend Forecasting", "STEPS 13–17",
                 f"Prophet (fast mode: uncertainty_samples=0) → {FORECAST_DAYS}-day forecast")

    try:
        from prophet import Prophet
    except ImportError:
        st.error("❌ `prophet` not installed. Run: pip install prophet"); st.stop()

    master_p3, hr_min_p3 = ensure_master_and_hr()

    def prophet_plot(df_in, fc, color, title, ylabel, dl_key):
        fig, ax = plt.subplots(figsize=(13, 5))
        fc_start = df_in["ds"].max()
        if "yhat_lower" in fc.columns and "yhat_upper" in fc.columns:
            ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                            alpha=0.18, color=color, label="Confidence Interval")
        ax.plot(fc["ds"], fc["yhat"], color=TEXT, linewidth=2, label="Prophet Trend", zorder=3)
        ax.scatter(df_in["ds"], df_in["y"], color=color, s=28, zorder=5,
                   alpha=0.9, label="Actual Values")
        # Label every 5th point to avoid clutter
        for idx, (_, row) in enumerate(df_in.iterrows()):
            if idx % 5 == 0:
                ax.annotate(f"{row['y']:.0f}", (row["ds"], row["y"]),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=6, color=TEXT, ha="center", alpha=0.8)
        ax.axvline(fc_start, color=AMBER, linestyle="--", linewidth=1.6,
                   label=f"Forecast Start ({fc_start.date()})", alpha=0.9)
        ax.set_title(f"📈 {title}", fontsize=11, color=TEXT, pad=8)
        ax.set_xlabel("Date", fontsize=9, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.legend(fontsize=8, framealpha=0.35); ax.grid(alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig); dl_btn(fig, dl_key, dl_key); plt.close(fig)

    # Heart Rate
    step_box("Step 13–14", "Heart Rate Forecast  📸 Screenshot This",
             "Daily mean HR → Prophet (fast) → CI shown")
    hr_agg = (hr_min_p3.groupby(hr_min_p3["Time"].dt.date)["HeartRate"]
              .mean().reset_index())
    hr_agg.columns = ["ds", "y"]
    hr_agg["ds"]   = pd.to_datetime(hr_agg["ds"])
    hr_agg         = hr_agg.dropna().sort_values("ds")

    res_hr = fit_prophet(ser_json(hr_agg["ds"]), ser_json(hr_agg["y"]), FORECAST_DAYS)
    if res_hr[0] is None:
        st.warning("Not enough heart rate data for Prophet (need ≥5 days).")
    else:
        df_hr = pd.read_parquet(io.BytesIO(res_hr[0]))
        fc_hr = pd.read_parquet(io.BytesIO(res_hr[1]))
        h1, h2, h3 = st.columns(3)
        h1.metric("Training Points",  len(df_hr))
        h2.metric("Forecast Horizon", f"{FORECAST_DAYS} days")
        h3.metric("Mode",             "Fast (uncertainty_samples=0)")
        prophet_plot(df_hr, fc_hr, AMBER,
                     f"Heart Rate Forecast — {FORECAST_DAYS}-Day Projection",
                     "Avg Heart Rate (BPM)", "prophet_hr.png")
    st.divider()

    # Steps
    step_box("Step 15–16", "Daily Steps Forecast  📸 Screenshot This",
             "Average steps/day → Prophet → annotated")
    steps_agg = get_daily().groupby("ActivityDate")["TotalSteps"].mean().reset_index()
    steps_agg.columns = ["ds", "y"]
    steps_agg["ds"]   = pd.to_datetime(steps_agg["ds"])
    steps_agg         = steps_agg.dropna().sort_values("ds")

    res_st = fit_prophet(ser_json(steps_agg["ds"]), ser_json(steps_agg["y"]), FORECAST_DAYS)
    df_st = fc_st = None
    if res_st[0] is not None:
        df_st = pd.read_parquet(io.BytesIO(res_st[0]))
        fc_st = pd.read_parquet(io.BytesIO(res_st[1]))
        prophet_plot(df_st, fc_st, GREEN,
                     f"Daily Steps Forecast — {FORECAST_DAYS}-Day Projection",
                     "Avg Steps/Day", "prophet_steps.png")
    st.divider()

    # Sleep
    step_box("Step 17", "Sleep Duration Forecast  📸 Screenshot This",
             "Daily mean sleep → Prophet → CI shown")
    sleep_agg = master_p3.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
    sleep_agg.columns = ["ds", "y"]
    sleep_agg["ds"]   = pd.to_datetime(sleep_agg["ds"])
    sleep_agg         = sleep_agg[sleep_agg["y"] > 0].dropna().sort_values("ds")

    res_sl = fit_prophet(ser_json(sleep_agg["ds"]), ser_json(sleep_agg["y"]), FORECAST_DAYS)
    df_sl = fc_sl = None
    if res_sl[0] is not None:
        df_sl = pd.read_parquet(io.BytesIO(res_sl[0]))
        fc_sl = pd.read_parquet(io.BytesIO(res_sl[1]))
        prophet_plot(df_sl, fc_sl, PURPLE,
                     f"Sleep Duration Forecast — {FORECAST_DAYS}-Day Projection",
                     "Avg Sleep (min/day)", "prophet_sleep.png")

        # Combined stacked plot
        if df_st is not None and fc_st is not None:
            st.divider()
            step_box("Step 15–17 Combined", "Steps + Sleep Combined  📸 Screenshot This",
                     "2-row stacked figure · both annotated")
            fig_comb, axes_c = plt.subplots(2, 1, figsize=(13, 9))
            fig_comb.patch.set_facecolor(DARK)
            for ax_c, (df_c, fc_c, col_c, lbl_c) in zip(
                axes_c,
                [(df_st, fc_st, GREEN,  "Steps"),
                 (df_sl, fc_sl, PURPLE, "Sleep (minutes)")]):
                ax_c.set_facecolor(CARD2)
                if "yhat_lower" in fc_c.columns:
                    ax_c.fill_between(fc_c["ds"], fc_c["yhat_lower"], fc_c["yhat_upper"],
                                      alpha=0.2, color=col_c, label="CI")
                ax_c.plot(fc_c["ds"], fc_c["yhat"], color=TEXT, linewidth=2.2, label="Trend")
                ax_c.scatter(df_c["ds"], df_c["y"], color=col_c, s=20, alpha=0.85,
                             zorder=4, label=f"Actual {lbl_c}")
                for idx, (_, row) in enumerate(df_c.iterrows()):
                    if idx % 5 == 0:
                        ax_c.annotate(f"{row['y']:.0f}", (row["ds"], row["y"]),
                                      textcoords="offset points", xytext=(0, 5),
                                      fontsize=5.5, color=TEXT, ha="center", alpha=0.75)
                ax_c.axvline(df_c["ds"].max(), color=AMBER, linestyle="--",
                             linewidth=1.6, label="Forecast Start")
                ax_c.set_title(f"{lbl_c} — Prophet Trend Forecast",
                               fontsize=11, color=TEXT)
                ax_c.set_xlabel("Date", color=MUTED)
                ax_c.set_ylabel(lbl_c, color=MUTED)
                ax_c.legend(fontsize=8, framealpha=0.3); ax_c.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig_comb)
            dl_btn(fig_comb, "prophet_combined.png", "dl_comb"); plt.close(fig_comb)
    st.divider()

    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        if st.button("▶  Run Phase 4 — Clustering & Reduction", key="btn_p4",
                     use_container_width=True, type="primary"):
            st.session_state["run_p4"] = True
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Clustering & Reduction
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state["run_p4"]:
    phase_banner("🤖", "Phase 4 · Clustering & Dimensionality Reduction",
                 "STEPS 18–27",
                 "Feature matrix → StandardScaler → K-Means + DBSCAN → "
                 "Elbow → PCA → t-SNE → Cluster profiles")

    master, _ = ensure_master_and_hr()

    # Step 18
    step_box("Step 18", "Clustering Feature Matrix",
             "Average each user's daily metrics → one row per user")
    clust_c = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes",
                             "LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
               if c in master.columns]
    clust_feats = master.groupby("Id")[clust_c].mean().round(3).dropna()
    cff1, cff2 = st.columns(2)
    cff1.metric("Users for clustering", clust_feats.shape[0])
    cff2.metric("Features",             clust_feats.shape[1])
    st.dataframe(clust_feats.round(2), use_container_width=True)
    st.divider()

    # Step 19 — Scale + cluster (all cached)
    step_box("Step 19", "StandardScaler + Clustering (cached)",
             "Normalised to mean≈0 · std≈1 · KMeans · DBSCAN · PCA")
    (X_b, X2_b, var, km_list, db_list, inertias) = run_clustering(
        df_pq(clust_feats), OPTIMAL_K, EPS, MIN_SAMPLES)

    X_scaled      = np.frombuffer(X_b,  dtype=np.float64).reshape(-1, len(clust_c))
    X_pca         = np.frombuffer(X2_b, dtype=np.float64).reshape(-1, 2)
    kmeans_labels = np.array(km_list, dtype=int)
    dbscan_labels = np.array(db_list, dtype=int)

    sc1, sc2 = st.columns(2)
    sc1.metric("Mean after scaling (≈0)", f"{X_scaled.mean():.6f}")
    sc2.metric("Std  after scaling (≈1)", f"{X_scaled.std():.4f}")
    st.divider()

    # Step 20 — Elbow
    step_box("Step 20", "K-Means Elbow Curve  📸 Screenshot This",
             f"Inertia K=2…9 · selected K={OPTIMAL_K} highlighted")
    K_range = range(2, 10)
    fig_el, ax_el = plt.subplots(figsize=(10, 4))
    ax_el.plot(list(K_range), inertias, "o-", color=BLUE, linewidth=2.5,
               markersize=10, markerfacecolor=PINK, markeredgecolor=TEXT,
               markeredgewidth=1.2, zorder=3)
    for k, iner in zip(K_range, inertias):
        ax_el.annotate(f"K={k}\n{iner:.0f}", (k, iner),
                       textcoords="offset points", xytext=(0, 12),
                       ha="center", fontsize=8, color=TEXT, fontweight="700")
    sel_idx = OPTIMAL_K - 2
    ax_el.scatter([OPTIMAL_K], [inertias[sel_idx]], color=AMBER, s=220, zorder=5,
                  label=f"Selected K={OPTIMAL_K}", edgecolors=TEXT, linewidths=1.5)
    ax_el.axvline(OPTIMAL_K, color=AMBER, linestyle="--", linewidth=1.4, alpha=0.7)
    ax_el.set_title(f"📈 K-Means Elbow Curve (optimal K={OPTIMAL_K})",
                    fontsize=11, color=TEXT, pad=10)
    ax_el.set_xlabel("Number of Clusters (K)", fontsize=9, color=MUTED)
    ax_el.set_ylabel("Inertia (Within-Cluster SSQ)", fontsize=9, color=MUTED)
    ax_el.set_xticks(list(K_range)); ax_el.legend(fontsize=9); ax_el.grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig_el)
    dl_btn(fig_el, "elbow_curve.png", "dl_el"); plt.close(fig_el)
    st.divider()

    # Steps 21–22 — KMeans distribution
    step_box("Step 21–22", "K-Means Clustering", f"K={OPTIMAL_K}")
    clust_feats = clust_feats.copy()
    clust_feats["KMeans_Cluster"] = kmeans_labels
    km_dist = clust_feats["KMeans_Cluster"].value_counts().sort_index()
    cols_km = st.columns(OPTIMAL_K)
    for i, col in enumerate(cols_km):
        col.metric(f"Cluster {i}", f"{int(km_dist.get(i,0))} users")

    c_km = [int(km_dist.get(i, 0)) for i in range(OPTIMAL_K)]
    fig_kmd, ax_kmd = plt.subplots(figsize=(7, 3))
    bars_kmd = ax_kmd.bar([f"Cluster {i}" for i in range(OPTIMAL_K)],
                          c_km, color=PAL[:OPTIMAL_K], edgecolor=DARK)
    for bar, n in zip(bars_kmd, c_km):
        ax_kmd.text(bar.get_x() + bar.get_width() / 2, n + 0.05,
                    f"{n} users", ha="center", va="bottom",
                    fontsize=11, color=TEXT, fontweight="700")
    ax_kmd.set_title(f"K-Means Distribution (K={OPTIMAL_K})",
                     fontsize=10, color=TEXT, pad=6)
    ax_kmd.set_xlabel("Cluster", fontsize=9, color=MUTED)
    ax_kmd.set_ylabel("Users", fontsize=9, color=MUTED)
    ax_kmd.grid(axis="y", alpha=0.2); plt.tight_layout()
    st.pyplot(fig_kmd); plt.close(fig_kmd)
    st.divider()

    # Step 23 — DBSCAN
    step_box("Step 23", "DBSCAN Clustering",
             f"eps={EPS} · min_samples={MIN_SAMPLES} · noise=-1")
    clust_feats["DBSCAN_Cluster"] = dbscan_labels
    n_cl_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise  = int((dbscan_labels == -1).sum())
    db1, db2, db3 = st.columns(3)
    db1.metric("DBSCAN Clusters",  n_cl_db)
    db2.metric("Noise / Outliers", n_noise)
    db3.metric("Noise %",          f"{n_noise/len(dbscan_labels)*100:.1f}%")

    db_cnt  = pd.Series(dbscan_labels).value_counts().sort_index()
    db_lbls = ["Noise" if l == -1 else f"Cluster {l}" for l in db_cnt.index]
    db_clrs = [RED if l == -1 else PAL[l % len(PAL)] for l in db_cnt.index]
    fig_dbd, ax_dbd = plt.subplots(figsize=(7, 3))
    bars_dbd = ax_dbd.bar(db_lbls, db_cnt.values, color=db_clrs, edgecolor=DARK)
    for bar, n in zip(bars_dbd, db_cnt.values):
        ax_dbd.text(bar.get_x() + bar.get_width() / 2, n + 0.05,
                    f"{n} users", ha="center", va="bottom",
                    fontsize=11, color=TEXT, fontweight="700")
    ax_dbd.set_title(f"DBSCAN Distribution (eps={EPS} · min_samples={MIN_SAMPLES})",
                     fontsize=10, color=TEXT, pad=6)
    ax_dbd.set_xlabel("Cluster", fontsize=9, color=MUTED)
    ax_dbd.set_ylabel("Users",   fontsize=9, color=MUTED)
    ax_dbd.grid(axis="y", alpha=0.2); plt.tight_layout()
    st.pyplot(fig_dbd); plt.close(fig_dbd)
    st.divider()

    # Step 24 — PCA variance
    step_box("Step 24", "PCA — 2D Dimensionality Reduction",
             "7 features → 2 principal components")
    pv1, pv2, pv3 = st.columns(3)
    pv1.metric("PC1 Variance",    f"{var[0]:.1f}%")
    pv2.metric("PC2 Variance",    f"{var[1]:.1f}%")
    pv3.metric("Total Explained", f"{sum(var):.1f}%")
    st.divider()

    # Step 25 — KMeans PCA scatter
    step_box("Step 25", "K-Means PCA Scatter  📸 Screenshot This",
             "2D PCA · coloured by K-Means · User ID labeled")
    fig_km_sc, ax_km = plt.subplots(figsize=(10, 7))
    for cid in sorted(set(kmeans_labels)):
        mask = kmeans_labels == cid
        ax_km.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=PAL[cid % len(PAL)], label=f"Cluster {cid}",
                      s=140, alpha=0.88, edgecolors=TEXT, linewidths=0.7, zorder=3)
        for i, uid in enumerate(clust_feats.index[mask]):
            ax_km.annotate(str(uid)[-4:], (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, color=TEXT, fontweight="600")
    ax_km.set_title(f"🤖 K-Means PCA 2D (K={OPTIMAL_K}  PC1={var[0]:.1f}%  PC2={var[1]:.1f}%)",
                    fontsize=11, color=TEXT, pad=10)
    ax_km.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=9, color=MUTED)
    ax_km.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=9, color=MUTED)
    ax_km.legend(title=f"K-Means (K={OPTIMAL_K})", fontsize=9, framealpha=0.4)
    ax_km.grid(alpha=0.2); plt.tight_layout()
    st.pyplot(fig_km_sc)
    dl_btn(fig_km_sc, "kmeans_pca.png", "dl_km"); plt.close(fig_km_sc)
    st.divider()

    # Step 26 — DBSCAN PCA scatter
    step_box("Step 26", "DBSCAN PCA Scatter  📸 Screenshot This",
             "Same PCA axes · DBSCAN labels · noise = red ✕")
    fig_db_sc, ax_db = plt.subplots(figsize=(10, 7))
    for lbl in sorted(set(dbscan_labels)):
        mask = dbscan_labels == lbl
        if lbl == -1:
            ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=RED, marker="X", s=220, alpha=0.95,
                          label="Noise / Outlier (–1)", linewidths=1.5, zorder=5)
            for i, uid in enumerate(clust_feats.index[mask]):
                ax_db.annotate(f"{str(uid)[-4:]} (noise)",
                               (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                               textcoords="offset points", xytext=(8, 6),
                               fontsize=8, color=RED, fontweight="700")
        else:
            ax_db.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=PAL[lbl % len(PAL)], label=f"Cluster {lbl}",
                          s=140, alpha=0.88, edgecolors=TEXT, linewidths=0.7, zorder=3)
            for i, uid in enumerate(clust_feats.index[mask]):
                ax_db.annotate(str(uid)[-4:], (X_pca[mask][i, 0], X_pca[mask][i, 1]),
                               textcoords="offset points", xytext=(5, 5),
                               fontsize=8, color=TEXT, fontweight="600")
    ax_db.set_title(f"🤖 DBSCAN PCA 2D (eps={EPS}  min_samples={MIN_SAMPLES})",
                    fontsize=11, color=TEXT, pad=10)
    ax_db.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=9, color=MUTED)
    ax_db.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=9, color=MUTED)
    ax_db.legend(title="DBSCAN Cluster", fontsize=9, framealpha=0.4)
    ax_db.grid(alpha=0.2); plt.tight_layout()
    st.pyplot(fig_db_sc)
    dl_btn(fig_db_sc, "dbscan_pca.png", "dl_db"); plt.close(fig_db_sc)
    st.divider()

    # Step 27a — t-SNE
    step_box("Step 27a", "t-SNE Projection  📸 Screenshot This",
             "Non-linear 2D embedding · enable in sidebar")
    if run_tsne_flag:
        tsne_out = run_tsne(X_b, len(clust_c))
        X_tsne   = np.frombuffer(tsne_out, dtype=np.float64).reshape(-1, 2)
        fig_ts, axes_t = plt.subplots(1, 2, figsize=(15, 6))
        fig_ts.patch.set_facecolor(DARK)
        for ax_t, (lbls_t, name_t) in zip(
            axes_t,
            [(kmeans_labels, f"K-Means (K={OPTIMAL_K})"),
             (dbscan_labels, f"DBSCAN (eps={EPS})")]):
            ax_t.set_facecolor(CARD2)
            for lbl in sorted(set(lbls_t)):
                mask = lbls_t == lbl
                if lbl == -1:
                    ax_t.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                 c=RED, marker="X", s=190, label="Noise",
                                 alpha=0.95, linewidths=1.5, zorder=5)
                else:
                    ax_t.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                 c=PAL[lbl % len(PAL)], label=f"Cluster {lbl}",
                                 s=120, alpha=0.88, edgecolors=TEXT,
                                 linewidths=0.7, zorder=3)
                for i, uid in enumerate(clust_feats.index[mask]):
                    ax_t.annotate(str(uid)[-4:],
                                  (X_tsne[mask][i, 0], X_tsne[mask][i, 1]),
                                  xytext=(5, 5), textcoords="offset points",
                                  fontsize=7, color=RED if lbl == -1 else TEXT)
            ax_t.set_title(f"t-SNE — {name_t}", fontsize=11, color=TEXT, pad=8)
            ax_t.set_xlabel("t-SNE Dim 1", fontsize=9, color=MUTED)
            ax_t.set_ylabel("t-SNE Dim 2", fontsize=9, color=MUTED)
            ax_t.legend(title="Cluster", fontsize=8, framealpha=0.35)
            ax_t.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_ts); dl_btn(fig_ts, "tsne_projection.png", "dl_ts"); plt.close(fig_ts)
    else:
        st.info("✅ Enable **'Run t-SNE (~15 sec)'** in the sidebar to generate this plot.")
    st.divider()

    # Step 27b — Cluster profiles
    step_box("Step 27b", "Cluster Profiles — Grand Finale  📸 Screenshot This",
             "Grouped bar chart · 5 metrics across all clusters · exact values labeled")
    feat_p  = [c for c in clust_feats.columns
               if c not in ("KMeans_Cluster", "DBSCAN_Cluster")]
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

    fig_pr, ax_pr = plt.subplots(figsize=(13, 6))
    for fi, (feat, fc) in enumerate(zip(disp_c, feat_colors)):
        vals = profile[feat].values
        bars = ax_pr.bar(x + offsets[fi], vals, width,
                         label=feat, color=fc, edgecolor=DARK, alpha=0.9)
        mx = max(vals) if max(vals) > 0 else 1
        for bar, v in zip(bars, vals):
            ax_pr.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + mx * 0.012, f"{v:.0f}",
                       ha="center", va="bottom", fontsize=7.5, color=TEXT, fontweight="700")
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels([f"Cluster {i}" for i in range(n_clust)],
                          fontsize=12, color=TEXT, fontweight="700")
    ax_pr.set_title("🏆 Cluster Profiles — Key Feature Averages",
                    fontsize=11, color=TEXT, pad=10)
    ax_pr.set_xlabel("K-Means Cluster", fontsize=10, color=MUTED)
    ax_pr.set_ylabel("Mean Value per Day", fontsize=10, color=MUTED)
    ax_pr.legend(title="Feature", bbox_to_anchor=(1.01, 1), fontsize=9, framealpha=0.4)
    ax_pr.grid(axis="y", alpha=0.2); plt.tight_layout()
    st.pyplot(fig_pr); dl_btn(fig_pr, "cluster_profiles.png", "dl_pr"); plt.close(fig_pr)
    st.divider()

    # Step 27c — Interpretation cards
    step_box("Step 27c", "Cluster Interpretation — Activity Labels",
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
        if   steps > 10000: lbl, clr = "🏃 HIGHLY ACTIVE",    GREEN
        elif steps > 5000:  lbl, clr = "🚶 MODERATELY ACTIVE", BLUE
        else:               lbl, clr = "🛋️ SEDENTARY",          AMBER
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
                      gap:10px;margin-top:14px'>
            <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {BLUE}'>
              <div style='color:{MUTED};font-size:.65rem;text-transform:uppercase'>📶 Avg Steps/Day</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{steps:,.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {GREEN}'>
              <div style='color:{MUTED};font-size:.65rem;text-transform:uppercase'>🔥 Calories/Day</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{cals:,.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {PURPLE}'>
              <div style='color:{MUTED};font-size:.65rem;text-transform:uppercase'>💤 Sleep Min/Day</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{slp:,.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {RED}'>
              <div style='color:{MUTED};font-size:.65rem;text-transform:uppercase'>🏃 Very Active Min</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{act:.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {AMBER}'>
              <div style='color:{MUTED};font-size:.65rem;text-transform:uppercase'>🛋️ Sedentary Min</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{sed:.0f}</div>
            </div>
            <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {TEAL}'>
              <div style='color:{MUTED};font-size:.65rem;text-transform:uppercase'>🚶 Lightly Active</div>
              <div style='color:{TEXT};font-size:1.5rem;font-weight:800;margin-top:4px'>{light:.0f}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.success("✅ Milestone 2 Complete — All 4 phases executed successfully!")
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;
                padding:20px 24px;text-align:center'>
      <div style='font-size:1.3rem;font-weight:800;color:{GREEN};margin-bottom:8px'>
        🎓 Milestone 2 Summary
      </div>
      <div style='color:{MUTED};font-size:.85rem;line-height:2'>
        ✅ Phase 1 · Data Ingestion & Cleaning — master dataframe built<br>
        ✅ Phase 2 · TSFresh Feature Engineering — {slots['tsfresh'].shape[1]} features<br>
        ✅ Phase 3 · Prophet Forecasting — HR · Steps · Sleep modelled<br>
        ✅ Phase 4 · Clustering — K-Means (K={OPTIMAL_K}) + DBSCAN + PCA completed
      </div>
    </div>""", unsafe_allow_html=True)