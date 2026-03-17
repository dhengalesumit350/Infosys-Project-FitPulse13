"""
FitPulse · Combined App — Milestone 1 & 2
==========================================
Sidebar navigation switches between:
  📋 Milestone 1 — Data Governance & Cleaning
  🧬 Milestone 2 — ML Pipeline (TSFresh · Prophet · Clustering)
Run:  streamlit run fitpulse_combined.py
"""
import io, warnings, logging, time
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

st.set_page_config(
    page_title="FitPulse · Pipeline",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
DARK=  "#0d1117"; CARD=  "#161b22"; CARD2= "#1c2128"; BORDER="#30363d"
TEXT=  "#e6edf3"; MUTED= "#8b949e"; BLUE=  "#58a6ff"; GREEN= "#3fb950"
AMBER= "#f0883e"; PURPLE="#bc8cff"; RED=   "#ff7b72"; PINK=  "#f778ba"
TEAL=  "#39d353"; PAL=   [BLUE,PINK,GREEN,AMBER,PURPLE,RED,TEAL,"#ffa657"]

plt.rcParams.update({
    "figure.facecolor":DARK,  "axes.facecolor":CARD2,  "axes.edgecolor":BORDER,
    "axes.labelcolor":MUTED,  "axes.titlecolor":TEXT,  "xtick.color":MUTED,
    "ytick.color":MUTED,      "grid.color":BORDER,     "text.color":TEXT,
    "legend.facecolor":CARD,  "legend.edgecolor":BORDER, "font.size":9,
})

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp{{background:{DARK}}}
.stSidebar>div:first-child{{background:{CARD}}}
[data-testid="metric-container"]{{background:{CARD2};border:1px solid {BORDER};border-radius:10px;padding:14px}}
h1,h2,h3,h4{{color:{TEXT}!important}}
hr{{border-color:{BORDER};margin:28px 0}}
.stDataFrame{{background:{CARD2}}}
.stAlert{{border-radius:8px}}
.stButton>button{{border-radius:8px;font-weight:bold;transition:0.3s}}
.glass-card{{background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);border:1px solid rgba(56,189,248,0.2);border-radius:16px;padding:24px;margin-bottom:24px;box-shadow:0 4px 30px rgba(0,0,0,0.1)}}
.status-badge{{padding:10px 15px;border-radius:8px;font-weight:700;font-size:.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}}
.pending{{background:#1e293b;color:#64748b;border:1px solid #334155}}
.complete{{background:#064e3b;color:#10b981;border:1px solid #059669}}
.step-box{{display:flex;align-items:flex-start;gap:14px;background:{CARD};border:1px solid {BORDER};border-left:4px solid {BLUE};border-radius:12px;padding:16px 20px;margin:24px 0 10px}}
.step-num{{background:{BLUE};color:{DARK};font-weight:800;font-size:.72rem;padding:4px 10px;border-radius:20px;letter-spacing:.08em;white-space:nowrap;margin-top:2px}}
.step-title{{font-size:1.05rem;font-weight:700;color:{TEXT}}}
.step-desc{{font-size:.78rem;color:{MUTED};margin-top:3px}}
.phase-banner{{background:linear-gradient(120deg,{CARD},{CARD2});border:1px solid {BLUE};border-left:5px solid {BLUE};border-radius:12px;padding:20px 26px;margin:32px 0 6px}}
.info-pill{{display:inline-block;background:{CARD2};border:1px solid {BORDER};border-radius:20px;padding:3px 12px;font-size:.72rem;color:{MUTED};margin:2px 4px 2px 0}}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "active_milestone":"milestone2",
    "m1_raw_df":None,"m1_clean_df":None,
    "m1_ingested":False,"m1_processed":False,
    "run_p1":False,"run_p2":False,"run_p3":False,"run_p4":False,
}
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Milestone 2 helpers ────────────────────────────────────────────────────────
def step_box(num,title,desc=""):
    st.markdown(
        f'<div class="step-box"><span class="step-num">{num}</span>'
        f'<div><div class="step-title">{title}</div>'
        f'<div class="step-desc">{desc}</div></div></div>',
        unsafe_allow_html=True)

def phase_banner(icon,title,steps,desc):
    st.markdown(
        f'<div class="phase-banner">'
        f'<div style="font-size:.65rem;font-weight:800;letter-spacing:.15em;color:{BLUE};text-transform:uppercase;margin-bottom:6px">{steps}</div>'
        f'<div style="font-size:1.4rem;font-weight:800;color:{TEXT}">{icon} {title}</div>'
        f'<div style="color:{MUTED};font-size:.82rem;margin-top:4px">{desc}</div>'
        f'</div>',unsafe_allow_html=True)

def fig_bytes(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=DARK,edgecolor="none")
    buf.seek(0); return buf

def dl(fig,fname,key):
    st.download_button(f"📥 Download {fname}",fig_bytes(fig),fname,"image/png",key=key)

@st.cache_data(show_spinner=False)
def read_csv(b:bytes)->pd.DataFrame: return pd.read_csv(io.BytesIO(b))

def detect_type(df):
    cols=set(df.columns)
    if "ActivityDate" in cols and "TotalSteps" in cols:      return "daily"
    if "ActivityHour" in cols and "StepTotal" in cols:       return "hourly_steps"
    if "ActivityHour" in cols and "TotalIntensity" in cols:  return "hourly_intensities"
    if "Time" in cols and "Value" in cols:                   return "heartrate"
    if "date" in cols and "value" in cols:                   return "sleep"
    if "value__sum_values" in cols or "value__mean" in cols: return "tsfresh"
    return "unknown"

def df_pq(df):
    buf=io.BytesIO(); df.to_parquet(buf); buf.seek(0); return buf.read()

def ser_json(s): return s.reset_index(drop=True).to_json().encode()

@st.cache_data(show_spinner="Resampling HR to 1-min…")
def resample_hr(b):
    hr=read_csv(b)
    hr["Time"]=pd.to_datetime(hr["Time"],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
    out=(hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index())
    out.columns=["Id","Time","HeartRate"]; return df_pq(out.dropna())

@st.cache_data(show_spinner="Building master dataframe…")
def build_master(daily_b,sleep_b,hr_min_b):
    daily=read_csv(daily_b); sleep=read_csv(sleep_b)
    hr_min=pd.read_parquet(io.BytesIO(hr_min_b))
    daily["ActivityDate"]=pd.to_datetime(daily["ActivityDate"],format="%m/%d/%Y",errors="coerce")
    sc="date" if "date" in sleep.columns else "Date"
    sleep[sc]=pd.to_datetime(sleep[sc],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
    if sc!="date": sleep=sleep.rename(columns={sc:"date"})
    hr_min["Date"]=hr_min["Time"].dt.date
    hr_d=(hr_min.groupby(["Id","Date"])["HeartRate"]
          .agg(AvgHR="mean",MaxHR="max",MinHR="min",StdHR="std").reset_index())
    sleep["Date"]=sleep["date"].dt.date
    sl_d=(sleep.groupby(["Id","Date"])
          .agg(TotalSleepMinutes=("value","count"),
               DominantSleepStage=("value",lambda x:x.mode()[0])).reset_index())
    m=daily.rename(columns={"ActivityDate":"Date"}).copy()
    m["Date"]=m["Date"].dt.date
    m=m.merge(hr_d,on=["Id","Date"],how="left").merge(sl_d,on=["Id","Date"],how="left")
    m["TotalSleepMinutes"]=m["TotalSleepMinutes"].fillna(0)
    m["DominantSleepStage"]=m["DominantSleepStage"].fillna(0)
    for c in ["AvgHR","MaxHR","MinHR","StdHR"]:
        m[c]=m.groupby("Id")[c].transform(lambda x:x.fillna(x.median()))
    return df_pq(m)

@st.cache_data(show_spinner="Fitting Prophet…")
def fit_prophet(ds_b,y_b,horizon):
    from prophet import Prophet
    ds=pd.read_json(io.BytesIO(ds_b),typ="series")
    y=pd.read_json(io.BytesIO(y_b),typ="series")
    df=pd.DataFrame({"ds":pd.to_datetime(ds),"y":y}).dropna().sort_values("ds")
    m=Prophet(daily_seasonality=False,weekly_seasonality=True,
              yearly_seasonality=False,interval_width=0.80,changepoint_prior_scale=0.1)
    m.fit(df); fc=m.predict(m.make_future_dataframe(periods=horizon))
    return df_pq(df),df_pq(fc)

@st.cache_data(show_spinner="Running clustering + PCA…")
def run_clustering(feat_b,k,eps,min_s):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans,DBSCAN
    from sklearn.decomposition import PCA
    feats=pd.read_parquet(io.BytesIO(feat_b))
    X=StandardScaler().fit_transform(feats)
    km=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
    db=DBSCAN(eps=eps,min_samples=min_s).fit_predict(X)
    pca=PCA(n_components=2,random_state=42); X2=pca.fit_transform(X)
    var=(pca.explained_variance_ratio_*100).tolist()
    inertias=[KMeans(n_clusters=ki,random_state=42,n_init=10).fit(X).inertia_ for ki in range(2,10)]
    return X.tobytes(),X2.tobytes(),var,km.tolist(),db.tolist(),inertias

@st.cache_data(show_spinner="Running t-SNE (~30 sec)…")
def run_tsne(X_b, n_feats):
    from sklearn.manifold import TSNE
    arr   = np.frombuffer(X_b, dtype=np.float64).copy()
    n_rows = arr.size // n_feats
    X      = arr.reshape(n_rows, n_feats).astype(np.float64)
    perp   = min(30, max(2, len(X) - 1))
    result = TSNE(n_components=2, random_state=42,
                  perplexity=perp, max_iter=1000).fit_transform(X)
    return result.astype(np.float64).tobytes(), n_rows

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:10px 0 4px'>
      <div style='font-size:1.6rem;font-weight:800;color:{BLUE}'>⚡ FitPulse</div>
      <div style='font-size:.72rem;color:{MUTED};margin-top:2px'>ML Pipeline · Dual Milestone</div>
    </div>""",unsafe_allow_html=True)
    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:.85rem;text-transform:uppercase;letter-spacing:.06em'>🗂 Select Milestone</p>",unsafe_allow_html=True)

    m1_on = st.session_state.active_milestone == "milestone1"
    m2_on = st.session_state.active_milestone == "milestone2"

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📋 Milestone 1", key="nav_m1",
                     use_container_width=True,
                     type="primary" if m1_on else "secondary"):
            st.session_state.active_milestone = "milestone1"
            st.rerun()
    with col_b:
        if st.button("🧬 Milestone 2", key="nav_m2",
                     use_container_width=True,
                     type="primary" if m2_on else "secondary"):
            st.session_state.active_milestone = "milestone2"
            st.rerun()

    now_label = "📋 Milestone 1 — Data Governance" if m1_on else "🧬 Milestone 2 — ML Pipeline"
    st.markdown(
        f"<div style='background:{CARD2};border:1px solid {BLUE};border-radius:8px;"
        f"padding:8px 12px;font-size:.72rem;color:{BLUE};font-weight:700;"
        f"text-align:center;margin:8px 0 0'>▶ {now_label}</div>",
        unsafe_allow_html=True)
    st.divider()

    # ── Milestone 1 controls ──────────────────────────────────────────────────
    if m1_on:
        st.markdown(f"<p style='color:{TEXT};font-weight:700'>🛡️ Pipeline Status</p>",unsafe_allow_html=True)
        t1 = "✅" if st.session_state.m1_ingested  else "⭕"
        t2 = "✅" if st.session_state.m1_processed else "⭕"
        c1 = "complete" if st.session_state.m1_ingested  else "pending"
        c2 = "complete" if st.session_state.m1_processed else "pending"
        st.markdown(f'<div class="status-badge {c1}">INGESTION <span>{t1}</span></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="status-badge {c2}">GOVERNANCE <span>{t2}</span></div>',unsafe_allow_html=True)
        if st.session_state.m1_processed:
            st.success("INTEGRITY: OPTIMIZED")
        st.divider()
        if st.button("🔄 Reset Milestone 1", use_container_width=True):
            for k in ["m1_raw_df","m1_clean_df","m1_ingested","m1_processed"]:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()

    # ── Milestone 2 controls ──────────────────────────────────────────────────
    else:
        st.markdown(f"<p style='color:{TEXT};font-weight:700'>⚙️ Model Parameters</p>",unsafe_allow_html=True)
        OPTIMAL_K     = st.slider("K-Means Clusters (K)",    2, 8, 3)
        EPS           = st.slider("DBSCAN  ε (eps)",         0.5, 5.0, 2.2, 0.1)
        MIN_SAMPLES   = st.slider("DBSCAN  min_samples",     1, 5, 2)
        FORECAST_DAYS = st.slider("Forecast horizon (days)", 7, 60, 30)
        run_tsne_flag = st.checkbox("Run t-SNE  (~30 sec)", value=False)
        st.divider()

        st.markdown(f"<p style='color:{TEXT};font-weight:700'>📊 Phase Progress</p>",unsafe_allow_html=True)
        for icon,label,pk in [("📂","Phase 1","run_p1"),("🧬","Phase 2","run_p2"),
                               ("📈","Phase 3","run_p3"),("🤖","Phase 4","run_p4")]:
            done=st.session_state.get(pk,False)
            clr=GREEN if done else MUTED; sym="✅" if done else "⭕"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 10px;"
                f"background:{CARD2};border-radius:6px;margin-bottom:4px;"
                f"font-size:.78rem;color:{clr}'><span>{icon} {label}</span><span>{sym}</span></div>",
                unsafe_allow_html=True)
        st.divider()
        if st.button("🔄 Reset Milestone 2", use_container_width=True):
            for k in ["run_p1","run_p2","run_p3","run_p4"]:
                st.session_state[k] = False
            st.rerun()

    st.markdown(f"<p style='color:{MUTED};font-size:.68rem;margin-top:8px'>Real Fitbit Dataset · Mar–Apr 2016</p>",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MILESTONE 1
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_milestone == "milestone1":

    st.markdown("<h1 style='text-align:center;color:#38bdf8'>🏃‍♂️ FitPulse Pro: Data Governance</h1>",unsafe_allow_html=True)

    # Step 1 — Ingestion
    st.markdown('<div class="glass-card">',unsafe_allow_html=True)
    st.subheader("📁 Step 1: Secure Data Ingestion")
    file=st.file_uploader("Upload FitPulse CSV",type="csv",key="m1_uploader",label_visibility="collapsed")
    if file:
        temp_df=pd.read_csv(file)
        if st.session_state.m1_raw_df is None or not st.session_state.m1_raw_df.equals(temp_df):
            st.session_state.m1_raw_df=temp_df
            st.session_state.m1_ingested=True
            st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

    if st.session_state.m1_ingested:
        # Step 2 — Null Diagnostics
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("📊 Step 2: Graphical Null Diagnostics")
        df=st.session_state.m1_raw_df
        null_counts=df.isnull().sum().reset_index()
        null_counts.columns=["Column","Count"]
        null_data=null_counts[null_counts["Count"]>0]
        if not null_data.empty:
            c1,c2=st.columns(2)
            with c1:
                st.write("Null Distribution by Column")
                bar=alt.Chart(null_data).mark_bar(cornerRadius=5,color="#38bdf8").encode(
                    x=alt.X("Column",sort="-y"),y="Count").properties(height=200)
                st.altair_chart(bar,use_container_width=True)
            with c2:
                st.write("Data Integrity Ratio")
                total=df.size; nulls=df.isnull().sum().sum()
                pie_df=pd.DataFrame({"Status":["Valid","Missing"],"Value":[total-nulls,nulls]})
                pie=alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                    theta="Value",
                    color=alt.Color("Status",scale=alt.Scale(range=["#10b981","#f43f5e"]))
                ).properties(height=200)
                st.altair_chart(pie,use_container_width=True)
        else:
            st.success("No anomalies found in source data.")
        st.markdown('</div>',unsafe_allow_html=True)

        # Step 3 — Governance
        st.markdown('<div class="glass-card">',unsafe_allow_html=True)
        st.subheader("⚙️ Step 3: Governance Pipeline")
        st.write("• **FUNC_01**: Drop Null Dates | **FUNC_02**: Impute 'Workout_Type' | **FUNC_03**: Mean Fill Metrics")
        if st.button("🚀 DEPLOY CLEANING PROTOCOL",key="m1_clean_btn",use_container_width=True):
            with st.status("Engaging Engine...",expanded=True) as status:
                clean=st.session_state.m1_raw_df.copy()
                if "Date" in clean.columns: clean=clean.dropna(subset=["Date"])
                if "Workout_Type" in clean.columns: clean["Workout_Type"]=clean["Workout_Type"].fillna("General")
                for col in clean.columns:
                    if clean[col].dtype in [np.float64,np.int64]:
                        clean[col]=clean[col].fillna(clean[col].mean())
                time.sleep(1)
                st.session_state.m1_clean_df=clean
                st.session_state.m1_processed=True
                status.update(label="System Optimized!",state="complete")
            st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

        if st.session_state.m1_processed:
            # Step 3.5 — Preview
            st.markdown('<div class="glass-card">',unsafe_allow_html=True)
            st.subheader("👀 Step 3.5: Data Integrity Preview")
            if st.checkbox("🔍 SHOW CLEANED DATA PREVIEW",key="m1_preview_cb"):
                st.dataframe(st.session_state.m1_clean_df,use_container_width=True,hide_index=True)
                st.info(f"Integrity Check: {len(st.session_state.m1_clean_df)} records ready for export.")
            st.markdown('</div>',unsafe_allow_html=True)

            # Step 4 — Analysis
            st.markdown('<div class="glass-card">',unsafe_allow_html=True)
            st.subheader("📈 Step 4: Processed Column Analysis")
            df_clean=st.session_state.m1_clean_df
            num_cols=df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                cols_ch=st.columns(min(len(num_cols),3))
                for i,col in enumerate(num_cols[:3]):
                    with cols_ch[i]:
                        st.write(f"**{col}** (Post-Optimization)")
                        chart=alt.Chart(df_clean).mark_area(
                            line={"color":"#38bdf8"},
                            color=alt.Gradient(gradient="linear",
                                stops=[alt.GradientStop(color="#0ea5e9",offset=0),
                                       alt.GradientStop(color="transparent",offset=1)],
                                x1=1,x2=1,y1=1,y2=0)
                        ).encode(alt.X(col,bin=alt.Bin(maxbins=20)),alt.Y("count()")).properties(height=180)
                        st.altair_chart(chart,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

            # Step 5 — EDA
            st.markdown('<div class="glass-card">',unsafe_allow_html=True)
            st.subheader("🔍 Step 5: Complete Governance EDA")
            tab_corr,tab_dist=st.tabs(["🔥 Correlation Matrix","📊 Feature Distribution"])
            with tab_corr:
                num_df=df_clean.select_dtypes(include=[np.number])
                if not num_df.empty:
                    corr=num_df.corr().reset_index().melt(id_vars="index")
                    heatmap=alt.Chart(corr).mark_rect().encode(
                        x="index:O",y="variable:O",
                        color=alt.Color("value:Q",scale=alt.Scale(scheme="viridis"))
                    ).properties(height=400)
                    st.altair_chart(heatmap,use_container_width=True)
            with tab_dist:
                cols_eda=st.columns(2)
                for idx,col in enumerate(df_clean.columns):
                    with cols_eda[idx%2]:
                        st.markdown(f"**Field:** `{col.upper()}`")
                        if df_clean[col].dtype in [np.float64,np.int64]:
                            c=alt.Chart(df_clean).mark_bar(color="#38bdf8").encode(x=alt.X(col,bin=True),y="count()").properties(height=150)
                        else:
                            c=alt.Chart(df_clean).mark_bar().encode(x="count()",y=alt.Y(col,sort="-x"),color=col).properties(height=150)
                        st.altair_chart(c,use_container_width=True)
            st.divider()
            buf=io.BytesIO()
            st.session_state.m1_clean_df.to_csv(buf,index=False)
            st.download_button("📥 DOWNLOAD FINAL OPTIMIZED DATASET",
                data=buf.getvalue(),file_name="FitPulse_Elite_Clean.csv",
                mime="text/csv",use_container_width=True,key="m1_download")
            st.markdown('</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MILESTONE 2
# ══════════════════════════════════════════════════════════════════════════════
else:
    if "OPTIMAL_K" not in dir():
        OPTIMAL_K=3; EPS=2.2; MIN_SAMPLES=2; FORECAST_DAYS=30; run_tsne_flag=False

    # Header
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{CARD},{CARD2});border:1px solid {BORDER};border-radius:14px;padding:28px 32px;margin-bottom:28px'>
      <div style='font-size:.65rem;font-weight:800;letter-spacing:.18em;color:{BLUE};text-transform:uppercase;margin-bottom:8px'>MILESTONE 2 · FEATURE EXTRACTION &amp; MODELING</div>
      <div style='font-size:2.2rem;font-weight:800;color:{TEXT};line-height:1.1'>⚡ FitPulse ML Pipeline</div>
      <div style='color:{MUTED};margin-top:10px;font-size:.85rem'>TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data</div>
    </div>""",unsafe_allow_html=True)

    # File upload
    FILE_DEFS=[
        ("daily","🏃","Daily Activity","dailyActivity_merged.csv"),
        ("heartrate","❤️","Heart Rate","heartrate_seconds_merged.csv"),
        ("hourly_intensities","⚡","Hourly Intensities","hourlyIntensities_merged.csv"),
        ("hourly_steps","👟","Hourly Steps","hourlySteps_merged.csv"),
        ("sleep","😴","Minute Sleep","minuteSleep_merged.csv"),
        ("tsfresh","🧬","TSFresh Features CSV","tsfresh_features.csv"),
    ]
    slots={k:None for k,*_ in FILE_DEFS}; raw={}

    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:22px 26px;margin-bottom:18px'>
      <div style='font-size:1.1rem;font-weight:800;color:{TEXT};margin-bottom:4px'>📂 Upload Your Fitbit CSV Files</div>
      <div style='font-size:.8rem;color:{MUTED}'>Click <b>Browse files</b> and select all 6 CSV files at once (hold <b>Ctrl/Cmd</b> to multi-select). Files are <b>auto-detected</b> — no renaming needed.</div>
    </div>""",unsafe_allow_html=True)

    uploaded_files=st.file_uploader("📁 Select all 6 CSV files at once",type=["csv"],accept_multiple_files=True,key="m2_bulk_upload")
    if uploaded_files:
        for f in uploaded_files:
            b=f.read(); df=read_csv(b); t=detect_type(df)
            if t in slots: slots[t]=df; raw[t]=b

    # Status cards
    st.markdown("<div style='margin-top:14px'></div>",unsafe_allow_html=True)
    card_cols=st.columns(6); n_ok=0
    for col,(key,icon,label,_) in zip(card_cols,FILE_DEFS):
        ready=slots[key] is not None
        if ready: n_ok+=1
        bg="rgba(63,185,80,.10)" if ready else CARD2
        bdr=GREEN if ready else BORDER
        stxt=(f'<span style="color:{GREEN};font-weight:800;font-size:.82rem">✅ Detected</span>'
              if ready else f'<span style="color:{MUTED};font-size:.78rem">⬜ Missing</span>')
        col.markdown(
            f'<div style="background:{bg};border:1px solid {bdr};border-radius:12px;padding:16px 10px;text-align:center">'
            f'<div style="font-size:1.8rem">{icon}</div>'
            f'<div style="font-size:.68rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.06em;margin:6px 0 4px">{label}</div>'
            f'{stxt}</div>',unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>",unsafe_allow_html=True)
    st.progress(n_ok/6,text=f"Files loaded: {n_ok} / 6")

    missing=[k for k in ["daily","heartrate","hourly_intensities","hourly_steps","sleep","tsfresh"] if slots[k] is None]
    if missing:
        nice={"daily":"Daily Activity","heartrate":"Heart Rate","hourly_intensities":"Hourly Intensities",
              "hourly_steps":"Hourly Steps","sleep":"Minute Sleep","tsfresh":"TSFresh Features CSV"}
        st.info(f"👆 Upload all 6 files.\n\n**Still needed:** {', '.join(nice[k] for k in missing)}")
        st.stop()

    st.success("✅ All 6 files uploaded and ready.")
    st.divider()

    # Data prep
    daily=slots["daily"].copy(); hourly_s=slots["hourly_steps"].copy()
    hourly_i=slots["hourly_intensities"].copy(); sleep=slots["sleep"].copy()
    hr=slots["heartrate"].copy(); features=slots["tsfresh"].copy()
    if features.columns[0] in ("Unnamed: 0",""):
        features=features.rename(columns={features.columns[0]:"UserId"}).set_index("UserId")
    daily["ActivityDate"]=pd.to_datetime(daily["ActivityDate"],format="%m/%d/%Y",errors="coerce")
    hourly_s["ActivityHour"]=pd.to_datetime(hourly_s["ActivityHour"],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
    hourly_i["ActivityHour"]=pd.to_datetime(hourly_i["ActivityHour"],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
    sc="date" if "date" in sleep.columns else "Date"
    sleep[sc]=pd.to_datetime(sleep[sc],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
    if sc!="date": sleep=sleep.rename(columns={sc:"date"})
    hr["Time"]=pd.to_datetime(hr["Time"],format="%m/%d/%Y %I:%M:%S %p",errors="coerce")
    hr_min_b=resample_hr(raw["heartrate"]); hr_min=pd.read_parquet(io.BytesIO(hr_min_b))
    master_b=build_master(raw["daily"],raw["sleep"],hr_min_b)
    master=pd.read_parquet(io.BytesIO(master_b))
    date_span=(daily["ActivityDate"].max()-daily["ActivityDate"].min()).days

    # Phase button helpers
    def launch_phase(icon,n,title,steps,clr,bk,sk):
        if not st.session_state[sk]:
            st.markdown(
                f'<div style="background:{CARD2};border:2px solid {clr};border-radius:14px;padding:22px 24px;text-align:center;margin:22px 0 10px">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-size:.68rem;font-weight:800;color:{clr};text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">Phase {n}</div>'
                f'<div style="font-size:.95rem;font-weight:700;color:{TEXT}">{title}</div>'
                f'<div style="font-size:.72rem;color:{MUTED};margin-top:4px">{steps}</div>'
                f'</div>',unsafe_allow_html=True)
            _c1,_c2,_c3=st.columns([2,3,2])
            with _c2:
                if st.button(f"▶  Run Phase {n}",key=bk,use_container_width=True,type="primary"):
                    st.session_state[sk]=True; st.rerun()

    def next_phase_btn(icon,n,title,steps,clr,bk,sk):
        st.divider()
        st.markdown(
            f'<div style="background:{CARD2};border:2px solid {clr};border-radius:14px;padding:22px 24px;text-align:center;margin:22px 0 10px">'
            f'<div style="font-size:2rem">{icon}</div>'
            f'<div style="font-size:.68rem;font-weight:800;color:{clr};text-transform:uppercase;letter-spacing:.08em;margin:6px 0 4px">🔜 Next — Phase {n}</div>'
            f'<div style="font-size:.95rem;font-weight:700;color:{TEXT}">{title}</div>'
            f'<div style="font-size:.72rem;color:{MUTED};margin-top:4px">{steps}</div>'
            f'</div>',unsafe_allow_html=True)
        _nc1,_nc2,_nc3=st.columns([2,3,2])
        with _nc2:
            if st.button(f"▶  Run Phase {n}",key=bk,use_container_width=True,type="primary"):
                st.session_state[sk]=True; st.rerun()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    launch_phase("📂",1,"Data Ingestion & Cleaning","Steps 1–9","#58a6ff","btn_p1","run_p1")

    if st.session_state["run_p1"]:
        phase_banner("📂","Phase 1 · Data Ingestion & Cleaning","STEPS 1 – 9",
                     "Parse timestamps → null audit → resample HR → merge master dataframe")
        step_box("Step 1–3","Files Loaded & Timestamps Parsed",
                 "5 CSVs auto-detected · all timestamps parsed")
        shape_df=pd.DataFrame({
            "Dataset":["dailyActivity","hourlySteps","hourlyIntensities","minuteSleep","heartrate"],
            "Rows":[f"{daily.shape[0]:,}",f"{hourly_s.shape[0]:,}",f"{hourly_i.shape[0]:,}",f"{sleep.shape[0]:,}",f"{hr.shape[0]:,}"],
            "Columns":[daily.shape[1],hourly_s.shape[1],hourly_i.shape[1],sleep.shape[1],hr.shape[1]],
            "Key Columns":["Id,ActivityDate,TotalSteps,Calories,VeryActiveMinutes,SedentaryMinutes",
                           "Id,ActivityHour,StepTotal","Id,ActivityHour,TotalIntensity,AverageIntensity",
                           "Id,date,value,logId","Id,Time,Value"],
        })
        st.dataframe(shape_df,use_container_width=True,hide_index=True); st.divider()
        step_box("Step 4","Null Value Check","Scan every column for missing values")
        null_rows=[]
        for name,df in [("dailyActivity",daily),("hourlySteps",hourly_s),
                        ("hourlyIntensities",hourly_i),("minuteSleep",sleep),("heartrate",hr)]:
            n=int(df.isnull().sum().sum())
            null_rows.append({"Dataset":name,"Total Nulls":n,"Shape":str(df.shape),
                              "Completeness":"100%" if n==0 else f"{(1-n/df.size)*100:.2f}%",
                              "Status":"✅ Clean" if n==0 else f"⚠️ {n} nulls"})
        st.dataframe(pd.DataFrame(null_rows),use_container_width=True,hide_index=True); st.divider()
        step_box("Step 5–6","Dataset Overview")
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Total Users",daily["Id"].nunique()); c2.metric("Sleep Users",sleep["Id"].nunique())
        c3.metric("HR Users",hr["Id"].nunique()); c4.metric("HR Records",f"{hr.shape[0]:,}")
        c5.metric("Date Span",f"{date_span} days")
        c6.metric("Total Rows",f"{sum(x.shape[0] for x in [daily,hourly_s,hourly_i,sleep,hr]):,}"); st.divider()
        step_box("Step 6","HR Resampling: Seconds → 1-Minute")
        r1,r2,r3=st.columns(3)
        r1.metric("Before",f"{hr.shape[0]:,}",delta="seconds-level")
        r2.metric("After",f"{hr_min.shape[0]:,}",delta="1-min intervals")
        r3.metric("Reduction",f"{(1-hr_min.shape[0]/hr.shape[0])*100:.0f}%",delta_color="off"); st.divider()
        step_box("Step 7–9","Cleaned Master Dataframe")
        cm1,cm2,cm3=st.columns(3)
        cm1.metric("Shape",str(master.shape)); cm2.metric("Users",master["Id"].nunique())
        cm3.metric("Nulls",int(master.isnull().sum().sum()))
        st.dataframe(master[["Id","Date","TotalSteps","Calories","AvgHR","TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]].head(15),use_container_width=True,hide_index=True); st.divider()
        step_box("Step 9a","Summary Statistics")
        st.dataframe(master[["TotalSteps","Calories","AvgHR","TotalSleepMinutes","VeryActiveMinutes","SedentaryMinutes"]].describe().round(2),use_container_width=True); st.divider()
        step_box("Step 9b","Activity Distribution Histograms","Counts labeled on every bar · mean line shown")
        dist_cfg=[("TotalSteps","Total Daily Steps",BLUE,"Steps/day"),("Calories","Calories Burned",GREEN,"Cal/day"),
                  ("TotalSleepMinutes","Sleep Duration",PURPLE,"Min/day"),("SedentaryMinutes","Sedentary Time",AMBER,"Min/day"),
                  ("VeryActiveMinutes","Very-Active Time",RED,"Min/day"),("AvgHR","Avg Heart Rate",PINK,"BPM")]
        for i in range(0,len(dist_cfg),2):
            cols_d=st.columns(2)
            for j,cd in enumerate(cols_d):
                if i+j>=len(dist_cfg): break
                key,title,color,xlabel=dist_cfg[i+j]; s=master[key].dropna()
                fig,ax=plt.subplots(figsize=(7,3.6))
                cnts,_,patches=ax.hist(s,bins=20,color=color,alpha=0.85,edgecolor=DARK,linewidth=0.4)
                top=max(cnts) if len(cnts)>0 else 1
                for patch,cnt in zip(patches,cnts):
                    if cnt>0:
                        ax.text(patch.get_x()+patch.get_width()/2,cnt+top*0.015,f"{int(cnt)}",ha="center",va="bottom",fontsize=7,color=TEXT)
                mv=s.mean(); ax.axvline(mv,color="white",linestyle="--",linewidth=1.4,label=f"Mean={mv:.0f}")
                ax.set_title(f"📊 {title}",fontsize=9,color=TEXT,pad=7)
                ax.set_xlabel(xlabel,fontsize=8,color=MUTED); ax.set_ylabel("Records",fontsize=8,color=MUTED)
                ax.legend(fontsize=8,framealpha=0.4); ax.grid(axis="y",alpha=0.2)
                plt.tight_layout(); cd.pyplot(fig); plt.close(fig)
        st.divider()
        step_box("Step 9c","Hourly Steps Heatmap","Exact values in every cell")
        hourly_s["Hour"]=hourly_s["ActivityHour"].dt.hour
        hourly_s["DayOfWeek"]=hourly_s["ActivityHour"].dt.day_name()
        pivot=hourly_s.groupby(["DayOfWeek","Hour"])["StepTotal"].mean().unstack()
        day_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot=pivot.reindex([d for d in day_order if d in pivot.index])
        fig_hw,ax_hw=plt.subplots(figsize=(16,5))
        sns.heatmap(pivot,ax=ax_hw,cmap="YlOrRd",annot=True,fmt=".0f",annot_kws={"size":6.5},
                    linewidths=0.25,linecolor=DARK,cbar_kws={"label":"Avg Steps/Hour","shrink":0.65})
        ax_hw.set_title("🕐 Average Steps by Day × Hour (each cell = exact avg count)",fontsize=11,color=TEXT,pad=10)
        ax_hw.set_xlabel("Hour (0–23)",fontsize=9,color=MUTED); ax_hw.set_ylabel("Day",fontsize=9,color=MUTED)
        plt.tight_layout(); st.pyplot(fig_hw); plt.close(fig_hw)
        next_phase_btn("🧬",2,"Feature Engineering","Steps 10–12","#3fb950","btn_p2","run_p2")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if st.session_state["run_p2"]:
        phase_banner("🧬","Phase 2 · Feature Engineering","STEPS 10 – 12",
                     "TSFresh features from CSV · 10 statistical features per user")
        step_box("Step 10–11","TSFresh Feature Matrix","Loaded from uploaded tsfresh_features.csv")
        ff1,ff2,ff3=st.columns(3)
        ff1.metric("Users",features.shape[0]); ff2.metric("Features",features.shape[1])
        ff3.metric("Source","tsfresh_features.csv")
        st.markdown("**Feature names:**")
        for i,c in enumerate(features.columns):
            st.markdown(f"<span class='info-pill'>{i+1}. {c.replace('value__','')}</span>",unsafe_allow_html=True)
        st.markdown(""); st.dataframe(features.round(4),use_container_width=True); st.divider()
        step_box("Step 12a","Feature Heatmap — Normalized 0–1  📸 Screenshot This","Exact value in every cell")
        from sklearn.preprocessing import MinMaxScaler as MMS
        feat_norm=pd.DataFrame(MMS().fit_transform(features),index=features.index,columns=features.columns)
        fd=feat_norm.rename(columns={c:c.replace("value__","") for c in feat_norm.columns})
        fd.index=[str(i)[-6:] for i in fd.index]
        fig_hm,ax_hm=plt.subplots(figsize=(max(12,len(features.columns)*1.5),max(6,len(features)*0.65)))
        sns.heatmap(fd,ax=ax_hm,cmap="coolwarm",annot=True,fmt=".2f",annot_kws={"size":8.5,"weight":"bold"},
                    linewidths=0.5,linecolor=DARK,cbar_kws={"label":"Normalized (0=min · 1=max)","shrink":0.8},vmin=0,vmax=1)
        ax_hm.set_title("🧬 TSFresh Feature Matrix (Normalized 0–1)",fontsize=11,color=TEXT,pad=12)
        ax_hm.set_xlabel("Feature",fontsize=9,color=MUTED); ax_hm.set_ylabel("User ID",fontsize=9,color=MUTED)
        plt.xticks(rotation=30,ha="right",fontsize=8); plt.tight_layout()
        st.pyplot(fig_hm); dl(fig_hm,"tsfresh_heatmap.png","dl_hm"); plt.close(fig_hm); st.divider()
        step_box("Step 12b","Per-Feature Bar Charts","Sorted · exact value on every bar")
        for feat in features.columns:
            fname=feat.replace("value__",""); vals=features[feat].sort_values()
            ulbls=[str(i)[-5:] for i in vals.index]
            fig_b,ax_b=plt.subplots(figsize=(12,3.4))
            bars_b=ax_b.bar(range(len(vals)),vals.values,color=[PAL[i%len(PAL)] for i in range(len(vals))],edgecolor=DARK,linewidth=0.4,zorder=3)
            mx=max(vals.values) if len(vals) else 1
            for bar,v in zip(bars_b,vals.values):
                ax_b.text(bar.get_x()+bar.get_width()/2,bar.get_height()+mx*0.014,f"{v:.2f}",ha="center",va="bottom",fontsize=7.5,color=TEXT,fontweight="700")
            ax_b.set_xticks(range(len(vals))); ax_b.set_xticklabels(ulbls,rotation=35,ha="right",fontsize=8)
            ax_b.set_title(f"📊 TSFresh: {fname} (sorted low→high · values labeled)",fontsize=10,color=TEXT,pad=8)
            ax_b.set_xlabel("User ID",fontsize=8,color=MUTED); ax_b.set_ylabel(fname,fontsize=8,color=MUTED)
            ax_b.grid(axis="y",alpha=0.2); ax_b.set_axisbelow(True); plt.tight_layout(); st.pyplot(fig_b); plt.close(fig_b)
        next_phase_btn("📈",3,"Prophet Forecasting","Steps 13–17","#f0883e","btn_p3","run_p3")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if st.session_state["run_p3"]:
        try: from prophet import Prophet
        except ImportError: st.error("❌ `prophet` not installed — `pip install prophet`"); st.stop()
        phase_banner("📈","Phase 3 · Prophet Forecasting",f"STEPS 13 – 17",
                     f"Prophet on HR · Steps · Sleep → {FORECAST_DAYS}-day forecast · 80% CI")

        def prophet_plot(df_in,fc,color,title,ylabel,dl_key):
            fig,ax=plt.subplots(figsize=(14,5.5))
            fc_start=df_in["ds"].max()
            ax.fill_between(fc["ds"],fc["yhat_lower"],fc["yhat_upper"],alpha=0.2,color=color,label="80% CI")
            ax.plot(fc["ds"],fc["yhat"],color=TEXT,linewidth=2.2,label="Prophet Trend",zorder=3)
            ax.scatter(df_in["ds"],df_in["y"],color=color,s=32,zorder=5,alpha=0.9,label="Actual")
            for _,row in df_in.iterrows():
                ax.annotate(f"{row['y']:.0f}",(row["ds"],row["y"]),textcoords="offset points",xytext=(0,6),fontsize=6.5,color=TEXT,ha="center",alpha=0.8)
            ax.axvline(fc_start,color=AMBER,linestyle="--",linewidth=1.8,label=f"Forecast Start",alpha=0.9)
            last=fc.iloc[-1]; ax.annotate(f"End:{last['yhat']:.1f}",(last["ds"],last["yhat"]),xytext=(-55,14),textcoords="offset points",fontsize=8,color=AMBER,fontweight="700",arrowprops=dict(arrowstyle="->",color=AMBER,lw=1.2))
            ax.set_title(f"📈 {title}\n(dots=actual · line=trend · band=80%CI · dashed=forecast start)",fontsize=11,color=TEXT,pad=10)
            ax.set_xlabel("Date",fontsize=9,color=MUTED); ax.set_ylabel(ylabel,fontsize=9,color=MUTED)
            ax.legend(fontsize=8,framealpha=0.35); ax.grid(alpha=0.15); plt.tight_layout()
            st.pyplot(fig); dl(fig,dl_key,dl_key); plt.close(fig)

        step_box("Step 13–14","Heart Rate Forecast  📸","Daily mean HR → Prophet → 80% CI")
        hr_agg=hr_min.groupby(hr_min["Time"].dt.date)["HeartRate"].mean().reset_index()
        hr_agg.columns=["ds","y"]; hr_agg["ds"]=pd.to_datetime(hr_agg["ds"]); hr_agg=hr_agg.dropna().sort_values("ds")
        df_hr_b,fc_hr_b=fit_prophet(ser_json(hr_agg["ds"]),ser_json(hr_agg["y"]),FORECAST_DAYS)
        df_hr=pd.read_parquet(io.BytesIO(df_hr_b)); fc_hr=pd.read_parquet(io.BytesIO(fc_hr_b))
        h1,h2,h3=st.columns(3); h1.metric("Training Points",len(df_hr)); h2.metric("Horizon",f"{FORECAST_DAYS}d"); h3.metric("CI","80%")
        prophet_plot(df_hr,fc_hr,AMBER,f"Heart Rate — {FORECAST_DAYS}-Day Forecast","Avg HR (BPM)","prophet_hr.png"); st.divider()

        step_box("Step 15–16","Daily Steps Forecast  📸","Steps → Prophet → annotated")
        steps_agg=daily.groupby("ActivityDate")["TotalSteps"].mean().reset_index()
        steps_agg.columns=["ds","y"]; steps_agg["ds"]=pd.to_datetime(steps_agg["ds"]); steps_agg=steps_agg.dropna().sort_values("ds")
        df_st_b,fc_st_b=fit_prophet(ser_json(steps_agg["ds"]),ser_json(steps_agg["y"]),FORECAST_DAYS)
        df_st=pd.read_parquet(io.BytesIO(df_st_b)); fc_st=pd.read_parquet(io.BytesIO(fc_st_b))
        prophet_plot(df_st,fc_st,GREEN,f"Daily Steps — {FORECAST_DAYS}-Day Forecast","Avg Steps/Day","prophet_steps.png"); st.divider()

        step_box("Step 17","Sleep Forecast  📸","Sleep minutes → Prophet → CI")
        sleep_agg=master.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
        sleep_agg.columns=["ds","y"]; sleep_agg["ds"]=pd.to_datetime(sleep_agg["ds"]); sleep_agg=sleep_agg[sleep_agg["y"]>0].dropna().sort_values("ds")
        df_sl_b,fc_sl_b=fit_prophet(ser_json(sleep_agg["ds"]),ser_json(sleep_agg["y"]),FORECAST_DAYS)
        df_sl=pd.read_parquet(io.BytesIO(df_sl_b)); fc_sl=pd.read_parquet(io.BytesIO(fc_sl_b))
        prophet_plot(df_sl,fc_sl,PURPLE,f"Sleep — {FORECAST_DAYS}-Day Forecast","Avg Sleep (min/day)","prophet_sleep.png"); st.divider()

        step_box("Step 15–17 Combined","Steps + Sleep Combined  📸","Notebook-style stacked plot")
        fig_comb,axes_c=plt.subplots(2,1,figsize=(14,10)); fig_comb.patch.set_facecolor(DARK)
        for ax_c,(df_c,fc_c,col_c,lbl_c) in zip(axes_c,[(df_st,fc_st,GREEN,"Steps"),(df_sl,fc_sl,PURPLE,"Sleep (minutes)")]):
            ax_c.set_facecolor(CARD2)
            ax_c.fill_between(fc_c["ds"],fc_c["yhat_lower"],fc_c["yhat_upper"],alpha=0.25,color=col_c,label="80% CI")
            ax_c.plot(fc_c["ds"],fc_c["yhat"],color=TEXT,linewidth=2.5,label="Trend")
            ax_c.scatter(df_c["ds"],df_c["y"],color=col_c,s=22,alpha=0.85,zorder=4,label=f"Actual {lbl_c}")
            for _,row in df_c.iterrows():
                ax_c.annotate(f"{row['y']:.0f}",(row["ds"],row["y"]),textcoords="offset points",xytext=(0,5),fontsize=6,color=TEXT,ha="center",alpha=0.75)
            ax_c.axvline(df_c["ds"].max(),color=AMBER,linestyle="--",linewidth=1.8,label="Forecast Start")
            ax_c.set_title(f"{lbl_c} — Prophet Forecast",fontsize=11,color=TEXT)
            ax_c.set_xlabel("Date",color=MUTED); ax_c.set_ylabel(lbl_c,color=MUTED)
            ax_c.legend(fontsize=8,framealpha=0.3); ax_c.grid(alpha=0.15)
        plt.tight_layout(); st.pyplot(fig_comb); dl(fig_comb,"prophet_combined.png","dl_comb"); plt.close(fig_comb)
        next_phase_btn("🤖",4,"Clustering & Reduction","Steps 18–27","#bc8cff","btn_p4","run_p4")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    if st.session_state["run_p4"]:
        phase_banner("🤖","Phase 4 · Clustering & Dimensionality Reduction","STEPS 18 – 27",
                     "Feature matrix → StandardScaler → K-Means + DBSCAN → Elbow → PCA → t-SNE → Profiles")
        step_box("Step 18","Clustering Feature Matrix","Average user daily metrics")
        clust_c=["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
        clust_feats=master.groupby("Id")[clust_c].mean().round(3).dropna()
        cff1,cff2=st.columns(2); cff1.metric("Users",clust_feats.shape[0]); cff2.metric("Features",clust_feats.shape[1])
        st.dataframe(clust_feats.round(2),use_container_width=True); st.divider()
        step_box("Step 19","StandardScaler Normalization")
        from sklearn.preprocessing import StandardScaler
        (X_b,X2_b,var,km_list,db_list,inertias)=run_clustering(df_pq(clust_feats),OPTIMAL_K,EPS,MIN_SAMPLES)
        X_scaled=np.frombuffer(X_b,dtype=np.float64).reshape(-1,len(clust_c))
        X_pca=np.frombuffer(X2_b,dtype=np.float64).reshape(-1,2)
        kmeans_labels=np.array(km_list,dtype=int); dbscan_labels=np.array(db_list,dtype=int)
        sc1,sc2=st.columns(2); sc1.metric("Mean (≈0)",f"{X_scaled.mean():.6f}"); sc2.metric("Std (≈1)",f"{X_scaled.std():.4f}"); st.divider()

        step_box("Step 20","K-Means Elbow Curve  📸",f"Inertia labeled on every point · K={OPTIMAL_K} selected")
        K_range=range(2,10)
        fig_el,ax_el=plt.subplots(figsize=(10,4.5))
        ax_el.plot(list(K_range),inertias,"o-",color=BLUE,linewidth=2.5,markersize=11,markerfacecolor=PINK,markeredgecolor=TEXT,markeredgewidth=1.2,zorder=3)
        for k,iner in zip(K_range,inertias):
            ax_el.annotate(f"K={k}\n{iner:.0f}",(k,iner),textcoords="offset points",xytext=(0,14),ha="center",fontsize=8.5,color=TEXT,fontweight="700")
        sel_idx=OPTIMAL_K-2
        ax_el.scatter([OPTIMAL_K],[inertias[sel_idx]],color=AMBER,s=230,zorder=5,label=f"Selected K={OPTIMAL_K}",edgecolors=TEXT,linewidths=1.5)
        ax_el.axvline(OPTIMAL_K,color=AMBER,linestyle="--",linewidth=1.5,alpha=0.7)
        ax_el.set_title(f"📈 K-Means Elbow Curve — Optimal K={OPTIMAL_K}",fontsize=11,color=TEXT,pad=12)
        ax_el.set_xlabel("Number of Clusters (K)",fontsize=9,color=MUTED); ax_el.set_ylabel("Inertia",fontsize=9,color=MUTED)
        ax_el.set_xticks(list(K_range)); ax_el.legend(fontsize=9,framealpha=0.4); ax_el.grid(alpha=0.2); plt.tight_layout()
        st.pyplot(fig_el); dl(fig_el,"elbow_curve.png","dl_el"); plt.close(fig_el); st.divider()

        step_box("Step 21–22","K-Means Clustering",f"K={OPTIMAL_K}")
        clust_feats=clust_feats.copy(); clust_feats["KMeans_Cluster"]=kmeans_labels
        km_dist=clust_feats["KMeans_Cluster"].value_counts().sort_index()
        cols_km=st.columns(OPTIMAL_K)
        for i,col in enumerate(cols_km): col.metric(f"Cluster {i}",f"{int(km_dist.get(i,0))} users")
        fig_kmd,ax_kmd=plt.subplots(figsize=(8,3.2))
        c_km=[int(km_dist.get(i,0)) for i in range(OPTIMAL_K)]
        bars_kmd=ax_kmd.bar([f"Cluster {i}" for i in range(OPTIMAL_K)],c_km,color=PAL[:OPTIMAL_K],edgecolor=DARK)
        for bar,n in zip(bars_kmd,c_km): ax_kmd.text(bar.get_x()+bar.get_width()/2,n+0.05,f"{n} users",ha="center",va="bottom",fontsize=11,color=TEXT,fontweight="700")
        ax_kmd.set_title(f"K-Means Distribution (K={OPTIMAL_K})",fontsize=10,color=TEXT,pad=8)
        ax_kmd.set_xlabel("Cluster",fontsize=9,color=MUTED); ax_kmd.set_ylabel("Users",fontsize=9,color=MUTED)
        ax_kmd.grid(axis="y",alpha=0.2); plt.tight_layout(); st.pyplot(fig_kmd); plt.close(fig_kmd); st.divider()

        step_box("Step 23","DBSCAN Clustering",f"eps={EPS} · min_samples={MIN_SAMPLES}")
        clust_feats["DBSCAN_Cluster"]=dbscan_labels
        n_cl_db=len(set(dbscan_labels))-(1 if -1 in dbscan_labels else 0)
        n_noise=list(dbscan_labels).count(-1)
        db1,db2,db3=st.columns(3); db1.metric("DBSCAN Clusters",n_cl_db); db2.metric("Noise",n_noise); db3.metric("Noise%",f"{n_noise/len(dbscan_labels)*100:.1f}%")
        db_cnt=pd.Series(dbscan_labels).value_counts().sort_index()
        db_lbls=["Noise" if l==-1 else f"Cluster {l}" for l in db_cnt.index]
        db_clrs=[RED if l==-1 else PAL[l%len(PAL)] for l in db_cnt.index]
        fig_dbd,ax_dbd=plt.subplots(figsize=(8,3.2))
        bars_dbd=ax_dbd.bar(db_lbls,db_cnt.values,color=db_clrs,edgecolor=DARK)
        for bar,n in zip(bars_dbd,db_cnt.values): ax_dbd.text(bar.get_x()+bar.get_width()/2,n+0.05,f"{n} users",ha="center",va="bottom",fontsize=11,color=TEXT,fontweight="700")
        ax_dbd.set_title(f"DBSCAN Distribution (eps={EPS})",fontsize=10,color=TEXT,pad=8)
        ax_dbd.set_xlabel("Cluster",fontsize=9,color=MUTED); ax_dbd.set_ylabel("Users",fontsize=9,color=MUTED)
        ax_dbd.grid(axis="y",alpha=0.2); plt.tight_layout(); st.pyplot(fig_dbd); plt.close(fig_dbd); st.divider()

        step_box("Step 24","PCA — 2D Reduction")
        pv1,pv2,pv3=st.columns(3); pv1.metric("PC1",f"{var[0]:.1f}%"); pv2.metric("PC2",f"{var[1]:.1f}%"); pv3.metric("Total",f"{sum(var):.1f}%"); st.divider()

        step_box("Step 25","K-Means PCA Scatter  📸","2D PCA · User ID labeled on every point")
        fig_km_sc,ax_km=plt.subplots(figsize=(11,8))
        for cid in sorted(set(kmeans_labels)):
            mask=kmeans_labels==cid
            ax_km.scatter(X_pca[mask,0],X_pca[mask,1],c=PAL[cid%len(PAL)],label=f"Cluster {cid}",s=150,alpha=0.88,edgecolors=TEXT,linewidths=0.8,zorder=3)
            for i,uid in enumerate(clust_feats.index[mask]):
                ax_km.annotate(str(uid)[-4:],(X_pca[mask][i,0],X_pca[mask][i,1]),textcoords="offset points",xytext=(5,5),fontsize=8,color=TEXT,fontweight="600")
        ax_km.set_title(f"🤖 K-Means PCA (K={OPTIMAL_K} · PC1={var[0]:.1f}% · PC2={var[1]:.1f}%)",fontsize=11,color=TEXT,pad=12)
        ax_km.set_xlabel(f"PC1 ({var[0]:.1f}%)",fontsize=9,color=MUTED); ax_km.set_ylabel(f"PC2 ({var[1]:.1f}%)",fontsize=9,color=MUTED)
        ax_km.legend(title=f"Cluster",fontsize=10,framealpha=0.4); ax_km.grid(alpha=0.2); plt.tight_layout()
        st.pyplot(fig_km_sc); dl(fig_km_sc,"kmeans_pca.png","dl_km"); plt.close(fig_km_sc); st.divider()

        step_box("Step 26","DBSCAN PCA Scatter  📸","Noise = red ✕ with label")
        fig_db_sc,ax_db=plt.subplots(figsize=(11,8))
        for lbl in sorted(set(dbscan_labels)):
            mask=dbscan_labels==lbl
            if lbl==-1:
                ax_db.scatter(X_pca[mask,0],X_pca[mask,1],c=RED,marker="X",s=230,alpha=0.95,label="Noise (–1)",linewidths=1.5,zorder=5)
                for i,uid in enumerate(clust_feats.index[mask]):
                    ax_db.annotate(f"{str(uid)[-4:]} (noise)",(X_pca[mask][i,0],X_pca[mask][i,1]),textcoords="offset points",xytext=(8,6),fontsize=8,color=RED,fontweight="700")
            else:
                ax_db.scatter(X_pca[mask,0],X_pca[mask,1],c=PAL[lbl%len(PAL)],label=f"Cluster {lbl}",s=150,alpha=0.88,edgecolors=TEXT,linewidths=0.8,zorder=3)
                for i,uid in enumerate(clust_feats.index[mask]):
                    ax_db.annotate(str(uid)[-4:],(X_pca[mask][i,0],X_pca[mask][i,1]),textcoords="offset points",xytext=(5,5),fontsize=8,color=TEXT,fontweight="600")
        ax_db.set_title(f"🤖 DBSCAN PCA (eps={EPS} · Red✕=noise)",fontsize=11,color=TEXT,pad=12)
        ax_db.set_xlabel(f"PC1 ({var[0]:.1f}%)",fontsize=9,color=MUTED); ax_db.set_ylabel(f"PC2 ({var[1]:.1f}%)",fontsize=9,color=MUTED)
        ax_db.legend(title="Cluster",fontsize=10,framealpha=0.4); ax_db.grid(alpha=0.2); plt.tight_layout()
        st.pyplot(fig_db_sc); dl(fig_db_sc,"dbscan_pca.png","dl_db"); plt.close(fig_db_sc); st.divider()

        step_box("Step 27a","t-SNE Projection  📸","Enable in sidebar · non-linear 2D")
        if run_tsne_flag:
            tsne_bytes, n_tsne_rows = run_tsne(X_b, len(clust_c))
            X_tsne = np.frombuffer(tsne_bytes, dtype=np.float64).copy().reshape(n_tsne_rows, 2)
            fig_ts,axes_t=plt.subplots(1,2,figsize=(16,7)); fig_ts.patch.set_facecolor(DARK)
            for ax_t,(lbls_t,name_t) in zip(axes_t,[(kmeans_labels,f"K-Means (K={OPTIMAL_K})"),(dbscan_labels,f"DBSCAN (eps={EPS})")]):
                ax_t.set_facecolor(CARD2)
                for lbl in sorted(set(lbls_t)):
                    mask=lbls_t==lbl
                    if lbl==-1:
                        ax_t.scatter(X_tsne[mask,0],X_tsne[mask,1],c=RED,marker="X",s=200,label="Noise",alpha=0.95,linewidths=1.5,zorder=5)
                        for i,uid in enumerate(clust_feats.index[mask]):
                            pts=X_tsne[mask]; ax_t.annotate(f"{str(uid)[-4:]} (noise)",(float(pts[i,0]),float(pts[i,1])),xytext=(5,5),textcoords="offset points",fontsize=7,color=RED,fontweight="700")
                    else:
                        ax_t.scatter(X_tsne[mask,0],X_tsne[mask,1],c=PAL[lbl%len(PAL)],label=f"Cluster {lbl}",s=130,alpha=0.88,edgecolors=TEXT,linewidths=0.8,zorder=3)
                        for i,uid in enumerate(clust_feats.index[mask]):
                            pts=X_tsne[mask]; ax_t.annotate(str(uid)[-4:],(float(pts[i,0]),float(pts[i,1])),xytext=(5,5),textcoords="offset points",fontsize=7,color=TEXT,fontweight="600")
                ax_t.set_title(f"t-SNE — {name_t}",fontsize=11,color=TEXT,pad=10)
                ax_t.set_xlabel("t-SNE Dim 1",fontsize=9,color=MUTED); ax_t.set_ylabel("t-SNE Dim 2",fontsize=9,color=MUTED)
                ax_t.legend(title="Cluster",fontsize=9,framealpha=0.35); ax_t.grid(alpha=0.2)
            plt.tight_layout(); st.pyplot(fig_ts); dl(fig_ts,"tsne_projection.png","dl_ts"); plt.close(fig_ts)
        else:
            st.info("Enable **'Run t-SNE'** in the sidebar."); st.divider()

        step_box("Step 27b","Cluster Profiles — Grand Finale  📸","Exact avg on every bar")
        feat_p=[c for c in clust_feats.columns if c not in ("KMeans_Cluster","DBSCAN_Cluster")]
        profile=clust_feats.groupby("KMeans_Cluster")[feat_p].mean().round(2)
        st.dataframe(profile,use_container_width=True)
        disp_c=[c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"] if c in profile.columns]
        feat_colors=[BLUE,GREEN,RED,AMBER,PURPLE]
        n_feat=len(disp_c); n_clust=len(profile); x=np.arange(n_clust); width=0.14
        offsets=np.linspace(-(n_feat-1)/2*width,(n_feat-1)/2*width,n_feat)
        fig_pr,ax_pr=plt.subplots(figsize=(14,6.5))
        for fi,(feat,fc) in enumerate(zip(disp_c,feat_colors)):
            vals=profile[feat].values
            bars=ax_pr.bar(x+offsets[fi],vals,width,label=feat,color=fc,edgecolor=DARK,alpha=0.9)
            mx=max(vals) if max(vals)>0 else 1
            for bar,v in zip(bars,vals):
                ax_pr.text(bar.get_x()+bar.get_width()/2,bar.get_height()+mx*0.012,f"{v:.0f}",ha="center",va="bottom",fontsize=7.5,color=TEXT,fontweight="700")
        ax_pr.set_xticks(x); ax_pr.set_xticklabels([f"Cluster {i}" for i in range(n_clust)],fontsize=12,color=TEXT,fontweight="700")
        ax_pr.set_title("🏆 Cluster Profiles — Key Feature Averages (values on every bar)",fontsize=11,color=TEXT,pad=12)
        ax_pr.set_xlabel("K-Means Cluster",fontsize=10,color=MUTED); ax_pr.set_ylabel("Mean Value/Day",fontsize=10,color=MUTED)
        ax_pr.legend(title="Feature",bbox_to_anchor=(1.01,1),fontsize=9,framealpha=0.4)
        ax_pr.grid(axis="y",alpha=0.2); plt.tight_layout()
        st.pyplot(fig_pr); dl(fig_pr,"cluster_profiles.png","dl_pr"); plt.close(fig_pr); st.divider()

        step_box("Step 27c","Cluster Interpretation","Auto-label based on avg daily steps")
        for i in range(OPTIMAL_K):
            if i not in profile.index: continue
            row=profile.loc[i]; steps=row.get("TotalSteps",0); sed=row.get("SedentaryMinutes",0)
            act=row.get("VeryActiveMinutes",0); cals=row.get("Calories",0)
            slp=row.get("TotalSleepMinutes",0); light=row.get("LightlyActiveMinutes",0)
            n_in=int((clust_feats["KMeans_Cluster"]==i).sum())
            if steps>10000: lbl,clr="🏃 HIGHLY ACTIVE",GREEN
            elif steps>5000: lbl,clr="🚶 MODERATELY ACTIVE",BLUE
            else: lbl,clr="🛋️ SEDENTARY",AMBER
            st.markdown(f"""
            <div style='background:{CARD2};border-left:5px solid {clr};border-radius:0 12px 12px 0;padding:18px 22px;margin-bottom:14px'>
              <div style='font-size:1.1rem;font-weight:800;color:{clr}'>Cluster {i} · {lbl} <span style='font-size:.75rem;color:{MUTED};font-weight:400'>({n_in} users)</span></div>
              <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px;font-size:.82rem'>
                <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {BLUE}'><div style='color:{MUTED};font-size:.68rem;text-transform:uppercase'>📶 Steps/Day</div><div style='color:{TEXT};font-size:1.5rem;font-weight:800'>{steps:,.0f}</div></div>
                <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {GREEN}'><div style='color:{MUTED};font-size:.68rem;text-transform:uppercase'>🔥 Calories/Day</div><div style='color:{TEXT};font-size:1.5rem;font-weight:800'>{cals:,.0f}</div></div>
                <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {PURPLE}'><div style='color:{MUTED};font-size:.68rem;text-transform:uppercase'>💤 Sleep Min</div><div style='color:{TEXT};font-size:1.5rem;font-weight:800'>{slp:,.0f}</div></div>
                <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {RED}'><div style='color:{MUTED};font-size:.68rem;text-transform:uppercase'>🏃 Very Active</div><div style='color:{TEXT};font-size:1.5rem;font-weight:800'>{act:.0f}</div></div>
                <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {AMBER}'><div style='color:{MUTED};font-size:.68rem;text-transform:uppercase'>🛋️ Sedentary</div><div style='color:{TEXT};font-size:1.5rem;font-weight:800'>{sed:.0f}</div></div>
                <div style='background:{CARD};border-radius:8px;padding:12px;border-top:2px solid {TEAL}'><div style='color:{MUTED};font-size:.68rem;text-transform:uppercase'>🚶 Light Active</div><div style='color:{TEXT};font-size:1.5rem;font-weight:800'>{light:.0f}</div></div>
              </div></div>""",unsafe_allow_html=True)

        # Summary
        st.divider()
        phase_banner("✅","Milestone 2 Complete","ALL STEPS DONE","All charts generated · use 📥 Download buttons")
        st.markdown(f"""
        <div style='background:{CARD2};border:1px solid {GREEN};border-radius:14px;padding:26px 32px;font-family:monospace;font-size:.82rem;line-height:2;color:{TEXT}'>
        <b style='color:{GREEN}'>{"="*50}</b><br>
        <b>   MILESTONE 2 SUMMARY — REAL FITBIT DATA</b><br>
        <b style='color:{GREEN}'>{"="*50}</b><br><br>
        ✅ Users: <b>{clust_feats.shape[0]}</b> · Date range: <b>{date_span} days</b> (Mar–Apr 2016)<br>
        ✅ TSFresh: <b>{features.shape[1]} features</b> · <b>{features.shape[0]} users</b><br>
        ✅ Prophet: <b>{FORECAST_DAYS}-day</b> horizon · 80% CI (HR · Steps · Sleep)<br>
        ✅ KMeans: <b>{OPTIMAL_K} clusters</b> · DBSCAN: <b>{n_cl_db} clusters</b> · <b>{n_noise}</b> noise user(s)<br>
        </div>""",unsafe_allow_html=True)