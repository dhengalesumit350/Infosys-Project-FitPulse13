import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
import time

# ---------------------------------------------
# 1. CORE SYSTEM CONFIG
# ---------------------------------------------
st.set_page_config(page_title="FitPulse Pro | Elite Governance", layout="wide", page_icon="⚡")

# Initialize Session States
for key, default in {
    "raw_df": None, "clean_df": None, 
    "ingested": False, "diag_run": False, "processed": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------
# 2. ADVANCED UI THEMING (CSS)
# ---------------------------------------------
st.markdown("""
<style>
    /* Main Background Gradient */
    [data-testid="stAppViewContainer"] { 
        background: linear-gradient(135deg, #020617 0%, #0f172a 100%); 
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid #38bdf8;
    }

    /* Professional Glass Action Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2); 
        border-radius: 16px;
        padding: 24px; 
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* Status Badge Styling for Sidebar */
    .status-badge {
        padding: 10px 15px; 
        border-radius: 8px; 
        font-weight: 700; 
        font-size: 0.75rem;
        letter-spacing: 1px; 
        text-transform: uppercase; 
        margin-bottom: 12px; 
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .pending { background: #1e293b; color: #64748b; border: 1px solid #334155; }
    .complete { background: #064e3b; color: #10b981; border: 1px solid #059669; }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 3. SIDEBAR: LIVE SYSTEM MONITOR
# ---------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #38bdf8; text-align: center;'>🛡️ CORE MONITOR</h2>", unsafe_allow_html=True)
    st.divider()
    
    t1 = "✅" if st.session_state.ingested else "⭕"
    c1 = "complete" if st.session_state.ingested else "pending"
    
    t2 = "✅" if st.session_state.processed else "⭕"
    c2 = "complete" if st.session_state.processed else "pending"

    st.markdown(f'<div class="status-badge {c1}">INGESTION <span>{t1}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-badge {c2}">GOVERNANCE <span>{t2}</span></div>', unsafe_allow_html=True)
    
    if st.session_state.processed:
        st.success("INTEGRITY: OPTIMIZED")
    
    st.divider()
    if st.button("🔄 HARD SYSTEM REBOOT", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------
# 4. MAIN PIPELINE
# ---------------------------------------------
st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🏃‍♂️ FitPulse Pro: Data Governance</h1>", unsafe_allow_html=True)

# --- STEP 1: INGESTION ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📁 Step 1: Secure Data Ingestion")
file = st.file_uploader("Upload FitPulse CSV", type="csv", label_visibility="collapsed")
if file:
    temp_df = pd.read_csv(file)
    if st.session_state.raw_df is None or not st.session_state.raw_df.equals(temp_df):
        st.session_state.raw_df = temp_df
        st.session_state.ingested = True
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.ingested:
    # --- STEP 2: DIAGNOSTICS ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Step 2: Graphical Null Diagnostics")
    
    df = st.session_state.raw_df
    null_counts = df.isnull().sum().reset_index()
    null_counts.columns = ['Column', 'Count']
    null_data = null_counts[null_counts['Count'] > 0]
    
    if not null_data.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Null Distribution by Column")
            bar = alt.Chart(null_data).mark_bar(cornerRadius=5, color='#38bdf8').encode(
                x=alt.X('Column', sort='-y'), y='Count'
            ).properties(height=200)
            st.altair_chart(bar, use_container_width=True)
        with c2:
            st.write("Data Integrity Ratio")
            total = df.size
            nulls = df.isnull().sum().sum()
            pie_df = pd.DataFrame({'Status': ['Valid', 'Missing'], 'Value': [total-nulls, nulls]})
            pie = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                theta="Value", color=alt.Color("Status", scale=alt.Scale(range=['#10b981', '#f43f5e']))
            ).properties(height=200)
            st.altair_chart(pie, use_container_width=True)
    else:
        st.success("No anomalies found in source data.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- STEP 3: GOVERNANCE PROTOCOL ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Step 3: Governance Pipeline")
    st.write("• **FUNC_01**: Drop Null Dates | **FUNC_02**: Impute 'Workout_Type' | **FUNC_03**: Mean Fill Metrics")
    
    if st.button("🚀 DEPLOY CLEANING PROTOCOL", use_container_width=True):
        with st.status("Engaging Engine...", expanded=True) as status:
            clean = st.session_state.raw_df.copy()
            if 'Date' in clean.columns: clean = clean.dropna(subset=['Date'])
            if 'Workout_Type' in clean.columns: clean['Workout_Type'] = clean['Workout_Type'].fillna("General")
            for col in clean.columns:
                if clean[col].dtype in [np.float64, np.int64]:
                    clean[col] = clean[col].fillna(clean[col].mean())
            
            time.sleep(1)
            st.session_state.clean_df = clean
            st.session_state.processed = True
            status.update(label="System Optimized!", state="complete")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- STEP 4: PREVIEW & ANALYSIS ---
    if st.session_state.processed:
        # --- SUB-STEP: PREVIEW DATA ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👀 Step 3.5: Data Integrity Preview")
        if st.checkbox("🔍 SHOW CLEANED DATA PREVIEW"):
            st.dataframe(st.session_state.clean_df, use_container_width=True, hide_index=True)
            st.info(f"Integrity Check: {len(st.session_state.clean_df)} records ready for export.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ANALYSIS ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📈 Step 4: Processed Column Analysis")
        df_clean = st.session_state.clean_df
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if num_cols:
            cols = st.columns(min(len(num_cols), 3))
            for i, col in enumerate(num_cols[:3]):
                with cols[i]:
                    st.write(f"**{col}** (Post-Optimization)")
                    chart = alt.Chart(df_clean).mark_area(
                        line={'color':'#38bdf8'}, 
                        color=alt.Gradient(
                            gradient='linear', 
                            stops=[alt.GradientStop(color='#0ea5e9', offset=0), alt.GradientStop(color='transparent', offset=1)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(alt.X(col, bin=alt.Bin(maxbins=20)), alt.Y('count()')).properties(height=180)
                    st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- STEP 5: EDA ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔍 Step 5: Complete Governance EDA")
        
        tab_corr, tab_dist = st.tabs(["🔥 Correlation Matrix", "📊 Feature Distribution"])
        
        with tab_corr:
            num_df = df_clean.select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr().reset_index().melt(id_vars='index')
                heatmap = alt.Chart(corr).mark_rect().encode(
                    x='index:O', y='variable:O', 
                    color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'))
                ).properties(height=400)
                st.altair_chart(heatmap, use_container_width=True)
        
        with tab_dist:
            cols_eda = st.columns(2)
            for idx, col in enumerate(df_clean.columns):
                with cols_eda[idx % 2]:
                    st.markdown(f"**Field:** `{col.upper()}`")
                    if df_clean[col].dtype in [np.float64, np.int64]:
                        c = alt.Chart(df_clean).mark_bar(color='#38bdf8').encode(x=alt.X(col, bin=True), y='count()').properties(height=150)
                    else:
                        c = alt.Chart(df_clean).mark_bar().encode(x='count()', y=alt.Y(col, sort='-x'), color=col).properties(height=150)
                    st.altair_chart(c, use_container_width=True)

        # Download
        st.divider()
        buf = io.BytesIO()
        st.session_state.clean_df.to_csv(buf, index=False)
        st.download_button("📥 DOWNLOAD FINAL OPTIMIZED DATASET", data=buf.getvalue(), file_name="FitPulse_Elite_Clean.csv", mime="text/csv", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    