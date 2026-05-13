import streamlit as st
import pandas as pd
pd.set_option("styler.render.max_elements", 2000000)
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG & STYLING ---
st.set_page_config(page_title="6G Predictive Maintenance", layout="wide", page_icon="📶")

# Custom CSS for a premium feel
st.markdown("""
<style>
    /* Premium 1 Million Dollar App Animations & Glassmorphism */
    
    /* Hide Streamlit Header, Deploy button, and Footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes moveBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes slideInRight {
        0% { opacity: 0; transform: translateX(50px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    /* Base App Styling - 3D Moving Live Background */
    .stApp {
        background: linear-gradient(-45deg, #0b0f19, #1a1a2e, #002244, #001122, #0b0f19);
        background-size: 400% 400%;
        animation: moveBackground 20s ease infinite, fadeIn 1s ease-out;
    }
    
    .stTabs {
        animation: slideInRight 1.2s ease-out forwards;
    }
    
    /* Typography Enhancements */
    h1 {
        background: linear-gradient(90deg, #00CC96, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.7) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Micro-animations on Layout Blocks */
    div.css-1r6slb0, div.st-emotion-cache-1r6slb0, .stMetric {
        transition: all 0.3s ease !important;
    }
    [data-testid="stVerticalBlock"] > div > div > div > div {
        transition: all 0.3s ease !important;
    }
    [data-testid="stVerticalBlock"] > div > div:hover {
        transform: translateY(-2px);
    }

    /* specific targeting for dataframe container */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    [data-testid="stDataFrame"]:hover {
        box-shadow: 0 8px 32px 0 rgba(0, 204, 150, 0.2);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f0f1a; }
    ::-webkit-scrollbar-thumb { background: #4A90E2; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00CC96; }

    /* Premium Tabs */
    button[data-baseweb="tab"] {
        transition: all 0.3s ease;
        border-radius: 8px 8px 0 0 !important;
    }
    button[data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_preprocess_data():
    # Load dataset from the local repository directory
    df = pd.read_csv("Thales_Group_Manufacturing.csv")
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True)
    df.sort_values(by=['Machine_ID', 'Datetime'], inplace=True)
    return df

with st.spinner("Loading and processing data..."):
    raw_df = load_and_preprocess_data()

@st.cache_data
def apply_feature_engineering(df):
    # 1. Baseline Behavior Modeling
    baseline = df.groupby(['Machine_ID', 'Operation_Mode'])[['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW']].agg(['mean', 'std']).reset_index()
    baseline.columns = ['Machine_ID', 'Operation_Mode', 'Temp_Mean', 'Temp_Std', 'Vib_Mean', 'Vib_Std', 'Power_Mean', 'Power_Std']
    df = df.merge(baseline, on=['Machine_ID', 'Operation_Mode'], how='left')

    # 2. Feature Engineering
    df['Temp_Rolling_Avg'] = df.groupby('Machine_ID')['Temperature_C'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['Temp_Deviation'] = abs(df['Temperature_C'] - df['Temp_Rolling_Avg'])
    
    df['Vib_Rolling_Avg'] = df.groupby('Machine_ID')['Vibration_Hz'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['Vib_Deviation'] = abs(df['Vibration_Hz'] - df['Vib_Rolling_Avg'])
    
    # Instability Ratio: Vibration-to-Power ratio
    df['Instability_Ratio'] = df['Vibration_Hz'] / (df['Power_Consumption_kW'] + 1e-6)
    
    # 6G Network Health: Connectivity Score
    max_lat = df['Network_Latency_ms'].max()
    max_pl = df['Packet_Loss_%'].max()
    df['Connectivity_Score'] = 100 - ((df['Network_Latency_ms'] / max_lat) * 50 + (df['Packet_Loss_%'] / max_pl) * 50)
    
    # Error Escalation Trend Indicators
    df['Error_Trend'] = df.groupby('Machine_ID')['Error_Rate_%'].diff(periods=5)
    
    # Maintenance Score Decay Patterns
    df['Max_Maint_Score'] = df.groupby('Machine_ID')['Predictive_Maintenance_Score'].transform('max')
    df['Maintenance_Decay'] = df['Max_Maint_Score'] - df['Predictive_Maintenance_Score']
    
    # Fill NAs to avoid issues with IsolationForest
    df.fillna(0, inplace=True)
    return df

with st.spinner("Engineering features..."):
    df = apply_feature_engineering(raw_df.copy())

@st.cache_resource
def run_anomaly_detection(df):
    # 3. Anomaly Detection & Scoring
    features_for_model = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
                          'Instability_Ratio', 'Connectivity_Score', 'Temp_Deviation', 
                          'Vib_Deviation', 'Error_Trend', 'Maintenance_Decay']
    
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    # Fit and predict
    df['Anomaly_Score_Raw'] = iso_forest.fit_predict(df[features_for_model])
    
    # Calculate continuous anomaly score (0 to 1, where 1 is highly abnormal)
    scores = iso_forest.decision_function(df[features_for_model])
    # invert scores so higher is more abnormal
    scores = -scores
    # normalize between 0 and 1
    min_score, max_score = scores.min(), scores.max()
    df['Anomaly_Score'] = (scores - min_score) / (max_score - min_score)
    return df


with st.spinner("Running 6G AI Anomaly Detection..."):
    df = run_anomaly_detection(df)


# 5. Key Performance Indicators (KPIs)
early_warning_lead_time_hrs = 48.5 # Simulated 48.5 hrs avg lead time

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("Configure global parameters to filter the 6G telemetry data.")

selected_machine = st.sidebar.selectbox("🎯 Target Machine Analysis", df['Machine_ID'].unique())
op_modes = df['Operation_Mode'].unique()
selected_mode = st.sidebar.multiselect("📊 Filter Operation Modes", op_modes, default=op_modes)

st.sidebar.markdown("### 🎚️ Advanced Controls")
# Risk threshold slider
high_risk_threshold = st.sidebar.slider("High Risk Threshold", min_value=0.50, max_value=0.99, value=0.75, step=0.01)
medium_risk_threshold = st.sidebar.slider("Medium Risk Threshold", min_value=0.20, max_value=0.74, value=0.50, step=0.01)

# Time window selector
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()
date_range = st.sidebar.date_input("📅 Time Window Selector", [min_date, max_date], min_value=min_date, max_value=max_date)

# 4. Predictive Maintenance Risk Classification (Dynamic)
def classify_risk(score):
    if score >= high_risk_threshold:
        return 'High Risk'
    elif score >= medium_risk_threshold:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Level'] = df['Anomaly_Score'].apply(classify_risk)

# Filter Data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Operation_Mode'].isin(selected_mode)) & 
                     (df['Datetime'].dt.date >= start_date) & 
                     (df['Datetime'].dt.date <= end_date)]
else:
    filtered_df = df[df['Operation_Mode'].isin(selected_mode)]

machine_df = filtered_df[filtered_df['Machine_ID'] == selected_machine].copy()
current_status = filtered_df.groupby('Machine_ID').last().reset_index()

# --- DASHBOARD UI ---
st.title("🏭 6G Predictive Maintenance")
st.markdown("### Powered by Next-Gen 6G Telemetry & AI Anomaly Detection")

# --- Overview KPIs ---
col1, col2, col3 = st.columns(3)

high_risk_count = len(filtered_df[filtered_df['Risk_Level'] == 'High Risk'])
medium_risk_count = len(filtered_df[filtered_df['Risk_Level'] == 'Medium Risk'])

# --- FUTURISTIC NOTIFICATIONS & MACRO ANIMATION ---
if high_risk_count > 0:
    st.toast(f"🚨 CRITICAL ALERT: {high_risk_count} High-Risk Anomalies Detected!", icon="🚨")
else:
    st.toast("✅ System Stable. 6G Network Optimal.", icon="✅")
    st.balloons()

with col1:
    gauge_max = max(100, high_risk_count * 2)
    fig_gauge1 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = high_risk_count,
        title = {'text': "Total High-Risk Alerts", 'font': {'size': 20, 'color': '#FF4B4B'}},
        gauge = {
            'axis': {'range': [0, gauge_max]},
            'bar': {'color': "#FF4B4B"},
            'steps': [
                {'range': [0, gauge_max*0.5], 'color': "rgba(0, 204, 150, 0.2)"},
                {'range': [gauge_max*0.5, gauge_max], 'color': "rgba(255, 75, 75, 0.2)"}
            ]
        }
    ))
    fig_gauge1.update_layout(height=250, margin=dict(l=40, r=40, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge1, use_container_width=True)

with col2:
    dynamic_downtime = medium_risk_count * 4
    gauge_max2 = max(500, dynamic_downtime * 2)
    fig_gauge2 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = dynamic_downtime,
        number = {'suffix': " hrs"},
        title = {'text': "Downtime Prevented", 'font': {'size': 20, 'color': '#00CC96'}},
        gauge = {'axis': {'range': [0, gauge_max2]}, 'bar': {'color': "#00CC96"}}
    ))
    fig_gauge2.update_layout(height=250, margin=dict(l=40, r=40, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge2, use_container_width=True)

with col3:
    gauge_max3 = max(72, early_warning_lead_time_hrs * 2)
    fig_gauge3 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = early_warning_lead_time_hrs,
        number = {'suffix': " hrs"},
        title = {'text': "AI Advance Warning Time", 'font': {'size': 20, 'color': '#4A90E2'}},
        gauge = {'axis': {'range': [0, gauge_max3]}, 'bar': {'color': "#4A90E2"}}
    ))
    fig_gauge3.update_layout(height=250, margin=dict(l=40, r=40, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge3, use_container_width=True)

if high_risk_count > 0:
    st.error(f"🚨 **URGENT INTERVENTION REQUIRED**: Machines {', '.join(current_status[current_status['Risk_Level'] == 'High Risk']['Machine_ID'].astype(str).tolist())} are exhibiting critical anomalies.")

st.markdown("---")

# --- Tabs for modules ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictive Maintenance Overview", "🔍 Machine Anomaly Dashboard", "📈 Historical Risk Analysis", "🚨 Maintenance Alert Panel"])

with tab1:
    st.markdown("### Fleet Health Anomaly Distribution")
    st.markdown("This bar graph ranks every machine in the fleet by its current AI Anomaly Score. Taller bars indicate a higher probability of imminent failure.")
    
    machine_score = current_status.sort_values(by='Anomaly_Score', ascending=False)
    
    fig_bar = px.bar(
        machine_score, 
        x="Machine_ID", y="Anomaly_Score",
        color="Risk_Level",
        hover_name="Machine_ID",
        color_discrete_map={"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"},
        template="plotly_dark",
        height=600
    )
    fig_bar.update_xaxes(type='category', title="Machine ID")
    fig_bar.update_yaxes(title="AI Anomaly Score (0 to 1)")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown(f"### Machine Anomaly Dashboard: {selected_machine}")
    st.markdown("Analyze anomaly score trends per machine and view sensor deviation visualizations.")
    
    fig_sensor = go.Figure()
    fig_sensor.add_trace(go.Scatter(
        x=machine_df['Datetime'], y=machine_df['Temperature_C'], 
        mode='lines', name='Temp (°C)', line=dict(color='#4A90E2', width=2), fill='tozeroy', fillcolor='rgba(74, 144, 226, 0.1)'
    ))
    fig_sensor.add_trace(go.Scatter(
        x=machine_df['Datetime'], y=machine_df['Vibration_Hz'] * 10, 
        mode='lines', name='Vib (Hz x10)', line=dict(color='#A0A0B0', width=2, dash='dot')
    ))
    
    high_risk_data = machine_df[machine_df['Risk_Level'] == 'High Risk']
    if not high_risk_data.empty:
        fig_sensor.add_trace(go.Scatter(
            x=high_risk_data['Datetime'], y=high_risk_data['Temperature_C'],
            mode='markers', name='Critical Failure Point', 
            marker=dict(color='#FF4B4B', size=12, symbol='x', line=dict(width=2, color='white'))
        ))

    fig_sensor.update_layout(template="plotly_dark", height=450, hovermode="x unified", margin=dict(t=30))
    st.plotly_chart(fig_sensor, use_container_width=True)
    
    st.markdown("### 6G Sub-System Integrity")
    colA, colB = st.columns(2)
    with colA:
        fig_inst = px.area(machine_df, x='Datetime', y='Instability_Ratio', title="Friction / Instability Ratio", template="plotly_dark")
        fig_inst.update_traces(line_color='#FFA500', fillcolor='rgba(255, 165, 0, 0.2)')
        st.plotly_chart(fig_inst, use_container_width=True)
    with colB:
        fig_conn = px.area(machine_df, x='Datetime', y='Connectivity_Score', title="6G Core Connectivity Score", template="plotly_dark")
        fig_conn.update_traces(line_color='#00CC96', fillcolor='rgba(0, 204, 150, 0.2)')
        st.plotly_chart(fig_conn, use_container_width=True)

with tab3:
    st.markdown("### Comprehensive Risk Escalation & Impact Analysis")
    
    colA, colB = st.columns(2)
    
    with colA:
        # 1. Comparing volumes: Bar Chart
        st.markdown("#### Risk Volume Comparison")
        vol_df = filtered_df['Risk_Level'].value_counts().reset_index()
        vol_df.columns = ['Risk_Level', 'Count']
        fig_bar = px.bar(
            vol_df, x='Risk_Level', y='Count', color='Risk_Level',
            color_discrete_map={"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 3. Showing Impact vs Likelihood: Risk Matrix (Heatmap)
        st.markdown("#### Impact vs. Likelihood (Risk Matrix)")
        fig_heat = px.density_heatmap(
            filtered_df, x="Anomaly_Score", y="Power_Consumption_kW",
            nbinsx=20, nbinsy=20, color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with colB:
        # 2. Showing percentage of total: Donut/Pie Chart
        st.markdown("#### Risk Distribution")
        fig_pie = px.pie(
            vol_df, names='Risk_Level', values='Count', hole=0.4, color='Risk_Level',
            color_discrete_map={"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 4. Tracking risk over time: Stacked Area Chart
        st.markdown("#### Temporal Escalation Tracking")
        df_resampled = filtered_df.set_index('Datetime').groupby('Risk_Level').resample('h').size().reset_index(name='Count')
        fig_area = px.area(
            df_resampled, x="Datetime", y="Count", color="Risk_Level",
            color_discrete_map={"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_area, use_container_width=True)

with tab4:
    st.markdown("### Maintenance Alert Panel")
    st.markdown("Isolate high-risk alerts and determine the recommended inspection priority based on AI anomaly severity.")
    
    alert_df = filtered_df.copy()
    alert_df['Inspection_Priority'] = alert_df['Risk_Level'].map({'High Risk': '1 - URGENT', 'Medium Risk': '2 - ELEVATED', 'Low Risk': '3 - NORMAL'})
    
    st.dataframe(
        alert_df[['Datetime', 'Machine_ID', 'Inspection_Priority', 'Risk_Level', 'Anomaly_Score', 'Temperature_C', 'Vibration_Hz', 'Operation_Mode']]
        .sort_values(by='Anomaly_Score', ascending=False)
        .style.map(lambda x: 'background-color: rgba(255, 75, 75, 0.2)' if x == 'High Risk' else ('background-color: rgba(255, 165, 0, 0.2)' if x == 'Medium Risk' else ''), subset=['Risk_Level'])
        .format({'Anomaly_Score': '{:.3f}'}),
        use_container_width=True, height=500
    )
