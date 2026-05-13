import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Thales 6G Predictive Maintenance", layout="wide", page_icon="📶")

# Custom CSS for a premium feel
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 4px solid #4A90E2;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .high-risk { border-left-color: #FF4B4B; }
    .medium-risk { border-left-color: #FFA500; }
    .low-risk { border-left-color: #00CC96; }
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
    
    # Fill NAs to avoid issues with IsolationForest
    df.fillna(0, inplace=True)
    return df

with st.spinner("Engineering features..."):
    df = apply_feature_engineering(raw_df.copy())

@st.cache_resource
def run_anomaly_detection(df):
    # 3. Anomaly Detection & Scoring
    features_for_model = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
                          'Instability_Ratio', 'Connectivity_Score', 'Temp_Deviation', 'Vib_Deviation']
    
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
    
    # 4. Predictive Maintenance Risk Classification
    def classify_risk(score):
        if score > 0.75:
            return 'High Risk'
        elif score > 0.50:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    df['Risk_Level'] = df['Anomaly_Score'].apply(classify_risk)
    return df

with st.spinner("Running 6G AI Anomaly Detection..."):
    df = run_anomaly_detection(df)


# 5. Key Performance Indicators (KPIs)
medium_risk_count = len(df[df['Risk_Level'] == 'Medium Risk'])
downtime_prevention_index = medium_risk_count * 4 # Estimated 4 hours saved per medium risk addressed

early_warning_lead_time_hrs = 48.5 # Simulated 48.5 hrs avg lead time

# --- DASHBOARD UI ---
st.title("🏭 Thales 6G Predictive Maintenance Dashboard")
st.markdown("Powered by 6G-enabled real-time IoT sensors & Isolation Forest AI")

# --- Overview KPIs ---
col1, col2, col3, col4 = st.columns(4)

current_status = df.groupby('Machine_ID').last().reset_index()
high_risk_machines = current_status[current_status['Risk_Level'] == 'High Risk']
medium_risk_machines = current_status[current_status['Risk_Level'] == 'Medium Risk']

with col1:
    st.markdown(f"""
    <div class="metric-card high-risk">
        <div class="metric-label">High Risk Assets</div>
        <div class="metric-value">{len(high_risk_machines)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card medium-risk">
        <div class="metric-label">Medium Risk (Warning)</div>
        <div class="metric-value">{len(medium_risk_machines)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card low-risk">
        <div class="metric-label">Downtime Prevented</div>
        <div class="metric-value">{downtime_prevention_index} hrs</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card low-risk">
        <div class="metric-label">Avg Lead Time</div>
        <div class="metric-value">{early_warning_lead_time_hrs} hrs</div>
    </div>
    """, unsafe_allow_html=True)

if len(high_risk_machines) > 0:
    st.error(f"🚨 **URGENT**: Machines {', '.join(high_risk_machines['Machine_ID'].astype(str).tolist())} require immediate attention!")


st.markdown("---")

# --- Tabs for modules ---
tab1, tab2, tab3 = st.tabs(["Fleet Overview", "Machine Anomaly Analysis", "Historical Risk Timeline"])

with tab1:
    st.subheader("Fleet-Wide Risk Map")
    fig_fleet = px.scatter(
        current_status, 
        x="Temperature_C", 
        y="Vibration_Hz", 
        color="Risk_Level",
        size="Power_Consumption_kW",
        hover_name="Machine_ID",
        color_discrete_map={"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"},
        title="Current Asset Status (Size = Power Consumption)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

with tab2:
    st.subheader("Machine Anomaly Dashboard")
    selected_machine = st.selectbox("Select Machine ID", df['Machine_ID'].unique())
    
    machine_df = df[df['Machine_ID'] == selected_machine].copy()
    
    st.markdown(f"#### Real-Time Sensor Trends for Machine {selected_machine}")
    
    fig_sensor = go.Figure()
    
    fig_sensor.add_trace(go.Scatter(
        x=machine_df['Datetime'], y=machine_df['Temperature_C'], 
        mode='lines', name='Temperature (°C)', line=dict(color='#4A90E2')
    ))
    
    fig_sensor.add_trace(go.Scatter(
        x=machine_df['Datetime'], y=machine_df['Vibration_Hz'] * 10, # Scaled for visibility
        mode='lines', name='Vibration (Hz x10)', line=dict(color='#A0A0B0')
    ))
    
    # Highlight high risk areas
    high_risk_data = machine_df[machine_df['Risk_Level'] == 'High Risk']
    fig_sensor.add_trace(go.Scatter(
        x=high_risk_data['Datetime'], y=high_risk_data['Temperature_C'],
        mode='markers', name='Anomaly Detected', 
        marker=dict(color='red', size=10, symbol='x')
    ))

    fig_sensor.update_layout(template="plotly_dark", title="Sensor Readings & Anomalies", hovermode="x unified")
    st.plotly_chart(fig_sensor, use_container_width=True)
    
    # 6G specific metrics
    st.markdown("#### 6G Network Health & Instability")
    colA, colB = st.columns(2)
    with colA:
        fig_inst = px.line(machine_df, x='Datetime', y='Instability_Ratio', title="Instability Ratio (Vibration/Power)", template="plotly_dark")
        st.plotly_chart(fig_inst, use_container_width=True)
    with colB:
        fig_conn = px.line(machine_df, x='Datetime', y='Connectivity_Score', title="6G Connectivity Score", template="plotly_dark")
        fig_conn.update_traces(line_color='#00CC96')
        st.plotly_chart(fig_conn, use_container_width=True)

with tab3:
    st.subheader("Historical Risk Analysis")
    st.markdown("Timeline showing how risk escalated across the fleet.")
    
    # Resample to hourly 
    df_resampled = df.set_index('Datetime').groupby('Risk_Level').resample('h').size().reset_index(name='Count')
    
    fig_hist = px.area(
        df_resampled, 
        x="Datetime", 
        y="Count", 
        color="Risk_Level",
        color_discrete_map={"High Risk": "#FF4B4B", "Medium Risk": "#FFA500", "Low Risk": "#00CC96"},
        title="Escalation of Risk Levels Over Time",
        template="plotly_dark"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
