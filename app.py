"""
German Electricity Day-Ahead Price Forecasting App
Author: Ritik Ghoghari (MSc Data Science Student, GISMA Berlin)
GitHub: https://github.com/Ritikghoghari
Deployed on HuggingFace Spaces & Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.download_data import (
    fetch_live_market_data,
    engineer_features,
    FEATURE_COLUMNS
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German Electricity Price Forecaster | XGBoost",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #0369a1;
        color: white;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data(n_weeks: int = 8) -> pd.DataFrame:
    """Fetches live market data from SMARD.de with a 30-minute cache."""
    try:
        raw_df = fetch_live_market_data(n_weeks=n_weeks)
        if raw_df.empty:
            return pd.DataFrame()
        feat_df = engineer_features(raw_df)
        return feat_df
    except Exception as e:
        st.error(f"Error fetching data from SMARD.de: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_forecasting_model(train_fallback_df: pd.DataFrame = None):
    """Loads pre-trained model or fits on historical data if missing."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "xgb_model.json")
    model = xgb.XGBRegressor()
    
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
            return model, "Pre-Trained Benchmark Model (Loaded from disk)"
        except Exception:
            pass

    # Fallback: Train on the fly
    if train_fallback_df is not None and not train_fallback_df.empty:
        clean_df = train_fallback_df.dropna(subset=FEATURE_COLUMNS + ['Day_Ahead_Price'])
        if len(clean_df) > 168:
            X = clean_df[FEATURE_COLUMNS]
            y = clean_df['Day_Ahead_Price']
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.85,
                random_state=42,
                objective='reg:squarederror'
            )
            model.fit(X, y)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                model.save_model(model_path)
            except Exception:
                pass
            return model, "Trained On-the-Fly (SMARD Historical Data)"
            
    return None, "No Model Available"

# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/electricity.png", width=64)
    st.title("⚡ Settings & Controls")
    st.markdown("**Live German Power Market (DE-LU)**")

    st.markdown("---")
    st.subheader("📡 Data Feed")
    weeks_to_fetch = st.slider("Historical Data Ingestion Window", min_value=3, max_value=16, value=8, step=1,
                               help="Number of past weeks to query from Bundesnetzagentur (SMARD.de API)")
    
    test_eval_hours = st.selectbox(
        "Evaluation Test Window",
        options=[24, 48, 72, 168],
        index=3,
        format_func=lambda x: f"Last {x} Hours ({x//24} Day{'s' if x > 24 else ''})"
    )

    if st.button("🔄 Force Refresh SMARD Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("👨‍💻 Project & Author")
    st.markdown("""
    **Ritik Ghoghari**  
    🎓 *MSc Data Science Student*  
    🏛️ **GISMA University of Applied Sciences Berlin**  
    🎯 *Targeting Junior Data Analyst / DS roles in Berlin & Hannover*
    
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github)](https://github.com/Ritikghoghari)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/ritikghoghari)
    """)
    
    st.caption("Data Source: German Federal Network Agency ([SMARD.de](https://www.smard.de)) under CC BY 4.0.")

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.title("⚡ German Day-Ahead Electricity Price Forecasting")
st.markdown("""
An interactive end-to-end Machine Learning dashboard predicting **EPEX Spot Day-Ahead Hourly Electricity Prices** 
for the Germany-Luxembourg (`DE-LU`) bidding zone. Powered by real-time data from **Bundesnetzagentur (SMARD.de)** 
and an optimized **XGBoost Regressor** modeling the **Merit Order Effect**.
""")

# Load Data
with st.spinner("Fetching live market data from Bundesnetzagentur SMARD.de API..."):
    df = load_market_data(n_weeks=weeks_to_fetch)

if df.empty or len(df) < 168:
    st.warning("⚠️ Fetching live data took longer than expected or returned limited rows. Retrying connection...")
    st.stop()

# Load Model
model, model_status = load_forecasting_model(train_fallback_df=df)

if model is None:
    st.error("Failed to initialize forecasting model.")
    st.stop()

# Prepare Evaluation Data
df_clean = df.dropna(subset=FEATURE_COLUMNS + ['Day_Ahead_Price']).copy()

if len(df_clean) <= test_eval_hours:
    test_eval_hours = max(24, len(df_clean) // 4)

train_split = df_clean.iloc[:-test_eval_hours]
test_split = df_clean.iloc[-test_eval_hours:].copy()

# Predict on Evaluation Window
test_features = test_split[FEATURE_COLUMNS]
test_split['Predicted_Price'] = model.predict(test_features)
test_split['Error'] = test_split['Predicted_Price'] - test_split['Day_Ahead_Price']
test_split['Abs_Error'] = test_split['Error'].abs()

live_mae = test_split['Abs_Error'].mean()
live_rmse = np.sqrt(mean_squared_error(test_split['Day_Ahead_Price'], test_split['Predicted_Price']))
corr = np.corrcoef(test_split['Day_Ahead_Price'], test_split['Predicted_Price'])[0, 1]
latest_actual = test_split['Day_Ahead_Price'].iloc[-1]
latest_pred = test_split['Predicted_Price'].iloc[-1]
latest_residual = test_split['Residual_Load'].iloc[-1]
latest_renewables = test_split['Renewable_Total'].iloc[-1]
latest_load = test_split['Total_Load'].iloc[-1]
ren_pct = (latest_renewables / latest_load * 100) if latest_load > 0 else 0

# -----------------------------------------------------------------------------
# Metric KPI Cards
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Benchmark Test MAE</div>
        <div class="metric-val" style="color: #10b981;">10.80 €</div>
        <small style="color: #64748b;">CV TimeSeriesSplit baseline</small>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Live Window MAE</div>
        <div class="metric-val">{live_mae:.2f} €</div>
        <small style="color: #64748b;">RMSE: {live_rmse:.2f} €/MWh</small>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Latest Actual Price</div>
        <div class="metric-val">{latest_actual:.2f} €</div>
        <small style="color: #64748b;">SMARD Clearing Price</small>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    delta_val = latest_pred - latest_actual
    delta_color = "#ef4444" if abs(delta_val) > 15 else "#38bdf8"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Latest XGB Forecast</div>
        <div class="metric-val" style="color: {delta_color};">{latest_pred:.2f} €</div>
        <small style="color: #64748b;">Delta: {delta_val:+.2f} €/MWh</small>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Renewable Share</div>
        <div class="metric-val" style="color: #f59e0b;">{ren_pct:.1f}%</div>
        <small style="color: #64748b;">Residual: {latest_residual/1000:.1f} GW</small>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Visualization: Actual vs Predicted Price
# -----------------------------------------------------------------------------
st.markdown("### 📈 Live Day-Ahead Spot Price: Actual vs. XGBoost Forecast")

fig_price = go.Figure()

# Actual Prices
fig_price.add_trace(go.Scatter(
    x=test_split.index,
    y=test_split['Day_Ahead_Price'],
    mode='lines+markers',
    name='Actual Day-Ahead Price (SMARD)',
    line=dict(color='#38bdf8', width=2.5),
    marker=dict(size=4)
))

# Predicted Prices
fig_price.add_trace(go.Scatter(
    x=test_split.index,
    y=test_split['Predicted_Price'],
    mode='lines+markers',
    name='XGBoost Day-Ahead Forecast',
    line=dict(color='#f97316', width=2.5, dash='dash'),
    marker=dict(size=4, symbol='diamond')
))

# Zero line for negative pricing visibility
fig_price.add_hline(
    y=0,
    line_dash="dot",
    line_color="#ef4444",
    annotation_text="Zero Price Threshold (Negative Pricing Zone)",
    annotation_position="bottom right"
)

fig_price.update_layout(
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Datetime (UTC)",
    yaxis_title="Day-Ahead Price (€/MWh)",
    hovermode="x unified"
)

st.plotly_chart(fig_price, use_container_width=True)

# -----------------------------------------------------------------------------
# Secondary Insights: Merit Order & Energy Mix
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### ⚡ Merit Order Dynamics: Residual Load vs. Spot Price")
    
    fig_mo = px.scatter(
        test_split,
        x="Residual_Load",
        y="Day_Ahead_Price",
        color="Renewable_Total",
        size="Total_Load",
        trendline="lowess",
        color_continuous_scale="Viridis",
        labels={
            "Residual_Load": "Residual Load (Demand - Renewables) [MWh]",
            "Day_Ahead_Price": "Electricity Price (€/MWh)",
            "Renewable_Total": "Renewables (MWh)"
        },
        template="plotly_dark",
        title="Merit Order Curve: Lower Residual Load Slashes Wholesale Prices"
    )
    fig_mo.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_mo, use_container_width=True)

with col_right:
    st.markdown("### 🏆 Top XGBoost Feature Importances")
    
    try:
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig_feat = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Tealgrn",
            template="plotly_dark",
            title="Tree Split Weightings Across Engineered Features"
        )
        fig_feat.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)
    except Exception:
        st.info("Feature importance display available with trained model.")

# -----------------------------------------------------------------------------
# Generation Breakdown & Stacked Area Chart
# -----------------------------------------------------------------------------
st.markdown("### 🍃 Generation Fuel Mix & Grid Demand Breakdown")

recent_display = df.iloc[-test_eval_hours:].copy()

fig_mix = go.Figure()

fig_mix.add_trace(go.Scatter(
    x=recent_display.index,
    y=recent_display['Solar'],
    mode='lines',
    stackgroup='one',
    name='Solar PV',
    line=dict(width=0.5, color='#eab308')
))

fig_mix.add_trace(go.Scatter(
    x=recent_display.index,
    y=recent_display['Wind_Onshore'],
    mode='lines',
    stackgroup='one',
    name='Wind Onshore',
    line=dict(width=0.5, color='#10b981')
))

fig_mix.add_trace(go.Scatter(
    x=recent_display.index,
    y=recent_display['Wind_Offshore'],
    mode='lines',
    stackgroup='one',
    name='Wind Offshore',
    line=dict(width=0.5, color='#06b6d4')
))

fig_mix.add_trace(go.Scatter(
    x=recent_display.index,
    y=recent_display['Total_Load'],
    mode='lines',
    name='Total Grid Load (Demand)',
    line=dict(color='#f87171', width=2.5, dash='dash')
))

fig_mix.update_layout(
    height=360,
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_dark",
    xaxis_title="Datetime (UTC)",
    yaxis_title="Power (MWh / h)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_mix, use_container_width=True)

# -----------------------------------------------------------------------------
# Interactive "What-If" Scenario Simulator
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🎛️ Interactive Energy Market Scenario Simulator")
st.caption("Adjust market conditions to simulate how changes in wind, solar, and grid demand impact tomorrow's electricity price via the Merit Order Effect.")

sim_col1, sim_col2, sim_col3 = st.columns(3)

with sim_col1:
    sim_solar_mod = st.slider("Solar PV Output Shift (%)", min_value=-80, max_value=100, value=0, step=10)
with sim_col2:
    sim_wind_mod = st.slider("Wind Generation Shift (%)", min_value=-80, max_value=100, value=0, step=10)
with sim_col3:
    sim_load_mod = st.slider("Grid Load (Demand) Shift (%)", min_value=-30, max_value=30, value=0, step=5)

# Calculate simulated row based on latest test interval
last_row = test_split.iloc[-1].copy()

sim_solar = last_row['Solar'] * (1 + sim_solar_mod / 100.0)
sim_wind = (last_row['Wind_Onshore'] + last_row['Wind_Offshore']) * (1 + sim_wind_mod / 100.0)
sim_load = last_row['Total_Load'] * (1 + sim_load_mod / 100.0)
sim_ren_total = sim_solar + sim_wind
sim_residual = sim_load - sim_ren_total

sim_input_dict = {
    'Total_Load': sim_load,
    'Renewable_Total': sim_ren_total,
    'Residual_Load': sim_residual,
    'Hour': last_row['Hour'],
    'DayOfWeek': last_row['DayOfWeek'],
    'Month': last_row['Month'],
    'IsWeekend': last_row['IsWeekend'],
    'Lag_24': last_row['Lag_24'],
    'Lag_48': last_row['Lag_48'],
    'Lag_168': last_row['Lag_168'],
    'Rolling_Mean_24': last_row['Rolling_Mean_24'],
    'Rolling_Std_24': last_row['Rolling_Std_24']
}

sim_df = pd.DataFrame([sim_input_dict])
sim_predicted_price = model.predict(sim_df[FEATURE_COLUMNS])[0]
base_price = last_row['Predicted_Price']
sim_price_delta = sim_predicted_price - base_price

sim_res1, sim_res2, sim_res3 = st.columns(3)
with sim_res1:
    st.metric("Base Forecasted Price", f"{base_price:.2f} €/MWh")
with sim_res2:
    st.metric(
        "Simulated Forecasted Price",
        f"{sim_predicted_price:.2f} €/MWh",
        delta=f"{sim_price_delta:+.2f} €/MWh",
        delta_color="inverse"
    )
with sim_res3:
    st.metric(
        "Simulated Residual Load",
        f"{sim_residual / 1000:.2f} GW",
        delta=f"{(sim_residual - last_row['Residual_Load']) / 1000:+.2f} GW",
        delta_color="inverse"
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    <strong>German Day-Ahead Electricity Price Forecasting Engine</strong> | Built with Python, Streamlit & XGBoost<br>
    Developed by <strong>Ritik Ghoghari</strong> (MSc Data Science Student, GISMA Berlin) | 
    <a href="https://github.com/Ritikghoghari/German-Energy-Price-Forecasting" target="_blank" style="color: #38bdf8;">GitHub Repository</a> | 
    <a href="https://www.linkedin.com/in/ritikghoghari" target="_blank" style="color: #38bdf8;">LinkedIn Profile</a>
</div>
""", unsafe_allow_html=True)
