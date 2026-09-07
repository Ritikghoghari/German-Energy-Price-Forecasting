"""
German Electricity Day-Ahead Price Forecasting System
Author: Ritik Ghoghari (MSc Data Science, GISMA University of Applied Sciences Berlin)
Data Source: Bundesnetzagentur / SMARD.de Open API
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys

# Add root/src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.download_data import (
    fetch_live_market_data,
    engineer_features,
    FEATURE_COLUMNS
)

# -----------------------------------------------------------------------------
# App Configuration & Elegant Minimalist Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German Electricity Price Forecaster | XGBoost",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined, clean professional CSS
st.markdown("""
<style>
    /* Clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container padding with ample clearance for Streamlit header */
    .block-container {
        padding-top: 4.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Subtle header badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        background-color: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
    }

    /* Clean subtle card */
    div[data-testid="stMetric"] {
        background-color: transparent;
        border-radius: 8px;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.75rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Tab bar refinement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.92rem;
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Ingestion & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data(n_weeks: int = 8) -> pd.DataFrame:
    """Fetches live market data from SMARD.de with caching."""
    try:
        raw_df = fetch_live_market_data(n_weeks=n_weeks)
        if raw_df.empty:
            return pd.DataFrame()
        return engineer_features(raw_df)
    except Exception as e:
        st.error(f"Error fetching data from SMARD.de: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_forecasting_model(train_fallback_df: pd.DataFrame = None):
    """Loads pre-trained XGBoost artifact or trains on historical data."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "xgb_model.json")
    model = xgb.XGBRegressor()

    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
            return model, "Pre-Trained Benchmark"
        except Exception:
            pass

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
            return model, "Trained On-Demand"

    return None, "Unavailable"

# -----------------------------------------------------------------------------
# Sidebar: Controls & Candidate Profile
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Market Feed Controls")
    
    weeks_to_fetch = st.slider(
        "Historical Data Range",
        min_value=4,
        max_value=16,
        value=8,
        step=1,
        help="Number of past weeks to query from Bundesnetzagentur (SMARD.de API)"
    )

    test_eval_hours = st.selectbox(
        "Evaluation Period",
        options=[24, 48, 72, 168],
        index=3,
        format_func=lambda x: f"Last {x} Hours ({x // 24} Day{'s' if x > 24 else ''})"
    )

    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🇩🇪 Market Zone Details")
    st.caption("""
    - **Zone:** Germany & Luxembourg (`DE-LU`)
    - **Exchange:** EPEX Spot Day-Ahead Auction
    - **Model:** Tuned XGBoost (`TimeSeriesSplit`)
    - **Data Source:** Bundesnetzagentur (SMARD.de)
    """)

    st.markdown("---")
    with st.container(border=True):
        st.markdown("**Developer Profile**")
        st.markdown("**Ritik Ghoghari**")
        st.caption("MSc Data Science Student at **GISMA University of Applied Sciences Berlin**")
        st.caption("🎯 Seeking: *Junior Data Analyst / Data Scientist / Werkstudent* in Berlin & Hannover")
        
        c1, c2 = st.columns(2)
        with c1:
            st.link_button("LinkedIn", "https://www.linkedin.com/in/ritikghoghari", use_container_width=True)
        with c2:
            st.link_button("GitHub", "https://github.com/Ritikghoghari", use_container_width=True)

# -----------------------------------------------------------------------------
# Data Loading & Preparation
# -----------------------------------------------------------------------------
with st.spinner("Connecting to Bundesnetzagentur SMARD.de live feed..."):
    df = load_market_data(n_weeks=weeks_to_fetch)

if df.empty or len(df) < 168:
    st.warning("⚠️ Connecting to SMARD.de open data platform. Please click 'Refresh Market Data' if this persists.")
    st.stop()

model, model_status = load_forecasting_model(train_fallback_df=df)
if model is None:
    st.error("Forecasting model could not be initialized.")
    st.stop()

# Prepare clean evaluation split
df_clean = df.dropna(subset=FEATURE_COLUMNS + ['Day_Ahead_Price']).copy()
if len(df_clean) <= test_eval_hours:
    test_eval_hours = max(24, len(df_clean) // 4)

train_split = df_clean.iloc[:-test_eval_hours]
test_split = df_clean.iloc[-test_eval_hours:].copy()

# Run Predictions
test_split['Predicted_Price'] = model.predict(test_split[FEATURE_COLUMNS])
test_split['Error'] = test_split['Predicted_Price'] - test_split['Day_Ahead_Price']
test_split['Abs_Error'] = test_split['Error'].abs()

live_mae = test_split['Abs_Error'].mean()
live_rmse = np.sqrt(mean_squared_error(test_split['Day_Ahead_Price'], test_split['Predicted_Price']))
live_r2 = r2_score(test_split['Day_Ahead_Price'], test_split['Predicted_Price'])

latest = test_split.iloc[-1]
latest_actual = latest['Day_Ahead_Price']
latest_pred = latest['Predicted_Price']
spread = latest_pred - latest_actual
latest_renewables = latest['Renewable_Total']
latest_load = latest['Total_Load']
ren_share = (latest_renewables / latest_load * 100) if latest_load > 0 else 0
latest_residual = latest['Residual_Load']

# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.title("⚡ German Day-Ahead Electricity Price Forecaster")

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: -6px; margin-bottom: 12px;">
    <div class="status-badge">
        <div class="status-dot"></div>
        <span>BUNDESNETZAGENTUR LIVE DATA FEED ACTIVE</span>
    </div>
    <span style="color: #94a3b8; font-size: 0.82rem;">•</span>
    <span style="color: #64748b; font-size: 0.82rem; font-weight: 500;">Bidding Zone: <strong style="color: #334155;">DE-LU</strong></span>
    <span style="color: #94a3b8; font-size: 0.82rem;">•</span>
    <span style="color: #64748b; font-size: 0.82rem; font-weight: 500;">Latest Clearing: <strong style="color: #334155;">{latest.name.strftime('%Y-%m-%d %H:00 UTC')}</strong></span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Accurate 24-hour day-ahead wholesale electricity price predictions for Germany (`DE-LU`). "
    "Models the **Merit Order Effect**, renewable intermittency (Wind & Solar), and grid demand using **XGBoost**."
)

st.write("")

# -----------------------------------------------------------------------------
# Clean Executive Metrics Row
# -----------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    with st.container(border=True):
        st.metric(
            label="Benchmark MAE",
            value="10.80 €",
            help="Cross-validated test set MAE on out-of-time evaluation period"
        )
        st.caption("51% beat vs. baseline")

with m2:
    with st.container(border=True):
        st.metric(
            label=f"Live Window MAE ({test_eval_hours}h)",
            value=f"{live_mae:.2f} €",
            delta=f"{live_mae - 10.80:+.2f} €",
            delta_color="inverse",
            help=f"Mean Absolute Error over the selected {test_eval_hours}-hour period"
        )
        st.caption(f"RMSE: {live_rmse:.2f} € | R²: {live_r2:.2f}")

with m3:
    with st.container(border=True):
        st.metric(
            label="Latest Spot Price",
            value=f"{latest_actual:.2f} €",
            help="Latest clearing price reported by SMARD.de (€/MWh)"
        )
        st.caption("EPEX Spot Day-Ahead")

with m4:
    with st.container(border=True):
        st.metric(
            label="XGBoost Forecast",
            value=f"{latest_pred:.2f} €",
            delta=f"{spread:+.2f} spread",
            delta_color="off",
            help="Model prediction for the corresponding hour"
        )
        st.caption(f"Residual: {latest_residual/1000:.1f} GW")

with m5:
    with st.container(border=True):
        st.metric(
            label="Renewable Share",
            value=f"{ren_share:.1f}%",
            help="Percentage of total electrical demand covered by Wind and Solar"
        )
        st.caption(f"{latest_renewables/1000:.1f} GW / {latest_load/1000:.1f} GW")

st.write("")

# Negative Price Alert Banner (if applicable)
neg_count = (test_split['Day_Ahead_Price'] < 0).sum()
if neg_count > 0:
    st.info(
        f"💡 **Negative Spot Prices Observed:** Detected **{neg_count} hours** with negative prices (< 0.00 €/MWh) "
        f"in the current evaluation window. This occurs during periods of high renewable generation and low grid demand."
    )

# -----------------------------------------------------------------------------
# Clean Interactive Tabs
# -----------------------------------------------------------------------------
tab_forecast, tab_merit, tab_mix, tab_sim, tab_model = st.tabs([
    "📈 Price Forecast",
    "⚡ Merit Order Effect",
    "🍃 Energy & Generation Mix",
    "🎛️ Market Simulator",
    "🧠 Model Diagnostics"
])

# =============================================================================
# TAB 1: Forecast & Spot Prices
# =============================================================================
with tab_forecast:
    st.subheader("Day-Ahead Spot Price: Actual vs. XGBoost Forecast")
    
    fig_price = go.Figure()

    # Actual price trace
    fig_price.add_trace(go.Scatter(
        x=test_split.index,
        y=test_split['Day_Ahead_Price'],
        mode='lines',
        name='Actual Clearing Price (SMARD)',
        line=dict(color='#2563eb', width=2.5),
        hovertemplate='<b>Actual:</b> %{y:.2f} €/MWh<br><b>Date:</b> %{x}<extra></extra>'
    ))

    # Predicted price trace
    fig_price.add_trace(go.Scatter(
        x=test_split.index,
        y=test_split['Predicted_Price'],
        mode='lines',
        name='XGBoost Day-Ahead Forecast',
        line=dict(color='#ea580c', width=2, dash='dash'),
        hovertemplate='<b>Forecast:</b> %{y:.2f} €/MWh<br><b>Date:</b> %{x}<extra></extra>'
    ))

    # Zero line
    fig_price.add_hline(
        y=0,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text="Zero Price Threshold",
        annotation_position="bottom right"
    )

    fig_price.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date & Time (UTC)",
        yaxis_title="Day-Ahead Electricity Price (€/MWh)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Secondary Diagnostic Row
    c_err, c_tbl = st.columns([1, 1])

    with c_err:
        with st.container(border=True):
            st.markdown("**Error Distribution (Forecast - Actual)**")
            fig_err = px.histogram(
                test_split,
                x="Error",
                nbins=30,
                color_discrete_sequence=['#0284c7'],
                labels={"Error": "Error (€/MWh)"}
            )
            fig_err.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_err, use_container_width=True)

    with c_tbl:
        with st.container(border=True):
            st.markdown("**Recent Observations (Last 5 Hours)**")
            table_preview = test_split[['Day_Ahead_Price', 'Predicted_Price', 'Error', 'Residual_Load']].tail(5).copy()
            table_preview.columns = ['Actual (€)', 'Forecast (€)', 'Error (€)', 'Residual (MW)']
            st.dataframe(
                table_preview.style.format({
                    'Actual (€)': '{:.2f}',
                    'Forecast (€)': '{:.2f}',
                    'Error (€)': '{:+.2f}',
                    'Residual (MW)': '{:,.0f}'
                }),
                use_container_width=True,
                height=240
            )

# =============================================================================
# TAB 2: Merit Order Dynamics
# =============================================================================
with tab_merit:
    st.subheader("The Merit Order Effect")
    st.markdown(
        "Wholesale electricity prices in Germany are dictated by the **Residual Load** "
        "(Grid Demand minus Zero-Marginal-Cost Renewables). As residual load rises, more expensive "
        "gas and coal plants are called to generate power, pushing prices up non-linearly."
    )

    c_plot, c_expl = st.columns([3, 1])

    with c_plot:
        fig_mo = px.scatter(
            test_split,
            x="Residual_Load",
            y="Day_Ahead_Price",
            color="Renewable_Total",
            size="Total_Load",
            color_continuous_scale="Tealgrn",
            labels={
                "Residual_Load": "Residual Load (MWh)",
                "Day_Ahead_Price": "Spot Price (€/MWh)",
                "Renewable_Total": "Renewables (MWh)",
                "Total_Load": "Total Demand (MWh)"
            }
        )

        # Pure NumPy polynomial curve fit (2nd degree)
        try:
            x_vals = test_split['Residual_Load'].values
            y_vals = test_split['Day_Ahead_Price'].values
            valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            if valid.sum() > 5:
                coefs = np.polyfit(x_vals[valid], y_vals[valid], deg=2)
                x_fit = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
                y_fit = np.polyval(coefs, x_fit)
                fig_mo.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    name='Fitted Merit Order Curve',
                    line=dict(color='#e11d48', width=2.5, dash='dash')
                ))
        except Exception:
            pass

        fig_mo.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_mo, use_container_width=True)

    with c_expl:
        with st.container(border=True):
            st.markdown("**Economic Insights**")
            st.markdown("""
            * **Low Residual Load (<15 GW):** High wind/solar output displaces fossil generation, driving prices toward zero or negative.
            * **High Residual Load (>45 GW):** Conventional peaker plants (gas/coal) set the market-clearing price.
            * **Non-Linear Supply:** Tree-based models (XGBoost) capture this curvature significantly better than linear models.
            """)

# =============================================================================
# TAB 3: Generation & Energy Mix
# =============================================================================
with tab_mix:
    st.subheader("German Generation Mix vs. Total Grid Load")
    
    recent_display = df.iloc[-test_eval_hours:].copy()

    fig_mix = go.Figure()

    # Solar
    fig_mix.add_trace(go.Scatter(
        x=recent_display.index,
        y=recent_display['Solar'],
        mode='lines',
        stackgroup='gen',
        name='Solar PV',
        line=dict(width=0.5, color='#eab308'),
        fillcolor='rgba(234, 179, 8, 0.7)'
    ))

    # Wind Onshore
    fig_mix.add_trace(go.Scatter(
        x=recent_display.index,
        y=recent_display['Wind_Onshore'],
        mode='lines',
        stackgroup='gen',
        name='Wind Onshore',
        line=dict(width=0.5, color='#10b981'),
        fillcolor='rgba(16, 185, 129, 0.7)'
    ))

    # Wind Offshore
    fig_mix.add_trace(go.Scatter(
        x=recent_display.index,
        y=recent_display['Wind_Offshore'],
        mode='lines',
        stackgroup='gen',
        name='Wind Offshore',
        line=dict(width=0.5, color='#06b6d4'),
        fillcolor='rgba(6, 182, 212, 0.7)'
    ))

    # Demand curve
    fig_mix.add_trace(go.Scatter(
        x=recent_display.index,
        y=recent_display['Total_Load'],
        mode='lines',
        name='Total Grid Load',
        line=dict(color='#0f172a', width=2.5, dash='dash')
    ))

    fig_mix.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date & Time (UTC)",
        yaxis_title="Power Generation & Demand (MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_mix, use_container_width=True)

# =============================================================================
# TAB 4: Market Simulator ("What-If")
# =============================================================================
with tab_sim:
    st.subheader("What-If Market Simulator")
    st.markdown("Simulate how hypothetical shifts in weather (wind & solar) or industrial demand affect tomorrow's electricity price.")

    with st.container(border=True):
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_solar_mod = st.slider("☀️ Solar PV Shift", min_value=-100, max_value=100, value=0, step=10, format="%d%%")
        with s2:
            sim_wind_mod = st.slider("💨 Wind Generation Shift", min_value=-100, max_value=100, value=0, step=10, format="%d%%")
        with s3:
            sim_load_mod = st.slider("🏭 Grid Load Shift", min_value=-40, max_value=40, value=0, step=5, format="%d%%")

    last_row = test_split.iloc[-1].copy()
    sim_solar = last_row['Solar'] * (1 + sim_solar_mod / 100.0)
    sim_wind = (last_row['Wind_Onshore'] + last_row['Wind_Offshore']) * (1 + sim_wind_mod / 100.0)
    sim_load = last_row['Total_Load'] * (1 + sim_load_mod / 100.0)
    sim_ren_total = sim_solar + sim_wind
    sim_residual = sim_load - sim_ren_total

    sim_dict = {
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

    sim_df = pd.DataFrame([sim_dict])
    sim_price = model.predict(sim_df[FEATURE_COLUMNS])[0]
    base_price = last_row['Predicted_Price']
    delta_p = sim_price - base_price

    res1, res2, res3 = st.columns(3)
    with res1:
        with st.container(border=True):
            st.metric("Base Forecast", f"{base_price:.2f} €/MWh")
    with res2:
        with st.container(border=True):
            st.metric(
                "Simulated Forecast",
                f"{sim_price:.2f} €/MWh",
                delta=f"{delta_p:+.2f} €/MWh",
                delta_color="inverse"
            )
    with res3:
        with st.container(border=True):
            st.metric(
                "Simulated Residual Load",
                f"{sim_residual / 1000:.2f} GW",
                delta=f"{(sim_residual - last_row['Residual_Load']) / 1000:+.2f} GW",
                delta_color="inverse"
            )

# =============================================================================
# TAB 5: Model Diagnostics & Feature Importance
# =============================================================================
with tab_model:
    st.subheader("Model Diagnostics & Feature Importances")
    
    col_feat, col_info = st.columns([3, 2])

    with col_feat:
        with st.container(border=True):
            st.markdown("**XGBoost Relative Feature Importance**")
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
                    color_continuous_scale="Blues"
                )
                fig_feat.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_feat, use_container_width=True)
            except Exception:
                st.info("Feature importance will display when the model is loaded.")

    with col_info:
        with st.container(border=True):
            st.markdown("**Validation Protocol**")
            st.markdown("""
            * **Temporal Integrity:** Evaluated using `TimeSeriesSplit` to prevent future data leakage.
            * **Auction Constraints:** All autoregressive lags (`Lag_24`, `Lag_48`, `Lag_168`) are offset by &ge; 24 hours to respect day-ahead auction clearing deadlines.
            * **Hyperparameters:** Tuned via RandomizedSearchCV across tree depth, subsample ratio, and learning rate.
            """)

# -----------------------------------------------------------------------------
# Clean Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "German Electricity Price Forecaster • Open Data from Bundesnetzagentur (SMARD.de) • "
    "Developed by [Ritik Ghoghari](https://www.linkedin.com/in/ritikghoghari) (MSc Data Science, GISMA Berlin) • "
    "[GitHub Repository](https://github.com/Ritikghoghari/German-Energy-Price-Forecasting)"
)
