# How I Forecasted German Electricity Prices with XGBoost and Open Data

### By Ritik Ghoghari
*MSc Data Science Student at GISMA University of Applied Sciences Berlin*

---

Electricity is one of the only commodities in the world that must be generated, transmitted, and consumed in the exact same fraction of a second. In Germany, the aggressive expansion of wind and solar capacity (*Energiewende*) has brought massive environmental benefits, but it has also turned the day-ahead wholesale electricity spot market into a rollercoaster.

On sunny, windy Sunday afternoons, spot prices on the EPEX Spot exchange regularly plummet below zero—meaning grid operators actually pay industrial consumers to absorb excess power. Conversely, during a *Dunkelflaute* (dark doldrums: zero wind, zero sun in the dead of winter), prices explode as high-marginal-cost gas turbines and coal plants fire up.

As an MSc Data Science student at GISMA Berlin, I wanted to move beyond toy datasets and tackle a real-world, high-stakes time-series challenge: **Can we accurately forecast Germany’s hourly Day-Ahead wholesale electricity prices 24 hours in advance using only open data and Machine Learning?**

Here is how I built an end-to-end forecasting pipeline using official data from the German Federal Network Agency (**Bundesnetzagentur / SMARD.de**) and **XGBoost**, achieving a test **Mean Absolute Error (MAE) of 10.80 €/MWh**.

---

## 1. The Data Source: Tapping into SMARD.de API

Rather than using static Kaggle CSVs, I wanted this project to reflect production data engineering. I tapped into the REST API of **SMARD.de**, the official transparency platform of the Bundesnetzagentur.

I extracted five synchronous hourly time-series for the Germany-Luxembourg (`DE-LU`) bidding zone:
1. **Day-Ahead Wholesale Electricity Price** (€/MWh, Filter `4169`)
2. **Total Electricity Consumption / Grid Load** (MWh, Filter `410`)
3. **Solar PV Generation** (MWh, Filter `4068`)
4. **Wind Onshore Generation** (MWh, Filter `4067`)
5. **Wind Offshore Generation** (MWh, Filter `4066`)

```python
import requests
import pandas as pd

def fetch_smard_series(filter_id, region="DE", resolution="hour", timestamp=1710000000000):
    url = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    response = requests.get(url)
    data = response.json().get('series', [])
    df = pd.DataFrame(data, columns=['timestamp', 'value']).dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')
```

---

## 2. Domain Knowledge: The Merit Order Effect

In power economics, the wholesale spot price is determined by the **Merit Order**. Generators are dispatched in ascending order of their short-run marginal costs:
1. **Zero Marginal Cost:** Solar, Onshore Wind, Offshore Wind, Hydro.
2. **Low-to-Medium Cost:** Nuclear, Lignite, Hard Coal.
3. **High Marginal Cost:** Open-Cycle Gas Turbines (OCGT), Oil peakers.

When renewable supply surges, it shifts the supply curve to the right, displacing expensive thermal plants. This is called the **Merit Order Effect**.

Therefore, raw load alone does not drive price. What truly dictates the clearing price is **Residual Load**:

$$\text{Residual Load} = \text{Total Grid Load} - (\text{Solar} + \text{Wind Onshore} + \text{Wind Offshore})$$

Residual load is the actual demand that conventional fossil-fuel plants must cover. When residual load is low (<15 GW), prices crater. When it is high (>55 GW), prices spike.

```python
# Feature Engineering: Fundamental Market State
df['Renewable_Total'] = df['Solar'] + df['Wind_Onshore'] + df['Wind_Offshore']
df['Residual_Load'] = df['Total_Load'] - df['Renewable_Total']
```

---

## 3. Strict Temporal Feature Engineering (Zero Lookahead Bias)

Electricity auctions for tomorrow take place at 12:00 CET today. Therefore, when predicting tomorrow's 24 hours, you cannot use any price lag shorter than 24 hours. A lag of 1 hour (`Lag_1`) would be fatal data leakage.

I engineered three tiers of strictly legal features:

```python
# 1. Calendar & Cyclical Features
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.day_of_week
df['Month'] = df.index.month
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 2. Autoregressive Lags (Day-ahead valid: offset >= 24h)
df['Lag_24'] = df['Day_Ahead_Price'].shift(24)    # Same hour yesterday
df['Lag_48'] = df['Day_Ahead_Price'].shift(48)    # Same hour 2 days ago
df['Lag_168'] = df['Day_Ahead_Price'].shift(168)  # Same hour last week

# 3. Rolling Volatility & Regime Indicators
df['Rolling_Mean_24'] = df['Day_Ahead_Price'].shift(24).rolling(24).mean()
df['Rolling_Std_24'] = df['Day_Ahead_Price'].shift(24).rolling(24).std()
```

---

## 4. Modeling & Validation: Why TimeSeriesSplit Matters

Standard random $k$-fold cross-validation is a cardinal sin in time-series forecasting because it shuffles future information into past predictions. 

Instead, I implemented **`TimeSeriesSplit` (Walk-Forward Validation)** to evaluate model generalization strictly forwards in time.

```python
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb

# Chronological Train-Test Split
train = df_model[df_model.index.year < 2024]
test = df_model[df_model.index.year == 2024]

tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.03, 0.08],
    'subsample': [0.8, 0.9]
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
search = RandomizedSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_iter=10)
search.fit(X_train, y_train)
```

---

## 5. Results & Market Insights

The results confirmed the superiority of gradient boosting in capturing the steep, non-linear curvature of the Merit Order supply function:

| Model Architecture | Validation | Test MAE (€/MWh) | Test RMSE (€/MWh) |
| :--- | :--- | :---: | :---: |
| Naive Persistence (`Lag_24`) | Chronological | 22.40 | 31.15 |
| Linear Regression | Chronological | 18.90 | 25.80 |
| Random Forest (Baseline) | TimeSeriesSplit | 12.50 | 18.20 |
| **XGBoost (Tuned)** | **TimeSeriesSplit** | **10.80** | **15.40** |

### Key Findings:
1. **Residual Load Dominance:** `Residual_Load` accounted for >35% of total tree split gain. 
2. **Weekly Seasonality Anchor:** `Lag_168` (exact same hour one week prior) outperformed `Lag_24` during transitional periods because industrial demand follows strict 7-day cyclicality.
3. **Negative Price Capture:** The model successfully anticipated 84% of zero-or-negative price events, primarily occurring on weekend afternoons with high wind generation.

---

## 6. Deployment on Hugging Face Spaces & Streamlit

To make the solution usable for energy traders and analysts, I developed an interactive **Streamlit dashboard** that fetches live data from SMARD.de and generates real-time predictions. It features:
- Live actual vs. predicted price plots with zoomable inspection.
- An interactive **"What-If" Scenario Simulator** allowing users to stress-test electricity prices against renewable surges.

Explore the live app on [Hugging Face Spaces](https://huggingface.co/spaces/Ritikghoghari/German-Energy-Price-Forecasting) and review the full source code on [GitHub](https://github.com/Ritikghoghari/German-Energy-Price-Forecasting).

---

*I am an MSc Data Science student at GISMA Berlin, actively seeking Junior Data Analyst / Data Scientist / Working Student roles in Berlin, Hannover, or Remote Germany. Connect with me on [LinkedIn](https://www.linkedin.com/in/ritikghoghari)!*
