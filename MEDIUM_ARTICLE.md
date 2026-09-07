# How I Forecasted German Electricity Prices with XGBoost and Open Data

### A master's student's deep dive into the Merit Order Effect, the Bundesnetzagentur SMARD API, and machine learning on the EPEX Spot market.

**By Ritik Ghoghari**  
*MSc Data Science Student at GISMA University of Applied Sciences Berlin*  
*Connect with me on [LinkedIn](https://www.linkedin.com/in/ritikghoghari) | Explore the project on [GitHub](https://github.com/Ritikghoghari/German-Energy-Price-Forecasting)*

---

![Header Image: German Energy Grid and Wind Turbines](https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=1200&q=80)
*The German power grid is undergoing the most ambitious clean energy transition in history. But with high renewables comes unprecedented price volatility.*

---

When I moved to Germany to pursue my Master’s in Data Science at **GISMA University of Applied Sciences Berlin**, one economic phenomenon immediately caught my attention: **negative electricity prices**.

On sunny, breezy Sunday afternoons in Germany, wholesale electricity prices on the EPEX Spot exchange don't just drop—they plunge into negative territory. Grid operators and generators literally pay industrial consumers up to -50 €/MWh to take excess electricity off the grid. Yet just three days later, during an overcast winter morning (*Dunkelflaute*), prices can spike above 200 €/MWh as expensive gas turbines ramp up.

Unlike stocks, electricity cannot be stored easily at national scale. It must be generated, transmitted, and consumed instantaneously. 

Rather than working on generic Kaggle datasets, I wanted to solve a real-world energy problem relevant to German industry: **Can we predict tomorrow's 24 hourly electricity spot prices with high accuracy using only open government data and machine learning?**

In this article, I walk through how I built an end-to-end forecasting engine using official data from the **Bundesnetzagentur (SMARD.de)** and **XGBoost**, achieving a validated **Mean Absolute Error (MAE) of 10.80 €/MWh** (and dropping to **8.33 €/MWh** in live production testing).

---

## 1. The Data Pipeline: Mining the Bundesnetzagentur (SMARD.de) API

Many academic papers rely on static CSV dumps. I wanted this pipeline to be production-ready and live.

The German Federal Network Agency (*Bundesnetzagentur*) maintains **SMARD.de**, an open transparency platform providing hourly time-series for the Germany-Luxembourg (`DE-LU`) bidding zone. 

Instead of downloading spreadsheets manually, I wrote a multi-threaded Python ingestion client in `requests` that pulls five synchronous market streams:

1. **Wholesale Day-Ahead Price** (€/MWh, Filter `4169`)
2. **Total Electrical Grid Demand** (MWh, Filter `410`)
3. **Solar PV Generation** (MWh, Filter `4068`)
4. **Wind Onshore Generation** (MWh, Filter `4067`)
5. **Wind Offshore Generation** (MWh, Filter `4066`)

```python
import requests
import pandas as pd

def fetch_smard_series(filter_id, region="DE-LU", resolution="hour", timestamp=1788732000000):
    url = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    response = requests.get(url, timeout=10)
    data = response.json().get('series', [])
    
    # Format into hourly time-indexed DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'value']).dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')
```

Because SMARD partitions its time-series by weekly epoch chunks, I implemented parallel fetching using `concurrent.futures.ThreadPoolExecutor`, allowing the model to ingest months of multi-gigawatt market data in under three seconds.

---

## 2. The Economic Secret: The "Merit Order Effect"

Data science without domain knowledge is just curve fitting. To predict electricity prices, you have to understand the **Merit Order**.

In European electricity markets, power stations are dispatched in ascending order of their **short-run marginal generation costs**:

1. **Zero Marginal Cost:** Solar PV, Wind Onshore, Wind Offshore (fuel is free).
2. **Low Marginal Cost:** Nuclear, Lignite, Run-of-River Hydro.
3. **Medium-to-High Marginal Cost:** Hard Coal, Combined-Cycle Gas Turbines (CCGT).
4. **Peak Marginal Cost:** Open-Cycle Gas Turbines (OCGT), Heavy Oil peakers.

The market-clearing price is determined by the **last and most expensive generator needed to satisfy demand**.

![Merit Order Curve Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Merit_Order_Curve_Diagram.svg/1000px-Merit_Order_Curve_Diagram.svg.png)
*The Merit Order: Renewable generation shifts the supply curve outward, leaving only the Residual Load to be covered by conventional fossil plants.*

When wind and sun generate 35,000 MW, they shift the supply curve to the right. The expensive gas plants are switched off, and wholesale prices collapse. This is the **Merit Order Effect**.

Therefore, the most critical engineered feature in my dataset is **Residual Load**:

$$\text{Residual Load} = \text{Total Grid Demand} - (\text{Solar} + \text{Wind Onshore} + \text{Wind Offshore})$$

Residual Load represents the exact conventional capacity needed from fossil plants. When Residual Load drops below 15 GW, prices plummet toward single digits. When it surges past 50 GW, prices explode.

```python
# Feature Engineering: Fundamental Market State
df['Renewable_Total'] = df['Solar'] + df['Wind_Onshore'] + df['Wind_Offshore']
df['Residual_Load'] = df['Total_Load'] - df['Renewable_Total']
```

---

## 3. Strict Temporal Feature Engineering (Avoiding Lookahead Bias)

Electricity day-ahead auctions on EPEX Spot close daily at **12:00 CET** for delivery throughout the next calendar day (hours 00:00 to 24:00). 

A common blunder in time-series modeling is using short lags like `Lag_1` (the price 1 hour ago). In real life, at 12:00 today, you do not know the price at 22:00 tonight, let alone tomorrow afternoon!

To guarantee **zero data leakage**, all autoregressive features were strictly constrained to $\ge 24$ hours:

```python
# 1. Calendar & Business Rhythm
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.day_of_week
df['Month'] = df.index.month
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 2. Autoregressive Lags (Strict Day-Ahead Offset >= 24h)
df['Lag_24'] = df['Day_Ahead_Price'].shift(24)    # Same hour yesterday
df['Lag_48'] = df['Day_Ahead_Price'].shift(48)    # Same hour 2 days ago
df['Lag_168'] = df['Day_Ahead_Price'].shift(168)  # Same hour last week (7 days)

# 3. Market Volatility & Momentum
df['Rolling_Mean_24'] = df['Day_Ahead_Price'].shift(24).rolling(24).mean()
df['Rolling_Std_24'] = df['Day_Ahead_Price'].shift(24).rolling(24).std()
```

Notice `Lag_168` (168 hours = 7 days prior). German manufacturing and heavy industry follow strict weekly operational cycles; Tuesday at 10:00 AM behaves remarkably like the previous Tuesday at 10:00 AM.

---

## 4. Modeling & Walk-Forward Validation

Standard random $k$-fold cross-validation is fatal for financial time-series because it trains on the future to predict the past. 

Instead, I used **`TimeSeriesSplit` (Walk-Forward Validation)** with $k=3$ folds to evaluate how the model generalises into unseen market regimes.

```python
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb

# Chronological Train-Test Split (Train: 2022-2023, Test: 2024+)
train = df_model[df_model.index.year < 2024]
test = df_model[df_model.index.year == 2024]

# Hyperparameter optimization on expanding chronological windows
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.02, 0.03, 0.05],
    'subsample': [0.8, 0.9]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
search = RandomizedSearchCV(xgb_model, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_iter=10)
search.fit(X_train, y_train)
```

---

## 5. The Results: Beating the Baseline by >50%

Gradient boosted trees excelled at capturing the sharp non-linear inflection points of the Merit Order curve:

| Model Architecture | Validation Strategy | Test MAE (€/MWh) | Test RMSE (€/MWh) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Naive Persistence (`Lag_24`)** | Chronological Split | 22.40 | 31.15 | 0.42 |
| **Linear Regression (Ridge)** | Chronological Split | 18.90 | 25.80 | 0.58 |
| **Random Forest Baseline** | TimeSeriesSplit ($k=3$) | 12.50 | 18.20 | 0.76 |
| **XGBoost (Tuned)** | **TimeSeriesSplit ($k=3$)** | **10.80** | **15.40** | **0.84** |

In live out-of-sample testing on the most recent 168-hour window from Bundesnetzagentur, the model reached an **MAE of 8.33 €/MWh** with an **$R^2$ of 0.97**.

### Three Key Takeaways from the Data:
1. **Residual Load is King:** In XGBoost's feature importances, `Residual_Load` accounted for over 38% of total split gain—far higher than gross load alone.
2. **Weekly Anchoring:** `Lag_168` proved to be the single most powerful autoregressive signal, outperforming `Lag_48` during transitional weather.
3. **Anticipating Negative Pricing:** The model successfully identified negative pricing hours during weekend midday solar peaks, providing actionable signals for battery storage operators looking to perform charge-discharge arbitrage.

---

## 6. From Model to Product: The Streamlit Dashboard

A model inside a Jupyter notebook provides zero business value. To make this actionable for energy traders, analysts, and grid operators, I built and deployed an interactive dashboard using **Streamlit**:

* 📡 **Live Bundesnetzagentur Ingestion:** Fetches real-time market data on the fly with smart caching.
* 📈 **Interactive Plotly Visuals:** Zoomable actual vs. forecast price curves and Merit Order scatter plots.
* 🎛️ **"What-If" Market Simulator:** Sliders allowing users to simulate how a +30% solar surge or -20% wind drop shifts tomorrow's price via the Merit Order Effect.

👉 **Try the Live App on Hugging Face Spaces:** [German Energy Price Forecaster](https://huggingface.co/spaces/Ritikghoghari/German-Energy-Price-Forecasting)  
👉 **Inspect the Full Source Code on GitHub:** [github.com/Ritikghoghari/German-Energy-Price-Forecasting](https://github.com/Ritikghoghari/German-Energy-Price-Forecasting)

---

## Final Thoughts & What's Next

Building this project taught me that data science in the European energy sector requires far more than algorithms—it demands an intimate understanding of market design, regulatory timelines, and physical grid constraints.

Next steps for this project include integrating **numerical weather predictions (DWD weather forecasts)** and carbon allowance prices (EU ETS EUA) to model long-term seasonal spreads.

---

### About the Author
I am **Ritik Ghoghari**, an MSc Data Science student at **GISMA University of Applied Sciences Berlin**. I am passionate about energy analytics, applied machine learning, and time-series forecasting. 

I am currently actively seeking **Junior Data Analyst / Junior Data Scientist / Working Student (Werkstudent)** opportunities in **Berlin, Hannover, or Remote Germany**. 

If your team is tackling exciting challenges in data analytics or energy transition, let's connect on [LinkedIn](https://www.linkedin.com/in/ritikghoghari) or reach out via [GitHub](https://github.com/Ritikghoghari)!
