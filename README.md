---
title: German Energy Price Forecasting
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# ⚡ German Day-Ahead Electricity Price Forecasting

> **End-to-End Machine Learning Pipeline & Interactive Dashboard Modeling the Merit Order Effect and Spot Market Volatility with SMARD Open Data**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0+-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App_Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#-live-interactive-demo)
[![Data: SMARD.de](https://img.shields.io/badge/Data-Bundesnetzagentur_SMARD.de-005A9C?style=for-the-badge)](https://www.smard.de/)
[![HuggingFace Spaces](https://img.shields.io/badge/Deploy-HuggingFace_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](#-hugging-face-spaces-deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)](#)

---

## 🌟 Executive Summary & Motivation

Germany's transition to clean energy (*Energiewende*) has fundamentally reshaped European power markets. With renewable energy sources regularly providing over 50% of the country's electricity, wholesale electricity prices on the European Power Exchange (EPEX Spot) have become notoriously volatile. 

During windy weekends or sunny midday hours, spot prices frequently crash toward zero or even plunge into **negative pricing**. Conversely, during overcast, windless winter periods (*Dunkelflaute*), expensive natural gas and coal plants are fired up, sending prices skyrocketing.

This project delivers an **end-to-end Machine Learning forecasting system** that predicts German-Luxembourg Day-Ahead wholesale electricity prices (`Day_Ahead_Price` in €/MWh) using official, live open market data from the **Bundesnetzagentur (SMARD.de API)**.

By capturing non-linear relationships such as the **Merit Order Effect** and accounting for strict day-ahead temporal constraints (preventing lookahead bias), our tuned **XGBoost model achieves a Mean Absolute Error (MAE) of 10.80 €/MWh**, beating standard baseline benchmarks by over 50%.

---

## 🚀 Live Interactive Demo

Try the interactive Streamlit dashboard directly:
* **HuggingFace Space:** *[Live App Demo Placeholder - German Energy Forecaster](https://huggingface.co/spaces/Ritikghoghari/German-Energy-Price-Forecasting)*
* **GitHub Repository:** [github.com/Ritikghoghari/German-Energy-Price-Forecasting](https://github.com/Ritikghoghari/German-Energy-Price-Forecasting)

Features of the live dashboard:
- 📡 **Live API Ingestion:** Fetches real-time market data directly from SMARD.de (Bundesnetzagentur).
- 📈 **Interactive Forecast vs. Actuals:** Zoomable Plotly visualizations of spot prices with error distributions.
- ⚡ **Merit Order Visualizer:** Live breakdown of Grid Load vs. Solar, Onshore Wind, and Offshore Wind generation.
- 🎛️ **Scenario Simulator ("What-If"):** Adjust residual demand or renewable spikes and observe predicted spot price elasticity in real time.

---

## 📊 Model Performance & Benchmark Comparison

Models were evaluated using a strict **time-series split** (preventing data leakage and lookahead bias). The model was trained on historical hourly market intervals and validated on unseen out-of-time test periods.

| Model | Validation Strategy | MAE (€/MWh) | RMSE (€/MWh) | $R^2$ Score | Key Characteristics & Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Naive Persistence (Lag 24h)** | Chronological Split | 22.40 | 31.15 | 0.42 | Assumes tomorrow's price equals yesterday's price at the same hour. |
| **Ridge / Linear Regression** | Chronological Split | 18.90 | 25.80 | 0.58 | Underfits non-linear curvature of the Merit Order supply curve. |
| **Random Forest Regressor** | TimeSeriesSplit ($k=3$) | 12.50 | 18.20 | 0.76 | Baseline ensemble tree model ($n=100, \text{depth}=20$). |
| **XGBoost Regressor (Tuned)** | **TimeSeriesSplit ($k=3$)** | **10.80** | **15.40** | **0.84** | **Best performer.** Tuned hyperparameters with tree depth regularization. |

> **Key Takeaway:** The tuned XGBoost model reduces prediction error by **>51%** relative to the naive market baseline and by **13.6%** relative to the unconstrained Random Forest regressor, effectively handling extreme market spikes and dips.

---

## 🧠 Feature Engineering Architecture

Forecasting electricity prices 24 hours in advance requires domain-specific feature engineering that respects market auction timelines. All lag features are strictly constructed with a minimum 24-hour offset to guarantee zero lookahead bias.

### 1. Fundamental Energy Market Features
* `Total_Load` (MWh): Total real-time electrical grid demand across Germany.
* `Renewable_Total` (MWh): Combined generation from `Wind_Onshore`, `Wind_Offshore`, and `Solar` (PV).
* `Residual_Load` (MWh): Defined as $\text{Residual Load} = \text{Total Load} - \text{Renewable Total}$. This represents the remaining demand that must be satisfied by conventional fossil-fuel or nuclear generation, directly dictating the marginal clearing price on the Merit Order supply curve.

### 2. Calendar & Temporal Signals
* `Hour` ($0-23$): Captures the diurnal consumption curve (morning commute peak at 08:00, midday solar dip at 13:00, evening household peak at 19:00).
* `DayOfWeek` ($0-6$): Encodes weekly industrial demand cycles (Monday through Friday vs. reduced weekend load).
* `Month` ($1-12$): Captures macroeconomic seasonality (winter heating demand and lower solar capacity vs. summer solar surges).
* `IsWeekend` (Binary $0/1$): Flags Saturdays and Sundays when industrial power demand drops by 20–30%.

### 3. Autoregressive Lags & Rolling Volatility
* `Lag_24`: Spot price 24 hours prior (same hour yesterday).
* `Lag_48`: Spot price 48 hours prior (two days prior).
* `Lag_168`: Spot price 168 hours prior (exact same hour last week, capturing strong 7-day cyclical autocorrelation).
* `Rolling_Mean_24`: 24-hour moving average of historical prices (lagged by 24h), capturing the macro price level and fuel/gas price regimes.
* `Rolling_Std_24`: 24-hour moving standard deviation (lagged by 24h), capturing short-term market volatility and grid stress.

---

## 💡 Key Energy Market Findings

1. **The Merit Order Effect in Action:**
   A strong non-linear inverse relationship exists between renewable generation and spot price. When `Residual_Load` drops below 15 GW, spot prices compress rapidly toward 0 €/MWh as zero-marginal-cost renewables set the clearing price.
2. **Predictability of Negative Pricing:**
   Negative spot prices are concentrated during weekend midday hours (Sunday 12:00–15:00) when low industrial load coincides with peak solar radiation and sustained onshore wind.
3. **Weekly Autoregressive Dominance:**
   `Lag_168` (one-week lag) proved to be one of the most predictive single features, confirming that German electricity consumption and pricing adhere strictly to 7-day business cycles.
4. **Peak Spread Dynamics:**
   The highest prediction errors and price volatility occur during the transition hours (07:00–08:00 and 18:00–20:00), driven by the rapid ramp-up of flexible gas turbine peakers.

---

## 📂 Repository Structure

```
German-Energy-Price-Forecasting/
├── app.py                     # 🚀 Production Streamlit interactive web application
├── models/
│   └── xgb_model.json         # 🤖 Pre-trained optimized XGBoost model artifact
├── notebooks/
│   └── EDA.ipynb              # 🔍 Exploratory Data Analysis, Feature Engineering & Tuning
├── src/
│   ├── data_processor.py      # 🧹 Data cleaning, formatting, and feature transformation
│   └── download_data.py       # 📥 Live SMARD.de Bundesnetzagentur API fetcher
├── requirements.txt           # 📦 Production dependencies
├── LICENSE                    # 📄 MIT License
└── README.md                  # 📖 Comprehensive project documentation
```

---

## 🛠️ Tech Stack

* **Language & Core:** Python 3.11, NumPy, Pandas
* **Machine Learning:** XGBoost (`xgboost.XGBRegressor`), Scikit-Learn (`TimeSeriesSplit`, `RandomizedSearchCV`)
* **Data Source & API:** SMARD.de API (German Federal Network Agency / *Bundesnetzagentur* REST service)
* **Visualization & UI:** Streamlit, Plotly Express & Graph Objects, Matplotlib, Seaborn
* **Deployment:** Hugging Face Spaces / Streamlit Community Cloud

---

## ⚡ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ritikghoghari/German-Energy-Price-Forecasting.git
cd German-Energy-Price-Forecasting
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard Locally
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

### 5. Explore the Jupyter Notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 🤗 Hugging Face Spaces Deployment

This repository is pre-configured for direct zero-friction deployment to **Hugging Face Spaces**:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Set Space SDK to **Streamlit**.
3. Link your GitHub repository (`github.com/Ritikghoghari/German-Energy-Price-Forecasting`) or push directly to the HF Git remote.
4. Hugging Face Spaces automatically reads the frontmatter in `README.md`, installs `requirements.txt`, and boots `app.py`.

---

## 👤 About the Author

**Ritik Ghoghari**  
🎓 **MSc Data Science Student** at **GISMA University of Applied Sciences Berlin**  
🎯 **Targeting:** Junior Data Analyst / Data Scientist / Working Student (*Werkstudent*) roles in **Berlin / Hannover / Remote Germany**  
💼 **Focus Areas:** Time-Series Forecasting, Applied Machine Learning, Energy Economics, Interactive Analytics Dashboards

* 🔗 **LinkedIn:** [linkedin.com/in/ritikghoghari](https://www.linkedin.com/in/ritikghoghari) *(connect or reach out!)*
* 🐙 **GitHub:** [github.com/Ritikghoghari](https://github.com/Ritikghoghari)
* 📧 **Email:** [Contact via LinkedIn](https://www.linkedin.com/in/ritikghoghari)

---

## 📜 License & Acknowledgments

* **License:** Distributed under the [MIT License](LICENSE).
* **Data Credit:** Electricity market data provided openly by the German Federal Network Agency (*Bundesnetzagentur*) via [SMARD.de](https://www.smard.de) under the [CC BY 4.0 International](https://creativecommons.org/licenses/by/4.0/) license.
