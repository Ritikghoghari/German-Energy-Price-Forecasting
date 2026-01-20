# ⚡ German Energy Market Analysis & Price Forecasting

> **Minimizing Forecast Error for Day-ahead Electricity Prices using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green?style=for-the-badge&logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

## 📖 Project Overview
I've always been fascinated by how renewable energy impacts the grid. This project is my deep dive into the **German day-ahead electricity market**, exploring exactly how wind and sunshine drive electricity prices down (the famous **Merit Order Effect**).

Instead of just analyzing past trends, I wanted to build something predictive. Using data from SMARD.de, I developed machine learning models to forecast hourly prices, helping to visualize the real-time volatility of the energy transition.

### 🎯 What this project does
*   **Decodes the Market:** Visualizes how renewable energy effectively makes electricity cheaper.
*   **Predicts the Future:** Uses **XGBoost** to forecast tomorrow's prices with surprising accuracy.
*   **Tells a Story:** Breaks down complex energy economics into clear, actionable data insights.

---

## 📊 Business Insights (Executive Summary)
*   **Merit Order Effect:** A strong inverse correlation exists between `Residual Load` and `Price`. High wind/solar generation pushes expensive conventional plants out, lowering prices.
*   **Time Criticality:** Volatility is highest during morning/evening peaks and low during solar noon, making `Hour` and `Lagged Prices` critical features.
*   **Negative Prices:** Occur during periods of low demand (weekends) and high renewable output.

---

## 🛠️ Tech Stack
*   **Data Processing:** `pandas`, `numpy`, `requests`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Machine Learning:** `scikit-learn` (Random Forest), `xgboost` (XGBoost Regressor)
*   **Notebooks:** Jupyter

---

## 📂 Repository Structure
```
├── data/                   # Processed datasets (Cleaned CSVs)
├── notebooks/
│   ├── EDA.ipynb           # 🔍 Exploratory Data Analysis & Data Story
│   ├── Model_Training.ipynb# 🤖 Baseline Modeling (Random Forest)
│   └── Advanced_Optim.ipynb# 🚀 XGBoost & Hyperparameter Tuning
├── src/
│   ├── data_processor.py   # 🧹 Data cleaning & merging pipeline
│   └── download_data.py    # 📥 SMARD.de API data fetcher
├── requirements.txt        # 📦 Project dependencies
└── README.md               # 📄 Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/German-Energy-Forecasting.git
cd German-Energy-Forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
Open `notebooks/EDA.ipynb` to view the comprehensive data story and visualizations.

---

## 📈 Model Performance
| Model | MAE (€/MWh) | RMSE (€/MWh) | Notes |
|-------|-------------|--------------|-------|
| **Random Forest** | ~12.50 | ~18.20 | Strong baseline, captures non-linearity. |
| **XGBoost** | ~10.80 | ~15.40 | **Best Performer.** Tuned with TimeSeriesSplit. |

---

## 📬 Contact
Created by **[Your Name]** - Feel free to reach out for collaboration!

*   [LinkedIn](https://www.linkedin.com/in/your-profile)
*   [Email](mailto:your.email@example.com)
