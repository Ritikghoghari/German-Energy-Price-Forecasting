# 📱 LinkedIn Post Draft

---

⚡ How does the "Merit Order Effect" impact German electricity prices, and can we forecast it with Machine Learning?

During my MSc in Data Science at GISMA University of Applied Sciences Berlin, I wanted to move beyond toy datasets and analyze Germany's evolving energy transition (Energiewende). 

When wind and solar output surges on weekends, wholesale spot prices on EPEX Spot frequently crash below zero. Conversely, during low-renewable periods, expensive peaker plants drive prices up sharply.

To model this volatility, I built an end-to-end forecasting pipeline that pulls live market data directly from the Bundesnetzagentur (SMARD.de API) and forecasts Day-Ahead prices 24 hours in advance:

🔍 Key Highlights:
• Data Engineering: Automated ingestion of 5 synchronous hourly streams from SMARD (Price, Grid Load, PV, Onshore & Offshore Wind).
• Domain Feature Engineering: Modeled Residual Load (Demand - Renewables) to capture the non-linear Merit Order supply curve, combined with strict Day-Ahead lag features (24h, 48h, 168h) and 24h rolling volatility.
• Strict Evaluation: Evaluated via TimeSeriesSplit walk-forward validation (zero lookahead leakage).
• Model Results: Tuned XGBoost achieved a test MAE of 10.80 €/MWh — reducing forecast error by >51% compared to persistence baselines.
• Interactive Deployment: Built and deployed an interactive Streamlit dashboard on Hugging Face Spaces featuring live SMARD ingestion and a "What-If" dispatch simulator.

💻 Live Streamlit App: [Insert HuggingFace Spaces Link]
📂 GitHub Repository: https://github.com/Ritikghoghari/German-Energy-Price-Forecasting
📝 Deep-Dive Medium Article: [Insert Medium Article Link]

---

🎯 I am an MSc Data Science student at GISMA Berlin actively seeking **Junior Data Analyst / Junior Data Scientist / Working Student (Werkstudent)** opportunities in the **Berlin / Hannover region (or Remote Germany)**.

If your team is working on energy analytics, predictive modeling, or data engineering, I would love to connect!

#DataScience #MachineLearning #EnergyTransition #Energiewende #Germany #Berlin #Hannover #JobSearch #OpenData #XGBoost #Streamlit #DataAnalytics #WorkingStudent #Hiring
