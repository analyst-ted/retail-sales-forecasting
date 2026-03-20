# 🛒 Retail Sales Forecasting — Store 1

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Prophet](https://img.shields.io/badge/Prophet-1.3-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)

A complete end-to-end time series forecasting project comparing
ARIMA, Prophet and LSTM on real retail sales data from Ecuador.

🔗 **[Live App →](YOUR_STREAMLIT_URL)**

---

## 📌 Business Problem

Store managers need accurate sales forecasts to:
- Plan inventory and avoid stockouts
- Optimise staffing levels
- Plan promotions during low seasons
- Manage supply chain effectively

This project builds and compares three forecasting models
to predict daily sales for Store 1 across 15 days.

---

## 📊 Dataset

**Source:** [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

| File | Description | Size |
|------|-------------|------|
| train.csv | Historical sales by store and family | 3M rows |
| stores.csv | Store metadata | 54 stores |
| oil.csv | Daily oil prices (Ecuador) | 1,218 rows |
| holidays_events.csv | National and local holidays | 350 rows |
| transactions.csv | Daily transaction counts | 83K rows |

**Scope:** Store 1 (Quito, Ecuador) — 1,684 days of daily sales

---

## 🔍 Key EDA Findings
```
Trend:       Sales doubled from 5,000 → 10,500 (2013-2017)
Seasonality: December highest (10,245) | February lowest (7,337)
Weekly:      Wednesday peak (+2,200) | Sunday lowest (-4,000)
Oil Impact:  Inverse relationship — oil crash 2015 → sales dip
ACF:         Strong spikes at lag 7 (weekly) and lag 365 (yearly)
```

---

## ⚙️ Project Pipeline
```
Data Cleaning
→ Oil: 43 missing values fixed with ffill + bfill
→ Focused on Store 1 (54 stores × 33 families = 1,782 series)
→ Aggregated 33 product families to daily total sales

Feature Engineering
→ Lag features: lag_7, lag_14, lag_30, lag_365
→ Rolling means: rolling_mean_7, rolling_mean_30
→ Calendar: dayofweek, month, quarter, dayofyear

Temporal Split (No Data Leakage)
→ Train: 2013-01-01 → 2017-07-31 (1,669 days)
→ Test:  2017-08-01 → 2017-08-15 (15 days)
```

---

## 🤖 Models

### ARIMA(7,1,1)
- p=7 captures weekly lag pattern
- d=1 removes gentle upward trend
- q=1 corrects for recent forecast errors

### Prophet
- Explicit yearly + weekly seasonality
- Additive seasonality mode
- Confidence intervals included

### LSTM Neural Network
- Architecture: LSTM(64) → Dropout → LSTM(32) → Dropout → Dense
- 14-day sequence window
- EarlyStopping with patience=10

---

## 📈 Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | 1,706 | 2,115 | 24.8% |
| Prophet | 1,325 | 1,858 | 18.9% |
| **LSTM** | **1,100** | **1,761** | **14.8%** |

---

## 💡 Recommendation

**Primary Model: Prophet**

Although LSTM achieved the best MAPE (14.8%), Prophet is
recommended for this use case because:

- 4.1% MAPE difference doesn't justify LSTM complexity
- Prophet components are explainable to store managers
- Built-in confidence intervals for uncertainty
- Handles holidays and weekly patterns automatically
- Easier to retrain and maintain in production

**LSTM recommended for:** High-stakes multi-store forecasting
where accuracy improvement justifies infrastructure cost.

---

## 📁 Project Structure
```
retail-sales-forecasting/
├── data/
│   ├── raw/                    ← original kaggle files
│   └── processed/              ← cleaned and engineered features
├── models/
│   ├── lstm_model.keras        ← trained LSTM
│   ├── prophet_model.pkl       ← trained Prophet
│   ├── scaler_X.pkl            ← feature scaler
│   ├── scaler_y.pkl            ← target scaler
│   └── feature_names.json      ← feature column names
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_arima.ipynb
│   ├── 05_prophet.ipynb
│   ├── 06_lstm.ipynb
│   └── 07_model_comparison.ipynb
├── reports/figures/            ← all EDA and model charts
├── app.py                      ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally
```bash
# Clone repo
git clone https://github.com/analyst-ted/retail-sales-forecasting
cd retail-sales-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Prophet](https://img.shields.io/badge/Prophet-1.3-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.8-orange)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-green)
![Pandas](https://img.shields.io/badge/Pandas-2.3-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)

---

## 👤 Author

**Arup Roy**
Data Scientist | ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-analyst--ted-black)](https://github.com/analyst-ted)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Arup%20Roy-blue)](https://linkedin.com/in/arup-roy-777925164)

---

*Open to Data Scientist roles — DM me on LinkedIn!*