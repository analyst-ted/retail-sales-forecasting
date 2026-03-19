import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title='Store 1 Sales Forecaster',
    page_icon='📈',
    layout='wide'
)

# ─────────────────────────────────────
# LOAD DATA AND MODELS
# ─────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv(
        'data/processed/train_raw.csv',
        index_col='date', parse_dates=True)
    test = pd.read_csv(
        'data/processed/test_raw.csv',
        index_col='date', parse_dates=True)
    train_fe = pd.read_csv(
        'data/processed/train_features.csv',
        index_col='date', parse_dates=True)
    test_fe = pd.read_csv(
        'data/processed/test_features.csv',
        index_col='date', parse_dates=True)
    return train, test, train_fe, test_fe

@st.cache_resource
def load_models():
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    lstm     = tf.keras.models.load_model(
        'models/lstm_model.keras')
    with open('models/prophet_model.pkl', 'rb') as f:
        prophet = pickle.load(f)
    with open('models/feature_names.json', 'r') as f:
        feature_cols = json.load(f)
    return scaler_X, scaler_y, lstm, prophet, feature_cols

train, test, train_fe, test_fe = load_data()
scaler_X, scaler_y, lstm_model, prophet_model, feature_cols = load_models()

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
st.title('📈 Store 1 — Sales Forecasting Dashboard')
st.markdown('Retail sales forecasting using ARIMA, Prophet and LSTM')
st.markdown('---')

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────
st.sidebar.header('Forecast Settings')

model_choice = st.sidebar.selectbox(
    'Select Forecasting Model',
    ['Prophet', 'LSTM', 'ARIMA'],
    index=0,
    help='Prophet recommended for best balance of accuracy and interpretability'
)

st.sidebar.markdown('---')
st.sidebar.markdown('### Model Performance')
st.sidebar.markdown('''
| Model   | MAPE  |
|---------|-------|
| ARIMA   | 24.8% |
| Prophet | 18.9% |
| LSTM    | 14.8% |
''')
st.sidebar.markdown('---')
st.sidebar.markdown('**Recommended:** Prophet')
st.sidebar.markdown('Best balance of accuracy and explainability')

# ─────────────────────────────────────
# SALES HISTORY SECTION
# ─────────────────────────────────────
st.subheader('📊 Sales History — Store 1 (2013-2017)')

full_history = pd.concat([train, test])

fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(train.index, train['total_sales'],
         color='#3498db', linewidth=0.8,
         alpha=0.8, label='Training Sales')
ax1.plot(test.index, test['total_sales'],
         color='#2ecc71', linewidth=2,
         label='Test Period Sales')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Sales')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Average Daily Sales',
            f"{train['total_sales'].mean():,.0f}")
col2.metric('Peak Sales Day',
            f"{train['total_sales'].max():,.0f}")
col3.metric('Lowest Sales Day',
            f"{train['total_sales'].min():,.0f}")
col4.metric('Total Days Tracked',
            f"{len(train):,}")

st.markdown('---')

# ─────────────────────────────────────
# FORECAST SECTION
# ─────────────────────────────────────
st.subheader(f'🔮 15-Day Forecast — {model_choice} Model')

def get_arima_forecast():
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train['total_sales'],
                  order=(7,1,1)).fit()
    pred  = model.forecast(steps=len(test))
    return pred.values

def get_prophet_forecast():
    future = prophet_model.make_future_dataframe(
        periods=len(test), freq='D')
    forecast = prophet_model.predict(future)
    return forecast[
        forecast['ds'] >= test.index.min()
    ]['yhat'].values

def get_lstm_forecast():
    TIMESTEPS  = 7
    full_data  = pd.concat([train_fe, test_fe])
    X_full     = scaler_X.transform(
        full_data[feature_cols])
    y_full     = scaler_y.transform(
        full_data[['total_sales']])

    def create_sequences(X, y, timesteps):
        Xs, ys = [], []
        for i in range(timesteps, len(X)):
            Xs.append(X[i-timesteps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(
        X_full, y_full, TIMESTEPS)
    train_size   = len(train_fe) - TIMESTEPS
    X_test_seq   = X_seq[train_size:]

    pred_scaled  = lstm_model.predict(X_test_seq)
    pred         = scaler_y.inverse_transform(
        pred_scaled).flatten()
    return pred

# Generate forecast based on selection
with st.spinner(f'Running {model_choice} forecast...'):
    if model_choice == 'ARIMA':
        predictions = get_arima_forecast()
        mape = 24.77
        color = '#e74c3c'
    elif model_choice == 'Prophet':
        predictions = get_prophet_forecast()
        mape = 18.87
        color = '#3498db'
    else:
        predictions = get_lstm_forecast()
        mape = 14.83
        color = '#9b59b6'

actual = test['total_sales'].values[:len(predictions)]
dates  = test.index[:len(predictions)]

# Forecast chart
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(dates, actual,
         color='#2ecc71', linewidth=2,
         marker='o', label='Actual Sales')
ax2.plot(dates, predictions,
         color=color, linewidth=2,
         marker='s', linestyle='--',
         label=f'{model_choice} Forecast')
ax2.fill_between(dates,
                 predictions * 0.85,
                 predictions * 1.15,
                 alpha=0.15, color=color,
                 label='±15% Range')
ax2.set_xlabel('Date')
ax2.set_ylabel('Total Sales')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

# Forecast metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric('Model', model_choice)
col2.metric('MAPE', f'{mape}%')
col3.metric('Avg Forecast',
            f"{predictions.mean():,.0f}")
col4.metric('Forecast Days', len(predictions))

st.markdown('---')

# ─────────────────────────────────────
# FORECAST TABLE
# ─────────────────────────────────────
st.subheader('📋 Day by Day Forecast')

forecast_df = pd.DataFrame({
    'Date':           dates.strftime('%Y-%m-%d'),
    'Actual Sales':   actual.round(0).astype(int),
    'Forecast':       predictions.round(0).astype(int),
    'Difference':     (predictions - actual).round(0).astype(int),
    'Error %':        (abs(predictions - actual) / actual * 100).round(1)
})

st.dataframe(
    forecast_df.style.background_gradient(
        subset=['Error %'],
        cmap='RdYlGn_r'
    ),
    use_container_width=True
)

st.markdown('---')

# ─────────────────────────────────────
# BUSINESS INSIGHTS
# ─────────────────────────────────────
st.subheader('💡 Business Insights')

col1, col2 = st.columns(2)

with col1:
    st.markdown('### Weekly Pattern')
    st.markdown('''
    - **Wednesday** → highest sales (+2,200 above avg)
    - **Sunday** → lowest sales (-4,000 below avg)
    - Stock up inventory before **Wednesday**
    - Schedule maintenance on **Sundays**
    ''')

with col2:
    st.markdown('### Seasonal Pattern')
    st.markdown('''
    - **December** → peak month (10,245 avg)
    - **February** → slowest month (7,337 avg)
    - Plan promotions in **Feb, May, Aug**
    - Increase staff in **Nov, December**
    ''')

st.markdown('---')
st.caption(
    'Built by Arup Roy | '
    'github.com/analyst-ted | '
    'Models: ARIMA, Prophet, LSTM'
)