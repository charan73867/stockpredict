import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pickle

from predict import predict_next_day, predict_7_days, get_accuracy

st.set_page_config(page_title="Stock Predictor", layout="wide")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("📊 Controls")

stocks = ["AAPL", "TSLA", "GOOG", "MSFT", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
stock = st.sidebar.selectbox("Select Stock", stocks)

predict_btn = st.sidebar.button("🔮 Predict Next Day")
forecast_btn = st.sidebar.button("📅 Predict 7 Days")
accuracy_btn = st.sidebar.button("🎯 Show Accuracy")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    "<h1 style='text-align:center;'>📈 AI Stock Price Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================================================
# NEXT DAY PREDICTION
# =========================================================
# =========================================================
# NEXT DAY PREDICTION
# =========================================================
if predict_btn:
    with st.spinner("Running LSTM model..."):

        price = predict_next_day(stock)

        df_today = yf.download(stock, period="5d")

        if df_today.empty:
            st.error("Could not fetch latest stock price.")
        else:
            current_price = float(df_today['Close'].iloc[-1])
            delta = price - current_price

            st.metric(
                label="Predicted Price",
                value=f"{price:.2f}",
                delta=round(delta,2)
            )
# =========================================================
# 7 DAY FORECAST
# =========================================================
if forecast_btn:
    with st.spinner("Generating 7-day forecast..."):
        preds = predict_7_days(stock)

        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(preds, marker='o')
        ax.set_title("Next 7 Days Forecast")
        st.pyplot(fig)

# =========================================================
# ACCURACY
# =========================================================
if accuracy_btn:
    acc = get_accuracy(stock)
    st.metric("Model Accuracy", f"{acc}%")

st.markdown("---")

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["📊 Stock Chart", "📉 Training Loss"])

# --------------------------
# STOCK CHART
# --------------------------
with tab1:
    st.subheader("Price Trend (Last 2 Years)")

    df_line = yf.download(stock, period="2y")
    st.line_chart(df_line['Close'])

# --------------------------
# LOSS GRAPH
# --------------------------
with tab2:
    if st.button("Show Training Loss"):
        history = pickle.load(open("model/history.pkl","rb"))

        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(history['loss'])
        ax.set_title("Training Loss")
        st.pyplot(fig)

st.markdown("---")

# =========================================================
# MODEL INFO
# =========================================================
with st.expander("Model Details"):
    st.write("""
    - Model: LSTM (Long Short-Term Memory)
    - Input Window: 60 days
    - Optimizer: Adam
    - Loss Function: Mean Squared Error
    - Data Source: Yahoo Finance
    """)



# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<p style='text-align:center;'>Built using LSTM + Streamlit</p>",
    unsafe_allow_html=True
)