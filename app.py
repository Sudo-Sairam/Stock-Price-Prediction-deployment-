import streamlit as st
from datetime import datetime
from model_utils import download_data, prepare_data, build_and_train_model, predict_price
import matplotlib.pyplot as plt
import pandas as pd

st.title("Stock Price Prediction using LSTM")
st.markdown("This app predicts the **next closing price** for a selected stock using a deep learning LSTM model.")

# User inputs
ticker = st.text_input("Enter Stock Symbol (e.g., INFY.NS, TCS.NS)", "INFY.NS")
start_date = st.date_input("Start Date", datetime(2018, 1, 1))
end_date = datetime.today().strftime('%Y-%m-%d')

if st.button("Predict"):
    with st.spinner("Training model and predicting..."):
        data, close_prices = download_data(ticker, start_date, end_date)
        scaled_data, X, y, scaler = prepare_data(close_prices)
        model = build_and_train_model(X, y)
        predicted_price, actual_prices, dates = predict_price(model, scaler, scaled_data, data)

        # Display result
        st.success(f"Predicted closing price for {datetime.today().strftime('%Y-%m-%d')}: ₹{predicted_price:.2f}")

        # Plot
        predicted_plot = [None] * (len(actual_prices) - 1) + [predicted_price]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, actual_prices, label='Actual Price', linewidth=2)
        ax.plot(dates, predicted_plot, label='Predicted Closing Price', linestyle='--', marker='o', color='orange')
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
