import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data, data[['Close']].values

def prepare_data(close_prices, lookback=60):
    scaler = MinMaxScaler()
    train_data = close_prices[:-1]
    scaled_train = scaler.fit_transform(train_data)
    scaled_data = np.concatenate([scaled_train, scaler.transform(close_prices[-1:])])

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return scaled_data, X, y, scaler

def build_and_train_model(X, y):
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)
    return model

def predict_price(model, scaler, scaled_data, original_data, lookback=60):
    last_60_days = scaled_data[-lookback:]
    X_test = last_60_days.reshape(1, lookback, 1)
    predicted_scaled = model.predict(X_test, verbose=0)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    actual_prices = original_data['Close'].values[-lookback:]
    dates = original_data.index[-lookback:]
    return predicted_price, actual_prices, dates
