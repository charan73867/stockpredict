import numpy as np
import yfinance as yf
import pickle
import math

from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# --------------------------
# LOAD MODEL + SCALER
# --------------------------
model = load_model("model/lstm_model.h5")
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# =========================================================
# NEXT DAY PREDICTION (LSTM)
# =========================================================
def predict_next_day(stock):
    df = yf.download(stock, start="2015-01-01")
    data = df[['Close']]

    scaled_data = scaler.transform(data)

    last_60 = scaled_data[-60:]
    X_test = np.array([last_60])

    pred = model.predict(X_test, verbose=0)
    price = scaler.inverse_transform(pred)

    return float(price[0][0])


# =========================================================
# NEXT 7 DAYS FORECAST
# =========================================================
def predict_7_days(stock):
    df = yf.download(stock, start="2015-01-01")
    data = df[['Close']]

    scaled_data = scaler.transform(data)

    temp = scaled_data[-60:].tolist()
    preds = []

    for _ in range(7):
        x = np.array([temp[-60:]])
        pred = model.predict(x, verbose=0)
        temp.append(pred[0])
        preds.append(pred[0])

    preds = scaler.inverse_transform(preds)
    return preds


# =========================================================
# MODEL ACCURACY (LSTM)
# =========================================================
def get_accuracy(stock):
    df = yf.download(stock, start="2015-01-01")
    data = df[['Close']]

    scaled_data = scaler.transform(data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    preds = model.predict(X, verbose=0)

    preds = scaler.inverse_transform(preds)
    y = scaler.inverse_transform(y)

    rmse = math.sqrt(mean_squared_error(y, preds))
    accuracy = 100 - (rmse / np.mean(y)) * 100

    return round(float(accuracy), 2)


# =========================================================
# GET FULL LSTM PREDICTIONS (for graph)
# =========================================================
def get_lstm_predictions(stock):
    df = yf.download(stock, start="2015-01-01")
    data = df[['Close']]

    scaled_data = scaler.transform(data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    preds = model.predict(X, verbose=0)

    preds = scaler.inverse_transform(preds)
    y = scaler.inverse_transform(y)

    return y.flatten(), preds.flatten()


# =========================================================
# LINEAR REGRESSION MODEL
# =========================================================
def linear_regression_prediction(stock):
    df = yf.download(stock, start="2015-01-01")
    data = df[['Close']]

    # create day index
    data['Day'] = np.arange(len(data))

    X = data[['Day']]
    y = data['Close']

    lr = LinearRegression()
    lr.fit(X, y)

    preds = lr.predict(X)

    return y.values, preds


# =========================================================
# ACCURACY CALCULATION FOR ANY MODEL
# =========================================================
def calculate_accuracy(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    accuracy = 100 - (rmse / np.mean(actual)) * 100
    return round(float(accuracy), 2)