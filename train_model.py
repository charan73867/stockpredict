import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

os.makedirs("model", exist_ok=True)

# --------------------------
# DOWNLOAD DATA
# --------------------------
stock = "AAPL"   # you can change later

df = yf.download(stock, start="2015-01-01", end="2024-01-01")

data = df[['Close']]

# --------------------------
# SCALE DATA
# --------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# --------------------------
# CREATE SEQUENCES
# --------------------------
X, y = [], []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# --------------------------
# TRAIN TEST SPLIT
# --------------------------
split = int(len(X)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------
# MODEL
# --------------------------
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# --------------------------
# TRAIN
# --------------------------
history = model.fit(X_train, y_train, epochs=15, batch_size=32)

# --------------------------
# SAVE MODEL
# --------------------------
model.save("model/lstm_model.h5")
pickle.dump(history.history, open("model/history.pkl","wb"))

# --------------------------
# LOSS GRAPH SAVE
# --------------------------
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.savefig("model/loss.png")

print("MODEL TRAINED & SAVED")
