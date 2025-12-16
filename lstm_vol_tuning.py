import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# -----------------------------
# 1. Data Loading
# -----------------------------
client = Client()

klines = client.get_historical_klines(
    "BTCUSDT",
    Client.KLINE_INTERVAL_1HOUR,
    "1 Jan, 2019"
)

df = pd.DataFrame(klines, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades",
    "taker_base_vol", "taker_quote_vol", "ignore"
])

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["close"] = df["close"].astype(float)

df = df[["timestamp", "close"]]


# -----------------------------
# 2. Feature Engineering
# -----------------------------
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

# 24h realized volatility
df["volatility"] = df["log_return"].rolling(window=24).std()
df = df.dropna().reset_index(drop=True)


# -----------------------------
# 3. Train / Test Split
# -----------------------------
train_df = df[df["timestamp"] < "2024-01-01"]
test_df  = df[df["timestamp"] >= "2024-01-01"]

train_vol = train_df[["volatility"]]
test_vol  = test_df[["volatility"]]


# -----------------------------
# 4. Scaling (No Leakage)
# -----------------------------
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_vol)
test_scaled  = scaler.transform(test_vol)


# -----------------------------
# 5. Sequence Builder
# -----------------------------
def create_sequences(data, lookback=48):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

LOOKBACK = 48

X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_test, y_test   = create_sequences(test_scaled, LOOKBACK)

X_train = X_train.reshape(X_train.shape[0], LOOKBACK, 1)
X_test  = X_test.reshape(X_test.shape[0], LOOKBACK, 1)


# -----------------------------
# 6. Baseline Model
# -----------------------------
baseline_pred = (
    test_vol["volatility"]
    .rolling(24)
    .mean()
    .iloc[LOOKBACK:]
    .values
)

baseline_actual = test_vol["volatility"].iloc[LOOKBACK:].values
baseline_rmse = np.sqrt(mean_squared_error(baseline_actual, baseline_pred))


# -----------------------------
# 7. LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# -----------------------------
# 8. Evaluation
# -----------------------------
pred_scaled = model.predict(X_test)
pred_vol = scaler.inverse_transform(pred_scaled)
actual_vol = scaler.inverse_transform(y_test.reshape(-1, 1))

lstm_rmse = np.sqrt(mean_squared_error(actual_vol, pred_vol))

print(f"Baseline RMSE: {baseline_rmse:.6f}")
print(f"LSTM RMSE:     {lstm_rmse:.6f}")


# -----------------------------
# 9. Visualization
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(actual_vol, label="Actual Volatility")
plt.plot(pred_vol, label="LSTM Forecast")
plt.title("BTC 24h Realized Volatility Forecast")
plt.legend()
plt.tight_layout()
plt.savefig("assets/vol_forecast.png", dpi=150)
plt.show()





