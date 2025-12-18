# monitoring/run_batch_inference.py

from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/btc_ohlcv_1h.csv")
MODEL_PATH = Path("models/lstm_volatility_model.keras")
LOG_PATH = Path("logs/predictions.csv")

LOOKBACK = 48
MONITOR_DAYS = 30


# -----------------------------
# Feature engineering
# -----------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(window=24).std()
    return df


# -----------------------------
# Sliding window inference
# -----------------------------
def main():
    # Load raw data
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # Feature engineering
    df = compute_features(df)
    df = df.dropna().reset_index(drop=True)

    # Split train / test (for scaler fit)
    train_df = df[df["timestamp"] < "2024-01-01"]
    test_df = df[df["timestamp"] >= "2024-01-01"]

    # Fit scaler ONLY on train
    scaler = MinMaxScaler()
    scaler.fit(train_df[["volatility"]])

    # Restrict to last N days of 2024
    end_time = test_df["timestamp"].max()
    start_time = end_time - pd.Timedelta(days=MONITOR_DAYS)

    monitor_df = test_df[
        test_df["timestamp"] >= start_time
    ].reset_index(drop=True)

    # Scale volatility
    scaled_vol = scaler.transform(monitor_df[["volatility"]])

    # Load trained model
    model = load_model(MODEL_PATH)

    rows = []

    for i in range(LOOKBACK, len(monitor_df)):
        X_window = scaled_vol[i - LOOKBACK:i]
        X_window = X_window.reshape(1, LOOKBACK, 1)

        # Predict (scaled)
        y_pred_scaled = model.predict(X_window, verbose=0)

        # Inverse scale
        y_pred = scaler.inverse_transform(y_pred_scaled)[0, 0]

        rows.append({
            "timestamp": monitor_df.loc[i, "timestamp"],
            "volatility": monitor_df.loc[i, "volatility"],
            "y_true": monitor_df.loc[i, "volatility"],
            "y_pred": y_pred
        })

    pred_df = pd.DataFrame(rows)

    # Append to log (schema-safe)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0

    pred_df.to_csv(
        LOG_PATH,
        mode="a",
        header=write_header,
        index=False
    )

    print(f"Logged {len(pred_df)} predictions to {LOG_PATH}")


if __name__ == "__main__":
    main()
