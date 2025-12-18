import pandas as pd
import numpy as np
from pathlib import Path

# Config
DATA_PATH = Path("data/btc_ohlcv_1h.csv")   # raw OHLCV
OUTPUT_PATH = Path("data/reference.csv")
 
    #Training period (example – adjust if needed)
TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31"

REFERENCE_MONTHS = 12
# Feature engineering
# -----------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features exactly as used during training.
    """
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["close"]).diff()

    # 24h realized volatility (24 * 1h candles)
    df["realized_vol_24h"] = (
        df["log_return"]
        .rolling(window=24)
        .std()
    )

    return df


# -----------------------------
# Main
# -----------------------------
def main():
    # Load raw OHLCV
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    # Restrict to training period
    df = df[
        (df["timestamp"] >= TRAIN_START) &
        (df["timestamp"] <= TRAIN_END)
    ]

    # Feature computation
    df = compute_features(df)

    # Drop initial NaNs from rolling windows
    df = df.dropna().reset_index(drop=True)

    # Take last N months as reference
    reference_df = df.tail(REFERENCE_MONTHS * 30 * 24)

    # Keep ONLY features + timestamp
    reference_df = reference_df[
        ["timestamp", "log_return", "realized_vol_24h"]
    ]

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save reference dataset
    reference_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Reference dataset saved to {OUTPUT_PATH}")
    print(f"Rows: {len(reference_df)}")


if __name__ == "__main__":
    main()