import pandas as pd
from decision.rules import volatility_regime, position_size


def apply_decision_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply volatility-based decision rules to predictions.
    Expects columns:
      - predicted_vol
      - return
    """
    df = df.copy()

    df["vol_regime"] = df["predicted_vol"].apply(volatility_regime)
    df["position"] = df["vol_regime"].apply(position_size)

    # Risk-adjusted strategy return
    df["strategy_return"] = df["position"] * df["return"]

    return df
