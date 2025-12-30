import numpy as np


def evaluate_strategy(df):
    """
    Evaluate decision-level impact of volatility-based strategy.
    """
    cumulative_returns = df["strategy_return"].cumsum()

    metrics = {
        "avg_return": float(df["strategy_return"].mean()),
        "volatility": float(df["strategy_return"].std()),
        "max_drawdown": float(
            (cumulative_returns.cummax() - cumulative_returns).max()
        ),
        "risk_off_rate": float((df["position"] == 0).mean()),
    }

    return metrics
