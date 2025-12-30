import numpy as np

def evaluate_strategy(df):
    return {
        "avg_return": df["strategy_return"].mean(),
        "volatility": df["strategy_return"].std(),
        "max_drawdown": (df["strategy_return"].cumsum().cummax() - 
                         df["strategy_return"].cumsum()).max(),
        "risk_off_rate": (df["position"] == 0).mean()
    }
