import pandas as pd
from decision.backtest import apply_decision_logic
from decision.evaluate import evaluate_strategy


def main():
    """
    Run decision-level evaluation on volatility data.
    Uses realized volatility as a proxy for predicted volatility.
    """

    # Load data
    df = pd.read_csv("data/reference.csv")

    # Map project-specific columns to decision schema
    df = df.rename(columns={
        "realized_vol_24h": "predicted_vol",
        "log_return": "return"
    })

    # Apply decision logic
    df = apply_decision_logic(df)

    # Evaluate decision impact
    metrics = evaluate_strategy(df)

    print("\nDecision-level metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
