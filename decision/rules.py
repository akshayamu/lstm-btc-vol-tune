def volatility_regime(vol, low=0.015, high=0.035):
    """
    Classify volatility into regimes.
    Thresholds are interpretable and tunable.
    """
    if vol < low:
        return "low"
    elif vol > high:
        return "high"
    else:
        return "medium"


def position_size(regime):
    """
    Risk-aware position sizing based on volatility regime.
    """
    if regime == "low":
        return 1.0      # full exposure
    elif regime == "medium":
        return 0.5      # reduced exposure
    else:
        return 0.0      # risk-off
