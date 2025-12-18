# monitoring/make_drift_report.py

import pandas as pd
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_PATH = Path("data/reference.csv")
PREDICTIONS_PATH = Path("logs/predictions.csv")
REPORT_PATH = Path("reports/data_drift_report.html")


def main():
    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(PREDICTIONS_PATH)

    reference_features = reference_df[["realized_vol_24h"]]
    current_features = current_df[["volatility"]].rename(
        columns={"volatility": "realized_vol_24h"}
    )

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    print(f"Data drift report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
