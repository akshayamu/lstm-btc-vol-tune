# monitoring/make_regression_report.py

import pandas as pd
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping

PREDICTIONS_PATH = Path("logs/predictions.csv")
REPORT_PATH = Path("reports/regression_report.html")


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    # Explicit column mapping (REQUIRED in Evidently 0.4.x)
    column_mapping = ColumnMapping(
        target="y_true",
        prediction="y_pred",
        numerical_features=["volatility"]
    )

    report = Report(
        metrics=[RegressionPreset()]
    )

    report.run(
        reference_data=None,
        current_data=df,
        column_mapping=column_mapping
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    print(f"Regression report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
