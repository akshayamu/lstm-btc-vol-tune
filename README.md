 
# LSTM BTC Volatility Tune
Project Summary
BTC 24h Realized Volatility Forecasting (LSTM)
This project forecasts 24-hour realized volatility of Bitcoin (BTC) using an LSTM neural network trained on historical price sequences from Binance BTCUSDT 1-hour candles.

The model is trained on data from 2019-01-01 to 2023-12-31 and evaluated on a strictly out-of-sample test period from 2024-01-01 to 2024-12-31.

The target variable is 24-hour realized volatility, defined as the rolling standard deviation of log returns. Model performance is benchmarked against a statistical baseline based on rolling average volatility.
## Overview

This project implements a volatility forecasting and decision system for BTC markets.
Model predictions are explicitly converted into risk-aware decisions and evaluated based
on impact metrics such as drawdown, exposure, and stability, rather than prediction accuracy alone.


Results:
Baseline RMSE = 0.001323
LSTM RMSE = 0.000410
→ 69.01% reduction in out-of-sample RMSE

Data
Source: Binance BTCUSDT spot market

Frequency: 1-hour candles

Training period: 2019-01-01 → 2023-12-31

Test period: 2024-01-01 → 2024-12-31

Target & Baseline
Log returns

𝑟
𝑡
=
ln
⁡
(
𝑃
𝑡
)
−
Project Summary
BTC 24h Realized Volatility Forecasting (LSTM)

This project forecasts 24-hour realized volatility of Bitcoin (BTC) using an LSTM neural network trained on historical price sequences from Binance BTCUSDT 1-hour candles.

The model is trained on data from 2019-01-01 to 2023-12-31 and evaluated on a strictly out-of-sample test period from 2024-01-01 to 2024-12-31.

The target variable is 24-hour realized volatility, defined as the rolling standard deviation of log returns. Model performance is benchmarked against a statistical baseline based on rolling average volatility.

Results:
Baseline RMSE = 0.001323
LSTM RMSE = 0.000410
→ 69.01% reduction in out-of-sample RMSE
Project Summary
BTC 24h Realized Volatility Forecasting (LSTM)

Data

Source: Binance BTCUSDT spot market

Frequency: 1-hour candles

Training period: 2019-01-01 → 2023-12-31

Test period: 2024-01-01 → 2024-12-31

Target & Baseline

Log returns  rt​=ln(Pt​)−ln(Pt−1​)

Realized volatility (target)
24-hour realized volatility computed as the rolling standard deviation of log returns: σt​=std(rt−23​,…,rt​)

Baseline model
Rolling mean of realized volatility over the previous 24 hours.

Results

The figure below compares actual 24-hour realized volatility with the LSTM forecast on the out-of-sample test set.

Actual vs. LSTM Forecast: assets/vol_forecast.png

The LSTM model closely tracks volatility regimes and materially outperforms the rolling-volatility baseline, achieving a ~69% reduction in RMSE.
=======
# BTC 24h Realized Volatility Forecasting with LSTM

End-to-end machine learning project for forecasting **24-hour realized volatility of Bitcoin** using an LSTM model, with **batch monitoring for data drift and model performance**.

This repository demonstrates the **full ML lifecycle**: data ingestion, feature engineering, model training, evaluation, batch inference, and post-deployment monitoring.

---

## Overview

- **Task:** Forecast 24-hour realized volatility of BTC  
- **Frequency:** 1-hour OHLCV candles  
- **Model:** LSTM (sequence-based time-series forecasting)  
- **Baseline:** Rolling mean of realized volatility  
- **Monitoring:** Batch data drift + regression performance using Evidently  

The project is designed to resemble a **production ML workflow**, not just an experiment.

---
## Latency & Cost Considerations

The volatility forecasting system is designed for offline training and batch inference.

- Model training is performed offline and does not impact runtime latency.
- Inference is executed in batch mode with predictable compute cost.
- Monitoring and drift detection are decoupled from inference to ensure stability and reproducibility.

## Data

- **Source:** Binance (BTCUSDT, 1-hour candles)
- **Period:**
  - Train: `2019-01-01 → 2023-12-31`
  - Test / Monitoring: `2024-01-01 → 2024-12-31`

Raw data is fetched once and stored locally to ensure **reproducibility**.

```bash
python data/fetch_btc_ohlcv.py

Feature Engineering

Log returns

24-hour realized volatility (rolling standard deviation of log returns)

This follows standard practices in quantitative volatility modeling.

Model

Architecture:

LSTM (64 units)

Dropout (0.2)

Dense output layer

Lookback window: 48 hours

Loss: Mean Squared Error

Optimizer: Adam

The trained model is saved as a reusable artifact.
python lstm_vol_tuning.py
models/lstm_volatility_model.keras

Evaluation

The LSTM is evaluated against a rolling mean baseline.

Results

Baseline RMSE: ~0.00132

LSTM RMSE: ~0.00043

Improvement: ~68–70% RMSE reduction

The LSTM captures volatility clustering and regime persistence more effectively.

Monitoring (Data Drift & Performance)

This project includes batch monitoring using Evidently, simulating production-grade post-deployment monitoring.

Data Drift

Compares 2024 feature distributions against a training reference window

Uses Wasserstein distance to detect regime shifts

Output

reports/data_drift_report.html

Regression Performance

Tracks MAE, MAPE, residuals, and prediction bias

Confirms model stability even under detected drift

Output

reports/regression_report.html

Run Monitoring
python monitoring/make_reference.py
python monitoring/run_batch_inference.py
python monitoring/make_drift_report.py
python monitoring/make_regression_report.py

Project Structure
lstm-btc-vol-tune/
│
├── data/
│   ├── fetch_btc_ohlcv.py
│   ├── btc_ohlcv_1h.csv
│   └── reference.csv
│
├── models/
│   └── lstm_volatility_model.keras
│
├── monitoring/
│   ├── make_reference.py
│   ├── run_batch_inference.py
│   ├── make_drift_report.py
│   └── make_regression_report.py
│
├── logs/
│   └── predictions.csv
│
├── reports/
│   ├── data_drift_report.html
│   └── regression_report.html
│
├── assets/
│   └── report_screenshots.png
│
├── lstm_vol_tuning.py
├── requirements.txt
└── README.md

Reproducibility

Python 3.11

Virtual environment recommended

Dependencies pinned for stability

python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

Key Takeaways

Volatility forecasting benefits from sequence-based models

Market regimes change, making drift detection essential

Monitoring complements accuracy metrics in real ML systems

Reproducibility and schema discipline are critical in production

Future Improvements

Quarterly drift reports (Q1–Q4)

Automated retraining triggers based on drift thresholds

Data versioning with DVC

Deployment as a scheduled batch job or API
aee34e7 (Add LSTM BTC volatility model with batch monitoring and drift reports)
## System Architecture

This project is designed as a decision-oriented volatility forecasting system
where model outputs are translated into risk-aware actions and evaluated
based on impact metrics.

![Volatility Decision Architecture](diagrams/volatility_decision_architecture.png)
