# LSTM BTC Volatility Tune
Project Summary
BTC 24h Realized Volatility Forecasting (LSTM)
This project forecasts 24-hour realized volatility of Bitcoin (BTC) using an LSTM neural network trained on historical price sequences from Binance BTCUSDT 1-hour candles.

The model is trained on data from 2019-01-01 to 2023-12-31 and evaluated on a strictly out-of-sample test period from 2024-01-01 to 2024-12-31.

The target variable is 24-hour realized volatility, defined as the rolling standard deviation of log returns. Model performance is benchmarked against a statistical baseline based on rolling average volatility.

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
