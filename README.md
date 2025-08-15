# LSTM BTC Volatility Tuning

A compact, reproducible notebook that:
1) Pulls BTC-USD prices (Yahoo Finance),
2) Builds a supervised dataset from realized volatility,
3) Trains an LSTM with dropout & batch normalization,
4) Evaluates and plots predictions.

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # For Git Bash on Windows
pip install -r requirements.txt
# Run Jupyter
jupyter notebook
