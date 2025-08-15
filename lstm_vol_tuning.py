#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

plt.style.use("seaborn-v0_8-darkgrid")


# In[20]:


# Download last 60 days of 5-min BTC-USD data
df = yf.download("BTC-USD", period="60d", interval="5m")
df = df[['Close']].dropna()
df.head()


# In[21]:


# Calculate rolling volatility (std dev of returns)
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=12).std()  # ~1 hour window
df = df.dropna()

plt.figure(figsize=(12,4))
plt.plot(df['Volatility'])
plt.title("BTC Rolling Volatility")
plt.show()


# In[22]:


scaler = MinMaxScaler()
scaled_vol = scaler.fit_transform(df[['Volatility']])

X, y = [], []
lookback = 24  # 2 hours of history

for i in range(lookback, len(scaled_vol)):
    X.append(scaled_vol[i-lookback:i, 0])
    y.append(scaled_vol[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM expects 3D

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train.shape, X_test.shape



# In[24]:


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)


# In[25]:


preds = model.predict(X_test)
preds = scaler.inverse_transform(preds.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12,4))
plt.plot(actual, label="Actual Volatility")
plt.plot(preds, label="Predicted Volatility")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




