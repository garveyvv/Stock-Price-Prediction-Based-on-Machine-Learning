#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import datetime as dt
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


yf.pdr_override()
df = pdr.get_data_yahoo("AAPL", start="2012-01-01", end="2019-12-17")

df


# In[3]:


df.shape


# In[4]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Close Price USD ($)', fontsize= 18)

plt.show()


# In[5]:


# Computed the 60-day Simple Moving Average (SMA)

df['SMA_60'] = df['Close'].rolling(window=60).mean()

# Predicted stock prices based on SMA
# Assuming that the simple strategy is: buy when the stock price crosses above the SMA and sell when the stock price crosses below the SMA
# The signal for buying is 1, and the signal for selling is -1

df['Signal'] = np.where(df['Close'] > df['SMA_60'], 1, -1)

# Plotted the trend chart of stock price and SMA line, as well as buy and sell signals

plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.plot(df['SMA_60'], label='60-day SMA', color='yellow')
plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'], label='Buy Signal', color='green', marker='^', alpha=1)
plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'], label='Sell Signal', color='red', marker='v', alpha=1)
plt.title('Stock Price with 60-day Simple Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.show()


# In[6]:


# Computed Relative Strength Index (RSI)

def calculate_rsi(data, window=60):
    delta = np.diff(data)
    gain = (delta >= 0) * delta
    loss = (delta < 0) * (-delta)
    
    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)
    
    avg_gain[window] = np.mean(gain[:window])  # The initial average upward movement
    avg_loss[window] = np.mean(loss[:window])  # The initial average downward movement
    
    for i in range(window + 1, len(data) - 1):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

rsi = calculate_rsi(df['Close'])
rsi


# In[7]:


df['RSI'] = rsi

# Plotted the trend chart of stock price, SMA, and RSI line, as well as buy and sell signals

plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.plot(df['SMA_60'], label='60-day SMA', color='yellow')
plt.plot(df['RSI'], label='RSI', color='grey')
plt.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['Close'], label='Buy Signal', color='green', marker='^', alpha=1)
plt.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['Close'], label='Sell Signal', color='red', marker='v', alpha=1)
plt.title('Stock Price with 60-day Simple Moving Average and RSI')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.show()


# In[8]:


data1 = df.filter(['Close'])
data2 = df.filter(['Signal'])
data3 = df.filter(['RSI'])

# Merged historical stock price data, buy and sell signal data and RSI data into the same DataFrame

df_merged = pd.concat([data1, data2, data3], axis=1)

# Removed missing values

df_merged_clean = df_merged.dropna()
df_merged_clean


# In[9]:


dataset = df_merged_clean.values
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[10]:


training_data_len = math.ceil(len(dataset) * 0.8)

training_data_len


# In[11]:


train_data = scaled_data[0 : training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60: i, :])
    y_train.append(train_data[i, 0])


# In[12]:


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))

x_train.shape


# In[13]:


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 3)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))


# In[14]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[15]:


model.fit(x_train, y_train, batch_size = 1, epochs = 1)


# In[16]:


test_data = scaled_data[training_data_len - 60: , :]

x_test = []
y_test = dataset[training_data_len: , 0]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60: i, :])


# In[17]:


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))

x_test.shape


# In[18]:


predictions = model.predict(x_test)
a, b = np.zeros_like(predictions), np.zeros_like(predictions)
predictions = np.hstack((predictions, a, b))
predictions = scaler.inverse_transform(predictions)

y_pred = predictions[:, 0]


# In[19]:


rmse = np.sqrt(np.mean(y_test - y_pred)** 2)
rmse


# In[20]:


mae = np.mean(abs(y_test - y_pred))
mae


# In[21]:


mape = np.mean(abs((y_test - y_pred)/y_test))
mape


# In[22]:


train = df_merged_clean[: training_data_len]
valid = df_merged_clean[training_data_len: ]
valid['Predictions'] = y_pred

valid


# In[23]:


plt.figure(figsize = (16, 8))
plt.title('Model 3')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')

plt.show()


# In[24]:


apple_quote = pdr.get_data_yahoo("AAPL", start="2012-01-01", end="2019-12-18")
new_df = apple_quote.filter(['Close'])

new_df['SMA_60'] = new_df['Close'].rolling(window=60).mean()
new_df['Signal'] = np.where(new_df['Close'] > new_df['SMA_60'], 1, -1)

def calculate_rsi(data, window=60):
    delta = np.diff(data)
    gain = (delta >= 0) * delta
    loss = (delta < 0) * (-delta)
    
    avg_gain = np.zeros_like(data)
    avg_loss = np.zeros_like(data)
    
    avg_gain[window] = np.mean(gain[:window])  # 初始的平均上涨幅度
    avg_loss[window] = np.mean(loss[:window])  # 初始的平均下跌幅度
    
    for i in range(window + 1, len(data) - 1):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

rsi = calculate_rsi(new_df['Close'])
new_df['RSI'] = rsi

data1 = new_df.filter(['Close'])
data2 = new_df.filter(['Signal'])
data3 = new_df.filter(['RSI'])
new_df_merged = pd.concat([data1, data2, data3], axis=1)
new_df_merged_clean = new_df_merged.dropna()

last_60_days = new_df_merged[-61: -1].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))

pred_price = model.predict(X_test)
a, b = np.zeros_like(pred_price), np.zeros_like(pred_price)
pred_price = np.hstack((pred_price, a, b))
pred_price = scaler.inverse_transform(pred_price)
pred_price_3 = pred_price[0, 0]

pred_price_3


# In[25]:


apple_quote2 = pdr.get_data_yahoo("AAPL", start="2019-12-18", end="2019-12-19")
real_price = apple_quote2['Close']

real_price

