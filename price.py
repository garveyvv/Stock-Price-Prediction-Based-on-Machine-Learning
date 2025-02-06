#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imported necessary libraries, including basic ones like numpy and pandas, as well as those for building deep learning models such as Sequential, Dense, LSTM
# Imported yfinance for fetching historical stock data and pandas_datareader for retrieving data from various internet sources

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


# Used the Yahoo Finance API to retrieve historical stock data for AAPL from January 1, 2012, to December 17, 2019

# Comment: To avoid potential data download failure caused by network issues, you can access the relevant data from the attached Excel file(AAPL.csv) locally. The corresponding code statement is as follows:
# df = pd.read_csv('AAPL.csv', index_col = 0)
# df

yf.pdr_override()
df = pdr.get_data_yahoo("AAPL", start="2012-01-01", end="2019-12-17")

df


# In[3]:


df.shape


# In[4]:


# Visualized the close prices

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Close Price USD ($)', fontsize= 18)

plt.show()


# In[5]:


# Filtered out the close prices and scaled the corresponding features to the range of 0 to 1

data = df.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[6]:


# Utilized the scaled data of the first 80% for training the model

training_data_len = math.ceil(len(dataset) * 0.8)

training_data_len


# In[7]:


# For each training instance, predicted the next closing price using the preceding 60 closing prices

train_data = scaled_data[0 : training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60: i, :])
    y_train.append(train_data[i, 0])


# In[8]:


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train.shape


# In[9]:


# Initialized a Sequential model
# Added two LSTM layers with 50 neurons each, followed by two fully connected layers with 25 neurons and 1 neuron respectively

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))


# In[10]:


# Compiled the model using the Adam optimizer and mean squared error loss function

model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[11]:


# Trained the model on the training data with a batch size of 1 and 1 epoch

model.fit(x_train, y_train, batch_size = 1, epochs = 1)


# In[12]:


# Utilized the scaled data of the last 20% for testing the model

test_data = scaled_data[training_data_len - 60: , :]

x_test = []
y_test = dataset[training_data_len: , 0]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60: i, :])


# In[13]:


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_test.shape


# In[14]:


# Predicted the closing prices of the test data and inverse transformed the predicted results back to their original scale

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

predictions


# In[15]:


# Computed Root Mean Squared Error (RMSE)

rmse = np.sqrt(np.mean(y_test - predictions)** 2)
rmse


# In[16]:


# Computed Mean Absolute Error (MAE)

mae = np.mean(abs(y_test - predictions))
mae


# In[17]:


# Computed Mean Absolute Percentage Error (MAPE)

mape = np.mean(abs((y_test - predictions)/y_test))
mape


# In[18]:


train = data[: training_data_len]
valid = data[training_data_len: ]
valid['Predictions'] = predictions

valid


# In[19]:


# Visualized the predicted values against the real closing prices for comparison

plt.figure(figsize = (16, 8))
plt.title('Model 1')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')

plt.show()


# In[20]:


# Predicted the closing price of the next day by obtaining the last 60 days of available data to evaluate the model

apple_quote = pdr.get_data_yahoo("AAPL", start="2012-01-01", end="2019-12-17")
new_df = apple_quote.filter(['Close'])

last_60_days = new_df[-60: ].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price_1 = pred_price[0, 0]

pred_price_1


# In[21]:


# Fetched the real price data and compared it with the predicted price

apple_quote2 = pdr.get_data_yahoo("AAPL", start="2019-12-18", end="2019-12-19")
real_price = apple_quote2['Close']

real_price

