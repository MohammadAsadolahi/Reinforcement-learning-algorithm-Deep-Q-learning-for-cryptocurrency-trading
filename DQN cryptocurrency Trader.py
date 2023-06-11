import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear
from keras.layers import Dense, Dropout, Conv1D, MaxPooling2D, Activation, Flatten, Embedding, Reshape,MaxPooling1D,LeakyReLU
!pip install yfinance
import yfinance as yf


Crypto_name = ["BTC-USD"] # replace with other crypto currency e.g. "ETH-USD" 'XRP-USD' "LTC-USD"
closing_price=pd.DataFrame()
for i in Crypto_name:
    data= yf.Ticker(i)
    data = data.history(start="2022-06-20" , end='2023-06-20',interval="1h")
    colse=pd.DataFrame(data.Close)
    closing_price[i] = colse
 

# plot the closing price changes in the given period
plt.plot(closing_price['BTC-USD'])
