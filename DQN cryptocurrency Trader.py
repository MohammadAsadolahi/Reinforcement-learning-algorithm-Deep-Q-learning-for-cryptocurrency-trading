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


Crypto_name = ["BTC-USD"] # Replace with other crypto currency e.g. "ETH-USD" 'XRP-USD' "LTC-USD"
start_date="2022-06-20"
end_date='2023-06-20'
closing_price=pd.DataFrame()
for i in Crypto_name:
    data= yf.Ticker(i)
    data = data.history(start=start_date , end=end_date,interval="1h")
    colse=pd.DataFrame(data.Close)
    closing_price[i] = colse
 

# Plot the closing price changes in the given period
plt.xlabel("date")
plt.ylabel("closing price")
plt.title(f"bitcoin closing prices from{start_date} to {end_date}")
plt.plot(closing_price['BTC-USD'])


# Generate the action space
import gym
from gym import spaces
action_choices = np.linspace(0.5, 20, num=25) # using linespace to generate 25 actions to buy or sell in [0.5$,20$] interval
actions = []
actions.append(0.0) # add 0 choice to actions for hold
for act in action_choices:
    actions.append(act) # add positive number to buy x amount
    actions.append(-act) # add negative number to sell x amount
actions.sort()

plt.xlabel("action id")
plt.ylabel("action value")
plt.title(f"generated discrete action space")
plt.scatter([act for act in range(len(actions))],actions)
