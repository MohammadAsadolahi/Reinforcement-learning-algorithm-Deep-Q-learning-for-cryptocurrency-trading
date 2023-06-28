import Agent
import ReplayBuffer
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
action_choices = np.linspace(-20, 20, num=51) # using linespace to generate 25 actions to buy or sell in [0.5$,20$] interval
print(action_choices)
plt.xlabel("action id")
plt.ylabel("action value")
plt.title(f"generated discrete action space")
plt.scatter([act for act in range(len(action_choices))],action_choices)


class TradingEnv(gym.Env) :
    def __init__(self, init_capital=2000, stock_price_history=[], window_size=30):
        self.init_capital = init_capital #amount of money we have at the initial step
        self.stock = 0 # initial amount of stock we have (eg. 0 Bitcoin at start)
        self.stock_price_history = stock_price_history # the full series of stock or currency values 
        self.window_size = window_size # amount of data we look at to predict the next price
        self.current_step = 0 # the inital location to start
        self.reset()
        
    def reset(self) :
        self.current_step = 0
        self.stock = 0
        self.capital = self.init_capital
        return self._next_observation()
    
    def _next_observation(self):
        prices = self.stock_price_history[self.current_step:self.current_step+self.window_size]
        return np.array(prices)
    
    def step(self, action):
        stock_price = self.stock_price_history[self.current_step+self.window_size]
        portfolio_value=(self.capital + self.stock * stock_price)
        if action > 0 and action <= self.capital :
            self.capital -= action
            self.stock += action/stock_price
        elif action < 0 and (self.stock * stock_price)>(-action):
            self.stock += action/stock_price                        
            self.capital -= action
        reward = (self.capital + self.stock * self.stock_price_history[self.current_step+self.window_size+1]) - portfolio_value
        self.current_step += 1
        done = self.current_step+self.window_size + 2 >= len(self.stock_price_history)
        return self._next_observation(), reward, done, (self.capital + self.stock * self.stock_price_history[self.current_step+self.window_size+1])
