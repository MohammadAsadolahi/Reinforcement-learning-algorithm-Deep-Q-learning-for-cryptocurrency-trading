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
        self.capital = self.init_capital #initial current capitaql to initial_capital
        return self._next_observation() # return the first observation
    
    def _next_observation(self):
        prices = self.stock_price_history[self.current_step:self.current_step+self.window_size] 
        #return the price seris according to current place and up to window_size eg. [23 to 23 + 30]
        return np.array(prices)
    
    def step(self, action):
        stock_price = self.stock_price_history[self.current_step+self.window_size]
        portfolio_value = (self.capital + self.stock * stock_price) # total portfolio value including cash and stocks
        if action > 0 and action <= self.capital :
            self.capital -= action
            self.stock += action/stock_price
        elif action < 0 and (self.stock * stock_price)>(-action):
            self.stock += action/stock_price                        
            self.capital -= action
        new_portfolie_value = self.capital + self.stock * self.stock_price_history[self.current_step+self.window_size+1]
        reward = new_portfolie_value - portfolio_value  # reward = protfolio value after commiting action - portfolio before commiting action
        self.current_step += 1
        done = self.current_step+self.window_size + 2 >= len(self.stock_price_history) # if we will reach the end of price series in next step of the environment
        return self._next_observation(), reward, done, new_portfolie_value # new_portfolie_value is returned due to track agent progress and its optional to log progress only


# Split data into training and testing sets
split_index = int(0.8 * len(closing_price))
train_prices = closing_price[:split_index]
test_prices = closing_price[split_index:]

# Initialize the trading environment and DQN agent
train_env= TradingEnv(stock_price_history=train_prices)
test_env = TradingEnv(stock_price_history=test_prices)

state_size = train_env.observation_space.shape[0]
action_size = train_env.action_space.n

agent = DQNAgent(state_size, action_size,batch_size=50,update_target_interval=100)

#main loop
agent1_value = []
for e in range(10):
    state = train_env.reset()
    done = False
    score = 0
    steps=0
    while not done:
        action = agent.act(state)
        next_state, reward, done ,value = train_env.step(actions[action][1])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        agent_value.append(value)
        steps+=1
        if (steps%500)==0:
            print(f"step{steps} value os far:{value}   cap:{train_env.capital} st:{train_env.stock} eps:{agent.epsilon}")
            plt.plot(agent_value)
            plt.show()
        agent1.train(50)
    print(f'Episode {e}, Score(total_reward): {score:.4f}')
