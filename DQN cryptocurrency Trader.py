import Agent
import ReplayBuffer
import Environment
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


# Split data into training and testing sets
split_index = int(0.8 * len(closing_price))
train_prices = closing_price[:split_index]
test_prices = closing_price[split_index:]

# Initialize the trading environment and DQN agent
train_env= TradingEnv(stock_price_history=train_prices,action_choices=action_choices)
test_env = TradingEnv(stock_price_history=test_prices,action_choices=action_choices)

state_size = train_env.observation_space.shape[0]
action_size = train_env.action_space.n

agent = DQNAgent(state_size, action_size,batch_size=50,update_target_interval=100)

#main loop
agent_value = []
for e in range(10):
    state = train_env.reset()
    done = False
    score = 0
    steps=0
    while not done:
        action = agent.act(state)
        next_state, reward, done ,value = train_env.step(action_choices[action])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        agent_value.append(value)
        steps+=1
        if (steps%500)==0:
            print(f"step{steps} value os far:{value}   cap:{train_env.capital} st:{train_env.stock} eps:{agent.epsilon}")
            plt.plot(agent_value)
            plt.show()
        agent.train(50)
    print(f'Episode {e}, Score(total_reward): {score:.4f}')




print("testing...")
agent_value = []
for e in range(1):
    state = test_env.reset()
    done = False
    score = 0
    steps=0
    while not done:
        action = agent.act(state)
        next_state, reward, done ,value = test_env.step(action_choices[action])
#         agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        agent_value.append(value)
        steps+=1
        if (steps%100)==0:
            print(f"step{steps} value os far:{value}   cap:{train_env.capital} st:{train_env.stock} eps:{agent.epsilon}")
            plt.plot(agent_value)
            plt.show()
#         agent.train(50)
    print(f'Episode {e}, Score(total_reward): {score:.4f}')
plt.plot(agent_value)
