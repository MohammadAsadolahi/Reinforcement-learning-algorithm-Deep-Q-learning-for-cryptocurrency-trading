class TradingEnv(gym.Env) :
    def __init__(self,action_choices, init_capital=2000, stock_price_history=[], window_size=30):
        self.init_capital = init_capital #amount of money we have at the initial step
        self.stock = 0 # initial amount of stock we have (eg. 0 Bitcoin at start)
        self.stock_price_history = stock_price_history # the full series of stock or currency values 
        self.window_size = window_size # amount of data we look at to predict the next price
        self.current_step = 0 # the inital location to start
        self.action_space = spaces.Discrete(len(action_choices))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size,))
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