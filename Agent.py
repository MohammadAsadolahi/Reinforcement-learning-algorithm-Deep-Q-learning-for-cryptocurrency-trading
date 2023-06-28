#Define the DQN agent 
class DQNAgent :
    def __init__(self, state_size, action_size,batch_size,update_target_interval=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(1000000,state_size,action_size)
        self.gamma = 0.90 # up to 0.99
        self.epsilon = 1.0
        self.epsilon_min =0.1
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.update_target_interval =update_target_interval
        self.update_target_counter=0
        
    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(128,8, input_shape=(self.state_size,1), padding='same'))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Conv1D(64,8, padding='same'))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(384))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def act(self, state,test_mode=False):
        if not test_mode:
            if np.random.rand() <= self.epsilon :
                return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(state,axis=0), verbose=0)
        return np.argmax(act_values[0])
    
    def train(self, batch_size) :
        if self.memory.mem_cntr  < batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)
        
        qState=self.model.predict(state,verbose=0)
        qNextState=self.model.predict(new_state,verbose=0)
        qNextStateTarget=self.target_model.predict(new_state,verbose=0)
        maxActions=np.argmax(qNextState,axis=1)
        batchIndex = np.arange(batch_size, dtype=np.int32)
        qState[batchIndex,action]=(reward+(self.gamma*qNextStateTarget[batchIndex,maxActions.astype(int)]*(1-done)))
        _=self.model.fit(x=state,y=qState,verbose=0,epochs=65)

        self.update_target_counter+=1
        if self.update_target_counter % self.update_target_interval==0 :
            self.target_model.set_weights(self.model.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
