import random
import numpy as np
from collections import deque
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


class DQNAgent:
    ''' Keras implementation of the paper '''
    def __init__(self, state_size, action_size):
        print(K.tensorflow_backend._get_available_gpus())
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=80000)
        
        self.gamma = 0.95
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.0000009
        self.epsilon_min = 0.1
        
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
    
        model.add(Conv2D(1, kernel_size=3, activation='relu', input_shape=(80, 80, 4)))
        model.add(Conv2D(16, kernel_size=8, strides=4, activation='relu'))
        model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            
            self.model.fit(state, target - target_f, epochs=1, verbose=0)
    
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
