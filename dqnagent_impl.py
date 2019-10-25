import random
import numpy as np
from collections import deque
from convolutional_neural_network import CNN_Impl


class DQNAgent:
    ''' Keras implementation of the paper '''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=40000)
        
        self.gamma = 0.95
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.0000009
        self.epsilon_min = 0.1
        
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        
    def _build_model(self):
        model = CNN_Impl()
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        
    def act(self, state):
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
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def load(self, name):
        ''' TODO: Load model weights from some input file '''
        
    def save(self, name):
        ''' TODO: Save model weights to some output file '''
