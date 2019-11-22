import random
import numpy as np
from collections import deque
from convolutional_neural_network import CNN_Impl

# Macros
UP_ACTION = 2
DOWN_ACTION = 3

class DQNAgent:
    ''' Keras implementation of the paper '''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=40000)
        
        self.gamma = 0.95
        
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.0000009
        # self.epsilon_min = 0.1
        
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        print(self.model)
        
    def _build_model(self):
        model = CNN_Impl((80, 80, 4), 1)
        return model
    
    def remember(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        
    def act(self, state):
        prob = self.model.predict(state)
        return UP_ACTION if np.random.uniform() < prob else DOWN_ACTION
    
    def replay(self, batch_size):
        x_train = [i[0] for i in self.memory]
        y_train = [1 if i[1] == UP_ACTION else 0 for i in self.memory]
        d_reward = discount_rewards([t[2] for t in self.memory], self.gamma)

        # self.model.fit(x=np.vstack(x_train), y=np.vstack(y_train), epochs=epochs, verbose=1, sample_weight=d_reward, callbacks=[self.tensorboard])
    
    def load(self, name):
        ''' TODO: Load model weights from some input file '''
        
    def save(self, name):
        ''' TODO: Save model weights to some output file '''
