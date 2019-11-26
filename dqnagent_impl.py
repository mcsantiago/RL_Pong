import random
import numpy as np
from collections import deque
from convolutional_neural_network import CNN_Impl

# Macros
UP_ACTION = 2
DOWN_ACTION = 3

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
        running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r) #normalizing the result
    discounted_r /= np.std(discounted_r) #idem
    return discounted_r

class DQNAgent:
    ''' Keras implementation of the paper '''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=80000)
        
        self.gamma = 0.95
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.0000009
        self.epsilon_min = 0.1
        
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        print(self.model)
        
    def _build_model(self):
        model = CNN_Impl((80, 80, 4), 1)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() <= self.epsilon:
            return UP_ACTION if np.random.uniform() < 0.5 else DOWN_ACTION
        prob = self.model.predict(state)
        return UP_ACTION if np.random.uniform() < prob else DOWN_ACTION
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        x_train=[]
        y_train=[]
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * self.model.predict(next_state)
            
            x_train.append(state)
            y_train.append(target)

        self.model.fit(x=x_train, y=y_train, epochs=1, sample_weight=None)
    
    def load(self, name):
        ''' TODO: Load model weights from some input file '''
        
    def save(self, name):
        ''' TODO: Save model weights to some output file '''
