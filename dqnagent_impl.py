import random
import numpy as np
from collections import deque
from neural_network import NeuralNetwork 
import pickle

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
        model = NeuralNetwork((6400, 1), self.action_size)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def forget(self):
        self.memory.clear()

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() <= self.epsilon:
            action = random.randrange(self.action_size) 
            return action
        act_values = self.model.predict(state)
        print('{}, action: {}'.format(act_values, np.argmax(act_values)))
        return np.argmax(act_values)
    
    def replay(self, batch_size):
        x_train=[]
        y_train=[]
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            x_train.append(state)
            y_train.append(target_f[0])

        error = self.model.fit(x=x_train, y=y_train, epochs=1, clip=1.0, sample_weight=None)
        return error

    
    def load(self, name):
        ''' TODO: Load model weights from some input file '''
        
    def save(self, name):
        pickle.dump(self, open(name, 'wb'))
