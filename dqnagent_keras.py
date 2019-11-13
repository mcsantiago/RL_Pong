import random
import numpy as np
from collections import deque
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

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
    def __init__(self, state_size, action_size, weight_file = None):
        print(K.tensorflow_backend._get_available_gpus())
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=80000)
        
        self.gamma = 0.99
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        
        if weight_file is not None:
            print("Loading weights: {}", weight_file)
            self.load(weight_file)
        
    def _build_model(self):
        model = Sequential()
    
        model.add(Conv2D(1, kernel_size=3, activation='relu', input_shape=(80, 80, 4)))
        model.add(Conv2D(16, kernel_size=8, strides=4, activation='relu'))
        model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer='RandomNormal'))
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def forget(self):
        self.memory.clear()
        
    def act(self, state):
        prob = self.model.predict(state)
        return UP_ACTION if np.random.uniform() < prob else DOWN_ACTION
    
    def replay(self, batch_size):
        x_train = []
        y_train = []
        d_reward = discount_rewards([t[2] for t in self.memory], self.gamma)
        i = 0
        for state, action, reward, next_state, done in self.memory:
            x_train.append(state)
            y_train.append(1 if action == UP_ACTION else 0)
            i += 1

        self.model.fit(x=np.vstack(x_train), y=np.vstack(y_train), epochs=1, verbose=1, sample_weight=d_reward)
    
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
