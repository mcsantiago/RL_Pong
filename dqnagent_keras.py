import random
import numpy as np
import tensorflow as tf
from collections import deque
# from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop

# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # if the game ended (in Pong), reset the reward sum
            running_add = 0
        # the point here is to use Horner's method to compute those rewards efficiently
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r)  # idem
    return discounted_r


class DQNAgent:
    ''' Keras implementation of the paper '''

    def __init__(self, state_size, action_size):
        # print(K.tensorflow_backend._get_available_gpus())
        print(tf.config.list_physical_devices('GPU'))
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=160000)

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.000009
        self.epsilon_min = 0.1

        self.learning_rate = 0.0001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=8, activation='relu',
                  kernel_initializer='glorot_uniform', input_shape=(80, 80, 4)))
        model.add(Conv2D(32, kernel_size=4, strides=2,
                  kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu',
                  kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print('{}, action: {}'.format(act_values, np.argmax(act_values[0])))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        # rewards = [x[2] for x in minibatch]
        # d_rewards = discount_rewards(rewards, self.gamma)
        # print(d_rewards)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
                # print('reward {} target {}'.format(reward, target))
                # print(target)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # print(target_f[0])
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
