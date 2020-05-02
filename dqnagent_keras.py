import random
import numpy as np
from collections import deque

# tensorflow stuff
import tensorflow as tf
from tensorflow import keras

from hyperparameters import TRAINING_BN_MOMENTUM, TRAINING_BN_EPSILON

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
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

        self.state_size = state_size
        self.action_size = action_size

        print('action_size: ' + str(self.action_size))

        self.memory = deque(maxlen=80000)

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.000009
        self.epsilon_min = 0.1

        self.learning_rate = 0.0001

        self.model = self._build_model()

    def bottleneck_block(self, k, t, c, n, s, x, padding='same'):
        """ Bottleneck block with:
            k input channels
            t expension
            c output channels
            n levels
            s strides on the first level
            x input layer
        """
        for n0 in range(n):
            residual = keras.layers.BatchNormalization(
                axis=-1, momentum=TRAINING_BN_MOMENTUM, epsilon=TRAINING_BN_EPSILON, center=True, scale=True)(x)
            residual = tf.nn.relu6(residual)
            residual = keras.layers.Conv2D(
                k, 1, strides=1, padding='same', activation=None, use_bias=False)(residual)

            residual = keras.layers.BatchNormalization(
                axis=-1, momentum=TRAINING_BN_MOMENTUM, epsilon=TRAINING_BN_EPSILON, center=True, scale=True)(residual)
            residual = tf.nn.relu6(residual)
            residual = keras.layers.DepthwiseConv2D(
                3, strides=1, padding='same', activation=None, use_bias=False, depth_multiplier=t)(residual)

            residual = keras.layers.BatchNormalization(
                axis=-1, momentum=TRAINING_BN_MOMENTUM, epsilon=TRAINING_BN_EPSILON, center=True, scale=True)(residual)
            residual = tf.nn.relu6(residual)
            residual = keras.layers.Conv2D(
                c, 1, strides=1, padding='same', activation=None, use_bias=False)(residual)

            x = keras.layers.Conv2D(
                c, 1, strides=1, padding='same', activation=None, use_bias=False)(x)
            x = keras.layers.Add()([x, residual])
        return x

    def _build_model(self):
        ''' Builds a MobileNetv2 model '''
        model_input = keras.Input(shape=(80, 80, 4), name='input_image')
        x = model_input

        x = keras.layers.Conv2D(32, 3, strides=2, padding='same',
                                activation=None, use_bias=False)(x)
        x = self.bottleneck_block(32, 1, 16, 1, 1, x)
        x = self.bottleneck_block(16, 6, 24, 2, 2, x)
        # x = self.bottleneck_block(24, 6, 32, 3, 2, x)
        # x = self.bottleneck_block(32, 6, 64, 4, 2, x)
        # x = self.bottleneck_block(64, 6, 96, 3, 1, x)
        # x = self.bottleneck_block(96, 6, 160, 3, 2, x)
        # x = self.bottleneck_block(96, 6, 320, 1, 1, x)

        x = keras.layers.Conv2D(
            256, 1, strides=1, padding='valid', activation='relu', use_bias=False)(x)

        # encoder - output
        encoder_output = x

        # decoder
        y = keras.layers.GlobalAveragePooling2D()(encoder_output)
        decoder_output = keras.layers.Dense(
            self.action_size, activation='softmax')(y)

        # forward path
        model = keras.Model(inputs=model_input,
                            outputs=decoder_output, name='mobilenetv2')

        # loss, backward path (implicit) and weight update
        model.compile(optimizer=tf.keras.optimizers.Adam(
            self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        # return model
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

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
                print('reward {} target {} action {}'.format(
                    reward, target, action))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, batch_size=32, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
