#####################################################################################################################
# Neural Network custom impl
#####################################################################################################################


import numpy as np
from scipy import signal
import sys
import math


class NeuralNetwork:
    def __init__(self, input_shape, output_layer_size):
        """
        Initializes the neural network. 
        """
        #np.random.seed(1)
        self.input_shape = input_shape

        self.w12 = 2 - np.random.random((6400, 3200)) - 1.5
        self.w23 = 2 - np.random.random((3200, 2048)) - 1.5
        self.w34 = 2 - np.random.random((2048, 1024)) - 1.5
        self.w45 = 2 - np.random.random((1024, 512)) - 1.5
        self.w56 = 2 - np.random.random((512, output_layer_size)) - 1.5


    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __linear(self, x):
        return x

    def __relu(self, x):
        return np.maximum(x, 0)

    def __sigmoid_derivative(self, x):
        return x * (1 - x) # assume that x has already been through sigmoid

    def __tanh_derivative(self, x):
        return 1 - np.power(x, 2) # assume that x has already been through tanh

    def __linear_derivative(self, x):
        return 1

    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x # assumes that x has alreayd been through the relu function

    # Below is the training function
    def fit(self, x, y, clip=None, epochs=1, sample_weight=None, learning_rate = 0.000005):
        if len(x) != len(y): 
            raise Exception('Length of X does not match Y')

        for iteration in range(epochs):
            for i in range(len(x)):
                out = self.forward_pass(x[i])
                error = 0.5 * np.power((out - y[i]), 2)
                # Backpropagation
                if (clip is None):
                    dOut56 = (self.x56 - y[i]) * (self.__linear_derivative(self.x56))
                    dOut45 = dOut56.dot(self.w56.T) * (self.__tanh_derivative(self.x45))
                    dOut34 = dOut45.dot(self.w45.T) * (self.__tanh_derivative(self.x34))
                    dOut23 = dOut34.dot(self.w34.T) * (self.__tanh_derivative(self.x23))
                    dOut12 = dOut23.dot(self.w23.T) * (self.__relu_derivative(self.x12))
                else:
                    dOut56 = np.clip((self.x56 - y[i])      * (self.__linear_derivative(self.x56)), None, clip)
                    dOut45 = np.clip(dOut56.dot(self.w56.T) * (self.__tanh_derivative(self.x45)),   None, clip)
                    dOut34 = np.clip(dOut45.dot(self.w45.T) * (self.__tanh_derivative(self.x34)),   None, clip)
                    dOut23 = np.clip(dOut34.dot(self.w34.T) * (self.__tanh_derivative(self.x23)),   None, clip)
                    dOut12 = np.clip(dOut23.dot(self.w23.T) * (self.__relu_derivative(self.x12)),   None, clip)

                update_layer5 = learning_rate * self.x45.T.dot(dOut56)
                update_layer4 = learning_rate * self.x34.T.dot(dOut45)
                update_layer3 = learning_rate * self.x23.T.dot(dOut34)
                update_layer2 = learning_rate * self.x12.T.dot(dOut23)
                update_layer1 = learning_rate * self.x01.T.dot(dOut12)

                self.w56 -= update_layer5
                self.w45 -= update_layer4
                self.w34 -= update_layer3
                self.w23 -= update_layer2
                self.w12 -= update_layer1

        print("After " + str(epochs) + " iterations, the total error is " + str(np.sum(error)))
        return np.sum(error)

    def predict(self, img):
        if img.shape != self.input_shape:
            print("")
            raise Exception("Image {} does not match expected input shape {}".format(img.shape, self.input_shape))
        return self.forward_pass(img)

    def forward_pass(self, img):
        # pass our inputs through our neural network
        self.x01 = img.T
        self.x12 = self.__relu(np.dot(self.x01, self.w12))
        self.x23 = self.__tanh(np.dot(self.x12, self.w23))
        self.x34 = self.__tanh(np.dot(self.x23, self.w34))
        self.x45 = self.__tanh(np.dot(self.x34, self.w45))
        self.x56 = self.__linear(np.dot(self.x45, self.w56))
        return self.x56

    def compute_output_delta(self, y):
        dx = (self.x23 - y) * (self.__relu_derivative(self.x23))
        return dx

    def compute_hidden_layer1_delta(self, dCost):
        dx = dCost.dot(self.w23.T) * (self.__linear_derivative(self.x12))
        return dx

if __name__ == '__main__':
    model = NeuralNetwork((6400, 1), 7)
    img = np.random.randint(2, size=(6400, 1))
    print(np.sum(img.T))

    model.fit([img], [0.5])
