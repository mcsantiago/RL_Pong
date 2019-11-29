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

        self.w12 = 2 - np.random.random((6400, 256))
        self.w23 = 2 - np.random.random((256, output_layer_size))


    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __linear(self, x):
        return x

    def __relu(self, x):
        return np.maximum(x, 0)

    def __sigmoid_derivative(self, x):
        return x * (1 - x) # assume that x has already been through sigmoid

    def __linear_derivative(self, x):
        return 1

    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x # assumes that x has alreayd been through the relu function

    # Below is the training function
    def fit(self, x, y, epochs=1, sample_weight=None, learning_rate = 0.005):
        if len(x) != len(y): 
            raise Exception('Length of X does not match Y')

        for iteration in range(epochs):
            for i in range(len(x)):
                out = self.forward_pass(x[i])
                error = 0.5 * np.power((out - y[i]), 2)

                # Backpropagation
                dOut23 = self.compute_output_delta(y[i])
                dOut12 = self.compute_hidden_layer1_delta(dOut23)

                update_layer2 = learning_rate * self.x12.T.dot(dOut23)
                update_layer1 = learning_rate * self.x01.dot(dOut12)

                self.w23 -= update_layer2
                self.w12 -= update_layer1

        print("After " + str(epochs) + " iterations, the total error is " + str(np.sum(error)))
        return error

    def predict(self, img):
        if img.shape != self.input_shape:
            print("")
            raise Exception("Image {} does not match expected input shape {}".format(img.shape, self.input_shape))
        return self.forward_pass(img)

    def forward_pass(self, img):
        # pass our inputs through our neural network
        self.x01 = img
        self.x12 = self.__relu(np.dot(self.x01.T, self.w12))
        self.x23 = self.__sigmoid(np.dot(self.x12, self.w23))
        return self.x23

    def compute_output_delta(self, y):
        dx = (self.x23 - y) * (self.__sigmoid_derivative(self.x23))
        return dx

    def compute_hidden_layer1_delta(self, dCost):
        dx = dCost.dot(self.w23.T) * (self.__relu_derivative(self.x12))
        return dx

if __name__ == '__main__':
    model = NeuralNetwork((6400, 1), 7)
    img = np.random.rand(6400, 1)
    out = model.predict(img)

    model.fit([img], [0.5])

    print('output {}'.format(out))