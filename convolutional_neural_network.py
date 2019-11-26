#####################################################################################################################
# Convolutional Neural Network custom impl
#####################################################################################################################


import numpy as np
from scipy import signal
import sys
import math

def convolve(input, filter, strides=1):
    """ Convolution operation """
    if input.shape[2] != filter.shape[2]:
        raise Exception("Input and filter must match dimensionality")
    if input.shape[0] < filter.shape[0] or input.shape[1] < filter.shape[1]:
        raise Exception("Filter is larger than input")

    feature_map = []
    y = 0
    while y < input.shape[1] - filter.shape[1]:
        row = []
        x = 0
        while x < input.shape[0] - filter.shape[0]:
            sum = np.sum(np.multiply(input[x:x+filter.shape[0], y:y+filter.shape[1]], filter))
            row.append(sum)
            x += strides
        feature_map.append(row)
        y += strides

    feature_map = np.matrix(feature_map)
    feature_map = np.expand_dims(feature_map, axis=2)
    return feature_map 

class CNN_Impl:
    def __init__(self, input_shape, output_layer_size):
        """
        Initializes the neural network. 
        """
        #np.random.seed(1)
        self.input_shape = input_shape

        # Conv2D(filters=16, kernel_size=8, strides=4, activation='relu', input_shape)
        self.filters12 = []
        for i in range(16):
            self.filters12.append(np.random.rand(8, 8, 4))
        self.filters12 = np.array(self.filters12)

        # Conv2D(filters=32, kernel_size=4, strides=4, activation='relu')
        self.filters23 = []
        for i in range(32):
            self.filters23.append(np.random.rand(4, 4, 1))
        self.filters23 = np.array(self.filters23)

        # Dense(units=256, activation='relu', kernel_initializer='glorot_uniform')
        self.w34 = 2 * np.random.random((25088, 256)) - 1 
        self.w45 = 2 * np.random.random((256, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __relu(self, x):
        return np.maximum(x, 0)

    def __sigmoid_derivative(self, x):
        return x * (1 - x) # assumes that x has already been through the sigmoid function

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
                delta5 = self.compute_output_delta(y[i], out)
                print(delta5)
                delta45 = self.compute_hidden_layer4_delta(delta5)
                delta34 = self.compute_hidden_layer3_delta(delta45)

                delta34_matrix = np.reshape(delta34, (512, 7, 7, 1))
                delta23 = self.compute_hidden_layer2_delta(delta34_matrix)
                # delta12 = self.compute_hidden_layer1_delta(delta23)

                # update_layer4 = learning_rate * self.X45.T.dot(delta5) # (256, 1) x (1, 1)
                # update_layer3 = learning_rate * self.l2_feature_map_flat.T.dot(delta45) # (4900, 1) x (1, 256)
                # update_layer2 = learning_rate * self.l2_feature_map_relu.T.dot(delta34)
                # update_layer1 = learning_rate * self.l1_feature_map_relu.T.dot(delta12)

                # self.w45 += update_layer4
                # self.w34 += update_layer3
                # self.filters23 += update_layer2
                # self.filters12 += update_layer1

        print("After " + str(epochs) + " iterations, the total error is " + str(np.sum(error)))

    def predict(self, img):
        if img.shape != self.input_shape:
            print("")
            raise Exception("Image {} does not match expected input shape {}".format(img.shape, self.input_shape))
        return self.forward_pass(img)

    def forward_pass(self, img):
        # pass our inputs through our neural network
        self.l1_feature_map = []
        for filter in self.filters12: 
            self.l1_feature_map.append(convolve(img, filter, strides=4))
        self.l1_feature_map_relu = self.__relu(self.l1_feature_map)
        self.l1_feature_map_relu = self.l1_feature_map_relu # Transpose (1, 18, 18, 16)

        self.l2_feature_map = []
        for filter in self.filters23:
            for feature_map in self.l1_feature_map_relu:
                self.l2_feature_map.append(convolve(feature_map, filter, strides=2))
        self.l2_feature_map_relu = self.__relu(self.l2_feature_map)
        self.l2_feature_map_relu = self.l2_feature_map_relu # (1, 7, 7, 512)

        self.l2_feature_map_flat = self.l2_feature_map_relu.flatten() 
        self.l2_feature_map_flat = np.expand_dims(self.l2_feature_map_flat, axis=1).T # (1, 25088)

        in2 = np.dot(self.l2_feature_map_flat, self.w34) # (1, 25088) x (25088, 256)
        self.X45 = self.__relu(in2) # (1, 256)

        in3 = np.dot(self.X45, self.w45) # (1, 256) x (256, 1)
        out = self.__sigmoid(in3) # (1, 1)
        return out

    def compute_output_delta(self, y, out):
        return (y - out) * (self.__sigmoid_derivative(out))

    def compute_hidden_layer4_delta(self, delta5):
        return (delta5.dot(self.w45.T)) * (self.__relu_derivative(self.X45))

    def compute_hidden_layer3_delta(self, delta45):
        return (delta45.dot(self.w34.T)) * (self.__relu_derivative(self.l2_feature_map_flat))

    def compute_hidden_layer2_delta(self, delta34):
        print('delta34 shape: {}'.format(delta34.shape)) # (512, 7, 7, 1)
        print('filters23 shape: {}'.format(self.filters23.shape)) # (32, 4, 4, 1)
        dw = [] # expecting (32, 4, 4, 1)
        dx = [] # expecting (16, 18, 18, 1)

        return dw, dx
            
        # return signal.convolve(np.rot90(np.rot90(delta34)), self.filters23, 'valid')

    def compute_hidden_layer1_delta(self, delta23):
        print('delta23 shape: {}'.format(delta23.shape))
        print('filters12 shape: {}'.format(self.filters12.shape))
        # return signal.convolve(np.rot90(np.rot90(delta23)), self.filters12, 'valid')

if __name__ == '__main__':
    model = CNN_Impl((80, 80, 4), 1)
    img = np.random.rand(80, 80, 4)
    out = model.predict(img)

    model.fit([img], [0.5])

    print('output {}'.format(out))