#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import sys

class CNN_Impl:
    def __init__(self, train, header = None, containsHeader = False, h1 = 4, h2 = 4, activation="sigmoid"):
        """
        Initializes the neural network. 
        """
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers


        self.activation = activation
        ''' TODO: Rewrite constructor to pass in expected dimensions as tuple '''
        #raw_input = pd.read_csv(train, header=header)
        #train_dataset = self.preprocess(raw_input)
        #ncols = len(train_dataset.columns)
        #nrows = len(train_dataset.index)
        #self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        #self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        ''' TODO: Convert this code into the neural network proposed in dqnagent_keras.py ''' 
        ''' TODO: Something to think about, what if we wanted to make a larger network? Do we keep hardcoding, or make it dynamic? '''
        #self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        #self.X01 = self.X
        #self.delta01 = np.zeros((input_layer_size, h1))
        #self.w12 = 2 * np.random.random((h1, h2)) - 1
        #self.X12 = np.zeros((len(self.X), h1))
        #self.delta12 = np.zeros((h1, h2))
        #self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        #self.X23 = np.zeros((len(self.X), h2))
        #self.delta23 = np.zeros((h2, output_layer_size))
        #self.deltaOut = np.zeros((output_layer_size, 1))

    def __activation(self, x):
        if self.activation.lower() == "sigmoid":
            return self.__sigmoid(x)
        elif self.activation.lower() == "relu":
            return self.__relu(x)
        elif self.activation.lower() == "tanh":
            return self.__tanh(x)
        else:
            raise Exception("Unknown activation function: " + activation)

    def __activation_derivative(self, x):
        if self.activation.lower() == "sigmoid":
            return self.__sigmoid_derivative(x)
        elif self.activation.lower() == "relu":
            return self.__relu_derivative(x)
        elif self.activation.lower() == "tanh":
            return self.__tanh_derivative(x)
        else:
            raise Exception("Unknown activation function: " + activation)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __relu(self, x):
        return np.maximum(x, 0)

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x) # assumes that x has already been through the sigmoid function

    def __relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x # assumes that x has alreayd been through the relu function
    
    def __tanh_derivative(self, x):
        return 1 - np.power(x, 2) # assumes that x has already been through the tanh function

    # Below is the training function
    def train(self, max_iterations = 100000, learning_rate = 0.005):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def fit(self, state, target_f, epochs=1, verbose=0):
        ''' 
        TODO: Implement this function.. I think it's a training function of sorts.
        Check Keras docs to be sure of functionality.
        '''


    def forward_pass(self):
        ''' 
        TODO: Current implementation applies activation function across all layers.. 
        But in keras implementation, some layers use different activations 
        '''
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01 )
        self.X12 = self.__activation(in1)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3)
        return out

    def backward_pass(self, out):
        # pass our inputs through our neural network
        self.compute_output_delta(out)
        self.compute_hidden_layer2_delta()
        self.compute_hidden_layer1_delta()

    def compute_output_delta(self, out):
        ''' 
        TODO: Current implementation applies activation function across all layers.. 
        But in keras implementation, some layers use different activations 
        '''
        delta_output = (self.y - out) * (self.__activation_derivative(out))
        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self):
        ''' 
        TODO: Current implementation applies activation function across all layers.. 
        But in keras implementation, some layers use different activations 
        '''
        delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__activation_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self):
        ''' 
        TODO: Current implementation applies activation function across all layers.. 
        But in keras implementation, some layers use different activations 
        '''
        delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__activation_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    def predict(self, test, header = None):
        ''' TODO: Forward pass through the network and return output vector '''


if __name__ == "__main__":
    train_path = 'data/'+sys.argv[1]
    test_path = 'data/'+sys.argv[2]
    activation = sys.argv[3]

    print("Initializing network with data: " + train_path)
    neural_network = NeuralNet(train_path, activation=activation)
    neural_network.train()
    testError = neural_network.predict(test_path)
    print("test error: " + str(testError))

