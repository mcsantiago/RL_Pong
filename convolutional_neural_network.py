#####################################################################################################################
# Convolutional Neural Network custom impl
#####################################################################################################################


import numpy as np
import sys

class CNN_Impl:
    def __init__(self, input_shape, output_layer_size):
        """
        Initializes the neural network. 
        """
        #np.random.seed(1)
        self.input_shape = input_shape

        # Conv2D(filters=16, kernel_size=8, strides=4, activation='relu', input_shape)
        self.filters1 = np.random.rand(8, 8, 4) # TODO: make 16 of these
        # Conv2D(filters=32, kernel_size=4, strides=4, activation='relu')
        self.filters2 = np.random.rand(4, 4, 4) # TODO: make 32 of these


        # Dense(units=256, activation='relu', kernel_initializer='glorot_uniform')
        # TODO: Figure out below dimensions
        self.X12 = np.zeros((1, 1)) # 
        self.w12 = 2 * np.random.random((18496, 256)) - 1 
        self.delta12 = np.zeros((1, 1))
        self.w23 = 2 * np.random.random((256, 1)) - 1
        self.X23 = np.zeros((1, 1))
        self.delta23 = np.zeros((256, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

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

    def create_feature_maps(self, img, filters):
        """ 
        Returns the feature maps from provided filters and input
        """ 
        # Check if number of image channels matches the filter depth
        if len(img.shape) > 2 or len(filters.shape) > 3: 
            if img.shape[-1] != filters.shape[-1]:
                raise Exception("Number of channels in image do not match filter")

        # Filter must be square
        if filters.shape[1] != filters.shape[2]:
            raise Exception("Filter must be a square matrix")

        feature_maps = numpy.zeros((img.shape[0]-filters.shape[1]+1, 
                                    img.shape[1]-filters.shape[2]+1, 
                                    filters.shape[0]))

        # Convolving the image by the filters
        for i in range(conv_filter.shape[0]):
            current_filter = filters[i, :]
            if len(current_filter.shape) > 2:
                # Feature map array
                conv_map = convolve(img[:, :, 0], current_filter[:, :, 0])

                for ch in range(1, current_filter.shape[-1]): 
                    conv_map = conv_map + convolve(img[:, :, ch],
                                                   current_filter[:, :, ch])
            else:
                conv_map = convolve(img, current_filter)

            feature_maps[:, :, i] = conv_map
        
        return feature_maps

    def convolve(self, img, conv_filter):
        filter_size = conv_filter.shape[1]
        conv_map = np.zeros((img.shape))
        #Looping through the image to apply the convolution operation.
        for r in np.uint16(np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1)):
            for c in np.uint16(np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1)):
                curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                                c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
                #Element-wise multipliplication between the current region and the filter.
                curr_result = curr_region * conv_filter
                conv_sum = np.sum(curr_result) #Summing the result of multiplication.
                conv_map[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
                
        #Clipping the outliers of the result matrix.
        final_result = conv_map[np.uint16(filter_size/2.0):conv_map.shape[0]-np.uint16(filter_size/2.0), 
                            np.uint16(filter_size/2.0):conv_map.shape[1]-np.uint16(filter_size/2.0)]
        return final_result

    def pooling(self, feature_map, size=2, stride=2):
        #Preparing the output of the pooling operation.
        pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride),
                                np.uint16((feature_map.shape[1]-size+1)/stride),
                                feature_map.shape[-1]))
        for map_num in range(feature_map.shape[-1]):
            r2 = 0
            for r in np.arange(0,feature_map.shape[0]-size-1, stride):
                c2 = 0
                for c in np.arange(0, feature_map.shape[1]-size-1, stride):
                    pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                    c2 = c2 + 1
                r2 = r2 +1
        return pool_out

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

    def fit(self, x, y, epochs=1, sample_weight=None):
        ''' 
        TODO: Implement this function.. I think it's a training function of sorts.
        Check Keras docs to be sure of functionality.
        '''

    def predict(self, img):
        if img.shape != self.input_shape:
            print("")
            raise Exception("Image {} does not match expected input shape {}".format(img.shape, self.input_shape))
        return self.forward_pass(img)

    def forward_pass(self, img):
        # pass our inputs through our neural network
        self.l1_feature_map = self.convolve(img, self.filters1)
        self.l1_feature_map_relu = self.__relu(self.l1_feature_map)

        self.l2_feature_map = self.convolve(self.l1_feature_map_relu, self.filters2)
        self.l2_feature_map_relu = self.__relu(self.l2_feature_map)

        self.l2_feature_map_flat = self.l2_feature_map_relu.flatten()
        self.l2_feature_map_flat = np.expand_dims(self.l2_feature_map_flat, axis=1)

        in2 = np.dot(np.transpose(self.l2_feature_map_flat), self.w12)
        self.X23 = self.__relu(in2)

        in3 = np.dot(self.X23, self.w23)
        out = self.__sigmoid(in3)
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


if __name__ == "__main__":
    train_path = 'data/'+sys.argv[1]
    test_path = 'data/'+sys.argv[2]
    activation = sys.argv[3]

    print("Initializing network with data: " + train_path)
    neural_network = NeuralNet(train_path, activation=activation)
    neural_network.train()
    testError = neural_network.predict(test_path)
    print("test error: " + str(testError))

