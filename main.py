import numpy as np
import nnfs
import matplotlib.pyplot as plt
from data import create_data
from nnfs.datasets import spiral_data

nnfs.init()
#np.random.seed(0)
E = 2.71828182846


# Simple Layer; Size of a single input (Num of neurons from previous layer), Number of neurons you want to use;
class Layer_Dense:
    def __init__(self, input_size, num_neurons):
        # Random weights for every connection
        self.weights = 0.01 * np.random.randn(input_size, num_neurons)
        # Array of 0s
        self.biases = np.zeros((1, num_neurons))
    def forward(self, inputs):
        #remember inputs
        self.inputs = inputs
        # Outputs a single value for a single input (single input is [input_size]) per neuron; n values for n neurons; m arrays of n values for n neurons and m size of batch
        # Output = [num_inputs][num_neurons] !!! num_inputs = batch size
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, values):
        # Values = values from ReLU backward
        # Gradients on weights, bias and inputs using chain rule
        # For more than 1 Neuron sum of Gradients is needed, hence np.dot/np.sum
        # single weight: dW0 = x[0] * dReLU
        self.dweights = np.dot(self.inputs.T, values)
        # single bias: dB0 = dReLU
        self.dbiases = np.sum(values, axis=0, keepdims=True)
        # single input: dX0 = w[0] * dReLU
        self.dinputs = np.dot(values, self.weights.T)


# Activation function 0 or X
# Outputs [num_inputs][num_neurons]
class AReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, values):
        self.dinputs = values.copy()
        # 0 Where inputs were < 0
        self.dinputs[self.inputs <= 0] = 0

# Softmax Activation function for output; For classification ~ Convert numbers to % Ex. [4, 2, 2] => [0.5, 0.25, 0.25]
# Outputs [num_inputs][num_neurons]
class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        # e^input for every input; axis=1 => only Rows; keepdim => Keep dimensions
        self.exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # previous values / sum
        self.probabilities = self.exp_vals/np.sum(self.exp_vals, axis=1, keepdims=True)
        self.output = self.probabilities
    def backward(self, values):
        self.dinputs = np.empty_like(values)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, values)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


# Classification Cross Entropy function for loss calculation. Inputs are [[],...,[]] from Softmax, targets are correct objects [] or [[],...,[]];
# One value for N batch size
class Loss:
    def calculate(self, output, targets):
        sample_losses = self.forward(output, targets)
        # Calculate average distance for whole batch
        data_loss = np.mean(sample_losses)
        return data_loss

class Cross_Entropy(Loss):
    # Calculate N values for N batch size; distance between Selected element and Wanted element
    def forward(self, inputs, targets):
        samples = len(inputs)
        # Clip to prevent division by 0
        inputs_clipped = np.clip(inputs, 1e-7, 1 - 1e-7)
        # For list of correct indices Ex. for 3x3 Input [0, 1, 1]
        if len(targets.shape) == 1:
            confidences = inputs_clipped[range(samples), targets]
        # For Full Matrix where 1 represents correct element Ex. for 3x3 [[0,0,1], [1,0,0], [0,0,1]]; correct values stay the same, wrong become 0 because of []*[]
        elif len(targets.shape) == 2:
            confidences = np.sum(inputs_clipped * targets, axis=1)
        # Calculate log of values; Smaller the value bigger the loss.
        self.log_of_max = -np.log(confidences)
        return self.log_of_max

    def backward(self, values, targets):
        lenOfBatch = len(values)
        lenOfValues = len(values[0])
        # If lenOfValues are sparse, turn them into one-hot vector
        if len(targets.shape) == 1:
            targets = np.eye(lenOfValues)[targets]
        self.dinputs = -targets / values
        self.dinputs = self.dinputs / lenOfBatch


# Softmax + CrossEntropy
class Softmax_CrossEntropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Cross_Entropy()

    def forward(self, inputs, targets):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, targets)

    def backward(self, values, targets):
        # Number of samples
        samples = len(values)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)
        # Copy so we can safely modify
        self.dinputs = values.copy()
        # Calculate gradient
        self.dinputs[range(samples), targets] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class SGD_Optimizer:
    # Learning rate decays over time to avoid local minimums
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
    def update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            self.iterations += 1
    def update_layer(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases



