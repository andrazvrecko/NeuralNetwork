import numpy as np

E = 2.71828182846


# Simple Layer; Size of a single input (Num of neurons from previous layer), Number of neurons you want to use;
class Layer_Dense:
    def __init__(self, input_size, num_neurons, weight_l1_lambda=0, weight_l2_lambda=0, bias_l1_lambda=0, bias_l2_lambda=0):
        # Random weights for every connection
        self.weights = 0.01 * np.random.randn(input_size, num_neurons)
        # Array of 0s
        self.biases = np.zeros((1, num_neurons))
        # L1 L2 Reqularization
        self.weight_l1 = weight_l1_lambda
        self.weight_l2 = weight_l2_lambda
        self.bias_l1 = bias_l1_lambda
        self.bias_l2 = bias_l2_lambda
    def forward(self, inputs):
        #remember inputs
        self.inputs = inputs
        # Outputs a single value for a single input (single input is [input_size]) per neuron; n values for n neurons; m arrays of n values for n neurons and m size of batch
        # Output = [num_inputs][num_neurons] !!! num_inputs = batch size
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, values):
        # Values = derivatives from ReLU or Softmax

        # For more than 1 Neuron sum of Gradients is needed, hence np.dot/np.sum
        # single weight: dW0 = x[0] * dReLU
        self.dweights = np.dot(self.inputs.T, values)
        # single bias: dB0 = dReLU
        self.dbiases = np.sum(values, axis=0, keepdims=True)
        # single input: dX0 = w[0] * dReLU
        self.dinputs = np.dot(values, self.weights.T)

        # L1 L2
        # Derivative of L1
        if self.weight_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_l1 * dL1
        # Derivative of L2
        if self.weight_l2 > 0:
            self.dweights += 2 * self.weight_l2 * self.weights
        if self.bias_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_l1 * dL1
        if self.bias_l2 > 0:
            self.dbiases += 2 * self.bias_l2 * self.biases


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
    def reqularize_loss(self, layer):
        regularization_loss = 0
        if layer.weight_l1 > 0:
            regularization_loss += layer.weight_l1* np.sum(np.abs(layer.weights))

        if layer.bias_l1 > 0:
            regularization_loss += layer.bias_l1* np.sum(np.abs(layer.biases))

        if layer.weight_l2 > 0:
            regularization_loss += layer.weight_l2* np.sum(layer.weights * layer.weights)

        if layer.bias_l2 > 0:
            regularization_loss += layer.bias_l2* np.sum(layer.biases * layer.biases)
        return regularization_loss

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

# Every output is binary (example: human or not human,...)
class Binary_Cross_Entropy(Loss):
    def forward(self, inputs, targets):
        inputs_clipped = np.clip(inputs, 1e-7, 1 - 1e-7)
        losses = -(targets * np.log(inputs_clipped) + (1 - targets) * np.log(1 - inputs_clipped))
        losses = np.mean(losses, axis=-1)
        return losses
    def backward(self, values, targets):
        lenOfBatch = len(values)
        lenOfValues = len(values[0])
        clipped_values = np.clip(values, 1e-7, 1 - 1e-7)

        self.dinputs = -(targets / clipped_values - (1 - targets) / (1 - clipped_values)) / lenOfValues
        # Normalize gradient
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
        # Calculate gradient; value - 1 where correct answer lies.
        self.dinputs[range(samples), targets] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

    def reqularize_loss(self, layer):
        regularization_loss = 0
        if layer.weight_l1 > 0:
            regularization_loss += layer.weight_l1* np.sum(np.abs(layer.weights))

        if layer.bias_l1 > 0:
            regularization_loss += layer.bias_l1* np.sum(np.abs(layer.biases))

        if layer.weight_l2 > 0:
            regularization_loss += layer.weight_l2* np.sum(layer.weights * layer.weights)

        if layer.bias_l2 > 0:
            regularization_loss += layer.bias_l2* np.sum(layer.biases * layer.biases)
        return regularization_loss

# TODO AdaGrad, RMSProp
class SGD_Optimizer:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0
    # if using decay, decay the learning rate based on iteration, to avoid local minimums
    def update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_layer(self, layer):
        # if we use momentum
        if self.momentum:
            # init weight momentums
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            # calculate updates from previous values; momentum ~ moving average from previous steps (1/2 step-1 + 1/4 step - 2 + 1/8 step - 3 + ....)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            # set new values
            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        # Vanilla SGD
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        # set new params
        layer.weights += weight_updates
        layer.biases += bias_updates

    def update_iteration(self):
        self.iterations += 1

class Adam_Optimizer:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    def update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            #self.iterations += 1
    def update_layer(self, layer):
        # init cache
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # update momentum with current gradients
        # Adaptive gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Update cache with squared current gradients, momentum for second order derivative
        # RMSProp
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Bias adjustment
        weight_momentums_adjusted = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_adjusted = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        weight_cache_adjusted = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_adjusted = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_adjusted / (np.sqrt(weight_cache_adjusted) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_adjusted / (np.sqrt(bias_cache_adjusted) + self.epsilon)

    def update_iteration(self):
        self.iterations += 1

# Dropout: rate - % of neurons you want to disable
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Generate array of 1s and 0s
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask
        self.output = inputs * self.binary_mask
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Sigmoid_Activation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, values):
        self.dinputs = values * (1 - self.output) * self.output