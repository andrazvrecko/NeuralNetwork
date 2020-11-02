import numpy as np
import cv2
import os
import pickle
import copy

E = 2.71828182846

# LAYERS

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
    def forward(self, inputs, training):
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
    def get_parameters(self):
        return self.weights, self.biases
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout: rate - % of neurons you want to disable
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        # Generate array of 1s and 0s
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask
        self.output = inputs * self.binary_mask
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

# ACTIVATION FUNCTIONS

# Activation function 0 or X
# Outputs [num_inputs][num_neurons]
class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, values):
        self.dinputs = values.copy()
        # 0 Where inputs were < 0
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self, outputs):
        return outputs

# Softmax Activation function for output; For classification ~ Convert numbers to % Ex. [4, 2, 2] => [0.5, 0.25, 0.25]
# Outputs [num_inputs][num_neurons]
class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        # e^input for every input; axis=1 => only Rows; keepdim => Keep dimensions
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # previous values / sum
        probabilities = exp_vals/np.sum(exp_vals, axis=1, keepdims=True)
        self.output = probabilities
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
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Sigmoid_Activation:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, values):
        self.dinputs = values * (1 - self.output) * self.output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


# Regression
class Linear_Activation:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
    def backward(self, values):
        self.dinputs = values.copy()
    def predictions(self, outputs):
        return outputs

# LOSS

# Classification Cross Entropy function for loss calculation. Inputs are [[],...,[]] from Softmax, targets are correct objects [] or [[],...,[]];
# One value for N batch size
class Loss:
    def calculate(self, output, targets, *, include_regularization=False):
        sample_losses = self.forward(output, targets)
        # Calculate average distance for whole batch
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        return data_loss, self.regularize_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularize_loss()

    def reset_accumulated(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularize_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
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
        self.activation = Activation_Softmax()
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
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Mean_Squared_Loss(Loss):
    def forward(self, inputs, targets):
        # average of (Point - Predicted point) squared
        loss = np.mean((targets - inputs)**2, axis=-1)
        return loss
    def backward(self, values, targets):
        samples = len(values)
        outputs = len(values[0])
        self.dinputs = -2 * (targets - values) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    def predictions(self, outputs):
        return outputs

class Mean_Absolute_Loss(Loss):
    def forward(self, inputs, targets):
        loss = np.mean(np.abs(targets - inputs), axis=-1)
        return loss
    def backward(self, values, targets):
        samples = len(values)
        outputs = len(values[0])
        self.dinputs = np.sign(targets - values) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# OPTIMIZERS

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

# ACCURACY

class Accuracy:
    def calculate(self, values, targets):
        comparisons = self.compare(values, targets)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        accuracy = np.mean(comparisons)
        return accuracy

    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def reset_accumulated(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, targets, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(targets) / 250
    def compare(self, values, targets):
        return np.absolute(values - targets) < self.precision

class Accuracy_Classification(Accuracy):
    def init(self, y):
        pass
    def compare(self, values, targets):
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)
        return values == targets


# MODEL #

class Model:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, X, y, *, batch_size=None ,epochs=1, print_every=1, validation_data=None):
        steps = 1
        self.accuracy.init(y)
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data


        if batch_size is not None:
            steps = len(X) //batch_size
            if steps * batch_size < X.shape[0]:
                steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        self.loss.reset_accumulated()
        self.accuracy.reset_accumulated()

        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')
            self.loss.reset_accumulated()
            self.accuracy.reset_accumulated()

            for step in range(steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_X, training=True)

                dataloss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = dataloss + regularization_loss
                predictions = self.output_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)
                self.optimizer.update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_layer(layer)
                self.optimizer.update_iteration()

                if not step % print_every or step == steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {dataloss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                   f'acc: {epoch_accuracy:.3f}, ' +
                   f'loss: {epoch_loss:.3f} (' +
                   f'data_loss: {epoch_data_loss:.3f}, ' +
                   f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                   f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.loss.reset_accumulated()
                self.accuracy.reset_accumulated()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[ step * batch_size:(step + 1) * batch_size]
                        batch_y = y_val[ step * batch_size:(step + 1) * batch_size]
                    output = self.forward(batch_X, training=False)
                    self.loss.calculate(output, batch_y)
                    predictions = self.output_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

    # Create references for prev and next layer. Used for forward/backward
    def connectLayers(self):

        self.input_layer = Layer_Input()
        count = len(self.layers)
        self.trainable_layers = []

        for i in range(count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_activation = self.layers[i]
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Cross_Entropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Softmax_CrossEntropy()
    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, values, targets):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(values, targets)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        self.loss.backward(values, targets)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        self.loss.reset_accumulated()
        self.accuracy.reset_accumulated()
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)

            self.loss.calculate(output, batch_y)

            predictions = self.output_activation.predictions(
                output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        output = []
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
            output = np.vstack(output)
            return output

    def get_parameters(self):
        params=[]
        for layer in self.trainable_layers:
            params.append(layer.get_parameters())
        return params

    def set_parameters(self, params):
        for param_layer, layer in zip(params, self.trainable_layers):
            layer.set_parameters(*param_layer)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.reset_accumulated()
        model.accuracy.reset_accumulated()
        # Remove properties
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

class Data:
    def load_mnist_dataset(self, dataset, path):
        labels = os.listdir(os.path.join(path, dataset))
        X = []
        y = []

        for label in labels:
            for file in os.listdir(os.path.join(path, dataset, label)):
                image = cv2.imread(os.path.join(
                    path, dataset, label, file
                ), cv2.IMREAD_UNCHANGED)

                X.append(image)
                y.append(label)

        # Convert the data to proper numpy arrays and return
        return np.array(X), np.array(y).astype('uint8')
    def create_data_mnist(self, path):
        X, y = self.load_mnist_dataset('train', path)
        X_test, y_test = self.load_mnist_dataset('test', path)

        # And return all the data
        return X, y, X_test, y_test
    def scale_data_mnist(self, X,):
        X = (X.astype(np.float32) - 127.5) / 127.5
        return X
    def flatten_data(self, X):
        X = X.reshape(X.shape[0], -1)
        return X
    def reshuffle_data(self, X, y):
        keys = np.array(range(X.shape[0]))
        np.random.shuffle(keys)
        X = X[keys]
        y = y[keys]
        return X,y
