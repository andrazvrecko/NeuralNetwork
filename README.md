# NeuralNetwork

<b>Simple Neural network</b>



Activation functions: ReLU, Softmax

Loss function: Cross Entropy

Optimizer: SGD (vanilla or using momentum)



Use:

```
# Create layer with 2 inputs and 64 neurons
layer1 = Layer_Dense(2,64)
# Create layer with 64 inputs and 3 neurons
layer2 = Layer_Dense(64,3)

# Rectified linear unit activation function
activation1 = AReLU()

# Class using Softmax activation function and cross entropy loss calculation
loss_activation = Softmax_CrossEntropy()
# SGD optimizer (learn_rate=1., decay=0., momentum=0.)
optimizer = SGD_Optimizer(decay=0.001)

# Pass data through first layer
layer1.forward(X)
# Pass output from first layer through ReLU
activation1.forward(layer1.output)
# Pass that through second layer
layer2.forward(activation1.output)

# Pass output from second layer through Softmax activation function and calculate loss based on array of correct values
loss = loss_activation.forward(layer2.output, y)

# Generate gradient for every layer and activation function
loss_activation.backward(loss_activation.output, y)
layer2.backward(loss_activation.dinputs)
activation1.backward(layer2.dinputs)
layer1.backward(activation1.dinputs)

# change learning_rate if using learning_rate decay
optimizer.pre_update_params()
# Optimize weights and biases based on gradient
optimizer.update_layer(layer1)
optimizer.update_layer(layer2)
```
