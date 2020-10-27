import main
import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

layer1 = main.Layer_Dense(2,64)
activation1 = main.AReLU()
layer2 = main.Layer_Dense(64,3)
loss_activation = main.Softmax_CrossEntropy()
optimizer = main.Adam_Optimizer(learning_rate=0.02, decay=0.00001)

#FORWARD
for epoch in range(10001):

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # BACKWARD
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.update_params()
    optimizer.update_layer(layer1)
    optimizer.update_layer(layer2)
    optimizer.update_iteration()


#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()