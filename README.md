# NeuralNetwork

<b>Simple Neural network</b>

Neural network supports Classification and Regression.



Activation functions: ReLU, Softmax, Sigmoid (Regression), Linear (Regression)
Loss function: Cross Entropy, Binary Cross Entropy, Mean Squared Loss (Regression), Mean Absolute Loss (Regression)
Optimizer: SGD (vanilla or using momentum), Adam
Regulizers: L1&L2, dropout



Use:

```
# Create Model
model = main.Model()

# Add layers to the model
model.add(main.Layer_Dense(2, 512))
model.add(main.Activation_ReLU())
model.add(main.Layer_Dense(512, 3))
model.add(main.Activation_Softmax())

# Set Loss function, Optimizer and Accuracy class (Accuracy_Regression for regression)
model.set(
    loss=main.Cross_Entropy(),
    optimizer=main.Adam_Optimizer(),
    accuracy=main.Accuracy_Classification()
)

# Prepare model for training
model.connectLayers()

# Train model
# X = data, y = actual results, print_every = Print Accuracy, Loss, ... every z epochs
model.train(X, y, epochs=10000, print_every=100)
```


Change learning rate and decay in Optimizer
```
optimizer=main.Adam_Optimizer(learning_rate=0.05, decay=5e-5)
```


To enable L1&L2 regularization
```
model.add(main.Layer_Dense(2, 512, weight_l2_lambda=5e-4, bias_l2_lambda=5e-4))
```
To enable Dropout
```
... Activation Layer
model.add(main.Layer_Dropout(0.1))
... Next Dense Layer
```
Inspired by https://nnfs.io/
