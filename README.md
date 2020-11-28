# Neural Network Framework

<b>This project helped me learn and understand how Neural Networks work. Inspired by https://nnfs.io/ and other online resources.</b>


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
To enable Dropout add it after activation layer and in front of next dense layer
```
... Activation Layer
model.add(main.Layer_Dropout(0.1))
... Next Dense Layer
```


To save/load model
```

# Whole Model

model.save('path')
# Call after model.connectLayers()
model = main.Model.load('path')

# Weights and Biases

model.save_parameters('path')
# Call after model.connectLayers()
model.load_parameters('path')
```

To predict
```
# Use parameter batch_size for custom sized batches
res = model.predict(X)
# Convert into human readable results based on Final Activation Function
prediction = model.output_activation.predictions(res)
print(prediction)
```



