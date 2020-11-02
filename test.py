import main
import numpy as np
import nnfs
import os
import cv2
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data, sine_data

nnfs.init()

data = main.Data()
BATCH_SIZE = 128
# Create dataset

#X, y, X_test, y_test = data.create_data_mnist('fashion_mnist_images')

#X = data.scale_data_mnist(X)
#X = data.flatten_data(X)
#X_test = data.scale_data_mnist(X_test)
#X_test = data.flatten_data(X_test)
#X, y = data.reshuffle_data(X, y)

image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28,28))
image_data = 255 - image_data
plt.imshow(image_data, cmap='gray')
plt.show()
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

model = main.Model()
model.add(main.Layer_Dense(28, 128))
model.add(main.Activation_ReLU())
model.add(main.Layer_Dense(128, 128))
model.add(main.Activation_ReLU())
model.add(main.Layer_Dense(128, 10))
model.add(main.Activation_Softmax())
model.set(
    loss=main.Cross_Entropy(),
    optimizer=main.Adam_Optimizer(decay=1e-3),
    accuracy=main.Accuracy_Classification()
)
model.connectLayers()
model.load_parameters('fashion_mnist.params')


# Train the model
#model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
#model.evaluate(X_test, y_test)


model2 = main.Model.load('fashion_mnist.model')

res = model2.predict(image_data)
pred = model2.output_activation.predictions(res)
print(pred)

