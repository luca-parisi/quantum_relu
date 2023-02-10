"""
Example of a simple MNIST image classifier using the QReLU or m-QReLU activation function in its two
convolutional layers.

Adapted from https://keras.io/examples/vision/mnist_convnet/
"""


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from tf_keras.m_qrelu import MQReLU
from tf_keras.qrelu import QReLU

# Choose either the 'qrelu' or 'm_qrelu' activation function to use in the two
# convolutional layers of a CNN
act_func = 'qrelu'

if act_func == 'qrelu':
    custom_layer = QReLU()
elif act_func == 'm_qrelu':
    custom_layer = MQReLU()

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
inputs_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=inputs_shape),

        # First convolutional layer with the QReLU or m-QReLU activation function
        layers.Conv2D(32, kernel_size=(3, 3)),
        # Instead of layers.Conv2D(64, kernel_size=(3, 3), activation="relu") when using the ReLU activation
        custom_layer,

        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer with the QReLU or m-QReLU activation function
        layers.Conv2D(64, kernel_size=(3, 3)),
        custom_layer,

        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 2

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
