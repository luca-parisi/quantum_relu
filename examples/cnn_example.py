"""
Example of a simple MNIST image classifier using the QReLU or m-QReLU activation function in its two
convolutional layers.

Adapted from https://keras.io/examples/vision/mnist_convnet/
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.tf_keras.constants import USE_M_QRELU
from src.tf_keras.quantum_activations import QuantumReLU

num_classes = 10
inputs_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def create_model(modified: bool = False) -> tf.keras.Model:
    """
    Create a convolutional neural network model with the QuantumReLU activation.

    Args:
        modified: bool, optional
            Whether to use the modified version of the QReLU or m-QReLU (False by default, i.e., using the QReLU).

    Returns:
        tf.keras.Model
            The constructed Keras model with QuantumReLU activations.
    """

    sequential_model = tf.keras.Sequential([
        tf.keras.Input(shape=inputs_shape),
        layers.Conv2D(32, kernel_size=(3, 3)),
        QuantumReLU(modified=modified),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3)),
        QuantumReLU(modified=modified),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return sequential_model


model = create_model(modified=USE_M_QRELU)
model.summary()

# Train the model
batch_size = 128
epochs = 2
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
