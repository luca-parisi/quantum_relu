"""
Example of a simple MNIST image classifier using the QReLU or m-QReLU activation function in its two
convolutional layers.

Adapted from https://keras.io/examples/vision/mnist_convnet/
"""

import numpy as np
import tensorflow as tf
from constants import (BATCH_SIZE, DROPOUT, IMAGE_DIM, KERNEL_SIZE_CONV,
                       KERNEL_SIZE_MAX_POOL, NUM_CLASSES, NUM_EPOCHS,
                       OUT_CHANNEL_CONV1, OUT_CHANNEL_CONV2)
from src.constants import USE_M_QRELU
from src.tf_keras.quantum_activations import QuantumReLU
from tensorflow.keras import layers

inputs_shape = (IMAGE_DIM, IMAGE_DIM, 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)


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
        layers.Conv2D(OUT_CHANNEL_CONV1, kernel_size=(
            KERNEL_SIZE_CONV, KERNEL_SIZE_CONV)),
        QuantumReLU(modified=modified),
        layers.MaxPooling2D(pool_size=(
            KERNEL_SIZE_MAX_POOL, KERNEL_SIZE_MAX_POOL)),
        layers.Conv2D(OUT_CHANNEL_CONV2, kernel_size=(
            KERNEL_SIZE_CONV, KERNEL_SIZE_CONV)),
        QuantumReLU(modified=modified),
        layers.MaxPooling2D(pool_size=(
            KERNEL_SIZE_MAX_POOL, KERNEL_SIZE_MAX_POOL)),
        layers.Flatten(),
        layers.Dropout(DROPOUT),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return sequential_model


model = create_model(modified=USE_M_QRELU)
model.summary()

# Train the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_split=0.1)

# Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
