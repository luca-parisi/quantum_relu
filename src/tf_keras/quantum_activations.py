"""Additional utility for Deep Learning models in TensorFlow and Keras"""

# The Quantum ReLU (QReLU) and the modified QReLU (m-QReLU) as custom activation functions in
# TensorFlow and Keras

# Author: Luca Parisi <luca.parisi@ieee.org>

import numpy as np
import tensorflow as tf
from src.constants import (FIRST_COEFFICIENT, M_QRELU_NAME, QRELU_NAME,
                           SECOND_COEFFICIENT_M_QRELU,
                           SECOND_COEFFICIENT_QRELU, USE_M_QRELU)
from src.tf_keras.utils import py_func
from tensorflow import Tensor
from tensorflow.keras.layers import Layer
from tensorflow.lite.python.op_hint import _LiteFuncCall

from .constants import DERIV_M_QRELU_NAME, DERIV_QRELU_NAME

# QReLU and m-QReLU as custom activation functions in TensorFlow

'''
# Example of usage of the QReLU or m-QReLU in TensorFlow as a custom activation function 
of a convolutional layer (#2).

convolutional_layer_2 = tf.layers.conv2d(
                        inputs=pooling_layer_1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding="same")
convolutional_layer_activation = tf_quantum_relu(convolutional_layer_2, False)  # True if using the m-QReLU (instead of the QReLU)
pooling_layer_2 = tf.layers.max_pooling2d(inputs=convolutional_layer_activation, pool_size=[2, 2], strides=2)
'''


def quantum_relu(x: float, modified: bool = USE_M_QRELU) -> float:
    """
    Apply the QReLU activation function or its modified version (m-QReLU) to transform inputs accordingly.

    Args:
        x: float
            The input to be transformed via the m-QReLU activation function.
        modified: bool
            Whether using the modified version of the QReLU or m-QReLU (no/False by default, i.e.,
            using the QReLU by default).

    Returns:
            The transformed x (float) via the QReLU or m-QReLU.
    """

    if x <= 0:
        second_coefficient = SECOND_COEFFICIENT_M_QRELU if modified else SECOND_COEFFICIENT_QRELU
        x = FIRST_COEFFICIENT * x - second_coefficient * x

    return x


# Vectorising the 'quantum_relu' function
np_quantum_relu = np.vectorize(quantum_relu)


def derivative_quantum_relu(x: float, modified: bool = USE_M_QRELU) -> float:
    """
    Compute the derivative of the QReLU activation function or its modified version (m-QReLU).

    Args:
        x: float
            The input from which the derivative is to be computed.
        modified: bool
            Whether using the modified version of the QReLU or m-QReLU (no/False by default, i.e.,
            using the QReLU by default).

    Returns:
        The derivative (float) of the QReLU or m-QReLU given an input.
    """

    if x <= 0:
        second_coefficient = SECOND_COEFFICIENT_M_QRELU if modified else SECOND_COEFFICIENT_QRELU
        x = FIRST_COEFFICIENT - second_coefficient

    return x


# Vectorising the derivative of the QReLU function
np_der_quantum_relu = np.vectorize(derivative_quantum_relu)


def quantum_relu_grad(op: _LiteFuncCall, grad: float, modified: bool = USE_M_QRELU) -> float:  # pragma: no cover
    """
    Define the gradient function of the QReLU or m-QReLU.

    Args:
        op: _LiteFuncCall
            A TensorFlow Lite custom function.
        grad:
            The input gradient.
        modified: bool
            Whether using the modified version of the QReLU or m-QReLU (no/False by default, i.e., using
            the QReLU by default).

    Returns:
            The gradient function of the QReLU or m-QReLU.
    """

    x = op.inputs[0]
    n_gr = tf_der_quantum_relu(x, modified)
    return grad * n_gr


def np_quantum_relu_float32(x: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Apply the QuantumReLU or modified QuantumReLU activation function to a NumPy array and return the result
    as a float32 NumPy array.

    Args:
        x: np.ndarray
            The input NumPy array to be transformed via the QuantumReLU or modified QuantumReLU activation function.

    Returns:
        np.ndarray
            The transformed NumPy array after applying the QuantumReLU or modified QuantumReLU activation function.
            The output array will have data type float32.
    """
    return np_quantum_relu(x).astype(np.float32)


def tf_quantum_relu(x: Tensor, modified: bool = USE_M_QRELU) -> Tensor:  # pragma: no cover
    """
    The QReLU activation function defined in TensorFlow.

    Args:
        x: Tensor
            The input tensor.
        modified: bool
            Whether using the modified version of the QReLU or m-QReLU (no/False by default, i.e., using
            the QReLU by default).

    Returns:
            The output tensor (Tensor) from the QReLU activation function.
    """

    name = QRELU_NAME
    if modified:
        name = M_QRELU_NAME

    y = py_func(
        np_quantum_relu_float32,  # Forward pass function
        [x],
        [tf.float32],
        name=name,
        grad=quantum_relu_grad
    )  # The function that overrides gradient
    y[0].set_shape(x.get_shape())  # To specify the rank of the input
    return y[0]


def np_der_quantum_relu_float32(x: np.ndarray) -> np.ndarray:  # pragma: no cover
    """
    Calculate the derivative of the QuantumReLU or modified QuantumReLU activation function for a NumPy array
    and return the result as a float32 NumPy array.

    Args:
        x: np.ndarray
            The input NumPy array for which the derivative is to be computed.

    Returns:
        np.ndarray
            The calculated derivative of the QuantumReLU or modified QuantumReLU activation function for the input
            array. The output array will have data type float32.
    """
    return np_der_quantum_relu(x).astype(np.float32)


def tf_der_quantum_relu(x: list[Tensor], modified: bool = USE_M_QRELU) -> float:  # pragma: no cover
    """
    The derivative of the QReLU or m-QReLU defined in TensorFlow.

    Args:
        x: list[Tensor]
            A list of input tensors.
        modified: bool
            Whether using the modified version of the QReLU or m-QReLU (no/False by default, i.e., using
            the QReLU by default).

    Returns:
            The output computed as the derivative of the QReLU activation function.
    """

    name = DERIV_QRELU_NAME
    if modified:
        name = DERIV_M_QRELU_NAME

    y = tf.py_func(
        np_der_quantum_relu_float32,
        [x],
        [tf.float32],
        name=name,
        stateful=False
    )
    return y[0]


# QuantumReLU as a custom layer in Keras

'''
# Example of usage of the QuantumReLU as a Keras layer in a sequential model between a convolutional layer and a pooling 
layer.

Either

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(QuantumReLU(modified=False))  # True if using the m-QReLU (instead of the QReLU)
model.add(layers.MaxPooling2D((2, 2)))

or

model = keras.Sequential(
        keras.Input(shape=(32, 32, 3)),

        # Convolutional layer with the QuantumReLU activation function
        layers.Conv2D(32, kernel_size=(3, 3)),
        QuantumReLU(modified=False),  # True if using the m-QReLU (instead of the QReLU)

        layers.MaxPooling2D(pool_size=(2, 2)),
    ]
)
'''


class QuantumReLU(Layer):  # pragma: no cover
    """
    A class defining the QuantumReLU activation function in Keras.
    """

    def __init__(self, modified: bool = USE_M_QRELU) -> None:
        """
        Initialise the QuantumReLU activation function.

        Args:
            modified: bool
                    Whether using the modified version of the QReLU or m-QReLU (no/False by default,
                    i.e., using the QReLU by default).
        """

        self.modified = modified
        self._name = M_QRELU_NAME if modified else QRELU_NAME
        super().__init__()

    def call(self, inputs: Tensor) -> Tensor:
        """
        Call the QuantumReLU activation function.

        Args:
            inputs: Tensor
                    The input tensor.

        Returns:
                The output tensor (Tensor) from the QuantumReLU activation function.
        """
        return tf_quantum_relu(x=inputs, modified=self.modified)
