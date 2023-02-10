"""Additional utility for Deep Learning models in TensorFlow and Keras"""

# The modified Quantum ReLU or 'm-QReLU' as a custom activation function in TensorFlow (tf_m_qrelu)
# and Keras (MQReLU)

# Author: Luca Parisi <luca.parisi@ieee.org>

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Layer
from tensorflow.lite.python.op_hint import _LiteFuncCall

from .constants import (
    DERIVATIVE,
    DERIV_M_QRELU_NAME,
    FIRST_COEFFICIENT,
    SECOND_COEFFICIENT_M_QRELU,
    M_QRELU_NAME,
)
from tf_keras.utils import py_func

# m-QReLU as a custom activation function in TensorFlow

'''
# Example of usage of the m-QReLU in TensorFlow as a custom activation function of a convolutional layer (#2).

convolutional_layer_2 = tf.layers.conv2d(
                        inputs=pooling_layer_1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding="same")
convolutional_layer_activation = tf_m_qrelu(convolutional_layer_2)
pooling_layer_2 = tf.layers.max_pooling2d(inputs=convolutional_layer_activation, pool_size=[2, 2], strides=2)
'''


def m_qrelu(x: float) -> float:
    """
    Apply the m-QReLU activation function to transform inputs accordingly.

    Args:
        x: float
            The input to be transformed via the m-QReLU activation function.

    Returns:
            The transformed x (float) via the m-QReLU activation function.
    """

    if x > 0:
        x = x
        return x
    else:
        x = FIRST_COEFFICIENT*x-x
        return x


# Vectorising the m-QReLU function
np_m_qrelu = np.vectorize(m_qrelu)


def d_m_qrelu(x: float) -> float:
    """
    Compute the derivative of the m-QReLU activation function.

    Args:
        x: float
            The input from which the derivative is to be computed.

    Returns:
            The derivative (float) of the m-QReLU activation function given an input.
    """

    if x > 0:
        x = DERIVATIVE
        return x
    else:
        x = FIRST_COEFFICIENT-SECOND_COEFFICIENT_M_QRELU
        return x


# Vectorising the derivative of the m-QReLU function
np_d_m_qrelu = np.vectorize(d_m_qrelu)


def m_qrelu_grad(op: _LiteFuncCall, grad: float) -> float:
    """
    Define the gradient function of the m-QReLU activation function.

    Args:
        op: _LiteFuncCall
            A TensorFlow Lite custom function.
        grad:
            The input gradient.

    Returns:
            The gradient function of the m-QReLU activation function.
    """

    x = op.inputs[0]
    n_gr = tf_d_m_qrelu(x)
    return grad * n_gr


def np_m_qrelu_32(x): return np_m_qrelu(x).astype(np.float32)


def tf_m_qrelu(x: Tensor, name: str = M_QRELU_NAME) -> Tensor:
    """
    The m-QReLU activation function defined in TensorFlow.

    Args:
        x: Tensor
            The input tensor.
        name: str
            The name of the activation function.

    Returns:
            The output tensor (Tensor) from the m-QReLU activation function.
    """

    y = py_func(
        np_m_qrelu_32,  # Forward pass function
        [x],
        [tf.float32],
        name=name,
        grad=m_qrelu_grad
    )  # The function that overrides gradient
    y[0].set_shape(x.get_shape())  # To specify the rank of the input
    return y[0]


def np_d_m_qrelu_32(x): return np_d_m_qrelu(x).astype(np.float32)


def tf_d_m_qrelu(x: list[Tensor], name: str = DERIV_M_QRELU_NAME) -> float:
    """
    The derivative of the m-QReLU activation function defined in TensorFlow.

    Args:
        x: list[Tensor]
            A list of input tensors.
        name: str
            The name of the derivative of the activation function.

    Returns:
            The output computed as the derivative of the m-QReLU activation function.
    """

    y = tf.py_func(
        np_d_m_qrelu_32,
        [x],
        [tf.float32],
        name=name,
        stateful=False
    )
    return y[0]


# m-QReLU as a custom layer in Keras

'''
# Example of usage of the m-QReLU as a Keras layer in a sequential model between a convolutional layer and a 
pooling layer.

Either

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MQReLU())
model.add(layers.MaxPooling2D((2, 2)))

or

model = keras.Sequential(
        keras.Input(shape=(32, 32, 3)),

        # Convolutional layer with the QReLU activation function
        layers.Conv2D(32, kernel_size=(3, 3)),
        MQReLU(),

        layers.MaxPooling2D(pool_size=(2, 2)),
    ]
)
'''


class MQReLU(Layer):
    """
    A class defining the m-QReLU activation function in keras.
    """

    def __init__(self) -> None:
        """
        Initialise the m-QReLU activation function.
        """
        super(MQReLU, self).__init__()

    def build(self, input_shape: tuple[int, int, int]) -> None:
        """
        Build the m-QReLU activation function given an input shape.

        Args:
            input_shape: tuple[int, int, int]
                        The shape of the input tensor considered.
        """
        super().build(input_shape)

    @staticmethod
    def call(inputs: Tensor, name: str = None) -> Tensor:
        """
        Call the QReLU activation function.

        Args:
            inputs: Tensor
                    The input tensor.
            name: str
                The name of the activation function.

        Returns:
                The output tensor (Tensor) from the m-QReLU activation function.
        """
        return tf_m_qrelu(inputs, name=name)

    def get_config(self) -> dict[list]:
        """
        Get the configs of the m-QReLU activation function.

        Returns:
                A dictionary of the configs of the m-QReLU activation function.
        """

        base_config = super(MQReLU, self).get_config()
        return dict(list(base_config.items()))
