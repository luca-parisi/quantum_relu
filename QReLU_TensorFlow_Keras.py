"""Additional utility for Deep Learning models in TensorFlow and Keras"""

# The Quantum ReLU or 'QReLU' as a custom activation function in TensorFlow (tf_q_relu)
# and Keras (QReLU)

# Author: Luca Parisi <luca.parisi@ieee.org>


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# QReLU as a custom activation function in TensorFlow

'''
# Example of usage of the QReLU in TensorFlow as a custom activation function of a convolutional layer (#2)

convolutional_layer_2 = tf.layers.conv2d(
                        inputs=pooling_layer_1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding="same")
convolutional_layer_activation = tf_q_relu(convolutional_layer_2)
pooling_layer_2 = tf.layers.max_pooling2d(inputs=convolutional_layer_activation, pool_size=[2, 2], strides=2)
'''

# Defining the QReLU function
def q_relu(x):
  if x>0:
    x = x
    return x
  else:
    x = 0.01*x-2*x
    return x

# Vectorising the QReLU function  
np_q_relu = np.vectorize(q_relu)

# Defining the derivative of the function QReLU
def d_q_relu(x):
  if x>0:
    x = 1
    return x
  else:
    x = 0.01-2
    return x

# Vectorising the derivative of the QReLU function  
np_d_q_relu = np.vectorize(d_q_relu)

# Defining the gradient function of the QReLU
def q_relu_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_q_relu(x)
    return grad * n_gr

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
# Generating a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

np_q_relu_32 = lambda x: np_q_relu(x).astype(np.float32)

def tf_q_relu(x,name=None):
    with tf.name_scope(name, "q_relu", [x]) as name:
        y = py_func(np_q_relu_32,   # Forward pass function
                        [x],
                        [tf.float32],
                        name=name,
                         grad= q_relu_grad) # The function that overrides gradient
        y[0].set_shape(x.get_shape())     # To specify the rank of the input.
        return y[0]

np_d_q_relu_32 = lambda x: np_d_q_relu(x).astype(np.float32)

def tf_d_q_relu(x,name=None):
    with tf.name_scope(name, "d_q_relu", [x]) as name:
        y = tf.py_func(np_d_q_relu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]


# QReLU as a custom layer in Keras 

'''
# Example of usage of the QReLU as a Keras layer in a sequential model between a convolutional layer and a pooling layer

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(QReLU())
model.add(layers.MaxPooling2D((2, 2)))
'''

class QReLU(Layer):

    def __init__(self):
        super(QReLU,self).__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs,name=None):
        return tf_q_relu(inputs,name=None)

    def get_config(self):
        base_config = super(QReLU, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
