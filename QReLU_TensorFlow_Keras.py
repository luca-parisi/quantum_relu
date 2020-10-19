# QReLU as a custom activation function in TensorFlow

import numpy as np
import tensorflow as tf

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

# Example of usage in TensorFlow with a QReLU layer between a convolutional layer (#2) and a pooling layer (#2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same")
  conv2_act = tf_q_relu(conv2)
  pool2 = tf.layers.max_pooling2d(inputs=conv2_act, pool_size=[2, 2], strides=2)
  
# QReLU as a custom layer in Keras 
from tensorflow.keras.layers import Layer

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

# Example of usage in a sequential model in Keras with a QReLU layer between a convolutional layer and a pool-ing layer

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(QReLU())
model.add(layers.MaxPooling2D((2, 2)))
