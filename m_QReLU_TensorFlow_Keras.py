# m-QReLU as a custom activation function in TensorFlow

# Defining the m-QReLU (modified QReLU) function
def m_q_relu(x):
  if x>0:
    x = x
    return x
  else:
    x = 0.01*x-x
    return x

# Vectorising the m-QReLU function  
np_m_q_relu = np.vectorize(m_q_relu)

# Defining the derivative of the function QReLU
def d_m_q_relu(x):
  if x>0:
    x = 1
    return x
  else:
    x = 0.01-1
    return x

# Vectorising the derivative of the m-QReLU function  
np_d_m_q_relu = np.vectorize(d_m_q_relu)

# Defining the gradient function of the QReLU
def m_q_relu_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_m_q_relu(x)
    return grad * n_gr

np_m_q_relu_32 = lambda x: np_m_q_relu(x).astype(np.float32)
def tf_m_q_relu(x,name=None):
    with tf.name_scope(name, "m_q_relu", [x]) as name:
        y = py_func(np_m_q_relu_32,   # Forward pass function
                        [x],
                        [tf.float32],
                        name=name,
                         grad= m_q_relu_grad) # The function that overrides gradient
        y[0].set_shape(x.get_shape())     # To specify the rank of the input
        return y[0]
np_d_m_q_relu_32 = lambda x: np_d_m_q_relu(x).astype(np.float32)
def tf_d_m_q_relu(x,name=None):
    with tf.name_scope(name, "d_m_q_relu", [x]) as name:
        y = tf.py_func(np_d_m_q_relu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]

# Example of usage in TensorFlow with a m-QReLU layer between a convolutional layer (#2) and a pooling layer (#2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same")
  conv2_act = tf_m_q_relu(conv2)
  pool2 = tf.layers.max_pooling2d(inputs=conv2_act, pool_size=[2, 2], strides=2)
  
# m-QReLU as a custom layer in Keras 
from tensorflow.keras.layers import Layer

class m_QReLU(Layer):

    def __init__(self):
        super(m_QReLU,self).__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs,name=None):
        return tf_m_q_relu(inputs,name=None)

    def get_config(self):
        base_config = super(m_QReLU, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

# Example of usage in a sequential model in Keras with a m-QReLU layer between a convolutional layer and a pool-ing layer

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
#model.add(QReLU())
model.add(m_QReLU())
model.add(layers.MaxPooling2D((2, 2)))
