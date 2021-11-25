# QReLU and m-QReLU in TensorFlow and Keras
## Two novel quantum activation functions

The Quantum ReLU **'QReLU'** and its modified version or **'m-QReLU'** are Python custom activation functions available for both shallow and deep neural networks in TensorFlow and Keras for Machine Learning- and Deep Learning-based classification. They are distributed under the [CC BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).

Details on this function, implementation and validation against gold standard activation functions for both shallow and deep neural networks are available at the following: **[Parisi, L., 2020](https://arxiv.org/abs/2010.08031)** and **[Parisi, L., et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0957417421012483)**. 


### Dependencies

Developed in Python 3.6, as they are compatible with TensorFlow (versions tested: 1.12 and 1.15) and Keras, please note the dependencies of TensorFlow (v1.12 or 1.15) and Keras to be able to use the 'QReLU' and 'm-QReLU' functions in shallow and deep neural networks.


### Usage

You can use the QReLU and m-QReLU functions as custom activation functions in Keras as a layer:

#### Example of usage in a sequential model in Keras with a `QReLU` (or a `m_QReLU`) layer between a convolutional layer and a pooling layer

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(QReLU())  # or model.add(m_QReLU()) 
model.add(layers.MaxPooling2D((2, 2)))
```

### Citation request

If you are using this function, please cite the paper by **[Parisi, L., 2020](https://arxiv.org/abs/2010.08031)** and **[Parisi, L., et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0957417421012483)**.
