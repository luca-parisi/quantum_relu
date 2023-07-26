# QReLU and m-QReLU in TensorFlow and Keras
## Two novel quantum activation functions

The Quantum ReLU **'QReLU'** and its modified version or **'m-QReLU'** are Python custom activation functions available for both shallow and deep neural networks in TensorFlow and Keras for Machine Learning- and Deep Learning-based classification. They are distributed under the [CC BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).

Details on these functions, implementations, and validations against gold standard activation functions for both shallow and deep neural networks are available at the papers: **[Parisi, L., 2020](https://arxiv.org/abs/2010.08031)** and **[Parisi, L., et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0957417421012483)**. 


### Dependencies

The dependencies are included in the `environment.yml` file. 
Run the following command to install the required version of Python (v3.9.16) and all dependencies in a conda virtual 
environment (replace `<env_name>` with your environment name):

- `conda create --name <env_name> --file environment.yml`


### Usage

You can use the `QuantumReLU` activation functions as a keras layer and set the `modified` attribute to either `False` 
or `True` if using the QReLU or the m-QReLU respectively:

#### Example of usage in a sequential model in Keras with a `QuantumReLU` layer between a convolutional layer and a pooling layer

Either

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(QuantumReLU(modified=False))  # True if using the m-QReLU (instead of the QReLU)
model.add(layers.MaxPooling2D((2, 2)))
```

or

```python
model = keras.Sequential(
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, kernel_size=(3, 3)),
        QuantumReLU(modified=False),  # True if using the m-QReLU (instead of the QReLU)

        layers.MaxPooling2D(pool_size=(2, 2)),
    ]
)
```

### Linting
`isort` is used to ensure a consistent order of imports, whilst `autopep8` to ensure adherence of the codes to PEP-8, 
via the following two commands respectively:

- `isort <folder_name>`
- `autopep8 --in-place --recursive .`

### Unit testing
Run `pytest --cov-report term-missing --cov=src tests/` to execute all unit tests and view the report with the test 
coverage in percentage and missing lines too.

### Citation request

If you use these activation functions, please cite the papers by **[Parisi, L., 2020](https://arxiv.org/abs/2010.08031)** and **[Parisi, L., et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S0957417421012483)**.
