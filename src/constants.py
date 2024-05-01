"""
This file contains constants leveraged across the two quantum activation functions
QReLU and m-QReLU regardless of the framework (TensorFlow or Keras or PyTorch)
of their implementation.
"""

import torch

FIRST_COEFFICIENT = 0.01
SECOND_COEFFICIENT_QRELU = 2
SECOND_COEFFICIENT_M_QRELU = 1

FIRST_COEFFICIENT_PYTORCH = torch.tensor(FIRST_COEFFICIENT)
SECOND_COEFFICIENT_QRELU_PYTORCH = torch.tensor(SECOND_COEFFICIENT_QRELU)
SECOND_COEFFICIENT_M_QRELU_PYTORCH = torch.tensor(SECOND_COEFFICIENT_M_QRELU)

M_QRELU_NAME = "modified_quantum_relu"
QRELU_NAME = "quantum_relu"

USE_M_QRELU = False
