"""Additional utility for Deep Learning models in PyTorch"""

# The Quantum ReLU (QReLU) and the modified QReLU (m-QReLU) as custom activation functions in
# PyTorch

# Author: Luca Parisi <luca.parisi@ieee.org>

import torch
import torch.nn as nn
from src.constants import (FIRST_COEFFICIENT_PYTORCH, M_QRELU_NAME, QRELU_NAME,
                           SECOND_COEFFICIENT_M_QRELU_PYTORCH,
                           SECOND_COEFFICIENT_QRELU_PYTORCH, USE_M_QRELU)


class QuantumReLU(nn.Module):  # pragma: no cover
    """
    A class defining the QuantumReLU activation function in PyTorch.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call the QuantumReLU activation function.

        Args:
            x: torch.Tensor
                The input tensor.

        Returns:
                The output tensor (torch.Tensor) from the QuantumReLU activation function.
        """
        return torch_quantum_relu(x=x, modified=self.modified)


def torch_quantum_relu(x: torch.Tensor, modified: bool = USE_M_QRELU) -> torch.Tensor:
    """
    Apply the QReLU activation function or its modified version (m-QReLU) to transform inputs accordingly.

    Args:
        x: torch.Tensor
            The input tensor to be transformed via the m-QReLU activation function.
        modified: bool
            Whether using the modified version of the QReLU or m-QReLU (no/False by default, i.e.,
            using the QReLU by default).

    Returns:
            The transformed x (torch.Tensor) via the QReLU or m-QReLU.
    """
    second_coefficient = SECOND_COEFFICIENT_M_QRELU_PYTORCH if modified else SECOND_COEFFICIENT_QRELU_PYTORCH
    return torch.where(
            x <= 0,
            FIRST_COEFFICIENT_PYTORCH * x - second_coefficient * x,
            x,
    )
