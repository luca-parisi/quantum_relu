"""Unit tests for the quantum activation functions"""

import unittest

from src.tf_keras.quantum_activations import (derivative_quantum_relu,
                                              quantum_relu)


class TestQuantumRelu(unittest.TestCase):
    """Class to unit test quantum activation functions"""

    def test_quantum_relu_with_positive_input(self):
        """Test with positive input and default (QReLU) activation"""
        x = 5.0
        expected_output = 5.0
        result = quantum_relu(x)
        self.assertAlmostEqual(result, expected_output)

    def test_quantum_relu_with_negative_input_qrelu(self):
        """Test with negative input and default (QReLU) activation"""
        x = -2.0
        expected_output = 3.98
        result = quantum_relu(x)
        self.assertAlmostEqual(result, expected_output)

    def test_quantum_relu_with_negative_input_mqrelu(self):
        """Test with negative input and modified (m-QReLU) activation"""
        x = -2.0
        expected_output = 1.98
        result = quantum_relu(x, modified=True)
        self.assertAlmostEqual(result, expected_output)

    def test_quantum_relu_with_zero_input_qrelu(self):
        """Test with zero input and default (QReLU) activation"""
        x = 0.0
        expected_output = 0.0
        result = quantum_relu(x)
        self.assertAlmostEqual(result, expected_output)

    def test_quantum_relu_with_zero_input_mqrelu(self):
        """Test with zero input and modified (m-QReLU) activation"""
        x = 0.0
        expected_output = 0.0
        result = quantum_relu(x, modified=True)
        self.assertAlmostEqual(result, expected_output)

    def test_derivative_quantum_relu_qrelu(self):
        """Test when using QReLU (not modified)"""
        x = 2.0
        expected_output = 2.0
        self.assertAlmostEqual(derivative_quantum_relu(
            x, modified=False), expected_output)

    def test_derivative_quantum_relu_mqrelu(self):
        """Test when using m-QReLU (modified)"""
        x = -1.0
        expected_output = -0.99
        self.assertAlmostEqual(derivative_quantum_relu(
            x, modified=True), expected_output)

    def test_derivative_quantum_relu_default(self):
        """Test with default modified (QReLU)"""
        x = 0.5
        expected_output = 0.5
        self.assertAlmostEqual(derivative_quantum_relu(x), expected_output)

    def test_derivative_quantum_relu_zero(self):
        """Test with x = 0"""
        x = 0.0
        expected_output = -1.99
        self.assertAlmostEqual(derivative_quantum_relu(x), expected_output)
