"""
Example of a simple MNIST image classifier using the QReLU or m-QReLU activation function in its two
convolutional layers.

Adapted from https://keras.io/examples/vision/mnist_convnet/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from constants import (BATCH_SIZE, DROPOUT, IMAGE_DIM, KERNEL_SIZE_CONV,
                       KERNEL_SIZE_MAX_POOL, NUM_CLASSES, NUM_EPOCHS,
                       OUT_CHANNEL_CONV1, OUT_CHANNEL_CONV2)
from src.constants import USE_M_QRELU
from src.pytorch.quantum_activations import QuantumReLU
from torch.utils.data import DataLoader
from torchvision import transforms

inputs_shape = (1, IMAGE_DIM, IMAGE_DIM)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class ConvNet(nn.Module):
    """
    Convolutional Neural Network model for image classification.

    Args:
        modified: bool
            Whether to use the modified version of the
            QReLU or m-QReLU (default: False).

    Attributes:
        conv1: nn.Conv2d
            First convolutional layer.
        relu1: QuantumReLU
            First Quantum ReLU activation.
        pool1: nn.MaxPool2d
            First max pooling layer.
        conv2: nn.Conv2d
            Second convolutional layer.
        relu2: QuantumReLU
            Second Quantum ReLU activation.
        pool2: nn.MaxPool2d
            Second max pooling layer.
        flatten: nn.Flatten
            Flatten layer.
        dropout: nn.Dropout
            Dropout layer.
        fc: nn.Linear
            Fully connected layer.
    """

    def __init__(self, modified=False):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, OUT_CHANNEL_CONV1,
                               kernel_size=KERNEL_SIZE_CONV)
        self.relu1 = QuantumReLU(modified=modified)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            OUT_CHANNEL_CONV1, OUT_CHANNEL_CONV2, kernel_size=KERNEL_SIZE_CONV)
        self.relu2 = QuantumReLU(modified=modified)
        self.pool2 = nn.MaxPool2d(kernel_size=KERNEL_SIZE_MAX_POOL)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(OUT_CHANNEL_CONV2 * 5 * 5, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: torch.Tensor
                The Input tensor.

        Returns:
            Output tensor (torch.Tensor).
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs: int = NUM_EPOCHS
) -> None:
    """
    Train the model.

    Args:
        model: nn.Module
            The neural network model to be trained.
        train_loader: DataLoader
            Data loader for training data.
        optimizer: optim.Optimizer
            Optimizer for training.
        criterion: nn.Module
            Loss function.
        epochs: int
            Number of epochs for training (default: NUM_EPOCHS).
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")


def test(
        model: nn.Module,
        test_loader: DataLoader
) -> None:
    """
    Evaluate the trained model on test data.

    Args:
        model: nn.Module
            The trained model to be evaluated.
        test_loader: DataLoader
            Data loader for test data.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


model = ConvNet(modified=USE_M_QRELU)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train(model, train_loader, optimizer, criterion)
test(model, test_loader)
