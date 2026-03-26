"""
Lightweight CNN for MNIST experiments.

Architecture:
    Conv(1,32,3) → ReLU → MaxPool(2)
    Conv(32,64,3) → ReLU → MaxPool(2)
    Flatten → Linear(64*5*5, 128) → ReLU → Linear(128, 10)

Expects input size (B, 1, 28, 28).

Also provides train_mnist_classifier() to train the model from scratch and
save the weights to disk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


class MNISTClassifier(nn.Module):
    """
    Three-layer CNN for MNIST (10-class digit classification).

    Input:  (B, 1, 28, 28)
    Output: (B, 10) logits
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After two pool operations on 28×28: 28→14→7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 7,  7)
        x = x.flatten(1)                      # (B, 64*7*7)
        x = F.relu(self.fc1(x))               # (B, 128)
        x = self.fc2(x)                       # (B, 10)
        return x


def train_mnist_classifier(
    save_path: str = "mnist_classifier.pth",
    data_root: str = "./data",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> MNISTClassifier:
    """
    Train MNISTClassifier from scratch and save weights.

    Args:
        save_path:  path to save the state dict (.pth file)
        data_root:  directory where MNIST is (or will be) downloaded
        epochs:     number of training epochs
        batch_size: mini-batch size
        lr:         Adam learning rate
        device:     'cpu' or 'cuda'

    Returns:
        Trained MNISTClassifier in eval mode.
    """
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    model = MNISTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved classifier weights to {save_path}")

    model.eval()
    return model
