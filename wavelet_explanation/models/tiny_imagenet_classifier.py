"""
ResNet-18 trained from scratch on Tiny-ImageNet-200 (64×64, 200 classes).

Architecture:
    Standard ResNet-18 (no pretrained weights) with the final FC layer
    replaced: Linear(512, 1000) → Linear(512, 200).

Also provides train_tiny_imagenet_classifier() to train the model from
scratch and save the weights to disk — mirrors train_mnist_classifier()
in models/custom_cnn.py.

Expected data layout (ImageFolder):
    data_path/
        train/
            n01443537/   ← 200 class folders, 500 images each
            ...
        val/
            n01443537/   ← 50 images per class (pre-reorganised by prepare_tiny_imagenet.py)
            ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import os


class TinyImageNetClassifier(nn.Module):
    """
    ResNet-18 for Tiny-ImageNet-200.

    Input:  (B, 3, 64, 64)
    Output: (B, 200) logits
    """

    def __init__(self):
        super().__init__()
        # Random init — no pretrained weights (Tiny-ImageNet ≠ ImageNet-1k label space)
        self.model = torchvision.models.resnet18(weights=None)
        # Replace the 1000-class head with a 200-class head
        self.model.fc = nn.Linear(512, 200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_tiny_imagenet_classifier(
    save_path: str = "tiny_imagenet_classifier.pth",
    data_path: str = "./data/tiny-imagenet",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TinyImageNetClassifier:
    """
    Train TinyImageNetClassifier from scratch and save weights.

    Args:
        save_path:  where to write the state dict (.pth)
        data_path:  root that contains train/ and val/ subdirs
        epochs:     training epochs (30 gives ~55-60% top-1 val accuracy)
        batch_size: mini-batch size
        lr:         initial Adam learning rate; decays by 0.1 at epochs 15 and 25
        device:     'cpu' or 'cuda'

    Returns:
        Trained TinyImageNetClassifier in eval mode.
    """
    # ── Transforms ─────────────────────────────────────────────────────────
    # Tiny-ImageNet is natively 64×64 — no resize needed, just augment lightly.
    train_transform = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_path, "train")
    val_dir   = os.path.join(data_path, "val")

    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_set   = torchvision.datasets.ImageFolder(root=val_dir,   transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = TinyImageNetClassifier().to(device)

    # ── Optimiser + LR schedule ─────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Step-LR: decay ×0.1 at epoch 15 and 25
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += images.size(0)

        train_acc  = correct / total
        train_loss = train_loss / total

        # -- Validation --
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved Tiny-ImageNet classifier weights to {save_path}")

    model.eval()
    return model
