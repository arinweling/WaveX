"""
ViT-B/16 classifier training for BACH H&E breast histopathology images.

Expected dataset layouts:
1) Pre-split ImageFolder:
   data_root/
     train/<class_name>/*.png
     val/<class_name>/*.png

2) Single ImageFolder root (auto split):
   data_root/
     <class_name>/*.png

The trainer fine-tunes torchvision ViT-B/16 and saves a checkpoint that
includes class mappings for later inference/loading.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.models as tvm
import torchvision.transforms as T


class BACHViTClassifier(nn.Module):
    """ViT-B/16 head-adapted for BACH classes."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = tvm.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = tvm.vit_b_16(weights=weights)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _build_transforms(pretrained: bool = True) -> Tuple[T.Compose, T.Compose]:
    """Build train/val transforms aligned with ViT-B/16 defaults."""
    if pretrained:
        weights = tvm.ViT_B_16_Weights.DEFAULT
        mean = weights.transforms().mean
        std = weights.transforms().std
        # ViT-B/16 default eval uses resize to 256 then center crop 224.
        resize_size = 256
        crop_size = 224
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        resize_size = 256
        crop_size = 224

    train_tf = T.Compose([
        T.Resize(resize_size),
        T.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    val_tf = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def _stratified_split_indices(
    targets: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Return stratified train/val indices without external dependencies."""
    generator = torch.Generator().manual_seed(seed)

    targets_t = torch.tensor(list(targets), dtype=torch.long)
    classes = torch.unique(targets_t).tolist()

    train_idx: List[int] = []
    val_idx: List[int] = []

    for cls in classes:
        cls_idx = torch.where(targets_t == int(cls))[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=generator)]

        n_val = max(1, int(round(len(perm) * val_ratio)))
        n_val = min(n_val, len(perm) - 1) if len(perm) > 1 else 1

        val_part = perm[:n_val]
        train_part = perm[n_val:]

        if len(train_part) == 0:
            # Safety for tiny classes.
            train_part = val_part[:1]
            val_part = val_part[1:]

        train_idx.extend(train_part.tolist())
        val_idx.extend(val_part.tolist())

    return train_idx, val_idx


def _build_datasets(
    data_root: str,
    val_ratio: float,
    seed: int,
    pretrained: bool,
):
    """Build train/val datasets from either split dirs or a single root."""
    train_tf, val_tf = _build_transforms(pretrained=pretrained)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
        val_set = torchvision.datasets.ImageFolder(val_dir, transform=val_tf)
        class_to_idx = train_set.class_to_idx
        return train_set, val_set, class_to_idx

    # Auto-split a single ImageFolder root.
    base_train = torchvision.datasets.ImageFolder(data_root, transform=train_tf)
    base_val = torchvision.datasets.ImageFolder(data_root, transform=val_tf)

    train_idx, val_idx = _stratified_split_indices(
        targets=base_train.targets,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_set = Subset(base_train, train_idx)
    val_set = Subset(base_val, val_idx)
    class_to_idx = base_train.class_to_idx
    return train_set, val_set, class_to_idx


class _HFDatasetWrapper(Dataset):
    """Wrap a Hugging Face split so DataLoader returns (tensor, label)."""

    def __init__(self, split, image_col: str, label_col: str, transform):
        self.split = split
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        row = self.split[idx]
        image = row[self.image_col]
        label = int(row[self.label_col])
        return self.transform(image), label


def _build_hf_datasets(
    hf_dataset_id: str,
    val_ratio: float,
    seed: int,
    pretrained: bool,
    download: bool,
    hf_cache_dir: Optional[str],
):
    """Build train/val datasets from a Hugging Face dataset id."""
    try:
        from datasets import ClassLabel, DownloadConfig, Image, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install 'datasets' to use --hf_dataset_id: pip install datasets"
        ) from exc

    train_tf, val_tf = _build_transforms(pretrained=pretrained)

    download_config = DownloadConfig(local_files_only=not download)
    data = load_dataset(
        hf_dataset_id,
        cache_dir=hf_cache_dir,
        download_config=download_config,
    )

    split_names = list(data.keys())
    train_split = data["train"] if "train" in split_names else data[split_names[0]]

    image_col = None
    label_col = None
    for col, feat in train_split.features.items():
        if image_col is None and isinstance(feat, Image):
            image_col = col
        if label_col is None and isinstance(feat, ClassLabel):
            label_col = col
    if image_col is None:
        for candidate in ["image", "img"]:
            if candidate in train_split.column_names:
                image_col = candidate
                break
    if label_col is None:
        for candidate in ["label", "labels", "class", "target"]:
            if candidate in train_split.column_names:
                label_col = candidate
                break

    if image_col is None or label_col is None:
        raise ValueError(
            f"Could not infer image/label columns for {hf_dataset_id}. "
            f"Columns: {train_split.column_names}"
        )

    if "validation" in split_names:
        val_split = data["validation"]
    elif "val" in split_names:
        val_split = data["val"]
    elif "test" in split_names:
        val_split = data["test"]
    else:
        split = train_split.train_test_split(
            test_size=val_ratio,
            seed=seed,
            stratify_by_column=label_col,
        )
        train_split = split["train"]
        val_split = split["test"]

    feat = train_split.features.get(label_col)
    if hasattr(feat, "names") and feat.names is not None:
        class_to_idx = {name: i for i, name in enumerate(feat.names)}
    else:
        unique_ids = sorted({int(x) for x in train_split[label_col]})
        class_to_idx = {str(i): i for i in unique_ids}

    train_set = _HFDatasetWrapper(train_split, image_col=image_col, label_col=label_col, transform=train_tf)
    val_set = _HFDatasetWrapper(val_split, image_col=image_col, label_col=label_col, transform=val_tf)
    return train_set, val_set, class_to_idx


def train_bach_vit_classifier(
    data_root: str = "",
    hf_dataset_id: str = "",
    save_path: str = "bach_vit_b16_classifier.pth",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    pretrained: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "cpu",
    download: bool = True,
    hf_cache_dir: Optional[str] = None,
) -> BACHViTClassifier:
    """Train ViT-B/16 on BACH and save best checkpoint by val accuracy."""
    if hf_dataset_id:
        train_set, val_set, class_to_idx = _build_hf_datasets(
            hf_dataset_id=hf_dataset_id,
            val_ratio=val_ratio,
            seed=seed,
            pretrained=pretrained,
            download=download,
            hf_cache_dir=hf_cache_dir,
        )
    else:
        if not data_root:
            raise ValueError("Provide either data_root or --hf_dataset_id")
        train_set, val_set, class_to_idx = _build_datasets(
            data_root=data_root,
            val_ratio=val_ratio,
            seed=seed,
            pretrained=pretrained,
        )

    num_classes = len(class_to_idx)
    model = BACHViTClassifier(num_classes=num_classes, pretrained=pretrained).to(device)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_acc = -1.0
    best_state = None

    print(
        f"Training ViT-B/16 on BACH: classes={num_classes}  train={len(train_set)}  "
        f"val={len(val_set)}  device={device}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = model(images).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        scheduler.step()

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}
    ckpt = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": num_classes,
        "backbone": "vit_b_16",
        "pretrained": pretrained,
        "best_val_acc": best_val_acc,
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"Saved BACH ViT checkpoint to {save_path}  (best_val_acc={best_val_acc:.4f})")

    model.eval()
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT-B/16 on BACH histopathology images")
    parser.add_argument("--data_root", type=str, default="", help="BACH dataset root")
    parser.add_argument("--hf_dataset_id", type=str, default="", help="Hugging Face dataset id, e.g. 1aurent/BACH")
    parser.add_argument("--hf_cache_dir", type=str, default="", help="Optional HF cache directory")
    parser.add_argument("--save_path", type=str, default="bach_vit_b16_classifier.pth", help="Output checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Used only if data_root has no train/val split")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="", help="cpu or cuda (auto if omitted)")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from random init")
    parser.add_argument("--download", dest="download", action="store_true", help="Allow downloading remote HF data")
    parser.add_argument("--no_download", dest="download", action="store_false", help="Use local cache only")
    parser.set_defaults(download=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    train_bach_vit_classifier(
        data_root=args.data_root,
        hf_dataset_id=args.hf_dataset_id,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
        download=args.download,
        hf_cache_dir=(args.hf_cache_dir or None),
    )


if __name__ == "__main__":
    main()
