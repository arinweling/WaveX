"""
Entry point for training the spectral-domain explanation mask.

Mirrors the structure of train.py but wires in SpectralExplanationTrainer
and the spectral visualization helpers.

Usage
-----
# Single image (avocado / any JPEG):
python train_spectral.py \\
    --config configs/resnet18_imagenet_spectral.yaml \\
    --data_path /path/to/imagenet \\
    --output_dir outputs/spectral_test/

# Full dataset epoch training:
python train_spectral.py \\
    --config configs/resnet18_imagenet_spectral.yaml \\
    --data_path /path/to/imagenette \\
    --output_dir outputs/spectral_imagenette/
"""

import argparse
import os
import sys

import numpy as np
import yaml
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))

from spectral.trainer import SpectralExplanationTrainer
from spectral.explanation import construct_spectral_explanation
from visualization.spectral_visualize import (
    visualize_spectral_explanation,
    visualize_radial_profile,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train spectral-domain explanation masks")
    p.add_argument("--config",      required=True,           help="Path to YAML config")
    p.add_argument("--data_path",   required=True,           help="Dataset root directory")
    p.add_argument("--output_dir",  default="outputs/",      help="Checkpoint & viz directory")
    p.add_argument("--device",      default="",              help="'cpu' or 'cuda' (auto if empty)")
    p.add_argument("--resume",      default="",              help="Path to checkpoint to resume from")
    return p.parse_args()


def build_dataloader(config: dict, data_path: str) -> DataLoader:
    """Construct the dataset and DataLoader (same logic as train.py)."""
    dataset_name = config.get("dataset", "imagenet")
    image_size   = config.get("image_size", 224)
    batch_size   = config.get("batch_size", 8)

    if dataset_name == "imagenet":
        tf = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=tf)

    elif dataset_name == "imagefolder":
        resize_ops = [] if image_size == 64 else [T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size)]
        tf = T.Compose([*resize_ops, T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, "train"), transform=tf
        )

    elif dataset_name == "stl10":
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.STL10(
            root=data_path, split="train", download=True, transform=tf
        )

    elif dataset_name == "mnist":
        tf = T.Compose([T.Resize(image_size), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=True, transform=tf
        )

    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    filter_class = config.get("filter_class", None)
    if filter_class is not None:
        labels = np.array(dataset.labels if hasattr(dataset, "labels") else dataset.targets)
        indices = np.where(labels == int(filter_class))[0].tolist()
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def _denorm(tensor: torch.Tensor, dataset: str) -> torch.Tensor:
    t = tensor.clone().cpu()
    if dataset in ("imagenet", "imagefolder", "stl10"):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = t * std + mean
    elif dataset == "mnist":
        t = t * 0.3081 + 0.1307
    return t.clamp(0, 1)


def main():
    args   = parse_args()
    config = yaml.safe_load(open(args.config))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    import shutil
    shutil.copy(args.config, os.path.join(args.output_dir, "config.yaml"))

    # ------------------------------------------------------------------
    # Single-image-path mode
    # ------------------------------------------------------------------
    single_image_path = config.get("single_image_path", None)
    if single_image_path:
        image_size = config.get("image_size", 224)
        tf = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img     = Image.open(single_image_path).convert("RGB")
        fixed_x = tf(img).unsqueeze(0).to(device)
        fixed_label = int(config.get("fixed_class_label", 0))
        fixed_y = torch.tensor([fixed_label], dtype=torch.long, device=device)
        dataloader = None
        print(f"Single-image mode: '{single_image_path}'  target class={fixed_label}")
    else:
        dataloader = build_dataloader(config, args.data_path)
        print(f"Dataset: {config['dataset']}  | {len(dataloader.dataset)} examples")

    # ------------------------------------------------------------------
    # Build trainer
    # ------------------------------------------------------------------
    trainer = SpectralExplanationTrainer(config, device=device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    # ------------------------------------------------------------------
    # Single-image mode: overfit on one fixed image
    # ------------------------------------------------------------------
    single_image_mode = config.get("single_image_mode", False)
    if single_image_mode and not single_image_path:
        image_index = config.get("single_image_index", 0)
        base_dataset = dataloader.dataset
        img_tensor, img_label = base_dataset[image_index]
        fixed_x = img_tensor.unsqueeze(0).to(device)
        fixed_label = int(config.get("fixed_class_label", img_label))
        fixed_y = torch.tensor([fixed_label], dtype=torch.long, device=device)
        print(f"Single-image mode: index={image_index}  target class={fixed_label}")
    single_image_mode = single_image_mode or bool(single_image_path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epochs    = config.get("epochs", 500)
    viz_every = config.get("viz_every", 20)

    for epoch in range(1, epochs + 1):
        if single_image_mode:
            losses, avg_losses = trainer.train_step(fixed_x, fixed_y), None
            avg_losses = losses
        else:
            avg_losses = trainer.train_epoch(dataloader)

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
        print(f"Epoch {epoch:4d}/{epochs}  {loss_str}")

        # Classifier confidence on explanation
        with torch.no_grad():
            sample_x = fixed_x if single_image_mode else next(iter(dataloader))[0][:1].to(device)
            _, _, _, e = trainer.predict(sample_x)
            logits_e = trainer.classifier(e)
            probs = torch.softmax(logits_e, dim=1)
            top_prob, top_class = probs[0].max(dim=0)
            fixed_label_conf = ""
            if config.get("fixed_class_label") is not None:
                fixed_label_conf = f"  | fixed target conf={probs[0, int(config['fixed_class_label'])].item():.4f}"
            print(f"           → top-1 class {top_class.item()}  conf={top_prob.item():.4f}{fixed_label_conf}")

        # Save checkpoint at end
        if epoch == epochs:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pth")
            trainer.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # Periodic visualisation
        if epoch % viz_every == 0:
            sample_x = fixed_x if single_image_mode else next(iter(dataloader))[0][:4].to(device)
            M_bin, M_cont, radial_profile, e = trainer.predict(sample_x)

            x_d = _denorm(sample_x, config["dataset"])
            e_d = _denorm(e,        config["dataset"])

            viz_path  = os.path.join(args.output_dir, f"explanation_epoch{epoch:04d}.png")
            profile_path = os.path.join(args.output_dir, f"radial_profile_epoch{epoch:04d}.png")

            visualize_spectral_explanation(x_d, M_bin, M_cont, e_d, save_path=viz_path)
            visualize_radial_profile(radial_profile, save_path=profile_path)
            print(f"  → Saved viz: {viz_path}")
            print(f"  → Saved radial profile: {profile_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
