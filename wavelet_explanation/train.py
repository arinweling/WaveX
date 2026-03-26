"""
Entry point for training the wavelet-domain explanation U-Net.

Usage
-----
ImageNet (ResNet-18):
    python train.py --config configs/resnet18_imagenet.yaml --data_path /path/to/imagenet

MNIST (custom CNN):
    python train.py --config configs/cnn_mnist.yaml --data_path /path/to/mnist

The training loop:
    - Loads the dataset and constructs a DataLoader
    - Instantiates WaveletExplanationTrainer
    - Runs for the configured number of epochs
    - Saves periodic visualisations and checkpoints to --output_dir
    - Prints per-epoch loss summary
"""

import argparse
import os
import numpy as np
import sys
import yaml
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset

# Make package root importable when run directly
sys.path.insert(0, os.path.dirname(__file__))

from training.trainer import WaveletExplanationTrainer
from visualization.visualize import visualize_explanation, visualize_subband_attribution


def parse_args():
    parser = argparse.ArgumentParser(description="Train wavelet-domain explanation masks")
    parser.add_argument("--config",     type=str, required=True,      help="Path to YAML config")
    parser.add_argument("--data_path",  type=str, required=True,      help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Save checkpoints & visualisations here")
    parser.add_argument("--device",     type=str, default="",         help="'cpu' or 'cuda' (auto-detected if omitted)")
    parser.add_argument("--resume",     type=str, default="",         help="Path to checkpoint to resume from")
    return parser.parse_args()


def build_dataloader(config: dict, data_path: str) -> DataLoader:
    """Construct the dataset and DataLoader from config."""
    dataset_name = config.get("dataset", "imagenet")
    image_size   = config.get("image_size", 224)
    batch_size   = config.get("batch_size", 8)

    if dataset_name == "imagenet":
        transform = T.Compose([
            T.Resize(int(image_size * 1.14)),   # slight oversize then crop
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=transform)

    elif dataset_name == "imagefolder":
        # Generic ImageFolder layout: data_path/train/<class>/<image>
        # Works for Tiny-ImageNet, ImageNet subsets, or any custom folder dataset.
        # Only resize if the dataset images aren't already the target size
        # (Tiny-ImageNet is natively 64×64 — no resize needed, avoids blur).
        resize_ops = [] if image_size == 64 else [T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size)]
        transform = T.Compose([
            *resize_ops,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dir = os.path.join(data_path, "train")
        dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

    elif dataset_name == "mnist":
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

    elif dataset_name == "stl10":
        # STL-10: natively 96×96, 10 classes.  ImageNet normalisation stats match
        # the ResNet-18 checkpoint (DuckyDuck123/resnet18-stl10) which was trained
        # with the same normalisation.
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.STL10(
            root=data_path, split="train", download=True, transform=transform
        )

    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Add it to build_dataloader().")

    # Optional: restrict to a single class (e.g. filter_class: 5 for STL-10 dog)
    filter_class = config.get("filter_class", None)
    if filter_class is not None:
        if hasattr(dataset, "labels"):          # STL10
            labels = np.array(dataset.labels)
        elif hasattr(dataset, "targets"):       # MNIST, ImageFolder, etc.
            labels = np.array(dataset.targets)
        else:
            raise ValueError("Dataset does not expose .labels or .targets — filter_class not supported.")
        indices = np.where(labels == int(filter_class))[0].tolist()
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    fixed_class_label = config.get("fixed_class_label", None)

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save a copy of the config used for this run into the output directory
    import shutil
    shutil.copy(args.config, os.path.join(args.output_dir, "config.yaml"))
    print(f"Config saved to {os.path.join(args.output_dir, 'config.yaml')}")

    # ------------------------------------------------------------------ #
    # Single-image-path mode: load one image directly from disk            #
    # (no dataset required — useful for ImageNet with a single image file) #
    # ------------------------------------------------------------------ #
    single_image_path = config.get("single_image_path", None)
    if single_image_path:
        image_size = config.get("image_size", 224)
        transform = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(single_image_path).convert("RGB")
        fixed_x = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
        fixed_label = int(fixed_class_label) if fixed_class_label is not None else 0
        fixed_y = torch.tensor([fixed_label], dtype=torch.long, device=device)
        dataloader = None
        if fixed_class_label is None:
            print(f"Single-image-path mode: loaded '{single_image_path}'  |  target class = model top-1")
        else:
            print(f"Single-image-path mode: loaded '{single_image_path}'  |  fixed target class = {fixed_label}")
    else:
        # Build dataset
        dataloader = build_dataloader(config, args.data_path)
        print(f"Dataset: {config['dataset']}  |  {len(dataloader.dataset)} examples  |  "
              f"batch_size={config['batch_size']}")

    # Build trainer
    trainer = WaveletExplanationTrainer(config, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    # ------------------------------------------------------------------ #
    # Single-image mode: overfit on one fixed image                        #
    # ------------------------------------------------------------------ #
    single_image_mode = config.get("single_image_mode", False)
    if single_image_mode and not single_image_path:
        image_index = config.get("single_image_index", 0)
        # Unwrap Subset if needed
        base_dataset = dataloader.dataset
        img_tensor, img_label = base_dataset[image_index]
        fixed_x = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
        target_label = int(fixed_class_label) if fixed_class_label is not None else int(img_label)
        fixed_y = torch.tensor([target_label], dtype=torch.long, device=device)
        print(f"Single-image mode: training on image index {image_index}  "
              f"(dataset label={img_label}, target class={target_label}) for {config.get('epochs', 50)} steps.")
    single_image_mode = single_image_mode or bool(single_image_path)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    epochs = config.get("epochs", 50)
    viz_every = config.get("viz_every", 1)

    for epoch in range(1, epochs + 1):
        if single_image_mode:
            losses = trainer.train_step(fixed_x, fixed_y)
            avg_losses = losses
        else:
            avg_losses = trainer.train_epoch(dataloader)

        # Log
        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
        print(f"Epoch {epoch:3d}/{epochs}  {loss_str}")

        # Classifier confidence on the explanation
        with torch.no_grad():
            if single_image_mode:
                conf_x = fixed_x
            else:
                conf_x, _ = next(iter(dataloader))
                conf_x = conf_x[:1].to(device)
            _, _, _, _, conf_e, _ = trainer.predict_masks(conf_x)
            logits_e = trainer.classifier(conf_e)
            probs = torch.softmax(logits_e, dim=1)
            top_prob, top_class = probs[0].max(dim=0)
            if fixed_class_label is None:
                print(f"           classifier on explanation → class {top_class.item()}  "
                    f"confidence {top_prob.item():.4f}")
            else:
                target_prob = probs[0, int(fixed_class_label)].item()
                print(f"           classifier on explanation → top-1 class {top_class.item()}  "
                    f"confidence {top_prob.item():.4f}  |  fixed target class {int(fixed_class_label)}  "
                    f"confidence {target_prob:.4f}")

        # Checkpoint (only at the very end)
        if epoch == epochs:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pth")
            trainer.save_checkpoint(ckpt_path)

        # Periodic visualisation
        if epoch % viz_every == 0:
            if single_image_mode:
                sample_x = fixed_x  # always visualise the training image
            else:
                sample_x, _ = next(iter(dataloader))
                sample_x = sample_x[:4].to(device)
            masks_and_e = trainer.predict_masks(sample_x)
            m_LL, m_LH, m_HL, m_HH, e, _ = masks_and_e

            # Denormalise for display
            viz_path = os.path.join(args.output_dir, f"explanation_epoch{epoch:04d}.png")
            x_denorm = _denorm(sample_x, config["dataset"])
            e_denorm = _denorm(e, config["dataset"])

            visualize_explanation(
                x_denorm, m_LL, m_LH, m_HL, m_HH, e_denorm,
                save_path=viz_path,
            )
            print(f"  → Saved visualisation: {viz_path}")

            # Colour-coded subband attribution map
            attr_path = os.path.join(args.output_dir, f"attribution_epoch{epoch:04d}.png")
            visualize_subband_attribution(
                x_denorm, m_LL, m_LH, m_HL, m_HH, e_denorm,
                save_path=attr_path,
            )
            print(f"  → Saved attribution map: {attr_path}")

    print("Training complete.")


def _denorm(tensor: torch.Tensor, dataset: str) -> torch.Tensor:
    """Reverse normalisation for display purposes."""
    t = tensor.clone().cpu()
    if dataset in ("imagenet", "imagefolder", "stl10"):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = t * std + mean
    elif dataset == "mnist":
        t = t * 0.3081 + 0.1307
    return t.clamp(0, 1)


if __name__ == "__main__":
    main()
