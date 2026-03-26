"""
Evaluation entry point.

Loads a trained checkpoint and evaluates the wavelet explanation model on
a held-out dataset, printing quantitative metrics and saving visualisations.

Usage
-----
    python evaluate.py --config configs/resnet18_imagenet.yaml \\
                       --data_path /path/to/imagenet \\
                       --checkpoint outputs/checkpoint_epoch0050.pth \\
                       --output_dir eval_outputs/

Metrics reported:
    - Per-subband and overall mask sparsity
    - Pixel-equivalent sparsity
    - Mean confidence delta: p_f(e)(y) - p_f(x)(y)
    - Label preservation rate
    - Subband activity profile (frequency bias)
"""

import argparse
import os
import sys

import torch
import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from training.trainer import WaveletExplanationTrainer
from evaluation.metrics import (
    compute_sparsity,
    compute_confidence_delta,
    compute_label_preservation_rate,
    compute_subband_activity_profile,
)
from visualization.visualize import visualize_explanation, visualize_frequency_profile


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate wavelet explanation model")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_outputs/")
    parser.add_argument("--device",     type=str, default="")
    parser.add_argument("--n_batches",  type=int, default=50, help="Number of batches to evaluate")
    return parser.parse_args()


def build_eval_dataloader(config: dict, data_path: str) -> DataLoader:
    dataset_name = config.get("dataset", "imagenet")
    image_size   = config.get("image_size", 224)
    batch_size   = config.get("batch_size", 8)

    if dataset_name == "imagenet":
        transform = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)

    elif dataset_name == "imagefolder":
        # Generic ImageFolder layout: data_path/val/<class>/<image>
        transform = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_dir = os.path.join(data_path, "val")
        dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

    elif dataset_name == "mnist":
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    elif dataset_name == "stl10":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.STL10(
            root=data_path, split="test", download=True, transform=transform
        )

    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build trainer and load checkpoint
    trainer = WaveletExplanationTrainer(config, device=device)
    trainer.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataloader = build_eval_dataloader(config, args.data_path)

    # Accumulators
    all_sparsity   = {"overall": 0, "LL": 0, "LH": 0, "HL": 0, "HH": 0, "pixel_equivalent": 0}
    conf_deltas    = []
    label_matches  = []
    activity_sum   = {"LL": 0, "LH": 0, "HL": 0, "HH": 0}
    n_evaluated    = 0

    first_batch_saved = False

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= args.n_batches:
            break

        x = x.to(device)
        y = y.to(device)

        m_LL, m_LH, m_HL, m_HH, e, _ = trainer.predict_masks(x)

        # Sparsity
        sp = compute_sparsity(m_LL, m_LH, m_HL, m_HH)
        for k in all_sparsity:
            all_sparsity[k] += sp[k]

        # Confidence delta
        delta = compute_confidence_delta(trainer.classifier, x, e, y)
        conf_deltas.append(delta)

        # Label preservation
        lpr = compute_label_preservation_rate(trainer.classifier, x, e)
        label_matches.append(lpr)

        # Activity profile
        ap = compute_subband_activity_profile(m_LL, m_LH, m_HL, m_HH)
        for k in activity_sum:
            activity_sum[k] += ap[k]

        n_evaluated += 1

        # Save a visualisation for the first batch
        if not first_batch_saved:
            from train import _denorm
            viz_path = os.path.join(args.output_dir, "sample_explanation.png")
            visualize_explanation(
                _denorm(x[:4], config["dataset"]),
                m_LL[:4], m_LH[:4], m_HL[:4], m_HH[:4],
                _denorm(e[:4], config["dataset"]),
                save_path=viz_path,
            )
            print(f"Saved sample visualisation: {viz_path}")
            first_batch_saved = True

    # Aggregate
    def mean(d, n):
        return {k: v / n for k, v in d.items()}

    avg_sparsity  = mean(all_sparsity, n_evaluated)
    avg_activity  = mean(activity_sum, n_evaluated)
    avg_conf_delta = sum(conf_deltas) / len(conf_deltas)
    avg_lpr        = sum(label_matches) / len(label_matches)

    print("\n===== Evaluation Results =====")
    print(f"Batches evaluated   : {n_evaluated}")
    print(f"\n-- Mask Sparsity (fraction zeroed) --")
    for k, v in avg_sparsity.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\n-- Subband Activity Profile (fraction kept) --")
    for k, v in avg_activity.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\n-- Fidelity --")
    print(f"  Mean confidence delta    : {avg_conf_delta:+.4f}")
    print(f"  Label preservation rate  : {avg_lpr:.4f}")

    # Frequency profile visualisation
    profile_path = os.path.join(args.output_dir, "frequency_profile.png")
    visualize_frequency_profile(
        {"Overall": avg_activity},
        class_names=["Overall"],
        save_path=profile_path,
    )
    print(f"\nSaved frequency profile: {profile_path}")


if __name__ == "__main__":
    main()
