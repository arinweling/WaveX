#!/usr/bin/env python3
"""
Download and prepare Tiny-ImageNet-200 for use with ImageFolder.

Usage:
    python prepare_tiny_imagenet.py --data_path ./data/tiny-imagenet

After running, data_path will contain:
    train/
        n01443537/   (200 class folders, 500 images each)
        ...
    val/
        n01443537/   (50 images per class, re-organised into class subfolders)
        ...

Total size: ~236 MB
"""

import argparse
import os
import shutil
import zipfile
import urllib.request


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_tiny_imagenet(data_path: str):
    os.makedirs(data_path, exist_ok=True)
    zip_path = os.path.join(data_path, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny-ImageNet-200 to {zip_path} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path, reporthook=_progress)
        print()
    else:
        print(f"Found existing archive: {zip_path}")

    extract_dir = os.path.join(data_path, "tiny-imagenet-200")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)
        print("Extracted.")
    else:
        print(f"Already extracted: {extract_dir}")

    return extract_dir


def reorganise_val(extract_dir: str):
    """
    Tiny-ImageNet val set ships flat:
        val/images/<image>.JPEG  +  val/val_annotations.txt

    Re-organise into ImageFolder layout:
        val/<class>/<image>.JPEG
    """
    val_dir      = os.path.join(extract_dir, "val")
    images_dir   = os.path.join(val_dir, "images")
    annot_file   = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(images_dir):
        print("val/images not found — already reorganised or missing.")
        return

    print("Reorganising val split into ImageFolder layout...")
    with open(annot_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(images_dir, fname)
            dst = os.path.join(class_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)

    shutil.rmtree(images_dir, ignore_errors=True)
    os.remove(annot_file) if os.path.exists(annot_file) else None
    print("val split reorganised.")


def link_to_target(extract_dir: str, data_path: str):
    """
    Move/symlink extract_dir/train and extract_dir/val to data_path/train, data_path/val
    so the ImageFolder loader finds them directly at data_path.
    """
    for split in ("train", "val"):
        src = os.path.join(extract_dir, split)
        dst = os.path.join(data_path, split)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"Moved {split}/ to {dst}")
        else:
            print(f"{dst} already exists, skipping move.")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    print(f"\r  {pct:.1f}%  ({downloaded/1e6:.1f} / {total_size/1e6:.1f} MB)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/tiny-imagenet",
                        help="Directory to download and prepare Tiny-ImageNet into")
    args = parser.parse_args()

    extract_dir = download_tiny_imagenet(args.data_path)
    reorganise_val(extract_dir)
    link_to_target(extract_dir, args.data_path)

    # Quick sanity check
    train_classes = len(os.listdir(os.path.join(args.data_path, "train")))
    val_classes   = len(os.listdir(os.path.join(args.data_path, "val")))
    print(f"\nReady: {train_classes} train classes, {val_classes} val classes")
    print(f"Use --data_path {args.data_path} with configs/resnet18_tiny_imagenet.yaml")


if __name__ == "__main__":
    main()
