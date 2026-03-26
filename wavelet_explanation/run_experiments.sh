#!/usr/bin/env bash
# run_experiments.sh
# Systematically runs all MNIST tuning experiments, then evaluates each.
#
# Usage (from wavelet_explanation/ directory):
#   bash run_experiments.sh [DATA_PATH] [DEVICE]
#
# Defaults:
#   DATA_PATH = ./data    (MNIST is downloaded here automatically)
#   DEVICE    = cuda      (falls back to cpu if CUDA unavailable)

set -euo pipefail

DATA_PATH="${1:-./data}"
TINY_DATA_PATH="${3:-./data/tiny-imagenet}"
INETTE_DATA_PATH="${4:-./data/imagenette2}"
DEVICE="${2:-cuda}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS=(
  "configs/mnist_run01_baseline.yaml          outputs/run01_baseline"
  "configs/mnist_run02_equal_area.yaml        outputs/run02_equal_area"
  "configs/mnist_run03_strong_fidelity.yaml   outputs/run03_strong_fidelity"
  "configs/mnist_run04_aggressive_sparsity.yaml outputs/run04_aggressive_sparsity"
)

TINY_CONFIG="configs/resnet18_tiny_imagenet.yaml"
TINY_OUT="outputs/resnet18_tiny_imagenet"

INETTE_CONFIG="configs/resnet18_imagenette.yaml"
INETTE_OUT="outputs/resnet18_imagenette"

echo "================================================================"
echo "  Wavelet Explanation — Experiment Suite"
echo "  ImageNette data:    $INETTE_DATA_PATH"
echo "  Device:             $DEVICE"
echo "================================================================"

# ── 0a. Ensure MNIST classifier weights exist ─────────────────────────────
# if [ ! -f "$SCRIPT_DIR/mnist_classifier.pth" ]; then
#   echo ""
#   echo ">>> [0a] Training MNIST classifier..."
#   python -c "
# import sys; sys.path.insert(0,'$SCRIPT_DIR')
# from models.custom_cnn import train_mnist_classifier
# train_mnist_classifier(save_path='$SCRIPT_DIR/mnist_classifier.pth',
#                        data_root='$DATA_PATH', epochs=5, device='$DEVICE')
# "
# else
#   echo ">>> MNIST classifier already trained: mnist_classifier.pth"
# fi

# ── 0b. No classifier pre-training needed ────────────────────────────────
# Using pretrained ImageNet-1k ResNet-18 with Option B label handling:
# L_CE / L_rob use the model's own top-1 prediction, so no label-space
# mismatch between ImageFolder (0-199) and ImageNet-1k (0-999) indices.
echo ">>> Using pretrained ResNet-18 (ImageNet-1k) — no classifier pre-training needed"

# ── 1. Train each MNIST config ───────────────────────────────────────────
# IDX=1
# EVAL_ARGS=()
#
# for entry in "${CONFIGS[@]}"; do
#   CFG=$(echo "$entry" | awk '{print $1}')
#   OUT=$(echo "$entry" | awk '{print $2}')
#   RUN_NAME=$(basename "$OUT")
#
#   echo ""
#   echo ">>> [$IDX/${#CONFIGS[@]}] Training: $RUN_NAME"
#   echo "    Config:  $CFG"
#   echo "    Output:  $OUT"
#   echo "----------------------------------------------------------------"
#
#   python train.py \
#     --config "$CFG" \
#     --data_path "$DATA_PATH" \
#     --output_dir "$OUT" \
#     --device "$DEVICE"
#
#   EVAL_ARGS+=("$CFG $OUT/checkpoint_epoch0050.pth eval_$RUN_NAME")
#   (( IDX++ ))
# done
#
# # ── 2. Evaluate all MNIST runs ────────────────────────────────────────────
# echo ""
# echo "================================================================"
# echo "  Evaluation Phase"
# echo "================================================================"
#
# for eval_entry in "${EVAL_ARGS[@]}"; do
#   CFG=$(echo "$eval_entry" | awk '{print $1}')
#   CKPT=$(echo "$eval_entry" | awk '{print $2}')
#   EVAL_OUT=$(echo "$eval_entry" | awk '{print $3}')
#   RUN_NAME=$(basename "$EVAL_OUT")
#
#   echo ""
#   echo ">>> Evaluating: $RUN_NAME"
#   python evaluate.py \
#     --config "$CFG" \
#     --data_path "$DATA_PATH" \
#     --checkpoint "$CKPT" \
#     --output_dir "$EVAL_OUT" \
#     --device "$DEVICE" \
#     --n_batches 100
# done

# ── 3. (disabled) Tiny-ImageNet experiment ───────────────────────────────
# python train.py \
#   --config "$TINY_CONFIG" \
#   --data_path "$TINY_DATA_PATH" \
#   --output_dir "$TINY_OUT" \
#   --device "$DEVICE"

# ── 4. Train U-Net explanation model on ImageNette ───────────────────────
echo ""
echo "================================================================"
echo "  ImageNette Experiment"
echo "================================================================"
echo ""
echo ">>> Training: resnet18_imagenette"
echo "    Config:  $INETTE_CONFIG"
echo "    Output:  $INETTE_OUT"
echo "----------------------------------------------------------------"

python train.py \
  --config "$INETTE_CONFIG" \
  --data_path "$INETTE_DATA_PATH" \
  --output_dir "$INETTE_OUT" \
  --device "$DEVICE"

# ── 5. Evaluate ImageNette run ────────────────────────────────────────────
echo ""
echo ">>> Evaluating: resnet18_imagenette"
python evaluate.py \
  --config "$INETTE_CONFIG" \
  --data_path "$INETTE_DATA_PATH" \
  --checkpoint "$INETTE_OUT/checkpoint_epoch0050.pth" \
  --output_dir "eval_resnet18_imagenette" \
  --device "$DEVICE" \
  --n_batches 100

echo ""
echo "================================================================"
echo "  All runs complete. Results in outputs/  and  eval_*/  dirs."
echo "================================================================"
