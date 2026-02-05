#!/usr/bin/env python3
"""
TransUNet Fine-tuning Script for Vesuvius Surface Detection

3D segmentation of papyrus surfaces in CT scans of Herculaneum scrolls.
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import pandas as pd
import tifffile
from medicai.models import TransUNet
from medicai.transforms import Compose, NormalizeIntensity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "data"
    output_dir: str = "checkpoints"
    experiment_name: str = ""

    # Model
    model_name: str = "TransUNet"
    encoder_name: str = "seresnext50"
    input_shape: tuple = (160, 160, 160)
    num_classes: int = 3
    classifier_activation: Optional[str] = None  # None for logits, "softmax" for probs

    # Training
    epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    loss: str = "combo"  # "dice", "ce", "combo"
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    label_smoothing: float = 0.0

    # Data
    val_split: float = 0.2
    num_patches_per_volume: int = 4  # Random patches per volume per epoch
    overlap: float = 0.25  # Patch overlap during validation

    # Augmentation
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotate_prob: float = 0.5

    # Misc
    seed: int = 42
    save_every: int = 5
    eval_every: int = 1

    # Derived fields
    full_input_shape: tuple = field(init=False)

    def __post_init__(self):
        self.full_input_shape = (*self.input_shape, 1)

        if not self.experiment_name:
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_volume(path: str) -> np.ndarray:
    """Load a TIFF volume."""
    vol = tifffile.imread(path)
    return vol.astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Apply intensity normalization."""
    data = {"image": volume}
    pipeline = Compose(
        [
            NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
        ]
    )
    result = pipeline(data)
    return result["image"]


def extract_random_patch(
    volume: np.ndarray, mask: np.ndarray, patch_size: tuple
) -> tuple:
    """Extract a random patch from volume and mask."""
    d, h, w = volume.shape
    pd, ph, pw = patch_size

    # Random start position
    d_start = random.randint(0, max(0, d - pd))
    h_start = random.randint(0, max(0, h - ph))
    w_start = random.randint(0, max(0, w - pw))

    # Extract patches
    vol_patch = volume[
        d_start : d_start + pd, h_start : h_start + ph, w_start : w_start + pw
    ]
    mask_patch = mask[
        d_start : d_start + pd, h_start : h_start + ph, w_start : w_start + pw
    ]

    # Pad if necessary
    if vol_patch.shape != patch_size:
        vol_padded = np.zeros(patch_size, dtype=np.float32)
        mask_padded = np.zeros(patch_size, dtype=np.uint8)
        vol_padded[: vol_patch.shape[0], : vol_patch.shape[1], : vol_patch.shape[2]] = (
            vol_patch
        )
        mask_padded[
            : mask_patch.shape[0], : mask_patch.shape[1], : mask_patch.shape[2]
        ] = mask_patch
        vol_patch = vol_padded
        mask_patch = mask_padded

    return vol_patch, mask_patch


def augment_patch(volume: np.ndarray, mask: np.ndarray, config: TrainConfig) -> tuple:
    """Apply random augmentations to a patch."""
    if not config.use_augmentation:
        return volume, mask

    # Random flips
    for axis in [0, 1, 2]:
        if random.random() < config.flip_prob:
            volume = np.flip(volume, axis=axis)
            mask = np.flip(mask, axis=axis)

    # Random 90-degree rotations in axial plane
    if random.random() < config.rotate_prob:
        k = random.randint(1, 3)
        volume = np.rot90(volume, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(1, 2))

    return np.ascontiguousarray(volume), np.ascontiguousarray(mask)


class VesuviusDataset:
    """Dataset for Vesuvius surface detection."""

    def __init__(
        self,
        data_dir: str,
        image_ids: list,
        config: TrainConfig,
        is_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_ids = image_ids
        self.config = config
        self.is_train = is_train

        # Pre-load and cache volumes (they're not too large)
        logger.info(f"Loading {len(image_ids)} volumes...")
        self.volumes = {}
        self.masks = {}

        for image_id in image_ids:
            vol_path = self.data_dir / "train_images" / f"{image_id}.tif"
            mask_path = self.data_dir / "train_labels" / f"{image_id}.tif"

            if vol_path.exists() and mask_path.exists():
                vol = load_volume(str(vol_path))
                vol = normalize_volume(vol)
                mask = tifffile.imread(str(mask_path)).astype(np.uint8)

                self.volumes[image_id] = vol
                self.masks[image_id] = mask
                logger.info(f"  Loaded {image_id}: {vol.shape}")

        self.valid_ids = list(self.volumes.keys())
        logger.info(f"Loaded {len(self.valid_ids)} valid volumes")

    def __len__(self):
        if self.is_train:
            return len(self.valid_ids) * self.config.num_patches_per_volume
        return len(self.valid_ids)

    def get_batch(self, batch_indices: list) -> tuple:
        """Get a batch of patches."""
        batch_x = []
        batch_y = []

        for idx in batch_indices:
            if self.is_train:
                # Get volume index and extract random patch
                vol_idx = idx % len(self.valid_ids)
                image_id = self.valid_ids[vol_idx]

                vol = self.volumes[image_id]
                mask = self.masks[image_id]

                vol_patch, mask_patch = extract_random_patch(
                    vol, mask, self.config.input_shape
                )
                vol_patch, mask_patch = augment_patch(
                    vol_patch, mask_patch, self.config
                )
            else:
                # For validation, use center crop
                image_id = self.valid_ids[idx]
                vol = self.volumes[image_id]
                mask = self.masks[image_id]

                # Center crop
                d, h, w = vol.shape
                pd, ph, pw = self.config.input_shape

                d_start = max(0, (d - pd) // 2)
                h_start = max(0, (h - ph) // 2)
                w_start = max(0, (w - pw) // 2)

                vol_patch = vol[
                    d_start : d_start + pd,
                    h_start : h_start + ph,
                    w_start : w_start + pw,
                ]
                mask_patch = mask[
                    d_start : d_start + pd,
                    h_start : h_start + ph,
                    w_start : w_start + pw,
                ]

                # Pad if necessary
                if vol_patch.shape != self.config.input_shape:
                    vol_padded = np.zeros(self.config.input_shape, dtype=np.float32)
                    mask_padded = np.zeros(self.config.input_shape, dtype=np.uint8)
                    vol_padded[
                        : vol_patch.shape[0], : vol_patch.shape[1], : vol_patch.shape[2]
                    ] = vol_patch
                    mask_padded[
                        : mask_patch.shape[0],
                        : mask_patch.shape[1],
                        : mask_patch.shape[2],
                    ] = mask_patch
                    vol_patch = vol_padded
                    mask_patch = mask_padded

            # Add channel dimension: (D, H, W) -> (D, H, W, 1)
            batch_x.append(vol_patch[..., np.newaxis])
            batch_y.append(mask_patch)

        return np.array(batch_x), np.array(batch_y)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """Compute Dice loss for multi-class segmentation."""
    # y_pred: (B, D, H, W, C) logits or probs
    # y_true: (B, D, H, W) integer labels

    y_pred = keras.ops.softmax(y_pred, axis=-1)
    y_true_onehot = keras.ops.one_hot(keras.ops.cast(y_true, "int32"), y_pred.shape[-1])

    # Ignore label 2 (unlabeled) by zeroing its contribution
    # Create mask for valid labels (0 and 1)
    valid_mask = keras.ops.cast(y_true < 2, y_pred.dtype)
    valid_mask = keras.ops.expand_dims(valid_mask, axis=-1)

    intersection = keras.ops.sum(y_true_onehot * y_pred * valid_mask, axis=(1, 2, 3))
    union = keras.ops.sum((y_true_onehot + y_pred) * valid_mask, axis=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Average over classes (excluding background optionally)
    return 1.0 - keras.ops.mean(dice[:, 1:])  # Exclude background class


def ce_loss(y_true, y_pred, label_smoothing=0.0):
    """Compute cross-entropy loss ignoring unlabeled pixels."""
    # y_pred: (B, D, H, W, C) logits
    # y_true: (B, D, H, W) integer labels

    # Create mask for valid labels (not 2 = unlabeled)
    valid_mask = keras.ops.cast(y_true < 2, "float32")

    # Flatten for loss computation
    y_pred_flat = keras.ops.reshape(y_pred, (-1, y_pred.shape[-1]))
    y_true_flat = keras.ops.reshape(y_true, (-1,))
    valid_mask_flat = keras.ops.reshape(valid_mask, (-1,))

    # Clamp labels to valid range for loss computation
    y_true_clamped = keras.ops.clip(y_true_flat, 0, y_pred.shape[-1] - 1)

    # Compute per-pixel loss
    loss = keras.ops.sparse_categorical_crossentropy(
        y_true_clamped,
        y_pred_flat,
        from_logits=True,
    )

    # Apply mask and compute mean
    masked_loss = loss * valid_mask_flat
    return keras.ops.sum(masked_loss) / (keras.ops.sum(valid_mask_flat) + 1e-6)


def combo_loss(y_true, y_pred, dice_weight=0.5, ce_weight=0.5, label_smoothing=0.0):
    """Combined Dice + Cross-Entropy loss."""
    d_loss = dice_loss(y_true, y_pred)
    c_loss = ce_loss(y_true, y_pred, label_smoothing)
    return dice_weight * d_loss + ce_weight * c_loss


def get_model(config: TrainConfig):
    """Create and return the model."""
    model = TransUNet(
        input_shape=config.full_input_shape,
        encoder_name=config.encoder_name,
        classifier_activation=config.classifier_activation,
        num_classes=config.num_classes,
    )
    return model


def compute_val_metrics(
    model, val_dataset: VesuviusDataset, config: TrainConfig
) -> dict:
    """Compute validation metrics."""
    total_loss = 0.0
    total_dice = 0.0
    num_samples = 0

    indices = list(range(len(val_dataset)))

    for i in range(0, len(indices), config.batch_size):
        batch_indices = indices[i : i + config.batch_size]
        x_batch, y_batch = val_dataset.get_batch(batch_indices)

        # Forward pass
        y_pred = model(x_batch, training=False)

        # Compute losses
        if config.loss == "dice":
            loss = dice_loss(y_batch, y_pred)
        elif config.loss == "ce":
            loss = ce_loss(y_batch, y_pred, config.label_smoothing)
        else:
            loss = combo_loss(
                y_batch,
                y_pred,
                config.dice_weight,
                config.ce_weight,
                config.label_smoothing,
            )

        d_loss = dice_loss(y_batch, y_pred)

        total_loss += float(loss) * len(batch_indices)
        total_dice += float(1.0 - d_loss) * len(batch_indices)
        num_samples += len(batch_indices)

    return {
        "val_loss": total_loss / num_samples,
        "val_dice": total_dice / num_samples,
    }


def print_config(config: TrainConfig):
    """Pretty print training configuration."""
    lines = [
        "",
        "=" * 60,
        "TRAINING CONFIGURATION",
        "=" * 60,
        "",
        "Model:",
        f"  model_name:       {config.model_name}",
        f"  encoder_name:     {config.encoder_name}",
        f"  input_shape:      {config.input_shape}",
        f"  num_classes:      {config.num_classes}",
        f"  activation:       {config.classifier_activation or 'None (logits)'}",
        "",
        "Paths:",
        f"  data_dir:         {config.data_dir}",
        f"  output_dir:       {config.output_dir}",
        f"  experiment_name:  {config.experiment_name}",
        "",
        "Training:",
        f"  epochs:           {config.epochs}",
        f"  batch_size:       {config.batch_size}",
        f"  learning_rate:    {config.learning_rate}",
        f"  warmup_epochs:    {config.warmup_epochs}",
        f"  weight_decay:     {config.weight_decay}",
        f"  loss:             {config.loss}",
        f"  dice_weight:      {config.dice_weight}",
        f"  ce_weight:        {config.ce_weight}",
        f"  label_smoothing:  {config.label_smoothing}",
        "",
        "Data:",
        f"  val_split:        {config.val_split}",
        f"  patches/volume:   {config.num_patches_per_volume}",
        "",
        "Augmentation:",
        f"  use_augmentation: {config.use_augmentation}",
        f"  flip_prob:        {config.flip_prob}",
        f"  rotate_prob:      {config.rotate_prob}",
        "",
        "Misc:",
        f"  seed:             {config.seed}",
        f"  save_every:       {config.save_every}",
        f"  eval_every:       {config.eval_every}",
        "",
        "=" * 60,
        "",
    ]
    for line in lines:
        logger.info(line)


def train(config: TrainConfig):
    """Main training loop."""
    logger.info(f"Starting experiment: {config.experiment_name}")
    print_config(config)

    # Set seed
    set_seed(config.seed)

    # Load data manifest
    train_csv = Path(config.data_dir) / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)
    image_ids = df["id"].tolist()
    logger.info(f"Found {len(image_ids)} samples in train.csv")

    # Train/val split
    random.shuffle(image_ids)
    val_size = int(len(image_ids) * config.val_split)
    val_ids = image_ids[:val_size]
    train_ids = image_ids[val_size:]

    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create datasets
    train_dataset = VesuviusDataset(config.data_dir, train_ids, config, is_train=True)
    val_dataset = VesuviusDataset(config.data_dir, val_ids, config, is_train=False)

    if len(train_dataset.valid_ids) == 0:
        raise ValueError("No valid training samples found!")

    # Create model
    logger.info("Creating model...")
    model = get_model(config)
    num_params = model.count_params()
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer with warmup
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.epochs * len(train_dataset) // config.batch_size,
        alpha=0.01,
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )

    # Training history
    history = {
        "config": {
            "model_name": config.model_name,
            "encoder_name": config.encoder_name,
            "input_shape": config.input_shape,
            "num_classes": config.num_classes,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "loss": config.loss,
            "seed": config.seed,
        },
        "train_loss": [],
        "val_metrics": [],
    }

    best_val_dice = 0.0
    experiment_dir = Path(config.output_dir) / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(config.epochs):
        model.trainable = True
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle training indices each epoch
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)

        for i in range(0, len(train_indices), config.batch_size):
            batch_indices = train_indices[i : i + config.batch_size]
            x_batch, y_batch = train_dataset.get_batch(batch_indices)

            # Forward pass with gradient tape
            with keras.backend.GradientTape() as tape:
                y_pred = model(x_batch, training=True)

                if config.loss == "dice":
                    loss = dice_loss(y_batch, y_pred)
                elif config.loss == "ce":
                    loss = ce_loss(y_batch, y_pred, config.label_smoothing)
                else:
                    loss = combo_loss(
                        y_batch,
                        y_pred,
                        config.dice_weight,
                        config.ce_weight,
                        config.label_smoothing,
                    )

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            epoch_loss += float(loss)
            num_batches += 1

            if num_batches % 10 == 0:
                logger.info(
                    f"  Epoch {epoch + 1} - Batch {num_batches} - Loss: {float(loss):.4f}"
                )

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.epochs:
            logger.info("Running validation...")
            val_metrics = compute_val_metrics(model, val_dataset, config)
            history["val_metrics"].append({"epoch": epoch + 1, **val_metrics})
            logger.info(
                f"Epoch {epoch + 1} - Val Loss: {val_metrics['val_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}"
            )

            # Save best model
            if val_metrics["val_dice"] > best_val_dice:
                best_val_dice = val_metrics["val_dice"]
                best_path = experiment_dir / "best.weights.h5"
                model.save_weights(str(best_path))
                logger.info(f"New best model saved! Val Dice: {best_val_dice:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = experiment_dir / f"epoch_{epoch + 1}.weights.h5"
            model.save_weights(str(ckpt_path))
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Save final model and history
    final_path = experiment_dir / "final.weights.h5"
    model.save_weights(str(final_path))

    history_path = experiment_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! Best Val Dice: {best_val_dice:.4f}")
    logger.info(f"Results saved to {experiment_dir}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TransUNet for Vesuvius surface detection"
    )

    # Paths
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default="")

    # Model
    parser.add_argument(
        "--encoder", type=str, default="seresnext50", help="Encoder backbone"
    )
    parser.add_argument(
        "--input-size", type=int, default=160, help="Input patch size (cubic)"
    )
    parser.add_argument("--num-classes", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--loss", type=str, default="combo", choices=["dice", "ce", "combo"]
    )
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # Data
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patches-per-volume", type=int, default=4)

    # Augmentation
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--rotate-prob", type=float, default=0.5)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        encoder_name=args.encoder,
        input_shape=(args.input_size, args.input_size, args.input_size),
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        loss=args.loss,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        label_smoothing=args.label_smoothing,
        val_split=args.val_split,
        num_patches_per_volume=args.patches_per_volume,
        use_augmentation=not args.no_augmentation,
        flip_prob=args.flip_prob,
        rotate_prob=args.rotate_prob,
        seed=args.seed,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )

    train(config)


if __name__ == "__main__":
    main()
