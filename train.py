#!/usr/bin/env python3
"""
TransUNet Fine-tuning Script for Vesuvius Surface Detection

3D segmentation of papyrus surfaces in CT scans of Herculaneum scrolls.
PyTorch + CUDA implementation.
"""

import argparse
import collections
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "data"
    output_dir: str = "checkpoints"
    experiment_name: str = ""

    # Model
    model_name: str = "Simple3DUNet"  # "Simple3DUNet" or "TransUNet"
    encoder_name: str = "seresnext50"
    input_shape: tuple = (128, 128, 128)  # Smaller default for faster baseline
    num_classes: int = 3
    base_channels: int = 24  # Base channel count for UNet
    classifier_activation: Optional[str] = None  # None for logits, "softmax" for probs

    # Training
    epochs: int = 15  # Fewer epochs for baseline
    batch_size: int = 2  # Can fit larger batch with smaller model
    learning_rate: float = 1e-4
    warmup_epochs: int = 2  # Shorter warmup
    weight_decay: float = 1e-5
    loss: str = "combo"  # "dice", "ce", "combo"
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    label_smoothing: float = 0.0

    # Data
    val_split: float = 0.2
    num_patches_per_volume: int = 8  # More patches for faster iteration
    overlap: float = 0.25  # Patch overlap during validation
    max_samples: int = 0  # 0 = use all samples, >0 = limit total samples (for memory)
    cache_size: int = 32  # Number of volumes to keep in LRU cache (0 = unlimited)

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
        self.full_input_shape = (1, *self.input_shape)  # (C, D, H, W)

        if not self.experiment_name:
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_volume(path: str) -> np.ndarray:
    """Load a TIFF volume."""
    vol = tifffile.imread(path)
    return vol.astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Apply intensity normalization (nonzero mean/std)."""
    nonzero_mask = volume > 0
    if nonzero_mask.sum() > 0:
        mean = volume[nonzero_mask].mean()
        std = volume[nonzero_mask].std()
        if std > 0:
            volume = (volume - mean) / std
    return volume


def extract_random_patch(volume: np.ndarray, mask: np.ndarray, patch_size: tuple) -> tuple:
    """Extract a random patch from volume and mask."""
    d, h, w = volume.shape
    pd, ph, pw = patch_size

    # Random start position
    d_start = random.randint(0, max(0, d - pd))
    h_start = random.randint(0, max(0, h - ph))
    w_start = random.randint(0, max(0, w - pw))

    # Extract patches
    vol_patch = volume[d_start : d_start + pd, h_start : h_start + ph, w_start : w_start + pw]
    mask_patch = mask[d_start : d_start + pd, h_start : h_start + ph, w_start : w_start + pw]

    # Pad if necessary
    if vol_patch.shape != patch_size:
        vol_padded = np.zeros(patch_size, dtype=np.float32)
        mask_padded = np.zeros(patch_size, dtype=np.uint8)
        vol_padded[: vol_patch.shape[0], : vol_patch.shape[1], : vol_patch.shape[2]] = vol_patch
        mask_padded[: mask_patch.shape[0], : mask_patch.shape[1], : mask_patch.shape[2]] = mask_patch
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

        # Lazy loading: only scan filesystem for valid pairs, load on demand
        self.valid_ids = []
        for image_id in image_ids:
            vol_path = self.data_dir / "train_images" / f"{image_id}.tif"
            mask_path = self.data_dir / "train_labels" / f"{image_id}.tif"
            if vol_path.exists() and mask_path.exists():
                self.valid_ids.append(image_id)

        logger.info(f"Found {len(self.valid_ids)} valid volumes (lazy loading, cache_size={config.cache_size})")

        # LRU cache: OrderedDict maintains insertion order, evict oldest when full
        self._cache: collections.OrderedDict[str, tuple[np.ndarray, np.ndarray]] = collections.OrderedDict()
        self._cache_size = config.cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_volume(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load a volume+mask pair, using LRU cache."""
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
            self._cache_hits += 1
            return self._cache[image_id]

        self._cache_misses += 1
        vol_path = self.data_dir / "train_images" / f"{image_id}.tif"
        mask_path = self.data_dir / "train_labels" / f"{image_id}.tif"

        vol = load_volume(str(vol_path))
        vol = normalize_volume(vol)
        mask = tifffile.imread(str(mask_path)).astype(np.uint8)

        # Evict oldest if cache is full
        if self._cache_size > 0 and len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

        self._cache[image_id] = (vol, mask)
        return vol, mask

    def __len__(self):
        if self.is_train:
            return len(self.valid_ids) * self.config.num_patches_per_volume
        return len(self.valid_ids)

    def get_batch(self, batch_indices: list, device: torch.device) -> tuple:
        """Get a batch of patches as tensors."""
        batch_x = []
        batch_y = []

        for idx in batch_indices:
            if self.is_train:
                # Get volume index and extract random patch
                vol_idx = idx % len(self.valid_ids)
                image_id = self.valid_ids[vol_idx]

                vol, mask = self._load_volume(image_id)

                vol_patch, mask_patch = extract_random_patch(vol, mask, self.config.input_shape)
                vol_patch, mask_patch = augment_patch(vol_patch, mask_patch, self.config)
            else:
                # For validation, use center crop
                image_id = self.valid_ids[idx]
                vol, mask = self._load_volume(image_id)

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
                    vol_padded[: vol_patch.shape[0], : vol_patch.shape[1], : vol_patch.shape[2]] = vol_patch
                    mask_padded[
                        : mask_patch.shape[0],
                        : mask_patch.shape[1],
                        : mask_patch.shape[2],
                    ] = mask_patch
                    vol_patch = vol_padded
                    mask_patch = mask_padded

            # Add channel dimension: (D, H, W) -> (1, D, H, W)
            batch_x.append(vol_patch[np.newaxis, ...])
            batch_y.append(mask_patch)

        x = torch.from_numpy(np.array(batch_x)).to(device)
        y = torch.from_numpy(np.array(batch_y)).long().to(device)
        return x, y


# =============================================================================
# 3D UNet with Transformer (TransUNet-style) for PyTorch
# =============================================================================


class ConvBlock3D(nn.Module):
    """3D convolutional block with batch norm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = x.view(b, c, -1).mean(dim=2)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y


class ResBlock3D(nn.Module):
    """Residual block with SE attention."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNet3D(nn.Module):
    """
    3D TransUNet: UNet with Transformer bottleneck.
    Simplified implementation for 3D medical image segmentation.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        classifier_activation=None,
    ):
        super().__init__()
        self.classifier_activation = classifier_activation

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ResBlock3D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck with Transformer
        self.bottleneck_conv = ConvBlock3D(base_channels * 8, base_channels * 16)

        # Transformer layers
        embed_dim = base_channels * 16
        self.transformer_layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_transformer_layers)])
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        # Output
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck_conv(self.pool4(e4))

        # Reshape for transformer: (B, C, D, H, W) -> (B, D*H*W, C)
        b_shape = b.shape
        b_flat = b.flatten(2).transpose(1, 2)

        # Transformer layers
        for layer in self.transformer_layers:
            b_flat = layer(b_flat)
        b_flat = self.transformer_norm(b_flat)

        # Reshape back: (B, D*H*W, C) -> (B, C, D, H, W)
        b = b_flat.transpose(1, 2).view(b_shape)

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)

        if self.classifier_activation == "softmax":
            out = F.softmax(out, dim=1)

        return out


class Simple3DUNet(nn.Module):
    """
    Simple 3D UNet for baseline experiments.
    No transformer layers, no SE-attention, just vanilla UNet.
    ~2.5M params with base_channels=24 (vs ~10M for TransUNet3D).
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_channels=24,
        classifier_activation=None,
    ):
        super().__init__()
        self.classifier_activation = classifier_activation
        bc = base_channels

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, bc)
        self.enc2 = ConvBlock3D(bc, bc * 2)
        self.enc3 = ConvBlock3D(bc * 2, bc * 4)
        self.enc4 = ConvBlock3D(bc * 4, bc * 8)

        # Bottleneck
        self.bottleneck = ConvBlock3D(bc * 8, bc * 16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(bc * 16, bc * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(bc * 16, bc * 8)

        self.up3 = nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(bc * 8, bc * 4)

        self.up2 = nn.ConvTranspose3d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(bc * 4, bc * 2)

        self.up1 = nn.ConvTranspose3d(bc * 2, bc, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(bc * 2, bc)

        # Output
        self.out_conv = nn.Conv3d(bc, num_classes, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)

        if self.classifier_activation == "softmax":
            out = F.softmax(out, dim=1)

        return out


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6):
    """
    Compute Dice loss for multi-class segmentation.

    Args:
        y_pred: (B, C, D, H, W) logits
        y_true: (B, D, H, W) integer labels
    """
    num_classes = y_pred.shape[1]
    y_pred_soft = F.softmax(y_pred, dim=1)

    # One-hot encode targets
    y_true_onehot = F.one_hot(y_true.clamp(0, num_classes - 1), num_classes)
    y_true_onehot = y_true_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

    # Create mask for valid labels (0 and 1, not 2=unlabeled)
    valid_mask = (y_true < 2).unsqueeze(1).float()  # (B, 1, D, H, W)

    # Compute dice per class
    intersection = (y_true_onehot * y_pred_soft * valid_mask).sum(dim=(2, 3, 4))
    union = ((y_true_onehot + y_pred_soft) * valid_mask).sum(dim=(2, 3, 4))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Average over classes, excluding background (class 0)
    return 1.0 - dice[:, 1:].mean()


def ce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, label_smoothing: float = 0.0):
    """
    Compute cross-entropy loss ignoring unlabeled pixels.

    Args:
        y_pred: (B, C, D, H, W) logits
        y_true: (B, D, H, W) integer labels
    """
    num_classes = y_pred.shape[1]

    # Create mask for valid labels (not 2 = unlabeled)
    valid_mask = (y_true < 2).float()

    # Clamp labels to valid range
    y_true_clamped = y_true.clamp(0, num_classes - 1)

    # Compute per-pixel loss
    loss = F.cross_entropy(y_pred, y_true_clamped, reduction="none", label_smoothing=label_smoothing)

    # Apply mask and compute mean
    masked_loss = loss * valid_mask
    return masked_loss.sum() / (valid_mask.sum() + 1e-6)


def combo_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    label_smoothing: float = 0.0,
):
    """Combined Dice + Cross-Entropy loss."""
    d_loss = dice_loss(y_pred, y_true)
    c_loss = ce_loss(y_pred, y_true, label_smoothing)
    return dice_weight * d_loss + ce_weight * c_loss


def get_model(config: TrainConfig, device: torch.device) -> nn.Module:
    """Create and return the model."""
    if config.model_name == "Simple3DUNet":
        model = Simple3DUNet(
            in_channels=1,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            classifier_activation=config.classifier_activation,
        )
    elif config.model_name == "TransUNet":
        model = TransUNet3D(
            in_channels=1,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            num_transformer_layers=4,
            num_heads=8,
            classifier_activation=config.classifier_activation,
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def compute_val_metrics(
    model: nn.Module,
    val_dataset: VesuviusDataset,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    """Compute validation metrics."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_samples = 0

    indices = list(range(len(val_dataset)))

    for i in range(0, len(indices), config.batch_size):
        batch_indices = indices[i : i + config.batch_size]
        x_batch, y_batch = val_dataset.get_batch(batch_indices, device)

        # Forward pass
        y_pred = model(x_batch)

        # Compute losses
        if config.loss == "dice":
            loss = dice_loss(y_pred, y_batch)
        elif config.loss == "ce":
            loss = ce_loss(y_pred, y_batch, config.label_smoothing)
        else:
            loss = combo_loss(
                y_pred,
                y_batch,
                config.dice_weight,
                config.ce_weight,
                config.label_smoothing,
            )

        d_loss = dice_loss(y_pred, y_batch)

        total_loss += loss.item() * len(batch_indices)
        total_dice += (1.0 - d_loss.item()) * len(batch_indices)
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
        f"  base_channels:    {config.base_channels}",
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

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled TF32 and cuDNN benchmark")

    # Load data manifest
    train_csv = Path(config.data_dir) / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)
    image_ids = df["id"].tolist()
    logger.info(f"Found {len(image_ids)} samples in train.csv")

    # Apply max_samples limit if set (for memory-constrained systems)
    if config.max_samples > 0 and len(image_ids) > config.max_samples:
        logger.info(f"Limiting to {config.max_samples} samples (from {len(image_ids)})")
        image_ids = image_ids[: config.max_samples]

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
    model = get_model(config, device)
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler with warmup
    total_steps = config.epochs * len(train_dataset) // config.batch_size
    warmup_steps = config.warmup_epochs * len(train_dataset) // config.batch_size

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Training history
    history = {
        "config": {
            "model_name": config.model_name,
            "encoder_name": config.encoder_name,
            "base_channels": config.base_channels,
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
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle training indices each epoch
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)

        for i in range(0, len(train_indices), config.batch_size):
            batch_indices = train_indices[i : i + config.batch_size]
            x_batch, y_batch = train_dataset.get_batch(batch_indices, device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)

            if config.loss == "dice":
                loss = dice_loss(y_pred, y_batch)
            elif config.loss == "ce":
                loss = ce_loss(y_pred, y_batch, config.label_smoothing)
            else:
                loss = combo_loss(
                    y_pred,
                    y_batch,
                    config.dice_weight,
                    config.ce_weight,
                    config.label_smoothing,
                )

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                logger.info(f"  Epoch {epoch + 1} - Batch {num_batches} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.epochs:
            logger.info("Running validation...")
            val_metrics = compute_val_metrics(model, val_dataset, config, device)
            history["val_metrics"].append({"epoch": epoch + 1, **val_metrics})
            logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['val_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}")

            # Save best model
            if val_metrics["val_dice"] > best_val_dice:
                best_val_dice = val_metrics["val_dice"]
                best_path = experiment_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dice": best_val_dice,
                        "config": config.__dict__,
                    },
                    best_path,
                )
                logger.info(f"New best model saved! Val Dice: {best_val_dice:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = experiment_dir / f"epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Save final model and history
    final_path = experiment_dir / "final.pt"
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
        },
        final_path,
    )

    history_path = experiment_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! Best Val Dice: {best_val_dice:.4f}")
    logger.info(f"Results saved to {experiment_dir}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train TransUNet for Vesuvius surface detection")

    # Paths
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default="")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Simple3DUNet",
        choices=["Simple3DUNet", "TransUNet"],
        help="Model architecture",
    )
    parser.add_argument("--encoder", type=str, default="seresnext50", help="Encoder backbone (TransUNet only)")
    parser.add_argument("--input-size", type=int, default=128, help="Input patch size (cubic)")
    parser.add_argument("--base-channels", type=int, default=24, help="Base channels for UNet")
    parser.add_argument("--num-classes", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", type=str, default="combo", choices=["dice", "ce", "combo"])
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # Data
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patches-per-volume", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to load (0=all, for memory limits)")
    parser.add_argument("--cache-size", type=int, default=32, help="LRU cache size for lazy volume loading (0=unlimited)")

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
        model_name=args.model,
        encoder_name=args.encoder,
        input_shape=(args.input_size, args.input_size, args.input_size),
        num_classes=args.num_classes,
        base_channels=args.base_channels,
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
        max_samples=args.max_samples,
        cache_size=args.cache_size,
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
