#!/usr/bin/env python3
"""Vesuvius Challenge Surface Detection - PyTorch Inference Script"""

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import remove_small_objects

# Configuration defaults
NUM_CLASSES = 3
INPUT_SHAPE = (128, 128, 128)
ROOT_DIR = "/kaggle/input/vesuvius-challenge-surface-detection"
TEST_DIR = f"{ROOT_DIR}/test_images"
OUTPUT_DIR = "/kaggle/working/submission_masks"
ZIP_PATH = "/kaggle/working/submission.zip"


# =============================================================================
# Model Definitions (must match train.py)
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


class Simple3DUNet(nn.Module):
    """Simple 3D UNet for baseline experiments."""

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
    """3D TransUNet: UNet with Transformer bottleneck."""

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


# =============================================================================
# Sliding Window Inference
# =============================================================================


class SlidingWindowInference:
    """
    Sliding window inference for 3D volumes with Gaussian weighting.
    """

    def __init__(
        self,
        model,
        roi_size,
        overlap=0.5,
        mode="gaussian",
        device=None,
    ):
        self.model = model
        self.roi_size = roi_size if isinstance(roi_size, tuple) else (roi_size,) * 3
        self.overlap = overlap
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create importance map for Gaussian weighting
        if mode == "gaussian":
            self.importance_map = self._create_gaussian_importance()
        else:
            self.importance_map = np.ones(self.roi_size, dtype=np.float32)

    def _create_gaussian_importance(self):
        """Create a Gaussian importance map for blending overlapping patches."""
        sigma = [s / 4 for s in self.roi_size]
        center = [(s - 1) / 2 for s in self.roi_size]

        importance = np.ones(self.roi_size, dtype=np.float32)

        for d in range(self.roi_size[0]):
            for h in range(self.roi_size[1]):
                for w in range(self.roi_size[2]):
                    dist = sum(((x - c) / s) ** 2 for x, c, s in zip([d, h, w], center, sigma))
                    importance[d, h, w] = np.exp(-0.5 * dist)

        # Normalize
        importance = importance / importance.max()
        importance = np.clip(importance, 0.01, 1.0)  # Avoid zeros

        return importance

    def __call__(self, volume):
        """
        Run sliding window inference on a volume.

        Args:
            volume: numpy array of shape (D, H, W) or (1, D, H, W)

        Returns:
            numpy array of shape (D, H, W, C) with logits
        """
        # Handle input shape
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]  # Add channel dim
        if volume.ndim == 4:
            # (C, D, H, W) -> keep as is
            pass

        _, D, H, W = volume.shape
        rd, rh, rw = self.roi_size

        # Calculate step sizes based on overlap
        step_d = max(1, int(rd * (1 - self.overlap)))
        step_h = max(1, int(rh * (1 - self.overlap)))
        step_w = max(1, int(rw * (1 - self.overlap)))

        # Get number of classes from model output
        with torch.no_grad():
            test_input = torch.zeros(1, 1, *self.roi_size, device=self.device)
            test_out = self.model(test_input)
            num_classes = test_out.shape[1]

        # Initialize output and count arrays
        output = np.zeros((D, H, W, num_classes), dtype=np.float32)
        count = np.zeros((D, H, W), dtype=np.float32)

        # Generate patch positions
        d_positions = list(range(0, max(1, D - rd + 1), step_d))
        h_positions = list(range(0, max(1, H - rh + 1), step_h))
        w_positions = list(range(0, max(1, W - rw + 1), step_w))

        # Ensure we cover the entire volume
        if D > rd and d_positions[-1] + rd < D:
            d_positions.append(D - rd)
        if H > rh and h_positions[-1] + rh < H:
            h_positions.append(H - rh)
        if W > rw and w_positions[-1] + rw < W:
            w_positions.append(W - rw)

        self.model.eval()
        with torch.no_grad():
            for d_start in d_positions:
                for h_start in h_positions:
                    for w_start in w_positions:
                        # Extract patch
                        patch = volume[
                            :,
                            d_start : d_start + rd,
                            h_start : h_start + rh,
                            w_start : w_start + rw,
                        ]

                        # Pad if necessary (for edge cases)
                        pad_d = rd - patch.shape[1]
                        pad_h = rh - patch.shape[2]
                        pad_w = rw - patch.shape[3]

                        if pad_d > 0 or pad_h > 0 or pad_w > 0:
                            patch = np.pad(
                                patch,
                                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                                mode="constant",
                            )

                        # Run inference
                        patch_tensor = torch.from_numpy(patch[np.newaxis, ...]).to(self.device)
                        pred = self.model(patch_tensor)
                        pred = pred.cpu().numpy()[0]  # (C, D, H, W)
                        pred = pred.transpose(1, 2, 3, 0)  # (D, H, W, C)

                        # Remove padding from prediction
                        if pad_d > 0:
                            pred = pred[:-pad_d, :, :, :]
                        if pad_h > 0:
                            pred = pred[:, :-pad_h, :, :]
                        if pad_w > 0:
                            pred = pred[:, :, :-pad_w, :]

                        # Get valid importance map slice
                        imp = self.importance_map[: pred.shape[0], : pred.shape[1], : pred.shape[2]]

                        # Accumulate weighted predictions
                        d_end = d_start + pred.shape[0]
                        h_end = h_start + pred.shape[1]
                        w_end = w_start + pred.shape[2]

                        output[d_start:d_end, h_start:h_end, w_start:w_end, :] += pred * imp[..., np.newaxis]
                        count[d_start:d_end, h_start:h_end, w_start:w_end] += imp

        # Normalize by count
        count = np.maximum(count, 1e-8)
        output = output / count[..., np.newaxis]

        return output


# =============================================================================
# Preprocessing and Postprocessing
# =============================================================================


def normalize_volume(volume):
    """Apply intensity normalization (nonzero mean/std)."""
    nonzero_mask = volume > 0
    if nonzero_mask.sum() > 0:
        mean = volume[nonzero_mask].mean()
        std = volume[nonzero_mask].std()
        if std > 0:
            volume = (volume - mean) / std
    return volume


def load_volume(path):
    """Load a TIFF volume and normalize."""
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = normalize_volume(vol)
    return vol


def build_anisotropic_struct(z_radius: int, xy_radius: int):
    """Build an anisotropic structuring element for morphological operations."""
    z, r = z_radius, xy_radius
    if z == 0 and r == 0:
        return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct
    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct


def topo_postprocess(
    probs,
    T_low=0.90,
    T_high=0.90,
    z_radius=2,
    xy_radius=0,
    dust_min_size=100,
):
    """Apply topological post-processing with hysteresis thresholding."""
    # Step 1: 3D Hysteresis
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


# =============================================================================
# TTA and Inference Pipeline
# =============================================================================


def predict_with_tta(volume, swi, use_tta=True):
    """
    Predict with test-time augmentation (flips and rotations).

    Args:
        volume: numpy array of shape (D, H, W)
        swi: SlidingWindowInference instance
        use_tta: whether to use TTA

    Returns:
        numpy array of shape (D, H, W) with class 1 probabilities
    """
    # Add channel dimension: (D, H, W) -> (1, D, H, W)
    inputs = volume[np.newaxis, ...]

    logits_list = []

    # Original
    out = swi(inputs)  # (D, H, W, C)
    logits_list.append(out)

    if use_tta:
        # Flips (spatial only, axis indices for (1, D, H, W))
        for axis in [1, 2, 3]:
            img_f = np.flip(inputs, axis=axis)
            p = swi(img_f)
            p = np.flip(p, axis=axis - 1)  # Output is (D, H, W, C), so adjust axis
            logits_list.append(p)

        # Axial rotations (H, W) - indices 2, 3 for input, 1, 2 for output
        for k in [1, 2, 3]:
            img_r = np.rot90(inputs, k=k, axes=(2, 3))
            p = swi(img_r)
            p = np.rot90(p, k=-k, axes=(1, 2))  # Output is (D, H, W, C)
            logits_list.append(p)

    # Average logits
    mean_logits = np.mean(logits_list, axis=0)

    # Get class 1 probability (foreground)
    probs = np.exp(mean_logits) / np.exp(mean_logits).sum(axis=-1, keepdims=True)
    class1_prob = probs[..., 1]

    return class1_prob


def inference_pipeline(
    volume,
    swi,
    use_tta=True,
    T_low=0.30,
    T_high=0.80,
    z_radius=4,
    xy_radius=2,
    dust_min_size=100,
):
    """Run the full inference pipeline with TTA and post-processing."""
    probs = predict_with_tta(volume, swi, use_tta=use_tta)
    final = topo_postprocess(
        probs,
        T_low=T_low,
        T_high=T_high,
        z_radius=z_radius,
        xy_radius=xy_radius,
        dust_min_size=dust_min_size,
    )
    return final


# =============================================================================
# Model Loading
# =============================================================================


def get_model(weights_path, device=None):
    """Load model from checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "Simple3DUNet")
    num_classes = config.get("num_classes", 3)
    base_channels = config.get("base_channels", 24)

    # Create model
    if model_name == "Simple3DUNet":
        model = Simple3DUNet(
            in_channels=1,
            num_classes=num_classes,
            base_channels=base_channels,
            classifier_activation=None,
        )
    elif model_name == "TransUNet":
        model = TransUNet3D(
            in_channels=1,
            num_classes=num_classes,
            base_channels=32,
            num_transformer_layers=4,
            num_heads=8,
            classifier_activation=None,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get input shape from config
    input_shape = config.get("input_shape", (128, 128, 128))
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)

    return model, input_shape, config


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Vesuvius Surface Detection - Inference")

    # Paths
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=TEST_DIR,
        help="Directory containing test TIFF volumes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to test.csv (optional, will process all TIFs in input-dir if not provided)",
    )

    # Inference settings
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap for sliding window inference",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")

    # Post-processing
    parser.add_argument("--t-low", type=float, default=0.30, help="Low threshold for hysteresis")
    parser.add_argument("--t-high", type=float, default=0.80, help="High threshold for hysteresis")
    parser.add_argument("--z-radius", type=int, default=4, help="Z radius for morphological closing")
    parser.add_argument("--xy-radius", type=int, default=2, help="XY radius for morphological closing")
    parser.add_argument("--dust-min-size", type=int, default=100, help="Min size for dust removal")

    # Output format
    parser.add_argument("--create-zip", action="store_true", help="Create submission ZIP")
    parser.add_argument("--zip-path", type=str, default=ZIP_PATH, help="Path for submission ZIP")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"Loading model from {args.weights}...")
    model, input_shape, config = get_model(args.weights, device)
    num_params = count_parameters(model)
    print(f"Model: {config.get('model_name', 'Unknown')}")
    print(f"Parameters: {num_params:,}")
    print(f"Input shape: {input_shape}")

    # Setup sliding window inference
    swi = SlidingWindowInference(
        model,
        roi_size=input_shape,
        overlap=args.overlap,
        mode="gaussian",
        device=device,
    )

    # Get list of volumes to process
    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        image_ids = df["id"].tolist()
        tif_paths = [os.path.join(args.input_dir, f"{id}.tif") for id in image_ids]
    else:
        # Process all TIF files in input directory
        tif_paths = sorted(Path(args.input_dir).glob("*.tif"))
        image_ids = [p.stem for p in tif_paths]

    print(f"Processing {len(tif_paths)} volumes...")

    # Setup ZIP if requested
    zip_file = None
    if args.create_zip:
        zip_file = zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED)

    # Run inference
    for image_id, tif_path in zip(image_ids, tif_paths):
        print(f"Processing {image_id}...")

        # Load and normalize volume
        volume = load_volume(str(tif_path))
        print(f"  Volume shape: {volume.shape}")

        # Run inference pipeline
        output = inference_pipeline(
            volume,
            swi,
            use_tta=not args.no_tta,
            T_low=args.t_low,
            T_high=args.t_high,
            z_radius=args.z_radius,
            xy_radius=args.xy_radius,
            dust_min_size=args.dust_min_size,
        )

        # Save output
        out_path = os.path.join(args.output, f"{image_id}.tif")
        tifffile.imwrite(out_path, output.astype(np.uint8))
        print(f"  Saved: {out_path}")

        # Add to ZIP if requested
        if zip_file:
            zip_file.write(out_path, arcname=f"{image_id}.tif")

    # Close ZIP
    if zip_file:
        zip_file.close()
        print(f"Submission ZIP: {args.zip_path}")

    print("Done!")


if __name__ == "__main__":
    main()
