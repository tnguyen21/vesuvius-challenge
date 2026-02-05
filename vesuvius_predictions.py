#!/usr/bin/env python3
"""Vesuvius Challenge Surface Detection - Inference Script"""

import os
import zipfile

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from matplotlib import pyplot as plt
from medicai.models import SegFormer, TransUNet
from medicai.transforms import Compose, NormalizeIntensity, ScaleIntensityRange
from medicai.utils.inference import SlidingWindowInference
from skimage.morphology import remove_small_objects

# Configuration
TTA = 1
NUM_CLASSES = 3
INPUT_SHAPE = (160, 160, 160)
ROOT_DIR = "/kaggle/input/vesuvius-challenge-surface-detection"
TEST_DIR = f"{ROOT_DIR}/test_images"
OUTPUT_DIR = "/kaggle/working/submission_masks"
ZIP_PATH = "/kaggle/working/submission.zip"
KAGGLE_MODEL_PATH = "/kaggle/input/vsd-model/keras/"


def val_transformation(image):
    """Apply validation transformations to input image."""
    data = {"image": image}
    pipeline = Compose([
        NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
    ])
    result = pipeline(data)
    return result["image"]


def get_model():
    """Load and return the TransUNet model with pretrained weights."""
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name="seresnext50",
        classifier_activation=None,
        num_classes=3,
    )
    model.load_weights(f"{KAGGLE_MODEL_PATH}/transunet/3/transunet.seresnext50.160px.comboloss.weights.h5")
    return model


def load_volume(path):
    """Load a TIFF volume and prepare it for inference."""
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = vol[None, ..., None]
    return vol


def predict_with_tta(inputs, swi):
    """Predict with test-time augmentation (flips and rotations)."""
    logits = []

    # Original
    logits.append(swi(inputs))

    # Flips (spatial only)
    for axis in [1, 2, 3]:
        img_f = np.flip(inputs, axis=axis)
        p = swi(img_f)
        p = np.flip(p, axis=axis)
        logits.append(p)

    # Axial rotations (H, W)
    for k in [1, 2, 3]:
        img_r = np.rot90(inputs, k=k, axes=(2, 3))
        p = swi(img_r)
        p = np.rot90(p, k=-k, axes=(2, 3))
        logits.append(p)

    mean_logits = np.mean(logits, axis=0)
    return mean_logits.argmax(-1).astype(np.uint8).squeeze()


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
    """Apply topological post-processing with hysteresis thresholding and morphological operations."""
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


def inference_pipeline(
    volume,
    swi,
    T_low=0.30,
    T_high=0.80,
    z_radius=4,
    xy_radius=2,
    dust_min_size=100,
):
    """Run the full inference pipeline with TTA and post-processing."""
    probs = predict_with_tta(volume, swi)
    final = topo_postprocess(
        probs,
        T_low=T_low,
        T_high=T_high,
        z_radius=z_radius,
        xy_radius=xy_radius,
        dust_min_size=dust_min_size,
    )
    return final


def plot_sample(x, y, sample_idx=0, max_slices=16):
    """Plot sample slices from volume and mask for visualization."""
    img = np.squeeze(x[sample_idx])
    mask = np.squeeze(y[sample_idx])
    D = img.shape[0]

    step = max(1, D // max_slices)
    slices = range(0, D, step)

    n_slices = len(slices)
    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))

    for i, s in enumerate(slices):
        axes[0, i].imshow(img[s], cmap="gray")
        axes[0, i].set_title(f"Slice {s}")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask[s], cmap="gray")
        axes[1, i].set_title(f"Mask {s}")
        axes[1, i].axis("off")

    plt.suptitle(f"Sample {sample_idx}")
    plt.tight_layout()
    plt.show()


def main():
    """Main inference function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(f"{ROOT_DIR}/test.csv")
    print(f"Loaded {len(test_df)} test samples")

    # Load model
    print("Loading model...")
    model = get_model()
    print(f"Model parameters: {model.count_params() / 1e6:.2f}M")

    # Setup sliding window inference
    swi = SlidingWindowInference(
        model,
        num_classes=NUM_CLASSES,
        roi_size=INPUT_SHAPE,
        sw_batch_size=1,
        mode="gaussian",
        overlap=0.40,
    )

    # Run inference and create submission
    print("Running inference...")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for image_id in test_df["id"]:
            tif_path = f"{TEST_DIR}/{image_id}.tif"
            print(f"Processing {image_id}...")

            volume = load_volume(tif_path)
            volume = val_transformation(volume)
            output = inference_pipeline(volume, swi)

            out_path = f"{OUTPUT_DIR}/{image_id}.tif"
            tifffile.imwrite(out_path, output.astype(np.uint8))

            z.write(out_path, arcname=f"{image_id}.tif")
            os.remove(out_path)

    print(f"Submission ZIP: {ZIP_PATH}")


if __name__ == "__main__":
    main()
