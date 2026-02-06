#!/usr/bin/env python3
"""
Competition Metrics Evaluation for Vesuvius Surface Detection

Computes the three competition metrics:
- TopoScore (Betti matching) - weight 0.30
- SurfaceDice@τ (τ=2.0) - weight 0.35
- VOI_score (instance consistency) - weight 0.35

Final score = 0.30 × topo_score + 0.35 × surface_dice + 0.35 × voi_score

Usage:
    python scripts/eval_metrics.py --pred output/experiment_name --gt data/train_labels
    python scripts/eval_metrics.py --pred output/baseline --gt data/train_labels --update-log baseline
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_volume(path: str) -> np.ndarray:
    """Load a TIFF volume."""
    return tifffile.imread(path)


def compute_surface_dice(pred: np.ndarray, gt: np.ndarray, tau: float = 2.0) -> float:
    """
    Compute Surface Dice coefficient with tolerance τ.

    Surface Dice measures how well predicted boundaries match ground truth boundaries,
    with tolerance for small displacements.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask
        tau: Tolerance distance in voxels (default 2.0)

    Returns:
        Surface Dice score in [0, 1]
    """
    # Get surface voxels (boundary pixels)
    # Note: XOR on uint8 returns uint8, need to convert to bool for proper indexing
    pred_surface = (pred ^ ndimage.binary_erosion(pred)).astype(bool)
    gt_surface = (gt ^ ndimage.binary_erosion(gt)).astype(bool)

    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_surface)
    gt_dist = distance_transform_edt(~gt_surface)

    # Count surface voxels within tolerance of each other
    pred_surface_coords = np.argwhere(pred_surface)
    gt_surface_coords = np.argwhere(gt_surface)

    if len(pred_surface_coords) == 0 and len(gt_surface_coords) == 0:
        return 1.0  # Both empty
    if len(pred_surface_coords) == 0 or len(gt_surface_coords) == 0:
        return 0.0  # One empty, one not

    # Predicted surface voxels within tau of GT surface
    pred_within_tau = np.sum(gt_dist[pred_surface] <= tau)

    # GT surface voxels within tau of predicted surface
    gt_within_tau = np.sum(pred_dist[gt_surface] <= tau)

    # Surface Dice
    surface_dice = (pred_within_tau + gt_within_tau) / (len(pred_surface_coords) + len(gt_surface_coords))

    return float(surface_dice)


def compute_voi(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Variation of Information (VOI) score.

    VOI measures instance-level consistency, penalizing splits and merges
    between predicted and ground truth components.

    Args:
        pred: Binary prediction mask (will be labeled into components)
        gt: Binary ground truth mask (will be labeled into components)

    Returns:
        VOI score in [0, 1] (higher is better, 1 - normalized VOI)
    """
    # Label connected components
    pred_labels, num_pred = ndimage.label(pred)
    gt_labels, num_gt = ndimage.label(gt)

    # Flatten for contingency table computation
    pred_flat = pred_labels.ravel()
    gt_flat = gt_labels.ravel()

    # Compute contingency table
    n = len(pred_flat)
    max_pred = pred_flat.max() + 1
    max_gt = gt_flat.max() + 1

    # Contingency matrix: C[i,j] = count of voxels with pred label i and gt label j
    contingency = np.zeros((max_pred, max_gt), dtype=np.float64)
    for p, g in zip(pred_flat, gt_flat):
        contingency[p, g] += 1

    # Marginals
    pred_marginal = contingency.sum(axis=1)
    gt_marginal = contingency.sum(axis=0)

    # Compute entropies
    def entropy(counts, n):
        """Compute entropy from counts."""
        p = counts[counts > 0] / n
        return -np.sum(p * np.log2(p))

    h_pred = entropy(pred_marginal, n)
    h_gt = entropy(gt_marginal, n)

    # Mutual information
    nonzero = contingency > 0
    mi = np.sum(contingency[nonzero] / n * np.log2(contingency[nonzero] * n / (pred_marginal[:, None] * gt_marginal)[nonzero]))

    # VOI = H(pred|gt) + H(gt|pred) = H(pred) + H(gt) - 2*MI
    voi = h_pred + h_gt - 2 * mi

    # Normalize to [0, 1] and invert (so higher is better)
    max_voi = h_pred + h_gt  # Maximum VOI when MI=0
    if max_voi == 0:
        return 1.0  # Both have single component
    voi_normalized = voi / max_voi
    voi_score = 1.0 - voi_normalized

    return float(voi_score)


def compute_betti_numbers(binary_mask: np.ndarray) -> tuple:
    """
    Compute Betti numbers (b0, b1, b2) for a 3D binary mask.

    - b0: Number of connected components
    - b1: Number of tunnels/handles
    - b2: Number of cavities/voids

    This is a simplified approximation using Euler characteristic.
    For exact computation, use a proper persistent homology library.

    Args:
        binary_mask: 3D binary array

    Returns:
        (b0, b1, b2) tuple
    """
    # b0: connected components in foreground
    _, b0 = ndimage.label(binary_mask)

    # For b1 and b2, we use Euler characteristic approximation
    # Euler = b0 - b1 + b2
    # This is a rough estimate; for competition use proper TDA library

    # Count Euler characteristic using local configurations
    # For now, use a simplified approximation
    _, b0_bg = ndimage.label(~binary_mask)

    # Cavities are background components fully enclosed by foreground
    # This is an approximation
    b2 = max(0, b0_bg - 1)  # Subtract 1 for the outer background

    # b1 is harder to compute without proper topology tools
    # Using Euler characteristic: chi = V - E + F for surface mesh
    # For binary volumes: chi ≈ b0 - b1 + b2
    # Estimate b1 as 0 for now (conservative estimate)
    b1 = 0

    return (b0, b1, b2)


def compute_topo_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute topological score via Betti number matching.

    Compares Betti numbers (b0, b1, b2) between prediction and ground truth.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        Topo score in [0, 1] (higher is better)
    """
    pred_betti = compute_betti_numbers(pred)
    gt_betti = compute_betti_numbers(gt)

    # Compute matching score for each Betti number
    scores = []
    for p, g in zip(pred_betti, gt_betti):
        if p == g == 0:
            scores.append(1.0)
        else:
            # Normalized difference
            diff = abs(p - g)
            max_val = max(p, g, 1)
            scores.append(1.0 - diff / max_val)

    # Weight: b0 (components) most important, then b1 (tunnels), then b2 (cavities)
    weights = [0.5, 0.3, 0.2]
    topo_score = sum(w * s for w, s in zip(weights, scores))

    return float(topo_score)


def evaluate_volume(pred_path: str, gt_path: str, tau: float = 2.0) -> dict:
    """
    Evaluate a single volume against ground truth.

    Args:
        pred_path: Path to prediction TIFF
        gt_path: Path to ground truth TIFF
        tau: Surface Dice tolerance

    Returns:
        Dictionary with all metrics
    """
    logger.info(f"Loading prediction: {pred_path}")
    pred = load_volume(pred_path)

    logger.info(f"Loading ground truth: {gt_path}")
    gt = load_volume(gt_path)

    # Binarize (handle multi-class masks)
    pred_binary = (pred == 1).astype(np.uint8)  # Foreground class
    gt_binary = (gt == 1).astype(np.uint8)  # Foreground class

    logger.info("Computing Surface Dice...")
    surface_dice = compute_surface_dice(pred_binary, gt_binary, tau=tau)

    logger.info("Computing VOI score...")
    voi_score = compute_voi(pred_binary, gt_binary)

    logger.info("Computing Topo score...")
    topo_score = compute_topo_score(pred_binary, gt_binary)

    # Combined score
    score = 0.30 * topo_score + 0.35 * surface_dice + 0.35 * voi_score

    return {
        "topo_score": round(topo_score, 4),
        "surface_dice": round(surface_dice, 4),
        "voi_score": round(voi_score, 4),
        "score": round(score, 4),
    }


def evaluate_directory(pred_dir: str, gt_dir: str, tau: float = 2.0) -> dict:
    """
    Evaluate all volumes in a directory.

    Args:
        pred_dir: Directory containing prediction TIFFs
        gt_dir: Directory containing ground truth TIFFs
        tau: Surface Dice tolerance

    Returns:
        Dictionary with per-volume and average metrics
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.tif"))
    if not pred_files:
        pred_files = sorted(pred_dir.glob("*.tiff"))

    if not pred_files:
        raise ValueError(f"No TIFF files found in {pred_dir}")

    results = {}
    all_metrics = {"topo_score": [], "surface_dice": [], "voi_score": [], "score": []}

    for pred_file in pred_files:
        vol_id = pred_file.stem
        gt_file = gt_dir / f"{vol_id}.tif"

        if not gt_file.exists():
            gt_file = gt_dir / f"{vol_id}.tiff"
        if not gt_file.exists():
            logger.warning(f"Ground truth not found for {vol_id}, skipping")
            continue

        logger.info(f"Evaluating volume: {vol_id}")
        metrics = evaluate_volume(str(pred_file), str(gt_file), tau=tau)
        results[vol_id] = metrics

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # Compute averages
    avg_metrics = {key: round(np.mean(values), 4) for key, values in all_metrics.items()}

    return {"per_volume": results, "average": avg_metrics}


def update_experiment_log(experiment_name: str, metrics: dict, log_path: str = "experiments/log.jsonl"):
    """
    Update an experiment's metrics in the log file.

    Args:
        experiment_name: Name of the experiment to update
        metrics: Dictionary with topo_score, surface_dice, voi_score, score
        log_path: Path to log.jsonl
    """
    log_path = Path(log_path)
    if not log_path.exists():
        logger.error(f"Log file not found: {log_path}")
        return

    # Read all entries
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Find and update the experiment
    found = False
    for entry in entries:
        if entry.get("name") == experiment_name:
            entry["metrics"]["topo_score"] = metrics["topo_score"]
            entry["metrics"]["surface_dice"] = metrics["surface_dice"]
            entry["metrics"]["voi_score"] = metrics["voi_score"]
            entry["metrics"]["score"] = metrics["score"]
            found = True
            logger.info(f"Updated metrics for experiment: {experiment_name}")
            break

    if not found:
        logger.warning(f"Experiment not found in log: {experiment_name}")
        return

    # Write back
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Log file updated: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate competition metrics for Vesuvius surface detection")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction directory or single TIFF file")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth directory or single TIFF file")
    parser.add_argument("--tau", type=float, default=2.0, help="Surface Dice tolerance (default: 2.0)")
    parser.add_argument("--update-log", type=str, help="Update experiment log entry with computed metrics")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    if pred_path.is_file():
        # Single volume evaluation
        results = evaluate_volume(str(pred_path), str(gt_path), tau=args.tau)
        print("\nResults:")
        print(f"  Topo Score:    {results['topo_score']:.4f} (weight: 0.30)")
        print(f"  Surface Dice:  {results['surface_dice']:.4f} (weight: 0.35)")
        print(f"  VOI Score:     {results['voi_score']:.4f} (weight: 0.35)")
        print(f"  Combined:      {results['score']:.4f}")
        avg_metrics = results
    else:
        # Directory evaluation
        results = evaluate_directory(str(pred_path), str(gt_path), tau=args.tau)

        print("\nPer-volume results:")
        for vol_id, metrics in results["per_volume"].items():
            print(
                f"  {vol_id}: score={metrics['score']:.4f} (topo={metrics['topo_score']:.4f}, surf={metrics['surface_dice']:.4f}, voi={metrics['voi_score']:.4f})"
            )

        print("\nAverage results:")
        avg = results["average"]
        print(f"  Topo Score:    {avg['topo_score']:.4f} (weight: 0.30)")
        print(f"  Surface Dice:  {avg['surface_dice']:.4f} (weight: 0.35)")
        print(f"  VOI Score:     {avg['voi_score']:.4f} (weight: 0.35)")
        print(f"  Combined:      {avg['score']:.4f}")
        avg_metrics = avg

    # Update experiment log if requested
    if args.update_log:
        update_experiment_log(args.update_log, avg_metrics)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
