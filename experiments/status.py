#!/usr/bin/env python3
"""
Show experiment status and suggest next steps.
Run this to see progress and get recommendations.
"""

import json
from pathlib import Path


def load_experiments() -> list[dict]:
    """Load all experiments from log."""
    log_path = Path("experiments/log.jsonl")
    if not log_path.exists():
        return []

    experiments = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments


def print_status():
    """Print experiment status."""
    experiments = load_experiments()

    print("=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80)

    if not experiments:
        print("\nNo experiments yet. Run your first experiment:")
        print("  ./scripts/run_experiment.sh baseline --epochs 15")
        print("=" * 80)
        return

    print(f"\nTotal experiments: {len(experiments)}")
    print()

    # Sort by combined score
    sorted_exps = sorted(
        experiments,
        key=lambda x: x.get("metrics", {}).get("score", 0) or 0,
        reverse=True,
    )

    # Print leaderboard
    print("LEADERBOARD (by Score = 0.30*Topo + 0.35*SurfDice + 0.35*VOI):")
    print("-" * 80)
    print(f"{'Rank':<5} {'Experiment':<20} {'Score':<8} {'Topo':<8} {'SurfDice':<10} {'VOI':<8} {'ValDice':<8}")
    print("-" * 80)

    for i, exp in enumerate(sorted_exps[:10], 1):
        metrics = exp.get("metrics", {})
        score = metrics.get("score", 0) or 0
        topo = metrics.get("topo_score", 0) or 0
        sdice = metrics.get("surface_dice", 0) or 0
        voi = metrics.get("voi_score", 0) or 0
        vdice = metrics.get("val_dice", 0) or 0
        print(f"{i:<5} {exp['name']:<20} {score:>6.4f}  {topo:>6.4f}  {sdice:>8.4f}  {voi:>6.4f}  {vdice:>6.4f}")

    print("-" * 80)

    # Best experiment details
    best = sorted_exps[0] if sorted_exps else None
    if best:
        print(f"\nBEST EXPERIMENT: {best['name']}")
        cfg = best.get("config", {})
        print(f"  Model:         {cfg.get('model', 'N/A')}")
        print(f"  Base Channels: {cfg.get('base_channels', 'N/A')}")
        print(f"  Input Shape:   {cfg.get('input_shape', 'N/A')}")
        print(f"  Epochs:        {cfg.get('epochs', 'N/A')}")
        print(f"  Batch Size:    {cfg.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {cfg.get('learning_rate', 'N/A')}")
        print(f"  Loss:          {cfg.get('loss', 'N/A')}")
        print(f"  Runtime:       {best.get('runtime_mins', 'N/A')} min")
        print(f"  Model Path:    {best.get('model_path', 'N/A')}")

    # Suggestions
    print("\n" + "=" * 80)
    print("SUGGESTED NEXT STEPS")
    print("=" * 80)

    suggestions = generate_suggestions(experiments)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['title']}")
        print(f"   {suggestion['description']}")
        if suggestion.get("command"):
            print(f"   Command: {suggestion['command']}")

    print("\n" + "=" * 80)


def generate_suggestions(experiments: list[dict]) -> list[dict]:
    """Generate suggestions based on experiment history."""
    suggestions = []

    if not experiments:
        suggestions.append(
            {
                "title": "Run baseline experiment",
                "description": "Start with Simple3DUNet to establish a baseline",
                "command": "./scripts/run_experiment.sh baseline --epochs 15",
            }
        )
        return suggestions

    best = max(
        experiments,
        key=lambda x: x.get("metrics", {}).get("score", 0) or 0,
    )
    best_score = best.get("metrics", {}).get("score", 0) or 0

    # Check what's been tried
    models_tried = {e.get("config", {}).get("model") for e in experiments}
    batch_sizes_tried = {e.get("config", {}).get("batch_size") for e in experiments}
    input_shapes = {tuple(e.get("config", {}).get("input_shape", [])) for e in experiments if e.get("config", {}).get("input_shape")}

    # Lazy loading (train on all samples)
    max_samples_used = max(
        (e.get("config", {}).get("max_samples", 0) or 0 for e in experiments),
        default=0,
    )
    if max_samples_used > 0:
        suggestions.append(
            {
                "title": "Implement lazy loading",
                "description": (f"Currently limited to {max_samples_used} samples. Lazy loading enables training on all 806 samples."),
                "command": None,
            }
        )

    # Larger batch size
    if batch_sizes_tried <= {1}:
        suggestions.append(
            {
                "title": "Increase batch size",
                "description": "Batch size 1 gives noisy gradients. Try batch 4 with scaled LR.",
                "command": ("./scripts/run_experiment.sh batch4 --batch-size 4 --lr 2e-4 --epochs 15"),
            }
        )

    # Model upgrade
    if "TransUNet" not in models_tried:
        suggestions.append(
            {
                "title": "Try TransUNet model",
                "description": ("TransUNet adds transformer bottleneck + SE attention (~10M params). May capture longer-range dependencies."),
                "command": ("./scripts/run_experiment.sh transunet --model TransUNet --base-channels 32 --batch-size 4 --lr 1e-4 --epochs 15"),
            }
        )

    # Larger patches
    if (160, 160, 160) not in input_shapes:
        suggestions.append(
            {
                "title": "Increase patch size to 160",
                "description": ("Larger patches provide more spatial context. May help with surface continuity and topology."),
                "command": ("./scripts/run_experiment.sh patch160 --input-size 160 --batch-size 2 --epochs 30"),
            }
        )

    # Longer training
    max_epochs = max(
        (e.get("config", {}).get("epochs", 0) or 0 for e in experiments),
        default=0,
    )
    if max_epochs < 30:
        suggestions.append(
            {
                "title": "Train longer",
                "description": f"Current max epochs: {max_epochs}. Try 30+ epochs.",
                "command": None,
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "title": "Continue experimenting",
                "description": (f"Best score: {best_score:.4f}. Try topology-aware loss, post-processing tuning, or ensembling."),
                "command": None,
            }
        )

    return suggestions


if __name__ == "__main__":
    print_status()
