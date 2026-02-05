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

    print("=" * 70)
    print("EXPERIMENT STATUS")
    print("=" * 70)

    if not experiments:
        print("\nNo experiments yet. Run your first experiment:")
        print("  ./run_experiment.sh baseline google/byt5-small 10 8 5e-5")
        print("=" * 70)
        return

    print(f"\nTotal experiments: {len(experiments)}")
    print()

    # Sort by geom_mean
    sorted_exps = sorted(experiments, key=lambda x: x.get("metrics", {}).get("geom_mean", 0), reverse=True)

    # Print leaderboard
    print("LEADERBOARD (by GeomMean):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Experiment':<25} {'BLEU':<10} {'chrF++':<10} {'GeomMean':<10}")
    print("-" * 70)

    for i, exp in enumerate(sorted_exps[:10], 1):
        metrics = exp.get("metrics", {})
        print(f"{i:<5} {exp['id']:<25} {metrics.get('bleu', 0):>8.2f}  {metrics.get('chrf++', 0):>8.2f}  {metrics.get('geom_mean', 0):>8.2f}")

    print("-" * 70)

    # Best experiment details
    best = sorted_exps[0] if sorted_exps else None
    if best:
        print(f"\nBEST EXPERIMENT: {best['id']}")
        print(f"  Model: {best.get('config', {}).get('model_name', 'N/A')}")
        print(f"  Epochs: {best.get('config', {}).get('epochs', 'N/A')}")
        print(f"  Batch Size: {best.get('config', {}).get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {best.get('config', {}).get('learning_rate', 'N/A')}")
        print(f"  Model Path: {best.get('model_path', 'N/A')}")

    # Suggestions
    print("\n" + "=" * 70)
    print("SUGGESTED NEXT STEPS")
    print("=" * 70)

    suggestions = generate_suggestions(experiments)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['title']}")
        print(f"   {suggestion['description']}")
        if suggestion.get("command"):
            print(f"   Command: {suggestion['command']}")

    print("\n" + "=" * 70)


def generate_suggestions(experiments: list[dict]) -> list[dict]:
    """Generate suggestions based on experiment history."""
    suggestions = []

    if not experiments:
        suggestions.append(
            {
                "title": "Run baseline experiment",
                "description": "Start with byt5-small to establish a baseline",
                "command": "./run_experiment.sh baseline google/byt5-small 10 8 5e-5",
            }
        )
        return suggestions

    best = max(experiments, key=lambda x: x.get("metrics", {}).get("geom_mean", 0))
    best_geom = best.get("metrics", {}).get("geom_mean", 0)
    best_config = best.get("config", {})

    # Check what's been tried
    models_tried = {e.get("config", {}).get("model_name") for e in experiments}
    epochs_tried = {e.get("config", {}).get("epochs") for e in experiments}
    lrs_tried = {e.get("config", {}).get("learning_rate") for e in experiments}

    # Model size suggestions
    if "google/byt5-small" in models_tried and "google/byt5-base" not in models_tried:
        suggestions.append(
            {
                "title": "Try larger model (byt5-base)",
                "description": "Current best uses byt5-small. byt5-base may improve quality.",
                "command": "./run_experiment.sh byt5_base google/byt5-base 10 4 3e-5",
            }
        )

    # Epoch suggestions
    max_epochs = max(epochs_tried) if epochs_tried else 0
    if max_epochs < 20:
        suggestions.append(
            {
                "title": "Train longer",
                "description": f"Current max epochs: {max_epochs}. Try more training.",
                "command": f"./run_experiment.sh longer_{max_epochs * 2}ep {best_config.get('model_name', 'google/byt5-small')} {max_epochs * 2} {best_config.get('batch_size', 8)} {best_config.get('learning_rate', 5e-5)}",
            }
        )

    # Learning rate suggestions
    if len(lrs_tried) < 3:
        suggestions.append(
            {
                "title": "Try different learning rates",
                "description": "Learning rate tuning often helps significantly",
                "command": "./run_experiment.sh lr_1e4 google/byt5-small 10 8 1e-4",
            }
        )

    # Data augmentation
    if best_geom > 0 and len(experiments) >= 3:
        suggestions.append(
            {
                "title": "Add data augmentation",
                "description": "Consider back-translation or noise injection for more training data",
                "command": None,
            }
        )

    # Ensemble
    if len(experiments) >= 5:
        suggestions.append(
            {
                "title": "Try ensembling",
                "description": "Combine predictions from top models",
                "command": None,
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "title": "Continue experimenting",
                "description": "Try different hyperparameters or model architectures",
                "command": None,
            }
        )

    return suggestions


if __name__ == "__main__":
    print_status()
