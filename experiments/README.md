# Experiment Tracking

This directory stores experiment logs and results.

## Files

- `log.jsonl` - Append-only experiment log (one JSON object per line)

## Log Schema

See `CLAUDE.md` for the authoritative schema. Each entry contains:

```json
{
  "name": "experiment_name",
  "branch": "exp/experiment_name",
  "config": {
    "model": "TransUNet",
    "encoder": "seresnext50",
    "input_shape": [160, 160, 160],
    "num_classes": 3,
    "epochs": 10,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "loss": "combo",
    "optimizer": "adam",
    "seed": 42
  },
  "seed": 42,
  "metrics": {
    "topo_score": 0.45,
    "surface_dice": 0.52,
    "voi_score": 0.48,
    "score": 0.485
  },
  "baseline_comparison": "+0.03 score vs baseline (0.455)",
  "runtime_mins": 120,
  "notes": "Description of what was tried",
  "timestamp": "2026-02-05T12:00:00Z",
  "model_path": "checkpoints/experiment_name/best.weights.h5"
}
```

### Metrics Reference

| Field | Description | Weight |
|-------|-------------|--------|
| `topo_score` | Topological correctness (Betti matching) | 0.30 |
| `surface_dice` | Surface proximity (τ=2.0) | 0.35 |
| `voi_score` | Instance consistency (split/merge) | 0.35 |
| `score` | Combined: `0.30×topo + 0.35×surface_dice + 0.35×voi` | — |

## Usage

```bash
# View all experiments
cat experiments/log.jsonl | jq .

# Get best experiment by combined score
cat experiments/log.jsonl | jq -s 'max_by(.metrics.score)'

# List experiments sorted by score
cat experiments/log.jsonl | jq -s 'sort_by(.metrics.score) | reverse | .[].name'

# Compare all experiments
cat experiments/log.jsonl | jq -s '.[] | {name, score: .metrics.score, topo: .metrics.topo_score, surface: .metrics.surface_dice, voi: .metrics.voi_score}'
```
