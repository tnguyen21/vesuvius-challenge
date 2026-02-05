# Cross-Session Notes

This file persists information across Claude sessions. Update after each session.

## VM Connection

```
Host: root@45.63.87.222
Port: 22
Key: ~/.ssh/primeintellect_ed25519
GPU: GRID A100D-40C (40GB)
Driver: 550.90.07
Provider: Prime Intellect
```

**SSH Command:**
```bash
ssh -i ~/.ssh/primeintellect_ed25519 root@45.63.87.222 -p 22
```

## Current Status

- [x] Code converted from Keras/JAX to PyTorch
- [x] Simple3DUNet baseline model implemented
- [x] VM connected (Prime Intellect A100 40GB)
- [x] Environment setup (uv sync on GPU VM)
- [x] Data downloaded and verified
- [x] Baseline experiment running (15 epochs, 300 samples)
- [ ] Competition metrics evaluation working

## Experiment Progress

| Experiment | Val Dice | vs Baseline | Status | Notes |
|------------|----------|-------------|--------|-------|
| baseline   | 0.1373   | —           | done   | Simple3DUNet, 128³, 15 epochs, 300 samples |

## Key Observations

- Simple3DUNet: ~12.7M params (vs ~35M for TransUNet3D)
- PyTorch inference script now fully standalone (no medicai dependency)
- Sliding window inference with Gaussian weighting implemented
- TTA: 7 augmentations (original + 3 flips + 3 rotations)

## Hypotheses to Test

1. **Topology-aware loss**: Adding a topology term may help with the TopoScore metric
2. **Larger input patches**: Current 128³ may miss context; try 160³ or 192³
3. **Post-processing tuning**: Hysteresis thresholds and morphological parameters

## Bugs / Issues Encountered

- **Fixed**: vesuvius_predictions.py was using Keras/JAX but train.py was PyTorch
  - Solution: Converted vesuvius_predictions.py to pure PyTorch

- **Fixed**: OOM when loading all 806 volumes into memory (~100 GB required)
  - Solution: Added `--max-samples` argument to limit samples loaded
  - VM has ~58 GB RAM, can safely load ~300 samples (~40 GB)
  - Future fix: Implement lazy loading with PyTorch DataLoader

- **Fixed**: run_experiment.sh using `python` instead of `uv run python`
  - Solution: Added check for `uv` and use `uv run python` if available

## Next Session Priorities

1. ~~Set up VM and verify GPU access~~ ✓ Done
2. Set up environment on VM (clone repo, uv sync)
3. Download competition data
4. Run baseline experiment (15 epochs)
5. Evaluate with competition metrics
6. Log results and plan next experiment

---

## Session History

### Session 2026-02-05

**Completed:**
- Implemented Simple3DUNet baseline model in train.py
- Converted vesuvius_predictions.py from Keras/JAX to PyTorch
- Implemented SlidingWindowInference with Gaussian weighting
- Updated pyproject.toml dependencies (torch>=2.0)
- Updated default hyperparameters for faster baseline iteration
- Added --model and --base-channels CLI arguments

**Completed:**
- Implemented Simple3DUNet baseline model in train.py
- Converted vesuvius_predictions.py from Keras/JAX to PyTorch
- Implemented SlidingWindowInference with Gaussian weighting
- Added --max-samples option to handle OOM issues
- Fixed run_experiment.sh to use uv run python
- **Baseline experiment completed**: Val Dice = 0.1373

**Results:**
```
Baseline: Val Dice 0.1373, Val Loss 0.6032, Train Loss 0.5822
Runtime: 28 minutes on A100 40GB
Samples: 300 (limited due to memory)
```

**Observations:**
- Simple3DUNet is ~3x smaller than TransUNet3D (12.7M vs ~35M params)
- GPU memory usage very low (~4.8 GB of 40 GB)
- Some batches have Loss=0.0 (empty/unlabeled patches)
- VM has ~58 GB RAM, can load ~300 samples safely
- Need lazy loading to use full 806-sample dataset
