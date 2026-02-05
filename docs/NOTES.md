# Cross-Session Notes

This file persists information across Claude sessions. Update after each session.

## VM Connection

```
Host: <not connected yet>
GPU: <pending>
CUDA: <pending>
```

## Current Status

- [x] Code converted from Keras/JAX to PyTorch
- [x] Simple3DUNet baseline model implemented
- [ ] Environment setup (uv sync on GPU VM)
- [ ] Data downloaded and verified
- [ ] Baseline experiment completed
- [ ] Competition metrics evaluation working

## Experiment Progress

| Experiment | Score | vs Baseline | Status | Notes |
|------------|-------|-------------|--------|-------|
| baseline   | —     | —           | pending | Simple3DUNet, 128³, 15 epochs |

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

## Next Session Priorities

1. Set up VM and verify GPU access
2. Download competition data
3. Run baseline experiment (15 epochs)
4. Evaluate with competition metrics
5. Log results and plan next experiment

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

**In Progress:**
- exp/baseline branch created and committed
- Waiting to push and run on GPU VM

**Next Steps:**
1. Connect to GPU VM
2. Pull exp/baseline branch
3. Run: `./scripts/run_experiment.sh baseline --epochs 15`
4. Evaluate metrics and log to experiments/log.jsonl

**Observations:**
- Simple3DUNet is ~3x smaller than TransUNet3D
- Default config now uses 128³ patches (vs 160³) and batch_size=2
