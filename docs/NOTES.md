# Cross-Session Notes

This file persists information across Claude sessions. Update after each session.

## VM Connection

```
Host: <not connected yet>
GPU: <pending>
CUDA: <pending>
```

## Current Status

- [ ] Environment setup (uv sync, GPU verified)
- [ ] Data downloaded and verified
- [ ] Baseline experiment completed
- [ ] Competition metrics evaluation working

## Experiment Progress

| Experiment | Score | vs Baseline | Status | Notes |
|------------|-------|-------------|--------|-------|
| baseline   | —     | —           | pending | Establish first |

## Key Observations

<!-- Add insights about data, model behavior, etc. -->

## Hypotheses to Test

1. **Topology-aware loss**: Adding a topology term may help with the TopoScore metric
2. **Larger input patches**: Current 160³ may miss context; try 192³ or 224³
3. **Post-processing**: Morphological operations to clean predictions

## Bugs / Issues Encountered

<!-- Document issues and fixes for future reference -->

## Next Session Priorities

1. Set up VM and verify GPU access
2. Run baseline experiment (10 epochs)
3. Evaluate with competition metrics
4. Log results and plan next experiment

---

## Session History

### Session [YYYY-MM-DD] (Template)

**Completed:**
- Item 1
- Item 2

**In Progress:**
- Experiment X running on VM

**Next Steps:**
1. Priority 1
2. Priority 2

**Observations:**
- Key finding
