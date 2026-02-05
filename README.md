# Vesuvius Challenge - Surface Detection

> Segment the scroll's surface in CT scans to help virtually unwrap ancient Herculaneum scrolls.

## Overview

Some ancient scrolls are too fragile to open, but that doesn't mean we can't read them. Before we can recover what's written between the lines, we first need to find the lines. This competition focuses on training a model that follows the scroll's surface—one of the trickiest and most essential parts of virtually unwrapping the text.

## Background

The library at the Villa dei Papiri is one-of-a-kind: it's the only classical antiquity known to survive. When Mount Vesuvius erupted in AD 79, most of its scrolls were turned into carbonized bundles of ash. Nearly 2,000 years later, many are still sealed shut, too delicate to unroll and too complex to decode.

Physical unrolling would destroy the scrolls, and today's digital methods can handle only the easy parts (clean, well-spaced layers). But the tightest, most tangled areas are often where the real discoveries hide. To recover those, we need better segmentation to handle noise and compression without distorting the scroll's shape.

## Dataset

The dataset includes 3D chunks of binary labeled CT scans of the closed and carbonized Herculaneum scrolls. Data was acquired at:
- ESRF synchrotron in Grenoble (beamline BM18)
- DLS synchrotron in Oxford (beamline I12)

**Key characteristics:**
- Chunk dimensions are not fixed and vary across the dataset
- A papyrus sheet has two layers: recto (horizontal fibers) and verso (vertical fibers)
- The ideal solution detects the **recto surface** (faces the umbilicus/center of scroll)
- Sheets may be partially damaged and frayed

**Critical constraint:** Avoid topological mistakes—artificial mergers between different sheets and holes that split a single entity into disconnected components.

See [docs/NOTES.md](docs/NOTES.md) for detailed data documentation.

## Evaluation

The competition uses a weighted average of three segmentation metrics:

```
Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score
```

| Metric | Purpose | Detects |
|--------|---------|---------|
| **SurfaceDice@τ** | Surface proximity (τ=2.0) | Boundary accuracy |
| **VOI_score** | Instance consistency | Splits/merges between components |
| **TopoScore** | Topological correctness | Holes, handles, bridges |

### Why Topology Matters

Classic metrics (e.g., Dice) can look great even when topology errors are severe. The leaderboard blends:
- Surface similarity
- Instance-level consistency
- Topological correctness

### Error Pattern Reference

| Error | SurfaceDice | VOI | TopoScore |
|-------|-------------|-----|-----------|
| Slight boundary misplacement (≤τ) | Tolerant | Unaffected | Unaffected |
| Bridge between parallel layers | May stay high | Merge ↓ | k=0/k=1 penalized |
| Split within same wrap | May stay high | Split ↓ | k=0 penalized |
| Spurious holes/handles | May stay high | Small effect | k=1/k=2 penalized |

## Submission

Submit a zip containing one `.tif` volume mask per test image:
- Each mask named `[image_id].tif`
- Must match source image dimensions exactly
- Same data type as train mask
- File must be named `submission.zip`

## Timeline

| Date | Milestone |
|------|-----------|
| Nov 13, 2025 | Start Date |
| Feb 6, 2026 | Entry & Team Merger Deadline |
| Feb 13, 2026 | Final Submission Deadline |

All deadlines at 11:59 PM UTC.

## Code Requirements

- CPU/GPU Notebook ≤ 9 hours runtime
- Internet access disabled
- External data allowed (freely & publicly available, including pre-trained models)

## References

- [Vesuvius Challenge](https://scrollprize.org/)
- [Digital unwrapping tutorial](https://www.youtube.com/watch?v=yHbpVcGD06U)
- Berger et al., "Pitfalls of topology-aware image segmentation", IPMI 2025
- Stucki et al., "Efficient Betti Matching Enables Topology-Aware 3D Segmentation", arXiv 2024
- Shit et al., "clDice - Topology-Preserving Loss Function", CVPR 2021
