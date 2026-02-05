# Claude Code Conventions for This Project

## Session Start Protocol

When starting a new session or after context reset:

1. Read `docs/NOTES.md` for context from previous sessions
2. Read `experiments/log.jsonl` to understand what's been tried
3. Summarize last 3-5 experiments (metrics, what worked/didn't)
4. Check current branch state: `git status`, `git branch`
5. Check VM status if connected: `nvidia-smi`, `ps aux | grep python`
6. Then proceed with next experiment

## Cross-Instance Notes

Use `docs/NOTES.md` to document information that should persist across Claude instances:

- Current VM connection details and status
- Bugs encountered and fixes applied
- Observations about the data or model behavior
- Next steps and hypotheses to test

Update `docs/NOTES.md` whenever significant state changes occur (experiment completes, bug found, new insight).

## Git Workflow

- **Work locally, run remotely**: Make code changes and commits here, push to branches, then pull on the VM to run experiments
- **Branch per experiment**: `git checkout -b exp/<experiment-name>`. Don't work directly on main
- **Only commit working code**: Every commit should be runnable. If something is broken, fix it before committing
- **Failed experiments stay on branches**: If a change doesn't improve val score, document the results in the PR/commit message but do NOT merge to main
- **PR before merge**: Open a PR for review before merging any experiment into main. Include metrics comparison vs baseline. Create a PR after experiment runs (or confirmation of a failed run).

## Using `gh` CLI for PRs

Always use the GitHub CLI (`gh`) for creating and managing PRs. This keeps everything scriptable and consistent.

**Creating a PR:**

```bash
gh pr create --title "Short imperative title" --body "$(cat <<'EOF'
## Summary
One or two sentences: what does this PR do and why?

## Changes
Describe *what* changed at a high level. Group related changes logically. For each significant change, briefly explain *why* it was necessary—this is more valuable than restating the diff.

## Testing
How was this tested? What commands were run? Include metrics comparisons where relevant.

## Review guidance
Point the reviewer to the most important files or functions. Flag anything risky, hacky, or worth extra scrutiny. If there are areas where you made judgment calls, explain your reasoning so the reviewer can evaluate it.

---
🤖 Generated with Claude
EOF
)"
```

**Other useful commands:**

- `gh pr list` — see open PRs
- `gh pr view <number>` — view PR details
- `gh pr checkout <number>` — check out a PR branch locally
- `gh pr merge <number> --squash` — squash and merge (preferred for experiment branches)

## PR Message Conventions

The goal of a PR message is to let you confidently approve or request changes _without_ reading every line of the diff. A good PR message answers: what changed, why it changed, and where to look if something seems off.

**Title:** Use imperative mood, keep it under 60 characters. The title should complete the sentence "This PR will \_\_\_." Examples: "Add learning rate warmup", "Fix tokenizer padding bug", "Increase batch size to 16".

**Summary:** One or two sentences describing the intent. Not a list of files changed—that's what the diff is for. Focus on the _goal_ of the change.

**Changes section:** Organize by logical grouping, not by file. Explain the "why" alongside the "what." For example: "Switched from AdamW to Adafactor to reduce memory usage, which allows us to increase batch size." This is more useful than "Changed optimizer in train.py."

**Review guidance:** This is the most important section for efficient human review. Include:

- **Key files to examine**: Which files contain the core logic changes? What should the reviewer focus on?
- **Risk areas**: Any tricky code, edge cases, or places where bugs might hide?
- **Judgment calls**: If you made a decision that could reasonably go another way, explain your reasoning so the reviewer can weigh in.
- **What NOT to worry about**: If there's boilerplate, auto-generated code, or trivial changes, say so—this saves review time.

**Metrics (for experiments):** Always include before/after metrics relative to baseline. Format consistently:

```
Baseline: score 0.455 (topo=0.42, surface_dice=0.48, voi=0.46)
This PR:  score 0.485 (topo=0.45, surface_dice=0.52, voi=0.48) (+0.03, +6.6%)
```

**Keep it scannable:** Use short paragraphs. A reviewer should be able to skim and understand the gist in 30 seconds, then dive deeper if needed.

## Experiment Discipline

- **Baseline first**: Must establish and document a baseline experiment before any other experimentation. All subsequent experiments compare against baseline metrics
- **One variable at a time**: Change only one variable per experiment unless explicitly bundling related changes. This enables clear attribution of what helped
- **Name experiments clearly**: `baseline`, `transunet_lr1e4`, `longer_20ep`, `topo_loss`, not `test1`, `final_v2`
- **Document negative results**: Failed experiments are valuable - note what was tried and why it didn't work
- **Reproducibility**: Always set and log random seeds. Commit `uv.lock` for dependency pinning

## Experiment Log Schema

**IMPORTANT**: After every experiment completes, log results to `experiments/log.jsonl` before doing anything else. This is critical for tracking progress and avoiding duplicate work.

All experiments logged to `experiments/log.jsonl` (append-only, one JSON object per line):

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
  "notes": "Tried larger input size, improved surface dice",
  "timestamp": "2026-02-05T12:00:00Z"
}
```

**Metrics:**
- `topo_score`: Topological correctness (Betti matching) — weight 0.30
- `surface_dice`: Surface proximity (τ=2.0) — weight 0.35
- `voi_score`: Instance consistency (split/merge) — weight 0.35
- `score`: Combined `0.30×topo + 0.35×surface_dice + 0.35×voi`

## VM/Remote Execution

- **One experiment at a time**: Run sequentially, don't parallelize training runs
- **Don't run destructive commands without confirmation**: No `rm -rf`, `git reset --hard`, etc. without asking
- **Max runtime**: Default 2-hour timeout per experiment. Abort if validation loss hasn't improved for 3 consecutive epochs

### Remote Execution Protocol

**VM Connection** (update these after connecting):
```
Host: <user>@<ip>
GPU: <type, e.g., A100 40GB>
CUDA: <version>
```

**Workflow:**
```bash
# 1. Push code changes locally
git add -A && git commit -m "..." && git push origin exp/<name>

# 2. Pull on VM
ssh <user>@<ip> 'cd vesuvius-challenge && git pull'

# 3. Run experiment on VM
ssh <user>@<ip> 'cd vesuvius-challenge && ./scripts/run_experiment.sh <name> --epochs 20'

# 4. Monitor progress
ssh <user>@<ip> 'tail -f vesuvius-challenge/logs/<name>/train.log'

# 5. Check GPU status
ssh <user>@<ip> 'nvidia-smi'
```

**First-time VM setup:**
```bash
# Clone repo
git clone <repo-url> vesuvius-challenge
cd vesuvius-challenge

# Setup environment and verify GPU
./scripts/setup_gpu.sh

# Download competition data
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip vesuvius-challenge-surface-detection.zip -d data/
```

### Long-Running Experiment Protocol

For experiments that may span multiple sessions:

1. **Before starting**: Note in `docs/NOTES.md`:
   - Experiment name and config
   - Expected runtime
   - VM connection details

2. **While running**: Periodically check and log:
   ```bash
   # Check if still running
   ssh vm 'ps aux | grep train.py'

   # Check latest metrics
   ssh vm 'tail -20 vesuvius-challenge/logs/<name>/train.log'
   ```

3. **If session ends mid-experiment**:
   - Next session should check `docs/NOTES.md` first
   - Check if experiment completed: `ls checkpoints/<name>/`
   - Check training log: `tail logs/<name>/train.log`

4. **After completion**:
   - Run evaluation: `python scripts/eval_metrics.py --pred output/<name> --gt data/train_labels --update-log <name>`
   - Update `docs/NOTES.md` with results
   - Update `experiments/log.jsonl` with competition metrics

## Escalation Protocol

- **Debugging limit**: If debugging an issue exceeds 30 minutes without progress, stop, document the issue clearly, and escalate to user
- **Unknown errors**: If an error is unclear or outside normal training failures, ask before attempting fixes
- **Git conflicts**: Ask before any force operations or conflict resolution

## Timeouts and Guardrails

- **Experiment timeout**: 2 hours max per training run
- **Early stopping**: Abort if validation loss hasn't improved in 3 epochs
- **Debugging timeout**: 30 minutes max, then escalate

## Competition Metrics Evaluation

The competition uses three metrics with specific weights:

| Metric | Weight | Tool | What it measures |
|--------|--------|------|------------------|
| TopoScore | 0.30 | `eval_metrics.py` | Topological correctness (Betti matching) |
| SurfaceDice@τ | 0.35 | `eval_metrics.py` | Surface proximity (τ=2.0 voxels) |
| VOI_score | 0.35 | `eval_metrics.py` | Instance consistency (splits/merges) |

**Final score**: `0.30 × topo_score + 0.35 × surface_dice + 0.35 × voi_score`

### Evaluation Workflow

After training completes:

```bash
# 1. Generate predictions on validation set
python vesuvius_predictions.py --weights checkpoints/<name>/best.weights.h5 --output output/<name>

# 2. Compute competition metrics
python scripts/eval_metrics.py --pred output/<name> --gt data/train_labels

# 3. Update experiment log with competition metrics
python scripts/eval_metrics.py --pred output/<name> --gt data/train_labels --update-log <name>
```

**Note**: The training script only computes internal metrics (val_dice, val_loss). Competition metrics must be computed separately after generating predictions.

### Metric Interpretation

| Error Type | Impact |
|------------|--------|
| Slight boundary misplacement (≤2 voxels) | Tolerant (SurfaceDice handles this) |
| Bridge between parallel layers | VOI↓, TopoScore↓ (merge error) |
| Split within same wrap | VOI↓, TopoScore↓ (split error) |
| Spurious holes/handles | TopoScore↓ (topological error) |

## Data Pipeline

### Directory Structure

```
data/
├── train.csv                    # Metadata: id, scroll_id
├── test.csv                     # Test metadata (public version is dummy)
├── train_images/
│   └── {id}.tif                 # 3D CT volumes (float32, variable dimensions)
└── train_labels/
    └── {id}.tif                 # Label masks (uint8: 0=bg, 1=fg, 2=unlabeled)
```

### Data Characteristics

- **Volume dimensions**: Variable (not fixed across dataset)
- **Label encoding**: 0=background, 1=foreground (recto surface), 2=unlabeled
- **Unlabeled pixels**: Most masks only cover a subset of the volume
- **Target**: Detect the recto surface (horizontal fiber layer facing scroll center)

### Preprocessing in train.py

1. Load TIFF volume (float32)
2. Intensity normalization (nonzero, channel-wise=False)
3. Random patch extraction (train) or center crop (val)
4. Augmentation: flips (3 axes), 90° rotations (axial plane)

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/run_experiment.sh` | Run training with auto-logging | `./scripts/run_experiment.sh <name> --epochs 20` |
| `scripts/setup_gpu.sh` | VM GPU environment setup | `./scripts/setup_gpu.sh` |
| `scripts/eval_metrics.py` | Compute competition metrics | `python scripts/eval_metrics.py --pred <dir> --gt <dir>` |

### run_experiment.sh

Automates the training workflow:
1. Runs `train.py` with given arguments
2. Extracts metrics from `history.json`
3. Appends entry to `experiments/log.jsonl`
4. Compares against baseline (if exists)
5. Logs runtime

```bash
# Basic usage
./scripts/run_experiment.sh baseline --epochs 10

# With notes
./scripts/run_experiment.sh higher_lr --epochs 20 --lr 5e-4 --notes "Testing 5x learning rate"

# Full config
./scripts/run_experiment.sh topo_loss --epochs 30 --loss dice --lr 1e-4 --notes "Pure dice loss"
```

### eval_metrics.py

Computes competition metrics on predictions:

```bash
# Evaluate directory of predictions
python scripts/eval_metrics.py --pred output/baseline --gt data/train_labels

# Evaluate single volume
python scripts/eval_metrics.py --pred output/vol1.tif --gt data/train_labels/vol1.tif

# Update experiment log with metrics
python scripts/eval_metrics.py --pred output/baseline --gt data/train_labels --update-log baseline

# Save results to JSON
python scripts/eval_metrics.py --pred output/baseline --gt data/train_labels --output results.json
```

## Code Quality

- **Format before commit**: `uvx ruff format . && uvx ruff check .`
- **Test before commit**: Run a quick sanity check (e.g., `--dry-run` or small data subset)
- **No large files in git**: Models, data, checkpoints stay in `.gitignore`
- **Pin dependencies**: Commit `uv.lock` for reproducibility

## Communication

- **Summarize after each iteration**: What was tried, what the metrics were, what to try next
- **Be explicit about uncertainty**: If I'm not sure something will help, say so
- **Compare to baseline**: Always report metrics relative to baseline

## File Structure

```
vesuvius-challenge/
├── train.py                    # Training script
├── vesuvius_predictions.py     # Inference script
├── pyproject.toml              # Project config
├── uv.lock                     # Dependency lockfile (committed)
├── CLAUDE.md                   # Claude instructions
├── scripts/
│   ├── setup_gpu.sh            # VM setup
│   ├── remote.sh               # Remote execution helper
│   └── run_experiment.sh       # Experiment runner
├── docs/
│   ├── NOTES.md                # Cross-instance session notes (dataset info)
│   └── DATA.md                 # Competition data documentation
├── experiments/
│   ├── log.jsonl               # Experiment log (append-only)
│   └── status.py               # Experiment status/suggestions
├── data/                       # Competition data (gitignored)
├── checkpoints/                # Model checkpoints (gitignored)
└── output/                     # Inference outputs (gitignored)
```

## Iteration Protocol

1. **Check status**: `python experiments/status.py` - review experiments, get suggestions
2. **Plan**: Decide what single variable to change next
3. **Branch**: `git checkout -b exp/<experiment-name>`
4. **Implement**: Make changes, commit locally
5. **Deploy**: Push, pull on VM
6. **Run**: `./scripts/run_experiment.sh <name> ...` (one at a time)
7. **Analyze**: Compare metrics to baseline
8. **Document**: If improved → PR to main. If not → document in commit, leave branch unmerged
9. **Repeat**

## Failure Handling

- **Training crash**: Check logs, if fix is obvious apply it, otherwise escalate after 30 min
- **VM disconnect**: Experiment logs persist in `experiments/log.jsonl` - can resume
- **OOM**: Reduce batch size, try gradient accumulation
- **Hung process**: Kill after timeout, log the failure, try with different config

## Autonomy Guidelines

### What Claude Can Do Without Asking

**Code changes:**
- Bug fixes that don't change model behavior
- Adding logging/monitoring
- Code formatting and cleanup
- Implementing planned experiments from NOTES.md

**Experiment execution:**
- Running experiments specified in the plan
- Re-running failed experiments with obvious fixes (OOM → smaller batch)
- Running evaluation scripts
- Updating experiment logs

**Documentation:**
- Updating NOTES.md with observations
- Logging experiment results
- Adding code comments

### What Requires Confirmation

**Significant changes:**
- Changing model architecture
- New loss functions or training strategies
- Modifying competition submission format

**Destructive operations:**
- `git reset`, `git clean`, force operations
- Deleting checkpoints or logs
- Killing long-running processes (unless clearly hung)

**Resource-intensive:**
- Starting experiments >2 hours expected runtime
- Running multiple experiments in parallel
- Large data preprocessing jobs

### Experiment Suggestion Priority

When suggesting next experiments, prioritize:

1. **Establish baseline** (if not done)
2. **Fix obvious issues** (training instability, bugs)
3. **Low-hanging fruit** (well-known improvements: augmentation, longer training)
4. **Hyperparameter tuning** (LR, batch size, loss weights)
5. **Architecture changes** (encoder, model size)
6. **Novel approaches** (topology-aware losses, post-processing)

### Session Handoff

When ending a session, always update `docs/NOTES.md` with:

```markdown
## Session [DATE]

### Completed
- [x] Experiment X: score=0.XX (vs baseline +0.XX)
- [x] Fixed bug Y

### In Progress
- [ ] Experiment Z running on VM (started HH:MM, ~N hours remaining)

### Next Steps
1. First priority item
2. Second priority item

### Observations
- Key insight or finding
```
