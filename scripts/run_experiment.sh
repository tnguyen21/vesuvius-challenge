#!/bin/bash
# Experiment runner with automatic logging to experiments/log.jsonl
#
# Usage:
#   ./scripts/run_experiment.sh <experiment_name> [options]
#
# Examples:
#   ./scripts/run_experiment.sh baseline --epochs 10
#   ./scripts/run_experiment.sh larger_lr --epochs 20 --lr 5e-4 --notes "Testing higher LR"
#   ./scripts/run_experiment.sh longer_training --epochs 50 --notes "Extended training run"
#
# The script will:
#   1. Run train.py with the given arguments
#   2. Extract metrics from the training history
#   3. Append results to experiments/log.jsonl
#   4. Compare against baseline (if exists)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/experiments/log.jsonl"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <experiment_name> [train.py options] [--notes \"description\"]"
    echo ""
    echo "Required:"
    echo "  experiment_name    Name for this experiment (e.g., baseline, larger_lr)"
    echo ""
    echo "Optional:"
    echo "  --notes \"...\"      Description of what this experiment tests"
    echo "  All other options are passed directly to train.py"
    echo ""
    echo "Examples:"
    echo "  $0 baseline --epochs 10"
    echo "  $0 higher_lr --epochs 20 --lr 5e-4 --notes \"Testing 5x learning rate\""
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

EXPERIMENT_NAME="$1"
shift

# Parse --notes separately from train.py args
NOTES=""
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --notes)
            NOTES="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Record start time
START_TIME=$(date +%s)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Get current git branch
GIT_BRANCH=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Create log directory for this run
LOG_DIR="$PROJECT_DIR/logs/$EXPERIMENT_NAME"
mkdir -p "$LOG_DIR"

# Run training
echo -e "${YELLOW}Starting training...${NC}"
echo "Command: python train.py --experiment-name $EXPERIMENT_NAME ${TRAIN_ARGS[*]}"
echo ""

cd "$PROJECT_DIR"

# Run train.py and capture output
if python train.py --experiment-name "$EXPERIMENT_NAME" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"; then
    TRAIN_SUCCESS=true
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    TRAIN_SUCCESS=false
    echo -e "${RED}Training failed!${NC}"
fi

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME_MINS=$(( (END_TIME - START_TIME) / 60 ))

# Extract metrics from history.json
HISTORY_FILE="$CHECKPOINTS_DIR/$EXPERIMENT_NAME/history.json"
if [ -f "$HISTORY_FILE" ]; then
    echo ""
    echo -e "${YELLOW}Extracting metrics from history...${NC}"

    # Extract best validation dice from history
    VAL_DICE=$(python3 -c "
import json
with open('$HISTORY_FILE') as f:
    h = json.load(f)
metrics = h.get('val_metrics', [])
if metrics:
    best = max(metrics, key=lambda x: x.get('val_dice', 0))
    print(f\"{best.get('val_dice', 0):.4f}\")
else:
    print('0.0000')
" 2>/dev/null || echo "0.0000")

    VAL_LOSS=$(python3 -c "
import json
with open('$HISTORY_FILE') as f:
    h = json.load(f)
metrics = h.get('val_metrics', [])
if metrics:
    best = min(metrics, key=lambda x: x.get('val_loss', float('inf')))
    print(f\"{best.get('val_loss', 0):.4f}\")
else:
    print('0.0000')
" 2>/dev/null || echo "0.0000")

    # Extract config from history
    CONFIG_JSON=$(python3 -c "
import json
with open('$HISTORY_FILE') as f:
    h = json.load(f)
config = h.get('config', {})
# Select key fields
out = {
    'model': config.get('model_name', 'TransUNet'),
    'encoder': config.get('encoder_name', 'seresnext50'),
    'input_shape': config.get('input_shape', [160, 160, 160]),
    'epochs': config.get('epochs', 0),
    'batch_size': config.get('batch_size', 1),
    'learning_rate': config.get('learning_rate', 1e-4),
    'loss': config.get('loss', 'combo'),
    'seed': config.get('seed', 42)
}
print(json.dumps(out))
" 2>/dev/null || echo '{}')

else
    echo -e "${RED}Warning: history.json not found at $HISTORY_FILE${NC}"
    VAL_DICE="0.0000"
    VAL_LOSS="0.0000"
    CONFIG_JSON='{}'
fi

# Get baseline comparison if baseline exists
BASELINE_COMPARISON=""
if [ -f "$LOG_FILE" ] && [ "$EXPERIMENT_NAME" != "baseline" ]; then
    BASELINE_DICE=$(grep '"name": "baseline"' "$LOG_FILE" 2>/dev/null | head -1 | python3 -c "
import sys, json
try:
    line = sys.stdin.read().strip()
    if line:
        data = json.loads(line)
        print(f\"{data.get('metrics', {}).get('val_dice', 0):.4f}\")
    else:
        print('')
except:
    print('')
" 2>/dev/null || echo "")

    if [ -n "$BASELINE_DICE" ] && [ "$BASELINE_DICE" != "" ]; then
        DIFF=$(python3 -c "print(f'{float($VAL_DICE) - float($BASELINE_DICE):+.4f}')" 2>/dev/null || echo "+0.0000")
        BASELINE_COMPARISON="$DIFF val_dice vs baseline ($BASELINE_DICE)"
    fi
fi

# Build the log entry
# Note: Competition metrics (topo_score, surface_dice, voi_score) must be filled in manually
# after running eval_metrics.py on the predictions
LOG_ENTRY=$(python3 -c "
import json
entry = {
    'name': '$EXPERIMENT_NAME',
    'branch': '$GIT_BRANCH',
    'config': $CONFIG_JSON,
    'metrics': {
        'val_dice': float('$VAL_DICE'),
        'val_loss': float('$VAL_LOSS'),
        'topo_score': None,
        'surface_dice': None,
        'voi_score': None,
        'score': None
    },
    'baseline_comparison': '$BASELINE_COMPARISON' if '$BASELINE_COMPARISON' else None,
    'runtime_mins': $RUNTIME_MINS,
    'notes': '$NOTES' if '$NOTES' else None,
    'timestamp': '$TIMESTAMP',
    'model_path': 'checkpoints/$EXPERIMENT_NAME/best.weights.h5',
    'status': 'success' if $TRAIN_SUCCESS else 'failed'
}
# Remove None values for cleaner output
entry = {k: v for k, v in entry.items() if v is not None}
if entry.get('metrics'):
    entry['metrics'] = {k: v for k, v in entry['metrics'].items() if v is not None}
print(json.dumps(entry))
")

# Append to log
echo "$LOG_ENTRY" >> "$LOG_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Experiment Complete: ${EXPERIMENT_NAME}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results:"
echo "  Val Dice: $VAL_DICE"
echo "  Val Loss: $VAL_LOSS"
echo "  Runtime:  ${RUNTIME_MINS} minutes"
if [ -n "$BASELINE_COMPARISON" ]; then
    echo "  vs Baseline: $BASELINE_COMPARISON"
fi
echo ""
echo "Log entry appended to: $LOG_FILE"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Run predictions: python vesuvius_predictions.py --weights checkpoints/$EXPERIMENT_NAME/best.weights.h5"
echo "  2. Evaluate competition metrics: python scripts/eval_metrics.py --pred output/$EXPERIMENT_NAME --gt data/train_labels"
echo "  3. Update experiments/log.jsonl with competition metrics (topo_score, surface_dice, voi_score)"
echo "  4. Run 'python experiments/status.py' to see leaderboard and suggestions"
