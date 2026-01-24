#!/bin/bash
# Q7 Ablation Study: Sampling Timesteps
set -e

LOG_FILE="hw_answers/hw1/q7_ablation_log.txt"
CHECKPOINT="logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt"

mkdir -p hw_answers/hw1

# 关键改动：从这里开始，所有输出都会实时累加到日志中
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting Q7 Ablation Study at $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "------------------------------------------"

# 1. Run KID evaluations
for steps in 100 300 500 700 900 1000; do
    echo ""
    echo ">>> Evaluating KID for $steps steps..."
    bash scripts/evaluate_modal_torch_fidelity.sh \
        --num-steps "$steps" \
        --num-samples 1000 \
        --checkpoint "$CHECKPOINT" \
        --override
done

echo ""
echo "------------------------------------------"
echo ">>> Generating qualitative samples (16-grid)..."

# 2. Generate grid samples for qualitative comparison
for steps in 100 300 500 700 900 1000; do
    echo "Generating sample for $steps steps..."
    modal run modal_app.py --action sample \
        --method ddpm \
        --checkpoint "$CHECKPOINT" \
        --num-samples 16 \
        --num-steps "$steps"
done

echo ""
echo "Ablation study complete at $(date)"