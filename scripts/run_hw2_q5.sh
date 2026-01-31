#!/bin/bash
# Q5: KID vs steps for Flow Matching and DDIM
set -e

LOG_FILE="hw_answers/hw2/q5/q5_kid_steps_log.txt"
FLOW_CHECKPOINT="logs/flow_matching/flow_matching_20260127_182922/checkpoints/flow_matching_final.pt"
DDPM_CHECKPOINT="logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt"

mkdir -p hw_answers/hw2

# Log everything
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting Q5 KID evaluation at $(date)"
echo "Flow Matching checkpoint: $FLOW_CHECKPOINT"
echo "DDPM checkpoint: $DDPM_CHECKPOINT"
echo "------------------------------------------"

STEPS_LIST=(1 5 10 50 100 200 1000)

for steps in "${STEPS_LIST[@]}"; do
    echo ""
    echo ">>> Flow Matching KID @ steps=$steps"
    bash scripts/evaluate_modal_torch_fidelity.sh \
        --method flow_matching \
        --checkpoint "$FLOW_CHECKPOINT" \
        --metrics kid \
        --num-samples 1000 \
        --num-steps "$steps" \
        --override

    echo ""
    echo ">>> DDIM KID @ steps=$steps"
    bash scripts/evaluate_modal_torch_fidelity.sh \
        --method ddpm \
        --sampler ddim \
        --eta 0.0 \
        --checkpoint "$DDPM_CHECKPOINT" \
        --metrics kid \
        --num-samples 1000 \
        --num-steps "$steps" \
        --override
done

echo ""
echo "Q5 evaluation complete at $(date)"
