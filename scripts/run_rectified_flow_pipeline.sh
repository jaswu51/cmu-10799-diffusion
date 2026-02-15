#!/bin/bash
# =============================================================================
# Rectified Flow Full Pipeline (Modal)
# =============================================================================
#
# Pipeline:
#   0. Download dataset (if needed)
#   1. Stage 1: Train 1-Rectified Flow (200K iters, DiT-S/4)
#   2. Generate sample grids from Stage 1 at NFE=1,2,5,10,20,50
#   3. Generate 60K reflow pairs using Stage 1 EMA model
#   4. Stage 2: Train 2-Rectified Flow / Reflow (100K iters)
#   5. Generate sample grids from Stage 2 at NFE=1,2,5,10,20,50
#   6. Evaluate KID+FID for both stages
#   7. Visualize trajectories (PCA + straightness)
#
# Usage:
#   ./scripts/run_rectified_flow_pipeline.sh              # Run all steps
#   ./scripts/run_rectified_flow_pipeline.sh 3             # Resume from step 3
#
# =============================================================================

set -e          # Exit on error
set -o pipefail # Catch errors in piped commands (e.g. cmd | tee)

# Resume from this step (default: 0 = run all)
START_STEP=${1:-0}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log directory
LOG_DIR="logs/pipeline_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Rectified Flow Pipeline${NC}"
echo -e "${BLUE}  Started: $(date)${NC}"
echo -e "${BLUE}  Logs: ${LOG_DIR}/${NC}"
echo -e "${BLUE}  Starting from step: ${START_STEP}${NC}"
echo -e "${BLUE}========================================${NC}"

# Helper function
run_step() {
    local step_num=$1
    local step_name=$2
    local log_file="${LOG_DIR}/step${step_num}_${step_name}.log"
    shift 2

    # Skip steps before START_STEP
    if [ "${step_num}" -lt "${START_STEP}" ] 2>/dev/null; then
        echo -e "${BLUE}  ⏭ Skipping step ${step_num} (${step_name})${NC}"
        return 0
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Step ${step_num}: ${step_name}${NC}"
    echo -e "${YELLOW}  $(date)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Run command and tee to log file (pipefail ensures we catch the real exit code)
    if "$@" 2>&1 | tee "${log_file}"; then
        echo -e "${GREEN}  ✓ Step ${step_num} completed successfully${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Step ${step_num} FAILED! Check ${log_file}${NC}"
        return 1
    fi
}

# Helper to find the latest checkpoint on Modal volume
find_latest_checkpoint() {
    local base_dir=$1  # e.g. "logs/rectified_flow"
    local filename=$2  # e.g. "rectified_flow_final.pt"
    # modal volume ls returns full paths like "logs/rectified_flow/rectified_flow_20260214_053610"
    local run_dir
    run_dir=$(modal volume ls cmu-10799-diffusion-data "${base_dir}/" 2>/dev/null \
        | grep -v '^$' \
        | sort \
        | tail -1 \
        | sed 's:/*$::')  # strip trailing slash only
    if [ -n "${run_dir}" ]; then
        echo "${run_dir}/checkpoints/${filename}"
    fi
}

# ─────────────────────────────────────────────────────────────
# Step 0: Download dataset
# ─────────────────────────────────────────────────────────────
run_step 0 "download_dataset" \
    modal run modal_app.py --action download

# ─────────────────────────────────────────────────────────────
# Step 1: Train Stage 1 (1-Rectified Flow)
# ─────────────────────────────────────────────────────────────
run_step 1 "train_stage1" \
    modal run modal_app.py \
        --action train \
        --method rectified_flow \
        --config configs/rectified_flow.yaml

# ─────────────────────────────────────────────────────────────
# Find Stage 1 checkpoint
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}Finding Stage 1 checkpoint...${NC}"
STAGE1_CKPT=$(find_latest_checkpoint "logs/rectified_flow" "rectified_flow_final.pt")
if [ -z "${STAGE1_CKPT}" ]; then
    echo -e "${RED}ERROR: Could not find Stage 1 checkpoint on Modal volume${NC}"
    echo "Try: modal volume ls cmu-10799-diffusion-data logs/rectified_flow/"
    exit 1
fi
echo -e "${GREEN}  Stage 1 checkpoint: ${STAGE1_CKPT}${NC}"

# ─────────────────────────────────────────────────────────────
# Step 2: Generate sample grids from Stage 1
# ─────────────────────────────────────────────────────────────
run_step 2 "sample_stage1" \
    modal run modal_app.py \
        --action sample_multi_steps \
        --method rectified_flow \
        --checkpoint "${STAGE1_CKPT}" \
        --step-counts "1,2,5,10,20,50" \
        --num-samples 64 \
        --seed 42 \
        --output-dir "samples/stage1"

# ─────────────────────────────────────────────────────────────
# Step 3: Generate 60K reflow pairs
# ─────────────────────────────────────────────────────────────
run_step 3 "generate_pairs" \
    modal run modal_app.py \
        --action generate_pairs \
        --checkpoint "${STAGE1_CKPT}" \
        --num-pairs 60000 \
        --batch-size 256 \
        --num-steps 50 \
        --seed 42

# ─────────────────────────────────────────────────────────────
# Step 4: Train Stage 2 (Reflow)
# ─────────────────────────────────────────────────────────────
run_step 4 "train_stage2" \
    modal run modal_app.py \
        --action train \
        --method reflow \
        --config configs/rectified_flow_reflow.yaml

# ─────────────────────────────────────────────────────────────
# Find Stage 2 checkpoint
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}Finding Stage 2 checkpoint...${NC}"
STAGE2_CKPT=$(find_latest_checkpoint "logs/rectified_flow_reflow" "reflow_final.pt")
if [ -z "${STAGE2_CKPT}" ]; then
    echo -e "${RED}ERROR: Could not find Stage 2 checkpoint on Modal volume${NC}"
    echo "Try: modal volume ls cmu-10799-diffusion-data logs/rectified_flow_reflow/"
    exit 1
fi
echo -e "${GREEN}  Stage 2 checkpoint: ${STAGE2_CKPT}${NC}"

# ─────────────────────────────────────────────────────────────
# Step 5: Generate sample grids from Stage 2
# ─────────────────────────────────────────────────────────────
run_step 5 "sample_stage2" \
    modal run modal_app.py \
        --action sample_multi_steps \
        --method rectified_flow \
        --checkpoint "${STAGE2_CKPT}" \
        --step-counts "1,2,5,10,20,50" \
        --num-samples 64 \
        --seed 42 \
        --output-dir "samples/stage2"

# ─────────────────────────────────────────────────────────────
# Step 6: Evaluate KID + FID for both stages (50-step sampling)
# ─────────────────────────────────────────────────────────────
run_step 6 "eval_stage1" \
    modal run modal_app.py \
        --action evaluate \
        --method rectified_flow \
        --checkpoint "${STAGE1_CKPT}" \
        --metrics "kid,fid" \
        --num-samples 1000 \
        --num-steps 50

run_step 6 "eval_stage2" \
    modal run modal_app.py \
        --action evaluate \
        --method rectified_flow \
        --checkpoint "${STAGE2_CKPT}" \
        --metrics "kid,fid" \
        --num-samples 1000 \
        --num-steps 50

# ─────────────────────────────────────────────────────────────
# Step 7: Visualize trajectories (PCA + straightness)
# ─────────────────────────────────────────────────────────────
run_step 7 "visualize" \
    modal run modal_app.py \
        --action visualize \
        --checkpoint "${STAGE1_CKPT}" \
        --checkpoint2 "${STAGE2_CKPT}" \
        --num-steps 50 \
        --seed 42

# ─────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}  Finished: $(date)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results on Modal volume (/data/):"
echo "  Stage 1 checkpoint: /data/${STAGE1_CKPT}"
echo "  Stage 2 checkpoint: /data/${STAGE2_CKPT}"
echo "  Reflow pairs:       /data/reflow_pairs/"
echo "  Stage 1 samples:    /data/samples/stage1/"
echo "  Stage 2 samples:    /data/samples/stage2/"
echo "  Visualizations:     /data/visualizations/"
echo ""
echo "Local logs: ${LOG_DIR}/"
echo ""
echo "To download results from Modal volume:"
echo "  modal volume get cmu-10799-diffusion-data samples/ ./results/samples/"
echo "  modal volume get cmu-10799-diffusion-data visualizations/ ./results/visualizations/"
echo "  modal volume ls cmu-10799-diffusion-data logs/"
