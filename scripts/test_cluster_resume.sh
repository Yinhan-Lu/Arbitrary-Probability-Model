#!/bin/bash
# ==========================================================================
# CLUSTER CHECKPOINT RESUME TEST
# ==========================================================================
# This script tests that checkpoint resume works correctly on the cluster.
#
# What it does:
# 1. Submits a test job (small config for fast testing)
# 2. Monitors for checkpoint to appear
# 3. Auto-scancels the job
# 4. Resubmits the SAME configuration
# 5. Verifies the resume worked (checks for "Using existing exp_name:" in logs)
#
# Usage:
#   ./scripts/test_cluster_resume.sh           # Run the full test
#   ./scripts/test_cluster_resume.sh --dry-run # Show what would be done
#   ./scripts/test_cluster_resume.sh --cleanup # Remove test experiments
#
# Expected time: ~10-15 minutes total
# ==========================================================================

set -e

# ==========================================================================
# CONFIGURATION
# ==========================================================================

# Test experiment configuration (small for fast testing)
MODEL_CONFIG="distilgpt2"
COND_MAX="0.2"
ORDERING="temporal"
THINKING="expectation"

# Use a fixed prefix for test experiments (easy to find and clean up)
TEST_PREFIX="TEST_RESUME"

# Polling interval (seconds)
POLL_INTERVAL=30

# Maximum wait time for checkpoint (seconds) - 10 minutes
MAX_WAIT=600

# Parse arguments
DRY_RUN=false
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--cleanup]"
            exit 1
            ;;
    esac
done

# ==========================================================================
# CLEANUP MODE
# ==========================================================================
if [ "$CLEANUP" == "true" ]; then
    echo "========================================="
    echo "CLEANUP MODE"
    echo "========================================="

    # Find and remove test experiments
    TEST_FOLDERS=$(ls -d experiments/${TEST_PREFIX}_* 2>/dev/null || true)

    if [ -z "$TEST_FOLDERS" ]; then
        echo "No test experiments found."
        exit 0
    fi

    echo "Found test experiments:"
    for folder in $TEST_FOLDERS; do
        echo "  $folder"
    done

    echo ""
    read -p "Delete these folders? (yes/no): " confirm
    if [ "$confirm" == "yes" ]; then
        for folder in $TEST_FOLDERS; do
            echo "Deleting: $folder"
            rm -rf "$folder"
        done
        echo "Done!"
    else
        echo "Aborted."
    fi
    exit 0
fi

# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

get_job_status() {
    local job_id=$1
    squeue -j "$job_id" -h -o "%t" 2>/dev/null || echo "GONE"
}

wait_for_checkpoint() {
    local exp_pattern=$1
    local start_time=$(date +%s)

    echo "Waiting for checkpoint to appear..."
    echo "  Pattern: experiments/${exp_pattern}/checkpoints/checkpoint_step_*.pt"
    echo "  Polling every ${POLL_INTERVAL}s (max ${MAX_WAIT}s)"
    echo ""

    while true; do
        # Find the experiment folder
        EXP_FOLDER=$(ls -dt experiments/${exp_pattern} 2>/dev/null | head -1)

        if [ -n "$EXP_FOLDER" ] && [ -d "$EXP_FOLDER/checkpoints" ]; then
            CHECKPOINT=$(ls -v "$EXP_FOLDER/checkpoints/checkpoint_step_"*.pt 2>/dev/null | tail -1)

            if [ -n "$CHECKPOINT" ]; then
                echo "✓ Checkpoint found: $CHECKPOINT"
                return 0
            fi
        fi

        # Check timeout
        local elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -ge $MAX_WAIT ]; then
            echo "✗ Timeout waiting for checkpoint (${elapsed}s elapsed)"
            return 1
        fi

        echo "  [$(date +%H:%M:%S)] No checkpoint yet (${elapsed}s elapsed)..."
        sleep $POLL_INTERVAL
    done
}

# ==========================================================================
# MAIN TEST
# ==========================================================================

echo "========================================="
echo "CLUSTER CHECKPOINT RESUME TEST"
echo "========================================="
echo "Config:"
echo "  Model: $MODEL_CONFIG"
echo "  Cond: 0.0-$COND_MAX"
echo "  Ordering: $ORDERING"
echo "  Thinking: $THINKING"
echo "  Test prefix: $TEST_PREFIX"
echo ""

if [ "$DRY_RUN" == "true" ]; then
    echo "[DRY-RUN MODE - No jobs will be submitted]"
    echo ""
fi

# Generate test experiment name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COND_PCT=$(python3 -c "print(int($COND_MAX * 100))")
EXP_NAME="${TEST_PREFIX}_cond0-${COND_PCT}_rope_${MODEL_CONFIG}_sigmagpt_${ORDERING}_think_${THINKING}_${TIMESTAMP}"
EXP_PATTERN="${TEST_PREFIX}_cond0-${COND_PCT}_rope_${MODEL_CONFIG}_sigmagpt_${ORDERING}_think_${THINKING}_*"

echo "Experiment name: $EXP_NAME"
echo "Pattern for resume: $EXP_PATTERN"
echo ""

# ==========================================================================
# PHASE 1: Submit initial job
# ==========================================================================

echo "========================================="
echo "PHASE 1: Submit Initial Job"
echo "========================================="

# Create a temporary job script (quick version for testing)
SCRIPT_FILE="/tmp/test_resume_${TIMESTAMP}.sh"

cat > "$SCRIPT_FILE" << 'SCRIPT_CONTENT'
#!/bin/bash
#SBATCH --job-name=test_resume
#SBATCH --output=logs/test_resume_%j.out
#SBATCH --error=logs/test_resume_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1

echo "========================================="
echo "TEST RESUME EXPERIMENT"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

nvidia-smi

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load modules (Mila cluster)
module load cuda/12.1.1/cudnn/8.9
source /cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/etc/profile.d/conda.sh
conda activate arbprob

echo "Python: $(which python3)"
echo "========================================="

SCRIPT_CONTENT

# Add experiment-specific variables
cat >> "$SCRIPT_FILE" << EOF

# Experiment configuration
EXP_NAME="$EXP_NAME"
MODEL_CONFIG="$MODEL_CONFIG"
COND_MAX="$COND_MAX"
ORDERING="$ORDERING"
THINKING="$THINKING"
EXP_PATTERN="$EXP_PATTERN"

EOF

# Add auto-resume logic
cat >> "$SCRIPT_FILE" << 'SCRIPT_CONTENT'

# =========================================================================
# AUTO-RESUME LOGIC (This is what we're testing!)
# =========================================================================
EXISTING_EXP=$(ls -dt ./experiments/${EXP_PATTERN} 2>/dev/null | head -1)
RESUME_ARG=""

if [ -n "$EXISTING_EXP" ] && [ -d "$EXISTING_EXP/checkpoints" ]; then
    LATEST_CKPT=$(ls -v "$EXISTING_EXP/checkpoints/checkpoint_step_"*.pt 2>/dev/null | tail -1)

    if [ -n "$LATEST_CKPT" ]; then
        # CRITICAL: Use the existing folder name, not the new timestamp!
        # This ensures we continue in the same folder and append to metrics.csv
        EXP_NAME=$(basename "$EXISTING_EXP")
        echo "========================================="
        echo "RESUMING FROM CHECKPOINT"
        echo "  Experiment: $EXISTING_EXP"
        echo "  Checkpoint: $LATEST_CKPT"
        echo "  Using existing exp_name: $EXP_NAME"
        echo "========================================="
        RESUME_ARG="--resume_from $LATEST_CKPT"
    fi
fi

# Run training (small config for testing)
# - Only 500 samples for quick iteration
# - Save checkpoint every 50 steps
# - 1 epoch only
python3 ./train.py \
    --model_type sigmagpt \
    --model_config ${MODEL_CONFIG} \
    --position_encoding_type rope \
    --sigmagpt_mode fair \
    --ordering_mode ${ORDERING} \
    --use_thinking_tokens \
    --thinking_token_mode ${THINKING} \
    --num_epochs 1 \
    --batch_size 8 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_samples 500 \
    --num_eval_samples 100 \
    --learning_rate 5e-4 \
    --warmup_steps 50 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --cond_pct_min 0.0 \
    --cond_pct_max ${COND_MAX} \
    --eval_pct_min 1.0 \
    --eval_pct_max 1.0 \
    --conditioning_sampling blockwise \
    --evaluation_sampling blockwise \
    --mode2_boundary_cond_pct_min 0.0 \
    --mode2_boundary_cond_pct_max ${COND_MAX} \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 50 \
    --early_stopping_patience 0 \
    --do_eval \
    --max_eval_batches 5 \
    --output_dir ./experiments \
    --exp_name ${EXP_NAME} \
    --device cuda \
    --num_workers 4 \
    $RESUME_ARG

EXIT_CODE=$?

echo "========================================="
echo "Training Completed"
echo "Exit code: $EXIT_CODE"
echo "Duration: $SECONDS seconds"
echo "========================================="

exit $EXIT_CODE
SCRIPT_CONTENT

echo "Created job script: $SCRIPT_FILE"

if [ "$DRY_RUN" == "true" ]; then
    echo "[DRY-RUN] Would submit: $SCRIPT_FILE"
    echo ""
    echo "Script contents:"
    echo "----------------------------------------"
    cat "$SCRIPT_FILE"
    echo "----------------------------------------"
    exit 0
fi

# Submit the job
JOB_ID_1=$(sbatch "$SCRIPT_FILE" | awk '{print $4}')
echo "Submitted job: $JOB_ID_1"
echo ""

# Wait for job to start
echo "Waiting for job to start..."
while true; do
    STATUS=$(get_job_status $JOB_ID_1)
    if [ "$STATUS" == "R" ]; then
        echo "✓ Job is running"
        break
    elif [ "$STATUS" == "GONE" ]; then
        echo "✗ Job disappeared (check logs for errors)"
        exit 1
    fi
    echo "  Status: $STATUS"
    sleep 10
done
echo ""

# ==========================================================================
# PHASE 2: Wait for checkpoint and cancel
# ==========================================================================

echo "========================================="
echo "PHASE 2: Wait for Checkpoint and Cancel"
echo "========================================="

if ! wait_for_checkpoint "$EXP_PATTERN"; then
    echo "Failed to find checkpoint. Cancelling job..."
    scancel $JOB_ID_1
    exit 1
fi

# Get the experiment folder name
EXP_FOLDER=$(ls -dt experiments/${EXP_PATTERN} 2>/dev/null | head -1)
FOLDER_NAME=$(basename "$EXP_FOLDER")

echo ""
echo "Experiment folder: $FOLDER_NAME"

# Count metrics.csv lines before cancel
METRICS_FILE="$EXP_FOLDER/logs/metrics.csv"
if [ -f "$METRICS_FILE" ]; then
    LINES_BEFORE=$(wc -l < "$METRICS_FILE")
    echo "metrics.csv lines before cancel: $LINES_BEFORE"
else
    LINES_BEFORE=0
    echo "metrics.csv not found before cancel"
fi

echo ""
echo "Cancelling job $JOB_ID_1..."
scancel $JOB_ID_1

# Wait for job to be fully cancelled
sleep 5
echo "✓ Job cancelled"
echo ""

# ==========================================================================
# PHASE 3: Resubmit and verify resume
# ==========================================================================

echo "========================================="
echo "PHASE 3: Resubmit and Verify Resume"
echo "========================================="

# Create a new timestamp for resubmit (to test that it uses the OLD folder)
TIMESTAMP_2=$(date +%Y%m%d_%H%M%S)
EXP_NAME_2="${TEST_PREFIX}_cond0-${COND_PCT}_rope_${MODEL_CONFIG}_sigmagpt_${ORDERING}_think_${THINKING}_${TIMESTAMP_2}"

echo "New EXP_NAME (should be overridden by resume): $EXP_NAME_2"
echo ""

# Update the script with new EXP_NAME
sed -i.bak "s/EXP_NAME=\".*\"/EXP_NAME=\"$EXP_NAME_2\"/" "$SCRIPT_FILE"

# Resubmit
JOB_ID_2=$(sbatch "$SCRIPT_FILE" | awk '{print $4}')
echo "Submitted job: $JOB_ID_2"
echo ""

# Wait for job to start
echo "Waiting for job to start..."
while true; do
    STATUS=$(get_job_status $JOB_ID_2)
    if [ "$STATUS" == "R" ]; then
        echo "✓ Job is running"
        break
    elif [ "$STATUS" == "GONE" ]; then
        echo "✗ Job disappeared (check logs for errors)"
        exit 1
    fi
    echo "  Status: $STATUS"
    sleep 10
done
echo ""

# Give it some time to start and log the resume message
echo "Waiting 60 seconds for job to initialize and log resume message..."
sleep 60

# ==========================================================================
# PHASE 4: Verification
# ==========================================================================

echo "========================================="
echo "PHASE 4: Verification"
echo "========================================="

# Find the log file for job 2
LOG_FILE=$(ls -t logs/test_resume_${JOB_ID_2}.out 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "WARNING: Could not find log file for job $JOB_ID_2"
    echo "Looking in logs/ for recent files..."
    ls -lt logs/*.out | head -5
else
    echo "Log file: $LOG_FILE"
    echo ""

    # Check for the magic resume message
    echo "--- Checking for resume message ---"
    if grep -q "Using existing exp_name:" "$LOG_FILE"; then
        echo "✓ PASS: Found 'Using existing exp_name:' in logs"
        grep "Using existing exp_name:" "$LOG_FILE"
    else
        echo "✗ FAIL: Did not find 'Using existing exp_name:' in logs"
        echo ""
        echo "Log contents:"
        cat "$LOG_FILE"
    fi
fi

echo ""

# Check that no new folder was created
MATCHING_FOLDERS=$(ls -d experiments/${EXP_PATTERN} 2>/dev/null | wc -l)
if [ "$MATCHING_FOLDERS" -eq 1 ]; then
    echo "✓ PASS: Only one experiment folder exists"
    ls -d experiments/${EXP_PATTERN}
else
    echo "✗ FAIL: Multiple experiment folders created!"
    ls -d experiments/${EXP_PATTERN}
fi

echo ""

# Let the job run a bit more for metrics
echo "Waiting 60 more seconds for metrics to accumulate..."
sleep 60

# Check metrics.csv
METRICS_FILE="$EXP_FOLDER/logs/metrics.csv"
if [ -f "$METRICS_FILE" ]; then
    LINES_AFTER=$(wc -l < "$METRICS_FILE")
    echo "metrics.csv lines after resume: $LINES_AFTER"

    if [ "$LINES_AFTER" -gt "$LINES_BEFORE" ]; then
        echo "✓ PASS: metrics.csv grew from $LINES_BEFORE to $LINES_AFTER lines"
    else
        echo "✗ FAIL: metrics.csv did not grow (before: $LINES_BEFORE, after: $LINES_AFTER)"
    fi
else
    echo "✗ FAIL: metrics.csv not found after resume"
fi

echo ""

# Cancel job 2 (cleanup)
echo "Cancelling job $JOB_ID_2 (test complete)..."
scancel $JOB_ID_2

echo ""
echo "========================================="
echo "TEST COMPLETE"
echo "========================================="
echo ""
echo "Results summary:"
echo "  - Job 1 (initial): $JOB_ID_1"
echo "  - Job 2 (resume): $JOB_ID_2"
echo "  - Experiment folder: $FOLDER_NAME"
echo ""
echo "To view logs:"
echo "  cat logs/test_resume_${JOB_ID_1}.out"
echo "  cat logs/test_resume_${JOB_ID_2}.out"
echo ""
echo "To cleanup test experiments:"
echo "  ./scripts/test_cluster_resume.sh --cleanup"
echo ""
