#!/bin/bash
#SBATCH --job-name=test-model
#SBATCH --nodes=1
#SBATCH --nodelist=test-node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --partition=hi
#SBATCH --error=/home/username/hugme/logs/job.%J.err
#SBATCH --output=/home/username/hugme/logs/job.%J.out


export DATASETS=""
export BERT_MODEL=""

export HF_TOKEN=""
export OPENAI_API_KEY=""

# for cultural open task
export PROVIDER_URL=""
export PROVIDER_API_KEY=""

# for local model serving
export MODEL_API_KEY="EMPTY"

# increase the per-attempt timeout for deepeval
# for longer texts (relevant for >32k context len)
export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300


echo "================ $SLURM_JOB_NAME ======================="

export MODEL_NAME="Qwen/Qwen3-4B"
export MODEL_URL="http://localhost:8001/v1"
export PARAMETERS_PATH="/home/username/hugme/parameters-model-name.json"
cat > "$PARAMETERS_PATH" << 'EOF'
{
    "max_completion_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "extra_body": {
        "top_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {"enable_thinking": true}
    }
}
EOF

echo "Model Name: $MODEL_NAME"
echo "Model URL: $MODEL_URL"
echo "Parameters Path: $PARAMETERS_PATH"

# ================== EXAMPLE TASKS =========================

export RUN_NUMBER=1
echo "Run $RUN_NUMBER"
export RESULTS_DIR="/home/osvathm/hugme/results/$SLURM_JOB_NAME/run_$RUN_NUMBER"

# note: some hugme eval can fail, e.g. due to judge model time out,
# therefore, it is recommended (for now) to run evals separately,
# i.e. fail in one does not affect the others' run
export TASK="mmlu"
hugme --model-name $MODEL_NAME --model-url $MODEL_URL --tasks $TASK \
        --parameters $PARAMETERS_PATH --save-results

