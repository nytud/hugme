#!/bin/bash
#SBATCH --job-name=test-model
#SBATCH --nodes=1
#SBATCH --nodelist=test-node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --partition=hi
#SBATCH --error=/home/username/hugme/logs/job.%J.err
#SBATCH --output=/home/username/hugme/logs/job.%J.out

export BERT_MODEL=""
export BERT_MODEL=""
export DATASETS=""

export MODEL_API_KEY=""
export OPENAI_API_KEY=""

# increase the per-attempt timeout for deepeval
# for longer texts (relevant for >32k context len)
export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300


echo "================ $SLURM_JOB_NAME ======================="

export MODEL_NAME=""
export MODEL_URL=""
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
export RESULTS_DIR="/home/username/results/$SLURM_JOB_NAME/run_$RUN_NUMBER"
export TASK="answer-relevancy"

# note: some hugme eval can fail, e.g. due to judge model time out, therefore
# it is recommended to run evals separately, so a fail in one does not affect the others' run
hugme --model-name $MODEL_NAME --model-url $MODEL_URL --tasks $TASK --parameters $PARAMETERS_PATH --batch-size 16 --save-results


for RUN_NUMBER in {1..5}; do

    export RUN_NUMBER
    export RESULTS_DIR="/home/username/results/$SLURM_JOB_NAME/run_$RUN_NUMBER"

    echo "=========== Run $RUN_NUMBER ==========="
    echo "Results Directory: $RESULTS_DIR"

    hugme --model-name $MODEL_NAME --model-url $MODEL_URL --tasks mmlu --parameters $PARAMETERS_PATH --batch-size 32 --save-results

done