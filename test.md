
```bash
pip install .

export LOG_LEVEL=4
# all: answer-relevancy`, `hallucination`, `needle-in-haystack`

# done:
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks answer-relevancy --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks summarization --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks faithfulness --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks toxicity --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks bias --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks spelling --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks readability --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks prompt-alignment --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks text-coherence --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=10GB hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks truthfulqa --sample-size 0.05 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks readability --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
# error:

# runs
srun -w lolka --gres=gpu:1 --mem=10GB hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json --batch-size 8
srun -w lolka --gres=gpu:1 --mem=10GB hugme --model-name meta-llama/Llama-3.1-8B-Instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json --batch-size 8
# runtime error:
srun -w lolka --gres=gpu:1 --mem=10GB hugme --model-name microsoft/Phi-4-mini-instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json

hugme --model-name microsoft/Phi-4-mini-instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json --cuda-ids 0,1,2,3
hugme --model-name google/gemma-3-4b-it --tasks mmlu --parameters /home/osvathm/hugme/parameters.json --cuda-ids 0,1,2,3
hugme --model-name NYTK/puli-llumix-instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json --cuda-ids 0,1,2,3 --use-alpaca-prompt
```