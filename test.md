
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
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name NYTK/PULI-LlumiX-Llama-3.1 --tasks mmlu --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks truthfulqa --sample-size 0.05 --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name Qwen/Qwen2.5-7B-Instruct --tasks readability --sample-size 0.1 --parameters /home/osvathm/hugme/parameters.json
# error:

srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name meta-llama/Llama-3.1-8B-Instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json
srun -w lolka --gres=gpu:1 --mem=5000 hugme --model-name microsoft/Phi-4-mini-instruct --tasks mmlu --parameters /home/osvathm/hugme/parameters.json
```