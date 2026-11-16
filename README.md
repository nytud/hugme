# HuGME: Hungarian Generative Model Evaluation benchmark

**HuGME** is an evaluation framework for Large Language Models (LLMs) with a focus on **Hungarian language proficiency and cultural understanding**. It measures model performance across multiple dimensions, from bias and toxicity to spelling, world knowledge and long-context retrieval. Several modules use an LLM-as-a-judge approach built on [DeepEval](https://docs.confident-ai.com/).

HuGME does not load models itself. It sends requests to any **OpenAI-compatible chat completions endpoint**, typically a model served with [vLLM](https://docs.vllm.ai/), or the OpenAI API directly.

More details and results: [hugme.nytud.hu](https://hugme.nytud.hu/)

## 📌 Installation

HuGME requires Python 3.9 or newer (CI runs on 3.11, 3.12 and 3.13).

Install from PyPI:

```bash
pip install hugme
```

Or install from source:

```bash
git clone https://github.com/nytud/hugme
cd hugme
pip install .
```

**Troubleshooting `huspacy` installation**

On first run, HuGME downloads the `hu_core_news_lg` spaCy model, which the `cola` task uses for sentence splitting. If the automatic download fails, install the model manually:

```bash
wget -O hu_core_news_lg-3.8.0-py3-none-any.whl "https://huggingface.co/huspacy/hu_core_news_lg/resolve/v3.8.0/hu_core_news_lg-any-py3-none-any.whl"
pip install ./hu_core_news_lg-3.8.0-py3-none-any.whl
```

## 🚀 Quick start

**1. Serve the model** with an OpenAI-compatible server. For example, with vLLM in Docker (see also `scripts/run_docker.sh`):

```bash
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:v0.27.1-cu129 \
    --model Qwen/Qwen3-4B
```

**2. Set the environment variables** (see [Configuration](#-configuration) for the full list):

```bash
export DATASETS=/path/to/datasets/      # note the trailing slash
export MODEL_API_KEY=EMPTY              # vLLM without --api-key accepts any value
export OPENAI_API_KEY=sk-...            # judge model for the LLM-as-a-judge tasks
```

**3. Run the evaluation:**

```bash
hugme --model-name Qwen/Qwen3-4B \
      --model-url http://localhost:8000/v1 \
      --tasks mmlu truthfulqa \
      --parameters parameters.json \
      --save-results
```

`scripts/run.sbatch` contains a complete Slurm job example. Some tasks can fail on a judge timeout, so running each task as a separate `hugme` call is recommended: a failure in one task then doesn't affect the others.

### Evaluating an OpenAI model

Pass `not_given` as the model URL to use the default OpenAI API endpoint, and set `MODEL_API_KEY` to your OpenAI key (if it isn't set, the OpenAI client falls back to `OPENAI_API_KEY`):

```bash
hugme --model-name gpt-4o-mini --model-url not_given --tasks mmlu --parameters parameters.json --save-results
```

The `nih` task is not supported this way, because it needs the server's `/tokenize` and `/detokenize` endpoints, which only vLLM provides.

## ⌨️ Command-line parameters

| Parameter | Description |
|---|---|
| `--model-name` | **Required.** Model name as the server knows it (for vLLM, the `--model` or `--served-model-name` value; for OpenAI, the model ID). |
| `--model-url` | **Required.** Base URL of the OpenAI-compatible server, e.g. `http://localhost:8000/v1`. Use `not_given` for the default OpenAI API. |
| `--parameters` | **Required.** Path to a JSON file with generation parameters. See [Generation parameters](#generation-parameters). |
| `--tasks` | One or more task names, separated by spaces. See [Evaluation tasks](#-evaluation-tasks) for the valid names. |
| `--judge` | Default: `gpt-4o`. Judge model for the LLM-as-a-judge tasks, the `cola` fallback check and `cultural-open`. |
| `--save-results` | Flag; off by default. Saves generation outputs, per-item evaluations and the final scores to `RESULTS_DIR`. |
| `--use-gen-results` | Path to a saved generation results file. Skips generation and evaluates these outputs instead. Use it with a single task. |
| `--use-eval-results` | Path to a saved evaluation results file of a judge-based metric (`bias`, `toxicity`, `faithfulness`, `summarization`, `answer-relevancy`). Already evaluated items are reused, and evaluation continues from where it stopped. |
| `--sample-size` | Default: `1.0`. Fraction of each task's dataset to sample randomly (at least one item). |
| `--batch-size` | Default: `8`. Number of requests sent to the model server concurrently. |

## 🛠 Configuration

HuGME is configured through environment variables.

| Variable | Used by | Default | Description |
|---|---|---|---|
| `DATASETS` | all tasks | `./datasets/` | Directory containing the dataset files. **Must end with `/`**, because file names are appended to it directly. |
| `RESULTS_DIR` | all tasks | `./results/` | Directory for the output files. |
| `MODEL_API_KEY` | all tasks | – | API key for the model server. For vLLM started without `--api-key`, any value works (e.g. `EMPTY`). |
| `OPENAI_API_KEY` | `bias`, `toxicity`, `faithfulness`, `summarization`, `answer-relevancy`, `prompt-alignment`, `cola` | – | Key for the OpenAI judge model. |
| `PROVIDER_URL` | `cultural-open` | – | Base URL of an OpenAI-compatible endpoint serving the judge model. |
| `PROVIDER_API_KEY` | `cultural-open` | – | API key for `PROVIDER_URL`. |
| `BERT_MODEL` | `cola` | – | Path or Hugging Face ID of the Hungarian BERT grammaticality classifier. |
| `MODEL_CONTEXT_LEN` | `nih` | `32768` | Largest context length to test. Must not exceed the server's maximum model length. |
| `N_TURNS` | `nih` | `2` | Number of random needle positions tested per depth band and context length. |

`HF_TOKEN` is not needed by HuGME itself; pass it to the vLLM server if the model is gated on Hugging Face.

For long outputs judged by DeepEval, you can raise DeepEval's own request timeout with `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` (in seconds).

### Generation parameters

The JSON file given in `--parameters` is passed unchanged to the OpenAI client's `chat.completions.create` call. Any parameter of the chat completions API works. Server-specific sampling options, such as vLLM's `top_k`, `min_p` or chat template settings, go under `extra_body`. Example (`parameters.json` in the repository):

```json
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
```

The parameters are printed to the log, so don't put credentials in this file. Use the environment variables above instead.

### Reasoning models

For models that think before answering, the text between `<think>` and `</think>` is removed before scoring and kept in the `thinking_output` field of the results. If an output has an opening `<think>` tag but no closing one (for example because `max_completion_tokens` ran out), the answer is treated as empty. Thinking tokens count toward `max_completion_tokens`, so leave enough room for the reasoning and the answer.

For hybrid models, thinking can be switched off through `chat_template_kwargs` (for Qwen3-style templates: `{"enable_thinking": false}`). Use non-thinking mode for the `nih` task.

## 🧠 Results

With `--save-results`, all files are written to `RESULTS_DIR`. In file names, `<model>` is the model name in lowercase, with `/` replaced by `_`.

| File | Content |
|---|---|
| `hugme-<model>-results-<timestamp>.json` | Final scores of all tasks in the run. |
| `<task>-<model>-generation-results.json` | Model outputs, saved after every batch. |
| `<task>-<model>-eval-results.json` | Per-item evaluation details. |
| `cultural-open-<model>-uncertain-cases.json` | Items the judge could not decide on. |
| `nih-<model>-plot.png` | Needle-in-the-haystack heatmap (always written). |

The judge-based metrics (`bias`, `toxicity`, `faithfulness`, `summarization`, `answer-relevancy`) save their evaluation file after every item, even without `--save-results`. If the judge connection fails mid-run, pass that file to `--use-eval-results` to continue from where it stopped.

## 📊 Evaluation tasks

Pass the task names in `--tasks`, e.g. `--tasks bias toxicity mmlu`.

### LLM-as-a-judge

These tasks are scored by the `--judge` model through DeepEval. Outputs that are empty or highly repetitive are scored as failures without calling the judge.

#### `bias`

Assesses outputs for biased content. The judge extracts the opinions from each response and checks them for gender, political, racial/ethnic and geographical bias. The dataset contains 100 queries crafted to potentially elicit biased responses. Score: percentage of responses passing the metric (threshold 0.5).

#### `toxicity`

Evaluates the tendency to generate harmful or offensive content. The prompt instructs the model to start its answer with an opinion marker (*Szerintem* 'I think', *Úgy gondolom* 'I believe', or *Véleményem szerint* 'In my opinion'), which makes the opinions easy to extract. An opinion counts as toxic if it contains personal attacks, mockery, hate speech, dismissive statements or threats. Score: percentage of responses passing the metric (threshold 0.5).

#### `answer-relevancy`

Evaluates whether responses stay on topic. The judge compares each statement of the output with the query, focusing on relevance rather than factual accuracy. The queries span history, logic and Hungarian idioms. Score: percentage of responses passing the metric (threshold 0.7).

#### `faithfulness`

Checks factual accuracy against a given context. Each query comes with a context, and the judge verifies that the claims in the output agree with it. Score: percentage of responses passing the metric (threshold 0.5).

#### `summarization`

Tests whether the model can condense Hungarian texts while keeping the key information. For each of the 50 texts, the judge checks whether four predefined yes/no questions can be answered from the generated summary. Score: percentage of summaries passing the metric (threshold 0.5).

#### `prompt-alignment`

Evaluates whether the model follows Hungarian instructions completely and precisely. Each query contains specific instructions, and the judge checks the output against them. Returns the success rate (judge score ≥ 0.5) and the average judge score.

### Language proficiency

#### `spelling`

Evaluates adherence to Hungarian orthography. The model summarizes texts, and `pyspellchecker` with a custom Hungarian dictionary (built from index.hu texts) counts the words it doesn't recognize. Score: percentage of words found in the dictionary.

#### `readability`

Evaluates how well the model matches the complexity of its input. The model continues 20 texts from four complexity levels (fairy tales, 6th grade, 10th grade and academic texts) in the same style. The Coleman–Liau index and textstat's `text_standard` of the input and the continuation are compared, weighted 60/40, into a similarity score from 0 to 100.

#### `cola`

Evaluates grammaticality. The model summarizes texts, huspacy splits each summary into sentences, and a fine-tuned Hungarian BERT classifier (`BERT_MODEL`) labels each sentence. Sentences the classifier marks as ungrammatical get a second check by the judge model. Score: percentage of grammatical sentences.

### World knowledge

#### `mmlu`

HuMMLU, a Hungarian adaptation of the MMLU benchmark: multiple-choice questions, machine-translated and manually refined for the Hungarian context. The model answers with the letter of the correct option. Scores are reported per subject and in total.

#### `truthfulqa`

HuTruthfulQA, an adaptation of TruthfulQA with translated questions and culturally specific Hungarian content: 747 questions in 37 categories. The model sees a list of correct and a list of incorrect answers, in random order, and has to return the number of the correct list. Scores are reported per category and in total.

#### `cultural-abcd`

Multiple-choice questions on Hungarian culture (literature, music, geography, foods and drinks, folk culture, sport and more). The answer letter is extracted from the output, also from formatted or explained answers. Scores are reported per category and in total.

#### `cultural-open`

Open-ended questions on Hungarian culture, with three answer types. *Entity* answers (a name of a person, place, dish, etc.) are matched against the gold answer and its accepted aliases. *Short answers* and *explanations* are graded by the judge model (served at `PROVIDER_URL`) against a rubric of required elements, optional elements and critical errors. The verdicts count as correct (1), partially correct (0.5), or incorrect and uncertain (0). Returns scores per category and a summary of the verdicts.

### Long context

#### `nih` (Needle in the Haystack)

Tests whether the model can retrieve a single fact hidden in a long Hungarian text. A sentence stating that a randomly chosen town celebrated a given anniversary is inserted into a Hungarian novel, and the model is asked which anniversary the town celebrated.

The test runs over context lengths from 1024 tokens to `MODEL_CONTEXT_LEN` in steps of 1024, and over ten depth bands (0–10%, 10–20%, …, 90–100% of the text). Within each band, the needle is placed at `N_TURNS` random positions. The prompt, the needle, the chat template and `max_completion_tokens` are all counted within each context length, so every request fits the model's window. An answer is correct if the first number in it equals the anniversary. The results are saved as a heatmap of success rate by context length and depth.

This task requires a vLLM server, because it uses the server's tokenizer endpoints. Use non-thinking mode and a small `max_completion_tokens` (for example 32), since the answer is a single number and the budget is reserved in every context length.

## 🧰 Scripts

| Script | Purpose |
|---|---|
| `scripts/run.sbatch` | Example Slurm job: sets the environment, writes a parameters file and runs a task. |
| `scripts/run_docker.sh` | Example command for serving a model with the vLLM Docker image. |
| `scripts/calculations.py` | Aggregates the final scores of repeated runs into mean and standard deviation tables. |
| `scripts/compare_cultural.py` | Compares `cultural-abcd` and `cultural-open` results of a model question by question. |

## 🤝 Contributing

Contributions to HuGME are welcome! If you find a bug, want to add new evaluation modules, or improve existing ones, please open an issue or submit a pull request.

For development, install the package in editable mode and set up the pre-commit hooks (ruff, mypy, pylint):

```bash
pip install -e .
pip install pre-commit ruff mypy pylint pytest
pre-commit install
```

Run the tests with:

```bash
python3 -m pytest tests/
```

## 📄 License

HuGME is licensed under the [Apache License 2.0](LICENCE).
