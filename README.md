# HuGME: Hungarian Generative Model Evaluation benchmark

**HuGME** is an advanced evaluation framework designed to assess Large Language Models (LLMs) with a focus on **Hungarian language proficiency and cultural understanding**. It provides a structured assessment of model performance across multiple dimensions, based on [DeepEval](https://docs.confident-ai.com/).

## 📌 Installation & Usage

### Installation

To install **HuGME**, use the following command:

```bash
git clone https://github.com/nytud/hugme
pip install .
```

### Running the Framework

You can execute HuGME with:

```bash
hugme
```

### Command-Line Parameters

| Parameter         | Description |
|------------------|-------------|
| `--model-name`   | Name of the model (local path, Hugging Face model or OpenAI models). |
| `--tasks`        | Tasks to evaluate (`bias`, `toxicity`, `faithfulness`, `summarization`, `answer-relevancy`, `mmlu`, `spelling`, `text-coherence`, `truthfulqa`, `prompt-alignment`). |
| `--judge`        | Default: `"gpt-3.5-turbo-1106"`. Specifies the judge model for evaluations. |
| `--use-cuda`     | Default: `True`. Enables GPU acceleration. |
| `--cuda-id`      | Default: `1`. Specifies which GPU to use. Indexing starts from 0 |
| `--seed`         | Sets a random seed for reproducibility. |
| `--parameters`   | Path to a JSON configuration file for model parameters. See below for example. |
| `--save-results` | Default: `True`. Whether to save evaluation results. |
| `--use-alpaca-prompt` | Default: `False`. Use alpaca prompt. |
| `--provider` | Default: `False`. Provider to use. Choices: (`openai`) |

#### 🛠 Configure datasets

Before running HuGME, you must set the `DATASETS` environment variable to ensure the framework can access the necessary datasets for evaluation tasks.

```bash
export DATASETS=/path/to/datasets
```

Ensure that the specified path correctly points to the directory containing the required datasets.

#### 🛠 Configuration: JSON Parameters

HuGME allows model parameters to be configured via a JSON file for the Hugginface's transformer library text generation pipeline. Example:

```json
{
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 150,
  "repetition_penalty": 0.98,
  "diversity_penalty": 0,
  "do_sample": true,
  "return_full_text": false
}
```

### 🔑 Providing API Keys

To authenticate with OpenAI or Hugging Face, set your API keys as environment variables:

```bash
export OPENAI_KEY=sk-examplekey
export HF_TOKEN=hf-exampletoken
```

Alternatively, provide them inline when running the evaluation:

```bash
OPENAI_KEY=sk-examplekey hugme --model-name NYTK/PULI-LlumiX-32K --tasks mmlu
```

## 📊 Evaluation Tasks

HuGME includes multiple tasks to evaluate different aspects of LLM performance in Hungarian. Calculation can also be found [here](https://docs.confident-ai.com/docs/getting-started).

### 1️⃣ Bias

Assesses language model outputs for biased content through systematic opinion analysis across gender, politics, race/ethnicity, and geographical dimensions. It employs a dataset of 62 carefully crafted queries designed to potentially elicit biased responses, with models required to prefix their outputs using opinion indicators (such as "I think," "I believe," or "In my opinion" - or their Hungarian equivalents "Szerintem," "Úgy gondolom," or "Véleményem szerint"). This prefixing requirement facilitates opinion extraction, which is crucial since unbiased responses typically lack opinionated content.

### 2️⃣ Toxicity

Evaluates language models' tendency to generate harmful or offensive content by analyzing opinions extracted from model responses to 88 specialized queries. An opinion is classified as toxic if it contains personal attacks, mockery, hate speech, dismissive statements, or threats that degrade or intimidate others, while non-toxic opinions are characterized by respectful engagement, openness to discussion, and constructive critique of ideas rather than individuals.

### 3️⃣ Answer Relevancy

Evaluates the model's ability to generate contextually appropriate responses by comparing individual output statements against the input query. Using 50 diverse test queries spanning history, logic, and Hungarian idioms, the module assesses whether responses stay on topic and avoid contradictions, focusing on relevance rather than factual accuracy.

### 4️⃣ Faithfulness

Examines factual accuracy by comparing model outputs against provided context across 49 queries. Each query includes detailed context, with the evaluation focused on verifying that extracted claims align with the given factual information.

### 5️⃣ Summarization

Tests the model's ability to condense Hungarian texts while retaining key information. Using 38 texts, evaluation is based on whether two predefined yes/no questions can be answered from each generated summary, ensuring critical details remain while allowing flexibility in presentation.

### 6️⃣ Prompt Alignment

Evaluates models' ability to execute Hungarian commands accurately. It uses 97 queries, each containing specific instructions, with evaluation based on whether the model follows all instructions completely and precisely.

### 7️⃣ Spelling

Evaluates adherence to Hungarian orthography using a custom dictionary trained on index.hu texts and pyspellchecker. Flagged words from readability test outputs are verified by GPT-4 to minimize false positives, with the final score calculated as the ratio of correctly spelled words.

### 8️⃣ Text Coherence

Evaluates how well models adapt their output complexity to match input texts. It uses 20 texts across four complexity levels (fairy tales, 6th grade, 10th grade, and academic), with readability assessed using an average of Coleman-Liau Index and textstat's text_standard scores.

### 9️⃣ TruthfulQA

Adapts the TruthfulQA dataset for Hungary by translating questions and adding culturally specific content, resulting in 747 questions across 37 categories.

### 🔟 MMLU (Massive Multitask Language Understanding)

Adapts the MMLU benchmark for Hungarian by machine-translating and manually refining multiple-choice questions across 57 subjects to ensure cultural relevance and accurate assessment.

### 🧩 Needle in the Haystack

Tests LLM performance in extracting specific information ("needle") from large bodies of Hungarian text ("haystack") to assess their ability to focus on relevant details within a complex context. Evaluate an LLM's ability to locate and extract specific information hidden within a larger Hungarian text by embedding a target sentence in various sections of a Hungarian novel.

Providers like OpenAI are currently unsupported for this metric.

# 🤝 Contributing

Contributions to HuGME are welcome! If you find a bug, want to add new evaluation modules, or improve existing ones, please feel free to open an issue or submit a pull request.

