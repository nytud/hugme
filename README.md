# HuGME library
HuGME is an easy-to-use LLM assessment framework, which explicitly rates the Hungarian language skills and cultural knowledge of the models. It can be used to evaluate LLM outputs based on metrics such as hallucination, relevance of response, bias and more.

## Installation
```bash
pip install .
```

## Usage
```bash
hugme
```

### Parameters for usage
--model-name: name of the model (local path or huggingface)
--tokenizer-name: optional, provide only if it is different than the model
--tasks: bias, toxicity, faithfulness, summarization, answer-relevancy, mmlu, spelling, text-coherence, truthfulqa, prompt-alignment
--judge: default="gpt-3.5-turbo-1106"
--use-cuda: default: True
--cuda-id: which gpu to use default=0 
--seed: random seed 
--parameters: path to JSON config file for model params
--save-results: default: True

## Tasks

### Bias
The Bias Evaluation module assesses language model outputs for biased content through systematic opinion analysis across gender, politics, race/ethnicity, and geographical dimensions. It employs a dataset of 62 carefully crafted queries designed to potentially elicit biased responses, with models required to prefix their outputs using opinion indicators (such as "I think," "I believe," or "In my opinion" - or their Hungarian equivalents "Szerintem," "Úgy gondolom," or "Véleményem szerint"). This prefixing requirement facilitates opinion extraction, which is crucial since unbiased responses typically lack opinionated content.

### Toxicity
The Toxicity module evaluates language models' tendency to generate harmful or offensive content by analyzing opinions extracted from model responses to 88 specialized queries. An opinion is classified as toxic if it contains personal attacks, mockery, hate speech, dismissive statements, or threats that degrade or intimidate others, while non-toxic opinions are characterized by respectful engagement, openness to discussion, and constructive critique of ideas rather than individuals.

### Answer Relevancy
The Relevance module evaluates the model's ability to generate contextually appropriate responses by comparing individual output statements against the input query. Using 50 diverse test queries spanning history, logic, and Hungarian idioms, the module assesses whether responses stay on topic and avoid contradictions, focusing on relevance rather than factual accuracy.

### Faithfulness
The Faithfulness module examines factual accuracy by comparing model outputs against provided context across 49 queries. Each query includes detailed context, with the evaluation focused on verifying that extracted claims align with the given factual information.

### Summarization
The Summarization module tests the model's ability to condense Hungarian texts while retaining key information. Using 38 texts, evaluation is based on whether two predefined yes/no questions can be answered from each generated summary, ensuring critical details remain while allowing flexibility in presentation.

### Prompt Alignment
The Instruction Following module evaluates models' ability to execute Hungarian commands accurately. It uses 97 queries, each containing specific instructions, with evaluation based on whether the model follows all instructions completely and precisely.

### Spelling
The Spelling sub-module evaluates adherence to Hungarian orthography using a custom dictionary trained on index.hu texts and pyspellchecker. Flagged words from readability test outputs are verified by GPT-4 to minimize false positives, with the final score calculated as the ratio of correctly spelled words.

### Text Coherence
This module evaluates how well models adapt their output complexity to match input texts. It uses 20 texts across four complexity levels (fairy tales, 6th grade, 10th grade, and academic), with readability assessed using an average of Coleman-Liau Index and textstat's text_standard scores.

### TruthfulQA
This module adapts the TruthfulQA dataset for Hungary by translating questions and adding culturally specific content, resulting in 747 questions across 37 categories.

### MMLU
This module adapts the MMLU benchmark for Hungarian by machine-translating and manually refining multiple-choice questions across 57 subjects to ensure cultural relevance and accurate assessment.

### Needle in the Haystack
This module tests LLM performance in extracting specific information ("needle") from large bodies of Hungarian text ("haystack") to assess their ability to focus on relevant details within a complex context.
This module evaluates an LLM's ability to locate and extract specific information hidden within a larger Hungarian text by embedding a target sentence in various sections of a Hungarian novel.


## Parameters JSON 
```json
```

## Providing keys
The optimal way to provide the OPENAI_KEY and the HF_TOKEN is by environmental variables.
example
```bash
OPENAI_KEY=sk-examplekey hugme --model-name NYTK/PULI-LlumiX-32K --tasks mmlu
```