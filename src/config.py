import os


HF_TOKEN = os.getenv("HF_TOKEN")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
DATASETS = os.getenv("DATASETS", "./datasets/")
BERT_MODEL = os.getenv("BERT_MODEL")
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY", None)
PROVIDER_URL = os.getenv("PROVIDER_URL", None)


BIAS_DATASET = DATASETS + os.getenv("BIAS_DATASET", "bias.json")
TOXICITY_DATASET = DATASETS + os.getenv("TOXICITY_DATASET", "toxicity.json")
FAITHFULNESS_DATASET = DATASETS + os.getenv("FAITHFULNESS_DATASET", "faithfulness.json")
SUMMARIZATION_DATASET = DATASETS + os.getenv("SUMMARIZATION_DATASET", "summarization.json")
ANSWER_RELEVANCY_DATASET = DATASETS + os.getenv("ANSWER_RELEVANCY_DATASET", "answer-relevancy.json")
SPELLING_DICT = DATASETS + os.getenv("SPELLING", "spell.json")
SPELLING_DATASET = DATASETS + os.getenv("SPELLING_DATASET", "summarization.json")
HAYSTACK_DATASET = DATASETS + os.getenv("HAYSTACK_DATASET", "nih.txt")
NEEDLE_FILE = DATASETS + os.getenv("NEEDLE_FILE", "nih_needle.txt")
TRUTHFUL_QA_DATASET = DATASETS + os.getenv("TRUTHFUL_QA_DATASET", "truthful-qa.json")
PROMPT_ALIGNMENT_DATASET = DATASETS + os.getenv("PROMPT_ALIGNMENT_DATASET", "prompt-alignment.json")
READABILITY_DATASET = DATASETS + os.getenv("READABILITY_DATASET", "readability.json")
COLA_DATASET = DATASETS + os.getenv("COLA_DATASET", "summarization.json")
MMLU_DATASET = DATASETS + os.getenv("MMLU_DATASET", "mmlu.json")


METRIC_DATASETES = {
    "bias": BIAS_DATASET,
    "toxicity": TOXICITY_DATASET,
    "faithfulness": FAITHFULNESS_DATASET,
    "summarization": SUMMARIZATION_DATASET,
    "needle-in-haystack": HAYSTACK_DATASET,
    "answer-relevancy": ANSWER_RELEVANCY_DATASET,
}

MMLU = "mmlu"
NIH = "needle-in-haystack"
SPELLING = "spelling"
TRUTHFUL_QA = "truthfulqa"
PROMPT_ALIGNMENT = "prompt-alignment"
READABILITY = "readability"
COLA = "cola"
METRICS = list(METRIC_DATASETES.keys())

HUSPACY_MODEL_NAME = "hu_core_news_lg"

MAX_NEW_TOKENS = {
    MMLU: 20,
    TRUTHFUL_QA: 20,
    PROMPT_ALIGNMENT: 256,
}
DEFAULT_MAX_NEW_TOKENS = 512