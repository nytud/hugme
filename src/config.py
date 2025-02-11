import os


HF_TOKEN = os.getenv("HF_TOKEN")

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

DATASETS = os.getenv("DATASETS", "./datasets/")

BIAS_DATASET = DATASETS + os.getenv("BIAS_DATASET", "bias.json")
TOXICITY_DATASET = DATASETS + os.getenv("TOXICITY_DATASET", "toxicity.json")
FAITHFULNESS_DATASET = DATASETS + os.getenv("FAITHFULNESS_DATASET", "faithfulness.json")
SUMMARIZATION_DATASET = DATASETS + os.getenv("SUMMARIZATION_DATASET", "summarization.json")
HALLUCINATION_DATASET = DATASETS + os.getenv("HALLUCINATION_DATASET", "hallucination.json")
ANSWER_RELEVANCY_DATASET = DATASETS + os.getenv("ANSWER_RELEVANCY_DATASET", "answer-relevancy.json")
SPELLING_DICT = DATASETS + os.getenv("SPELLING", "spell.json")
SPELLING_DATASET = DATASETS + os.getenv("SPELLING_DATASET", "spelling.json")
HAYSTACK_DATASET = DATASETS + os.getenv("HAYSTACK_DATASET", "nih.txt")
NEEDLE_FILE = DATASETS + os.getenv("NEEDLE_FILE", "nih_needle.txt")


METRIC_DATASETES = {
    "bias": BIAS_DATASET,
    "toxicity": TOXICITY_DATASET,
    "faithfulness": FAITHFULNESS_DATASET,
    "hallucination": HALLUCINATION_DATASET,
    "summarization": SUMMARIZATION_DATASET,
    "answer-relevancy": ANSWER_RELEVANCY_DATASET,
}

MMLU = "mmlu"
SPELLING = "spelling"
TRUTHFUL_QA = "truthfulqa"
NEEDLE_IN_THE_HAYSTACK = "needle-in-the-haystack"
METRICS = list(METRIC_DATASETES.keys())
