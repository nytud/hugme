import os


HF_TOKEN = os.getenv("HF_TOKEN")

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

DATASETS = os.getenv("DATASETS", "./datasets/")

BIAS_DATASET = DATASETS + os.getenv("BIAS_DATASET", "bias.json")
FAITHFULNESS_DATASET = DATASETS + os.getenv("FAITHFULNESS_DATASET", "faithfulness.json")
HALLUCINATION_DATASET = DATASETS + os.getenv("HALLUCINATION_DATASET", "hallucination.json")
ANSWER_RELEVANCY_DATASET = DATASETS + os.getenv("ANSWER_RELEVANCY_DATASET", "answer-relevancy.json")

METRIC_DATASETES = {
    "bias": BIAS_DATASET,
    "faithfulness": FAITHFULNESS_DATASET,
    "hallucination": HALLUCINATION_DATASET,
    "answer-relevancy": ANSWER_RELEVANCY_DATASET,
}
BENCHMARK_DATASETES = {
    "mmlu": None,
    "truthfulqa": None,
}

MMLU = "mmlu"
TRUTHFUL_QA = "truthfulqa"
METRICS = list(METRIC_DATASETES.keys())
