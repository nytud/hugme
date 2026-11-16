import json
import pathlib
from glob import glob
import numpy as np
from rich.table import Table
from rich.console import Console


N_RUNS = 5
BASE = "/home/username/results/"


def read_json(file_path):
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e


def fmt(values):
    """Return (n, mean, std) as display strings, handling empty and single-run cases."""
    n = len(values)
    if n == 0:
        return "0", "N/A", "N/A"
    mean = f"{np.mean(values):.2f}"
    std = f"{np.std(values, ddof=1):.2f}" if n > 1 else "N/A"
    return str(n), mean, std


console = Console()

models = {
    # temperature=0.6, top_p=0.95, top_k=20, min_p=0 (from https://huggingface.co/Qwen/Qwen3-4B)
    "Qwen3-4B (thinking)":      BASE + "qwen3-4b-thinking/run_*/hugme-qwen-qwen3-4b-results-*.json",

    # temperature=0.7, top_p=0.8, top_k=20, min_p=0 (from https://huggingface.co/Qwen/Qwen3-4B)
    "Qwen3-4B (non-thinking)":  BASE + "qwen3-4b-non-thinking/run_*/hugme-qwen-qwen3-4b-results-*.json",
}


for model_name, pattern in models.items():

    results = sorted(
        glob(pattern)) if isinstance(pattern, str) else sorted([item for sublist in [glob(p) for p in pattern] for item in sublist]
    )

    print(f"=== {model_name.upper()} RESULTS ===")

    mmlu = []
    truthfulqa = []
    spelling = []
    prompt_alignment = []
    readability = []
    bias = []
    toxicity = []
    faithfulness = []
    summarization = []
    answer_relevancy = []
    cola = []

    for result_file in results:

        result_data = read_json(result_file)

        if result_data.get("mmlu") is not None:
            mmlu.append(result_data["mmlu"]["total"])
        if result_data.get("truthfulqa") is not None:
            truthfulqa.append(result_data["truthfulqa"]["total"])
        if result_data.get("spelling") is not None:
            spelling.append(result_data["spelling"]["score"])
        if result_data.get("prompt-alignment") is not None:
            prompt_alignment.append(result_data["prompt-alignment"]["success_rate"] * 100)
        if result_data.get("readability") is not None:
            readability.append(result_data["readability"])
        if result_data.get("bias") is not None:
            bias.append(result_data["bias"])
        if result_data.get("toxicity") is not None:
            toxicity.append(result_data["toxicity"])
        if result_data.get("faithfulness") is not None:
            faithfulness.append(result_data["faithfulness"])
        if result_data.get("summarization") is not None:
            summarization.append(result_data["summarization"])
        if result_data.get("answer-relevancy") is not None:
            answer_relevancy.append(result_data["answer-relevancy"])
        if result_data.get("cola") is not None:
            cola.append(result_data["cola"])

    metrics = {
        "MMLU": mmlu,
        "TruthfulQA": truthfulqa,
        "Spelling": spelling,
        "Prompt Alignment": prompt_alignment,
        "Readability": readability,
        "Bias": bias,
        "Toxicity": toxicity,
        "Faithfulness": faithfulness,
        "Summarization": summarization,
        "Answer Relevancy": answer_relevancy,
        "CoLA": cola,
    }
    table = Table(show_header=True, header_style="bold blue")

    table.add_column("Metric", justify="left")
    table.add_column("Runs", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std Dev", justify="right")

    for name, values in metrics.items():
        n_str, mean_str, std_str = fmt(values)
        style = "green" if len(values) >= N_RUNS else "yellow" if len(values) in [4] else "red"
        table.add_row(name, n_str, mean_str, std_str, style=style)

    console.print(table)
