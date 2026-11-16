from __future__ import annotations

import random
from typing import Any

from tqdm import tqdm

import config
import generation
import helper


def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.MMLU_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    dataset = helper.preprocess(dataset)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def post_process_llama(output: str):
    keyword = "válasz"
    index = output.lower().find(keyword)

    if index != -1:

        if index + len(keyword) < len(output) and output[index + len(keyword)] == ":":
            return output[index + len(keyword) + 1:].strip()
        return output[index + len(keyword):].strip()
    return output


def format_result(entry: dict[str, Any], prompt: Any, output: generation.ModelOutput) -> dict:
    actual_output_text = post_process_llama(output.text)
    return {
        "prompt": prompt,
        "output": actual_output_text,
        "target": entry['target'],
        "category": entry['subject'],
        "total_tokens": output.total_tokens
    }


def compute_scores(args, results: list):
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        if entry['output'].strip() == entry['target'] or entry['output'].startswith(entry['target']):
            entry['score'] = 1.0
            score += 1.0
        else:
            entry['score'] = 0.0
    total_score = score / len(results)

    print(f"MMLU benchmark score: {round(total_score * 100, 2)}%")
    if args.save_results:
        model_name = helper.cleanup_model_name(args.model_name)
        helper.save_json(results, config.RESULTS_DIR, f"{config.MMLU}-{model_name}-eval-results.json")

    return helper.group_by_category(results, total_score)
