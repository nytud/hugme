from __future__ import annotations

import random
from typing import Any

from tqdm import tqdm

import config
import generation
import helper


def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.CULTURAL_ABCD_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    dataset = helper.preprocess(dataset)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def format_result(entry: dict[str, Any], prompt: Any, output: generation.ModelOutput) -> dict:
    return {
        "question_id": entry["question_id"],
        "category": entry['category'],
        "prompt": prompt,
        "output": output.text,
        "correct_answer": entry['correct'],
        "total_tokens": output.total_tokens
    }


def compute_scores(args, results: list):
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):

        output_extraced = helper.extract_abcd_answer(entry['output'])
        entry['output_extracted'] = output_extraced

        if entry['output_extracted'].strip() == entry['correct_answer']:
            entry['score'] = 1.0
            score += 1.0
        else:
            entry['score'] = 0.0
    total_score = score / len(results)

    print(f"Cultural benchmark score: {round(total_score * 100, 2)}%")
    if args.save_results:
        model_name = helper.cleanup_model_name(args.model_name)
        helper.save_json(results, config.RESULTS_DIR, f"{config.CULTURAL_ABCD}-{model_name}-eval-results.json")
    return helper.group_by_category(results, total_score)
