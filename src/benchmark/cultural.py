from typing import Any, Dict

import random
import logging
from tqdm import tqdm

import config
import helper
import generation


def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.CULTURAL_ABCD_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    dataset = helper.preprocess(dataset)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def format_result(entry: Dict[str, Any], prompt: Any, output: generation.ModelOutput) -> Dict:
    actual_output_text = helper.post_process_llama(output.text)
    return {
        "question_id": entry["question_id"],
        "category": entry['category'],
        "prompt": prompt,
        "output_raw": output.text,
        "output": actual_output_text,
        "correct_answer": entry['correct_answer'],
        "total_tokens": output.total_tokens
    }


def compute_scores(args, results: list):
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        if entry['output'].strip() == entry['correct_answer'] or entry['output'].startswith(entry['correct_answer']):
            entry['score'] = 1.0
            score += 1.0
        else:
            entry['score'] = 0.0
    total_score = score / len(results)

    logging.info(f"Cultural benchmark score: {round(total_score * 100, 2)}%")
    if args.save_results:
        helper.save_json(
            results,
            config.RESULTS_DIR,
            f"{config.CULTURAL_ABCD}-{args.model_name}-{str(args.thinking).lower()}-eval-results.json"
        )
    return helper.group_by_category(results, total_score)
