from __future__ import annotations

import random
from typing import Any

from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase

import config
import generation
import helper

THRESHOLD = 0.5


def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.PROMPT_ALIGNMENT_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset)))
    dataset = random.sample(dataset, sample_size)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def format_result(entry: dict, prompt: str, output: generation.ModelOutput) -> dict:
    return {
        "prompt": prompt,
        "input": prompt if isinstance(prompt, str) else prompt[1]["content"], # see template.py 24-35 rows
        "output": output.text,
        "prompt_instructions": entry["prompt_instructions"],
        "token_usage": output.total_tokens
    }


def compute_scores(args, results: list[dict]) -> dict:
    measurement_results = []
    model_name = helper.cleanup_model_name(args.model_name)
    filename = f"{config.PROMPT_ALIGNMENT}-{model_name}-eval-results.json"

    for i, entry in enumerate(results):
        if i < len(measurement_results):
            print(f"Using preloaded / precomputed evaluation result for index {i}")
            continue

        measurement_results.append(compute_entry(args, entry, i))

        if args.save_results:
            print(f"Saved {i + 1}/{len(results)} result for task {config.PROMPT_ALIGNMENT}.")
            helper.save_json(measurement_results, config.RESULTS_DIR, filename)

    total_score = sum(r["score"] for r in measurement_results)
    passed = sum(bool(r["success"]) for r in measurement_results)
    return {"success_rate": passed / len(results), "average_score": total_score / len(results)}


def compute_entry(args, entry: dict, index: int) -> dict[str, Any]:
    output = entry["output"]
    if not output.strip() or helper.is_repetitive(
        output, ngram=5, min_words=50, unique_ratio_threshold=0.35, compression_threshold=0.12
    ):
        success, score, reason = False, 0.0, "Model output is empty or repetitive."
    else:
        metric = PromptAlignmentMetric(
            prompt_instructions=entry["prompt_instructions"],
            model=args.judge,
            include_reason=True,
        )
        metric.measure(LLMTestCase(input=entry["input"], actual_output=output))
        if metric.score is None:
            raise ValueError(
                f"Metric score is None for entry {index}. "
                "Please check the model output and prompt instructions."
            )
        if metric.reason is None:
            raise ValueError(
                f"Metric reason is None for entry {index}. "
                "Please check the model output and prompt instructions."
            )
        success, score, reason = metric.score >= THRESHOLD, metric.score, metric.reason

    return {
        "index": index,
        "success": success,
        "score": score,
        "reason": reason,
        "input": entry["input"],
        "output": output,
        "prompt_instructions": entry["prompt_instructions"],
        "token_usage": entry["token_usage"],
    }


__all__ = ["compute_metric"]
