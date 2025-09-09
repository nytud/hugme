from typing import List, Dict
import random
import logging
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

import config
import helper
import generation


THRESHOLD = 0.5


def compute_metric(args, task_name: str) -> Dict:
    dataset = helper.read_json(config.PROMPT_ALIGNMENT_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset)))
    dataset = random.sample(dataset, sample_size)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def format_result(entry: dict, prompt: str, output: generation.ModelOutput) -> dict:
    return {
        "prompt": prompt,
        "output": output.text,
        "prompt_instructions": entry["prompt_instructions"],
        "token_usage": output.total_tokens
    }


def compute_scores(args, results: List[Dict]) -> Dict:

    passed = 0
    total_score = 0.0

    for entry in results:
        test_case = LLMTestCase(input=entry["prompt"], actual_output=entry["output"])
        metric = PromptAlignmentMetric(
            prompt_instructions=entry["prompt_instructions"],
            model=args.judge,
            include_reason=True
        )
        metric.measure(test_case)

        total_score += metric.score
        if metric.score >= THRESHOLD:
            passed += 1

        entry["reason"] = metric.reason

    avg_score = total_score / len(results)
    success_rate = passed / len(results)

    logging.info(f"Average score: {avg_score:.2f}")
    logging.info(f"Success rate ({THRESHOLD}+): {success_rate:.2%}")

    if args.save_results:
        helper.save_json(
            results,
            config.RESULTS_DIR,
            f"{config.PROMPT_ALIGNMENT}-{args.model_name}-{str(args.thinking).lower()}-eval-results.json"
        )
    return {"success_rate": success_rate, "average_score": avg_score }


__all__ = ["compute_metric"]
