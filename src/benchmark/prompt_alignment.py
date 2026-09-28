from typing import List, Dict
import random
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
        "input": prompt if isinstance(prompt, str) else prompt[1]["content"], # see template.py 24-35 rows
        "output": output.text,
        "prompt_instructions": entry["prompt_instructions"],
        "token_usage": output.total_tokens
    }


def compute_scores(args, results: List[Dict]) -> Dict:

    passed = 0
    total_score = 0.0
    measurement_results = []

    for i, entry in enumerate(results):

        if i < len(measurement_results):
            print(f"Using preloaded / precomputed evaluation result for index {i}")
            total_score += measurement_results[i]["score"]
            passed += int(bool(measurement_results[i]["success"]))
            continue

        # no output or repetitive output
        if not entry["output"].strip() or helper.is_degenerate(
            entry["output"], ngram=5, min_words=50, unique_ratio_threshold=0.35, compression_threshold=0.12
        ):
            success = False
            score = 0.0
            reason = "Model output is empty or repetitive (ngram=5, min_words=50, unique_ratio_threshold=0.35, compression_threshold=0.12)"
            total_score += 0.0

        else:
            pass

            test_case = LLMTestCase(input=entry["input"], actual_output=entry["output"])
            metric = PromptAlignmentMetric(
                prompt_instructions=entry["prompt_instructions"],
                model=args.judge,
                include_reason=True
            )
            metric.measure(test_case)

            total_score += metric.score

            success = metric.score >= THRESHOLD
            score = metric.score
            reason = metric.reason

            if success:
                passed += 1

        measurement_results.append(
            {
                "index": i,
                "success": success,
                "score": score,
                "reason": reason,
                "input": entry["input"],
                "output": entry["output"],
                "prompt_instructions": entry["prompt_instructions"],
                "token_usage": entry["token_usage"]
            }
        )

        print(f"Saved {i + 1}/{len(results)} result for task {config.PROMPT_ALIGNMENT}.")
        helper.save_json(
            measurement_results,
            config.RESULTS_DIR,
            f"{config.PROMPT_ALIGNMENT}-{args.model_name.replace('/', '_').lower()}-eval-results.json"
        )

    avg_score = total_score / len(results)
    success_rate = passed / len(results)

    print(f"Average score: {avg_score:.2f}")
    print(f"Success rate ({THRESHOLD}+): {success_rate:.2%}")

    if args.save_results:
        helper.save_json(
            measurement_results,
            config.RESULTS_DIR,
            f"{config.PROMPT_ALIGNMENT}-{args.model_name.replace('/', '_').lower()}-eval-results.json"
        )
    return {"success_rate": success_rate, "average_score": avg_score}


__all__ = ["compute_metric"]
