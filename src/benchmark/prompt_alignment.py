import random
import config
import helper

from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

import template


MAX_NEW_TOKENS = 256


def compute_metric(task_name, args, generate_fn):

    dataset = helper.read_json(config.PROMPT_ALIGNMENT_DATASET)

    sample_size = max(1, int(args.sample_size * len(dataset)))
    dataset = random.sample(dataset, sample_size)

    results = []
    for entry in dataset:
        prompt_instructions = entry['prompt_instructions']
        query = template.get_prompt("prompt-alignment", entry)

        actual_output = generate_fn(
            query,
            max_new_tokens=MAX_NEW_TOKENS,
            alpaca_prompt=args.use_alpaca_prompt
        )

        results.append({
            "query": str(query),
            "actual_output": str(actual_output),
            "prompt_instructions": prompt_instructions
        })

    threshold = 0.5
    total_score = 0.0
    passed = 0

    for item in results:
        test_case = LLMTestCase(
            input=item["query"],
            actual_output=item["actual_output"]
        )

        metric = PromptAlignmentMetric(
            prompt_instructions=item["prompt_instructions"],  #a prompt instruction lista legyen!
            model="gpt-4o",
            include_reason=True
        )
        metric.measure(test_case)

        total_score += metric.score
        if metric.score >= threshold:
            passed += 1

    avg_score = total_score / len(results)
    success_rate = passed / len(results)

    print(f"Average score: {avg_score:.2f}")
    print(f"Success rate ({threshold}+): {success_rate:.2%}")


    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.PROMPT_ALIGNMENT}-eval-results.json")
    return results
