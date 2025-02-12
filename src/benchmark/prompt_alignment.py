from deepeval import evaluate
from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase

import config
import helper


def aggregate_metric_pass_rates(test_results) -> float:
    total_metrics = 0
    total_successes = 0

    for result in test_results:
        for metric in result.metrics_data or []:
            total_metrics += 1
            if metric.success:
                total_successes += 1

    return total_successes / total_metrics if total_metrics > 0 else 0.0


def compute_metric(args, generation_pipeline):
    dataset = helper.read_json(config.PROMPT_ALIGNMENT_DATASET)
    metrics = []
    cases = []
    for entry in dataset:
        prompt_instructions, query = entry['prompt_instructions'], entry['query']
        output = generation_pipeline(query, max_new_tokens=20 )[0]['generated_text']
        metrics.append(PromptAlignmentMetric(prompt_instructions=prompt_instructions ,include_reason=True))
        cases.append(LLMTestCase(input=query,actual_output=output,))

    result = evaluate(cases, metrics)

    if args.save_results:
        helper.save_json(result.test_results, config.RESULTS_DIR, f"{config.PROMPT_ALIGNMENT}-eval-results.json")
    return aggregate_metric_pass_rates(result.test_results)
