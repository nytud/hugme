import config
import helper

from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

def aggregate_metric_pass_rates(test_results) -> float:
    metric_counts = {}
    metric_successes = {}

    for result in test_results:
        if result.metrics_data:
            for metric_data in result.metrics_data:
                metric_name = metric_data.name
                if metric_name not in metric_counts:
                    metric_counts[metric_name] = 0
                    metric_successes[metric_name] = 0
                metric_counts[metric_name] += 1
                if metric_data.success:
                    metric_successes[metric_name] += 1

    metric_pass_rates = {
        metric: (metric_successes[metric] / metric_counts[metric])
        for metric in metric_counts
    }

    for _, pass_rate in metric_pass_rates.items():
        return pass_rate


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
    