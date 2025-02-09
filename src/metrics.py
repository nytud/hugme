from tqdm import tqdm
from deepeval import metrics
from deepeval.test_case import LLMTestCase

import config
import helper


def compute_metric(task_name, args, generation_pipeline):
    _metrics = {
        "bias": metrics.BiasMetric(threshold=0.5, model=args.judge),
        "faithfulness": metrics.FaithfulnessMetric(threshold=0.5, model=args.judge),
        "hallucination": metrics.HallucinationMetric(threshold=0.5, model=args.judge),
        "answer-relavancy": metrics.AnswerRelevancyMetric(threshold=0.7, model=args.judge),
    }
    metric = _metrics.get(task_name)
    dataset_name = config.METRIC_DATASETES.get(task_name)
    dataset = helper.read_json(dataset_name)
    gen_results = generate_results(args, generation_pipeline, dataset, task_name)
    results = compute_score(args, gen_results, metric, task_name)
    return results


def generate_results(args, generation_pipeline, dataset, task_name):
    results = []
    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):
        query = helper.get_metric_prompt(task_name, entry["query"], entry.get("context"))
        prompt = helper.get_model_prompt(args.model_name, query)
        output = generation_pipeline(prompt)[0]['generated_text']
        results.append({"input": prompt, "output": output, "context": entry.get("context")})
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{task_name}-results.json")
    return results


def compute_score(args, results, metric, task_name):
    total_score = 0.0
    measurement_results = []
    for i, entry in enumerate(results):
        test_case = LLMTestCase(
            input = entry["input"], actual_output = entry["output"],
            retrieval_context = entry.get("context"), context = entry.get("context")
        )
        metric.measure(test_case)
        total_score += float(metric.score)
        measurement_results.append({"index": i, "score": metric.score, "reason": metric.reason})
    final_score = total_score / len(results)
    print(f"final score: {final_score}")
    if args.save_results:
        helper.save_json(measurement_results, config.RESULTS_DIR, f"{task_name}-eval-results.json")
    return final_score
