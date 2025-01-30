from tqdm import tqdm
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

import config
import helper


def compute_metric(args, generation_pipeline) -> float:
    dataset = helper.read_json(config.DATASETS + "answer-relevancy.json")
    results = generate_results(args, generation_pipeline, dataset)
    relevancy_score = compute_score(args, results)
    return relevancy_score


def generate_results(args, generation_pipeline, dataset):
    results = []
    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):
        prompt = helper.get_model_prompt(args.model_name, entry["query"])
        output = generation_pipeline(prompt)[0]['generated_text']
        results.append({"input": prompt, "output": output})
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, "answer-relevancy-results.json")
    return results


def compute_score(args, results):
    total_score = 0.0
    measurement_results = []
    for i, entry in enumerate(results):
        test_case = LLMTestCase(input = entry["input"], actual_output = entry["output"])
        metric = AnswerRelevancyMetric(threshold=0.5, model=args.judge, include_reason=True)
        metric.measure(test_case)
        total_score += float(metric.score)
        measurement_results.append({"index": i, "score": metric.score, "reason": metric.reason})
    answer_relevancy_score = total_score / len(results)
    print(f"Answer relevancy score: {answer_relevancy_score}")
    if args.save_results:
        helper.save_json(measurement_results, config.RESULTS_DIR, "answer-relevancy-eval-results.json")
    return answer_relevancy_score
