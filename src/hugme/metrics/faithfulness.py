from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from tqdm import tqdm

import hugme.config as config
import hugme.helper as helper


def compute_metric(args, generation_pipeline) -> float:

    dataset = helper.read_json(config.DATASETS + "faithfulness.json")

    results = generate_results(args, generation_pipeline, dataset)

    faithfulness_score = compute_score(args, results)

    return faithfulness_score


def generate_results(args, generation_pipeline, dataset):

    results = []

    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):

        query, context = entry["query"], entry["context"]
        query = f"Válaszolj a kérdésre a megadott kontextus alapján! Kérdés: {query},\n Kontextus: {context}"

        prompt = helper.get_model_prompt(args.model_name, query)

        output = generation_pipeline(prompt)[0]["generated_text"]

        results.append({"input": query, "output": output, "context": context})

    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, "faithfulness-results.json")

    return results


def compute_score(args, results):

    total_score = 0.0
    measurement_results = []

    for i, entry in enumerate(results):

        test_case = LLMTestCase(
            input=entry["input"],
            actual_output=entry["output"],
            retrieval_context=[entry["context"]],
        )
        metric = FaithfulnessMetric(
            threshold=0.7, model=args.judge, include_reason=True
        )

        metric.measure(test_case)

        total_score += float(metric.score)

        measurement_results.append(
            {"index": i, "score": metric.score, "reason": metric.reason}
        )

    faithfulness_score = total_score / len(results)

    print(f"Faithfulness score: {faithfulness_score}")

    if args.save_results:
        helper.save_json(
            measurement_results, config.RESULTS_DIR, "faithfulness-eval-results.json"
        )

    return faithfulness_score
