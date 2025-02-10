from collections import defaultdict

from tqdm import tqdm

from hugme import config
from hugme import helper


def benchmark(args, generation_pipeline) -> dict:
    dataset = helper.read_json(config.DATASETS + "mmlu.json")
    dataset = preprocess(dataset)
    results = generate_results(args, generation_pipeline, dataset)
    return compute_scores(args, results)


def preprocess(dataset: list):
    for entry in dataset:
        entry["A"] = "A " + entry["A"]
        entry["B"] = "B " + entry["B"]
        entry["C"] = "C " + entry["C"]
        entry["D"] = "D " + entry["D"]
    return dataset


def generate_results(args, generation_pipeline, dataset: list):
    results = []
    for entry in tqdm(dataset, desc="Generating responses", unit="query"):
        question, target = entry["input"], entry["target"]
        a, b, c, d = entry["A"], entry["B"], entry["C"], entry["D"]
        query = (
            "Alább van egy kérdés, és négy válasz. Kizárólag a helyes választ előtti betűt add vissza!"
            f"Kérdés: {question} Válaszok: {a}, {b}, {c}, {d}"
        )
        prompt = helper.get_model_prompt(args.model_name, query)
        output = generation_pipeline(prompt)[0]["generated_text"]
        results.append(
            {
                "query": question,
                "output": output,
                "target": target,
                "category": entry["category"],
            }
        )
    return results


def compute_scores(args, results: list):
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        if entry["output"].strip() == entry["target"] or entry["output"].startswith(
            entry["target"]
        ):
            entry["score"] = 1.0
            score += 1.0
        else:
            entry["score"] = 0.0
    total_score = score / len(results)

    print(f"MMLU benchmark score: {score}")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, "mmlu-results.json")

    grouped_scores = defaultdict(float)
    for entry in results:
        grouped_scores[entry["category"]] += entry["score"]
    grouped_scores["total"] = total_score
    return dict(grouped_scores)
