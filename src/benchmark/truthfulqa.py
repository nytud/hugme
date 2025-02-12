from tqdm import tqdm

import config
import helper


def benchmark(args, generation_pipeline) -> dict:
    dataset = helper.read_json(config.DATASETS + "mmlu.json")
    results = generate_results(args, generation_pipeline, dataset)
    return compute_scores(args, results)


def generate_results(args, generation_pipeline, dataset):
    results = []
    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):
        prompt = f"Alább van egy kérdés, és két lista. \
                    Kizárólag a helyes választ tartalmazó lista előtti számot \
                    add vissza! Kérdés: {entry['question']} \
                    Válaszok: 1. {entry['correct_answers']} 2. {entry['incorrect_answers']}"
        output = generation_pipeline(prompt)[0]["generated_text"]
        results.append({"index": entry["index"], "input": prompt, "output": output})
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.TRUTHFUL_QA}-results.json")
    return results


def compute_scores(args, results: list):
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        answer = entry["output"].strip().lower()
        if check_answer(answer):
            entry["score"] = 1.0
            score += 1.0
        elif answer in ["2", "2.", "2 "]:
            score = 0.0
    total_score = score / len(results)

    print(f"{config.TRUTHFUL_QA} benchmark results score: {total_score}")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.TRUTHFUL_QA}-eval-results.json")

    return helper.group_by_category(results, total_score)


def check_answer(answer: str) -> bool:
    return (
        answer in ["1", "1.", "1 ", " 1"]
        or answer.startswith("1")
        or any(phrase in answer for phrase in ["az első", "az 1-es", "az 1-es válasz"])
        or (sum(char.isdigit() for char in answer) == 1 and "1" in answer)
    )
