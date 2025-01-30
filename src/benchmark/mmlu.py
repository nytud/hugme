from tqdm import tqdm

import config
import helper


def benchmark(args, generation_pipeline) -> float:
    dataset = helper.read_json(config.DATASETS + "mmlu.json")
    dataset = preprocess(dataset)
    results = generate_results(args, generation_pipeline, dataset)
    mmlu_score = compute_score(args, results)
    return mmlu_score


def preprocess(dataset: dict):
    for entry in dataset:
        entry["A"] = "A " + entry["A"]
        entry["B"] = "B " + entry["B"]
        entry["C"] = "C " + entry["C"]
        entry["D"] = "D " + entry["D"]
    return dataset


def generate_results(args, generation_pipeline, dataset):
    results = []
    for entry in tqdm(dataset, desc="Generating responses", unit="query"):
        question, target = entry['input'], entry['target']
        a, b, c, d = entry['A'], entry['B'], entry['C'], entry['D']
        query = (
            "Alább van egy kérdés, és négy válasz. Kizárólag a helyes választ előtti betűt add vissza!"
            f"Kérdés: {question} Válaszok: {a}, {b}, {c}, {d}"
        )
        prompt = helper.get_model_prompt(args.model_name, query)
        output = generation_pipeline(prompt)[0]['generated_text']

        print("prompt, output")
        print(prompt, output)

        results.append({"query": question, "actual_output": output, "target": target})
    return results


def compute_score(args, results):
    score = 0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        if entry['output'].strip() == entry['target']:
            entry['score'] = 1
            score += 1
        else:
            entry['score'] = 0
    total_score = score / len(results)
    print(f"MMLU score: {score}")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, "mmlu-results.json")
    return total_score
