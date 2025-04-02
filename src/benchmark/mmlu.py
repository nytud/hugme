import random
from tqdm import tqdm

import config
import helper
import template


MAX_NEW_TOKENS = 20


def benchmark(task_name, args, generate) -> dict:
    dataset = helper.read_json(config.DATASETS + "mmlu.json")
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    dataset = preprocess(dataset)
    if args.use_gen_results:
        print("Using generation results from path: ", args.use_gen_results)
        results = helper.read_json(args.use_gen_results)
    else:
        results = generate_results(args, generate, dataset, task_name)
    return compute_scores(args, results)


def preprocess(dataset: list):
    for entry in dataset:
        entry["A"] = "A " + entry["A"]
        entry["B"] = "B " + entry["B"]
        entry["C"] = "C " + entry["C"]
        entry["D"] = "D " + entry["D"]
    return dataset


def generate_results(args, generate, dataset: list, task_name):
    results = []
    for entry in tqdm(dataset, desc="Generating responses", unit="query"):
        prompt = template.get_prompt(task_name, entry, args.use_alpaca_prompt)
        output = generate(prompt, max_new_tokens=MAX_NEW_TOKENS, alpaca_prompt=args.use_alpaca_prompt)
        results.append(
            {"query": entry['input'], "output": output, "target": entry['target'], "category": entry['category']}
        )
    return results


def compute_scores(args, results: list):
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        if entry['output'].strip() == entry['target'] or entry['output'].startswith(entry['target']):
            entry['score'] = 1.0
            score += 1.0
        else:
            entry['score'] = 0.0
    total_score = score / len(results)

    print(f"MMLU benchmark score: {score}")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, "mmlu-results.json")

    return helper.group_by_category(results, total_score)
