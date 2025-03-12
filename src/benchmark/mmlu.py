import time

from tqdm import tqdm

import config
import helper
from answer_provider import AbstractGenerator, GenerationInput


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


def generate_results(args, generation_pipeline: AbstractGenerator, dataset: list):
    results = []
    for entry in tqdm(dataset, desc="Generating responses", unit="query"):
        question, target = entry['input'], entry['target']
        mmlu_answers = {'a': entry['A'], 'b': entry['B'], 'c': entry['C'], 'd': entry['D']}
        generation_input = GenerationInput(prompt=question, mmlu_answers=mmlu_answers, task_name=args.task_name)
        output = generation_pipeline.generate_for_task(generation_input)
        results.append({"query": question, "output": output, "target": target, "category": entry['category']})
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
        helper.save_json(results, config.RESULTS_DIR, f"{config.MMLU}-{args.model_name}-{int(time.time())}-results.json")

    return helper.group_by_category(results, total_score)
