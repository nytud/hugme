from tqdm import tqdm

import config
import helper
import template


MAX_NEW_TOKENS = 20


def benchmark(args, generation_pipeline) -> dict:
    dataset = helper.read_json(config.DATASETS + "mmlu.json")
    dataset = preprocess(dataset)
    if args.use_gen_results:
        print("Using generation results from path: ", args.use_gen_results)
        results = helper.read_json(args.use_gen_results)
    else:
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
        prompt = template.get_prompt(args.task_name, entry=entry)
        output = generation_pipeline(prompt, batch_size=args.batch_size, max_new_tokens=MAX_NEW_TOKENS
        )
        output = output[0]['generated_text']
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
