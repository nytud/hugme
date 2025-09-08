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
        entry["A"] = "A " + str(entry["A"])
        entry["B"] = "B " + str(entry["B"])
        entry["C"] = "C " + str(entry["C"])
        entry["D"] = "D " + str(entry["D"])
    return dataset


# TODO move to helper.py
def generate_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def post_process_llama(output):
    keyword = "válasz"
    index = output.lower().find(keyword)

    if index != -1:

        if index + len(keyword) < len(output) and output[index + len(keyword)] == ":":
            return output[index + len(keyword) + 1:].strip()
        else:
            return output[index + len(keyword):].strip()
    else:
        return output

def generate_results(args, generate, dataset: list, task_name):
    results = []
    total_batches = len(dataset) // args.batch_size + (1 if len(dataset) % args.batch_size != 0 else 0)
    for batch_entry in tqdm(generate_batches(dataset, args.batch_size), desc="Generating responses", total=total_batches, unit="query"):
        prompts = [template.get_prompt(task_name, entry, args.use_alpaca_prompt) for entry in batch_entry]
        if args.provider is not None: # TODO fix for openai batch generation
            outputs = [generate(prompt, max_new_tokens=MAX_NEW_TOKENS) for prompt in prompts]
        else:
            outputs = generate(prompts, max_new_tokens=MAX_NEW_TOKENS, alpaca_prompt=args.use_alpaca_prompt, batch_size=args.batch_size)

        for output, entry in zip(outputs, batch_entry):
            actual_output_text = post_process_llama(str(output))
            results.append(
                {"query": entry['input'], "output": actual_output_text, "target": entry['target'], "category": entry['category']}
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

    print(f"MMLU benchmark score: {round(total_score * 100, 2)}%")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, "mmlu-results.json")

    return helper.group_by_category(results, total_score)
