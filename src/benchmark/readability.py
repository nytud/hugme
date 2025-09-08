import re
import random
from statistics import mean
from textstat import textstat
from tqdm import tqdm

import helper
import config
import template


TEMPERATURE = 0.4
MAX_NEW_TOKENS = 256


def compute_metric(task_name, args, generate):

    dataset = helper.read_json(config.READABILITY_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)

    if args.use_gen_results:
        print("Using generation results from path: ", args.use_gen_results)
        results = helper.read_json(args.use_gen_results)
    else:
        results = generate_results(args, generate, dataset, task_name)

    return compute_scores(args, results)


def generate_results(args, generate, dataset, task_name):
    results = []
    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):
        prompt = template.get_prompt(task_name, entry, alpaca_prompt=args.use_alpaca_prompt)

        try:
            temperature = args.parameters.get("temperature", TEMPERATURE)
            max_new_tokens = args.parameters.get("max_new_tokens", MAX_NEW_TOKENS)
            do_sample = args.parameters.get("do_sample", True)

            output = generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample)
        except RuntimeError as e:
            print(f"Error during text generation: {e}")
            output = None

        results.append({"query": entry.get("query"), "prompt": prompt, "output": output})
    return results


def calculate_scores(text):
    col_score = int(textstat.coleman_liau_index(text))
    std_score = mean(map(float, re.findall(r'\d+(?:\.\d+)?', textstat.text_standard(text))))

    return col_score, std_score


def calculate_similarity_score(original_coleman, original_std, generated_coleman, generated_std):
    weight_coleman = 0.6
    weight_std = 0.4

    coleman_diff = abs(generated_coleman - original_coleman)
    std_diff = abs(generated_std - original_std)

    coleman_score = max(0, 100 - coleman_diff * 10)
    std_score = max(0, 100 - std_diff * 10)

    similarity_score = (coleman_score * weight_coleman) + (std_score * weight_std)
    return round(similarity_score, 2)


def compute_scores(args, results):

    similarity_scores = []

    for item in tqdm(results, desc="Calculating scores", unit="query"):

        query, output = item["query"], item["output"]

        original_mean_coleman, original_mean_std = calculate_scores(query)
        generated_mean_coleman, generated_mean_std = calculate_scores(output)

        similarity_score = calculate_similarity_score(
            original_mean_coleman, original_mean_std,
            generated_mean_coleman, generated_mean_std
        )
        similarity_scores.append(similarity_score)
        item["similarity_score"] = similarity_score

    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.READABILITY}-eval-results.json")

    print(f"{config.READABILITY} benchmark results accuracy: {mean(similarity_scores)}")

    return mean(similarity_scores)
