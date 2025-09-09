import re
import random
import logging
from tqdm import tqdm
from statistics import mean
from textstat import textstat

import helper
import config
import generation


def compute_metric(args, task_name: str):
    dataset = helper.read_json(config.READABILITY_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, results)


def format_result(entry: dict, prompt: str, output: generation.ModelOutput) -> dict:
    return {
        "query": entry["query"],
        "prompt": prompt,
        "output": output.text,
        "total_tokens": output.total_tokens
    }


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
        helper.save_json(
            results, config.RESULTS_DIR, f"{config.READABILITY}-{args.model_name}-{args.thinking}-eval-results.json"
        )
    logging.info(f"{config.READABILITY} benchmark results accuracy: {mean(similarity_scores)}")

    return mean(similarity_scores)


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