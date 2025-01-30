

import config
import helper


def benchmark(args, generation_pipeline) -> None:
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
    pass


def compute_score(args, results):

    score = 0.0

    return score

