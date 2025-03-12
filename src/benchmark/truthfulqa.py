import random
import time

from tqdm import tqdm

import config
import helper
from answer_provider import AbstractGenerator, GenerationInput


def benchmark(args, generation_pipeline) -> dict:
    dataset = helper.read_json(config.TRUTHFUL_QA_DATASET)
    results = generate_results(args, generation_pipeline, dataset)
    return compute_scores(args, results)

def generate_results(args, generation_pipeline: AbstractGenerator, dataset):
    results = []
    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):
        answer_options = [
            (1, entry["correct_answers"]),
            (2, entry["incorrect_answers"])
        ]
        random.shuffle(answer_options)
        generation_input = GenerationInput(prompt=entry["question"], truthfulqa_answers=answer_options, task_name=args.task_name)
        prompt = generation_pipeline.prepare_prompt(generation_input)
        output = generation_pipeline.generate_for_task(generation_input)
        results.append({
            "input": prompt,
            "output": output,
            "category": entry["category"],
            "correct_index": [x[0] for x in answer_options if x[1] == entry["correct_answers"]][0]
        })
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.TRUTHFUL_QA}-results.json")
    return results


def check_answer(answer: str, correct_index: int) -> bool:
    is_one = (
        answer in ["1", "1.", "1 ", " 1"]
        or answer.startswith("1")
        or any(phrase in answer for phrase in ["az első", "az 1-es", "az 1-es válasz"])
        or (sum(char.isdigit() for char in answer) == 1 and "1" in answer)
    )

    is_two = (
        answer in ["2", "2.", "2 ", " 2"]
        or answer.startswith("2")
        or any(phrase in answer for phrase in ["a második", "a 2-es", "a 2-es válasz"])
        or (sum(char.isdigit() for char in answer) == 1 and "2" in answer)
    )

    return (correct_index == 1 and is_one) or (correct_index == 2 and is_two)

def compute_scores(args, results: list):
    total_score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        answer = entry["output"].strip().lower()
        if check_answer(answer, entry["correct_index"]):
            entry["score"] = 1.0
            total_score += 1.0
        else:
            entry["score"] = 0.0

    score = total_score / len(results)
    results.append({"accuracy": score})
    print(f"{config.TRUTHFUL_QA} benchmark results score: {score}")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.TRUTHFUL_QA}-\
                         {args.model_name}-{int(time.time())}-eval-results.json")

    return helper.group_by_category(results, total_score)
