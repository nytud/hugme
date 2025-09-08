import logging
import random
from tqdm import tqdm

import config
import helper
import template
import generation


def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.TRUTHFUL_QA_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    gen_results = generate_results(args, task_name, dataset)
    return compute_scores(args, gen_results)


def generate_results(args, task_name, dataset):

    client = generation.load_model(args, task_name)
    parameters = generation.create_parameters(args, task_name)

    if args.use_gen_results:
        logging.info("Using generation results from path: ", args.use_gen_results)
        gen_results = helper.read_json(args.use_gen_results)
        return gen_results

    results = []
    for entry in tqdm(dataset, desc="Generating responses...", unit="query"):
        prompt = template.get_prompt(task_name, entry, args.use_alpaca_prompt)
        output = generation.generate(prompt, client, parameters, model_name=args.model_name, provider=args.provider)
        results.append({
            "input": prompt,
            "output": output.text,
            "category": entry["category"],
            "correct_index": [x[0] for x in entry["answer_options"] if x[1] == entry["correct_answers"]][0],
            "total_tokens": output.total_tokens
        })
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{task_name}-generation-results.json")
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

    acc = total_score / len(results)
    print(f"{config.TRUTHFUL_QA} benchmark results accuracy: {round(acc * 100, 2)}")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.TRUTHFUL_QA}-eval-results.json")
    return helper.group_by_category(results, acc)
