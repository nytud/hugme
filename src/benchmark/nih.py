import os
import random
from collections import defaultdict
from statistics import mean
from typing import Dict

import config
import generation
import helper
import template

BAD_SOLUTION = 0.0
GOOD_SOLUTION = 1.0
N_TURNS = int(os.getenv("N_TURNS", 2))
FRACTION_STEP = 0.1
FRACTIONS = [i / 10 for i in range(10)]  # [0.0, 0.1, ..., 0.9]
# MODEL_CONTEXT_LEN = int(os.getenv("MODEL_CONTEXT_LEN", 32_768))
MODEL_CONTEXT_LEN = int(os.getenv("MODEL_CONTEXT_LEN", 2048))
# [1024,2048,...,MODEL_CONTEXT_LEN]
CONTEXT_LENS = list(range(1024, MODEL_CONTEXT_LEN, 1024)) + [MODEL_CONTEXT_LEN]
# estimate chat template around the message + space for generation
# use non-thinking mode for this task
CHAT_TEMPLATE_TOKENS = 32



def compute_metric(args, task_name: str | None = None):
    # task_name is unused, kept for a common interface with other metrics
    client = generation.load_model(args)
    parameters = generation.load_parameters(args)
    results = generate_results(args, client, parameters)
    scores = compute_scores(args, results)
    return scores


def generate_results(args, client, parameters: dict):
    if args.use_gen_results:
        print(f"Using generation results from path: {args.use_gen_results}")
        return helper.read_json(args.use_gen_results)

    data = create_needle_and_haystack(args)
    prompt = create_prompt(args, data)
    results = _generate_results(args, client, parameters, data, prompt)

    if args.save_results:
        model_name = helper.cleanup_model_name(args.model_name)
        helper.save_results(results, config.NIH, model_name, False)
    return results


def create_needle_and_haystack(args):
    haystack = create_haystack(args)
    needle, city, anniversary = create_needle(args)
    return {"haystack": haystack, "needle": needle, "city": city, "anniversary": anniversary}


def create_needle(args):
    cities_text = helper.read_file(config.NEEDLE_FILE)
    assert isinstance(cities_text, str), "Needle dataset must be a string."
    cities = cities_text.split("\n")
    city = random.choice(cities)
    anniversary = random.randint(1, 100)
    needle_text = f"Ezen a napon ünnepelte {city} város a {anniversary}. évfordulóját."
    tokenized_needle = generation.tokenize(needle_text, args)
    print(f"Created and tokenized needle for city: {city}, anniversary: {anniversary}")
    return tokenized_needle, city, anniversary


def create_haystack(args):
    haystack = helper.read_file(config.HAYSTACK_DATASET)
    assert isinstance(haystack, str), "Haystack dataset must be a string."
    tokenized_haystack = generation.tokenize(haystack, args)
    print("Created and tokenized haystack.")
    return tokenized_haystack


def create_prompt(args, data):
    prompt = template.get_needle_in_haystack_template({"city": data["city"]})
    tokenized_prompt = generation.tokenize(prompt, args)
    return tokenized_prompt


def _generate_results(args, client, parameters, data, prompt):
    # everything besides the haystack also has to fit into the context length
    reserved = len(prompt) + len(data["needle"]) + CHAT_TEMPLATE_TOKENS
    reserved += parameters.get("max_completion_tokens", 0) # space for the answer

    results = []
    for context_len in CONTEXT_LENS:
        haystack = data["haystack"][:context_len - reserved]
        assert len(haystack) == context_len - reserved, "Cannot build haystack for this context length."
        for fraction in FRACTIONS:
            positions = create_insertion_positions(fraction, len(haystack), N_TURNS)
            messages = [create_message(args, prompt, haystack, data["needle"], position) for position in positions]
            outputs = generation.generate_batch(client, messages, args.model_name, parameters)
            results.extend(
                format_result(context_len, fraction, pos, output, data) for pos, output in zip(positions, outputs)
            )
            print(f"Evaluated context length: {context_len}, fraction: {fraction} with positions {positions}.")
    return results


def create_insertion_positions(fraction: float, haystack_len: int, n_turns: int) -> list[int]:
    """random needle positions, one per turn, within [fraction, fraction + FRACTION_STEP)."""
    if n_turns <= 0:
        raise ValueError(f"n_turns must be a positive integer, got {n_turns}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    upper = min(fraction + FRACTION_STEP, 1.0)
    return [round(random.uniform(fraction, upper) * haystack_len) for _ in range(n_turns)]


def create_message(args, prompt, haystack, needle, position: int):
    tokens = prompt + haystack[:position] + needle + haystack[position:]
    return [
        {"role": "system", "content": "Te egy segítőkész asszisztens vagy."},
        {"role": "user", "content": generation.detokenize(tokens, args)}
    ]


def format_result(context_len: int, fraction: float, position: int, output, data: dict) -> Dict:
    answer = helper.grab_first_number(output.text)
    return {
        "context_len": context_len,
        "fraction": f"{fraction:0.1f}-{fraction + 0.1:0.1f}",
        "position": position,
        "output": output.text,
        "output_cleaned": answer,
        "correct_answer": data["anniversary"],
    }


def compute_scores(args, results: list[dict]) -> dict:
    for result in results:
        is_correct = str(result["output_cleaned"]) == str(result["correct_answer"])
        result["score"] = GOOD_SOLUTION if is_correct else BAD_SOLUTION

    model_name = helper.cleanup_model_name(args.model_name)
    if args.save_results:
        filename = f"{config.NIH}-{model_name}-eval-results.json"
        helper.save_json(results, config.RESULTS_DIR, filename)

    scores = average_scores(results)
    plot_path = f"{config.RESULTS_DIR}/{config.NIH}-{model_name}-plot.png"
    helper.plot_needle_in_haystack(scores, save_path=plot_path)

    average_score = mean(result["score"] for result in results)
    print(f"Average score: {average_score:.2f}")
    return {"average_score": average_score, "scores": scores}


def average_scores(results: list[dict]) -> dict[str, dict[str, float]]:
    """Average score per context length and fraction: {context_len: {fraction: score}}."""
    grouped = defaultdict(lambda: defaultdict(list))
    for result in results:
        grouped[str(result["context_len"])][result["fraction"]].append(result["score"])
    return {
        context_len: {fraction: mean(scores) for fraction, scores in fractions.items()}
        for context_len, fractions in grouped.items()
    }
