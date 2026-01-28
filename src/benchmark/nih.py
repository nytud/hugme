from typing import Dict
import os
import random
import logging

import config
import helper
import template
import generation


MAX_NEW_TOKENS = 20
MODEL_MIN_CONTEXT_LEN = 1024
GOOD_SOLUTION = 1.0
BAD_SOLUTION = 0.0


def compute_metric(args, task_name: str):

    check_prerequisites(args)

    client = generation.load_model(args, task_name)
    parameters = generation.create_parameters(args, task_name)
    parameters["max_new_tokens"] = MAX_NEW_TOKENS

    results = generate_results(args, client, parameters)
    scores = compute_scores(args, results)
    return scores


def check_prerequisites(args, n_turns: int = 5, model_context_len: int = 2048):
    if args.provider == config.OPENAI_PROVIDER_NAME:
        raise ValueError("The NIH task is not supported with OpenAI API. Please use local model with `transformers`.")
    if config.N_TURNS is None:
        os.environ["N_TURNS"] = str(n_turns)
        logging.warning(f"N_TURNS env variable is not set. N_TURNS={n_turns} is set.")
    if config.MODEL_CONTEXT_LEN is None:
        os.environ["MODEL_CONTEXT_LEN"] = str(model_context_len)
        logging.warning("MODEL_CONTEXT_LEN env variable is not set. MODEL_CONTEXT_LEN={model_context_len} is set")


def generate_results(args, client, parameters: dict):

    if args.use_gen_results:
        logging.info(f"Using generation results from path: {args.use_gen_results}")
        results = helper.read_json(args.use_gen_results)
        return results

    data = create_needle_and_haystack(client.tokenizer.tokenize)
    results = generate_results_for_context_lengths(data, client, parameters)

    if args.save_results:
        generation.save_results(results, config.NIH, args.model_name, False)

    return results


def generate_results_for_context_lengths(data, client, parameters):

    fractions = [i / 10 for i in range(0, 10)] # [0.0, 0.1, ..., 0.9]
    context_lengths = create_context_lengths()
    logging.info(f"Created context lengths for evaluation: {context_lengths}")

    results = []
    for context_lenth in context_lengths:
        for fraction in fractions:
            insertion_positions = create_needle_insertion_depths(data, context_lenth, fraction)
            for insertion_position in insertion_positions:
                output = generate_result(insertion_position, data, client, parameters)
                result = format_result(context_lenth, fraction, output, data)
                results.append(result)

        helper.cleanup()
        logging.info(f"Completed evaluation for context length: {context_lenth}\n")

    return results


def create_needle_and_haystack(tokenize_fn):

    cities = helper.read_file(config.NEEDLE_FILE).split("\n")
    random_city = random.choice(cities)
    anniversary = random.randint(1, 100)
    needle_text = f"Ezen a napon ünnepelte {random_city} város a {anniversary}. évfordulóját."
    tokenized_needle = tokenize_fn(needle_text)
    logging.info(f"Created and tokenzied needle for city: {random_city}, anniversary: {anniversary}")

    haystack = helper.read_file(config.HAYSTACK_DATASET)
    tokenized_haystack = tokenize_fn(haystack)
    logging.info("Created and tokenized haystack.")

    return {
        "haystack": tokenized_haystack, "needle": tokenized_needle,
        "city": random_city, "anniversary": anniversary
    }


def create_context_lengths() -> list[int]:
    # [1024, 2048, ..., max_context_len]
    max_context_len = os.getenv("MODEL_CONTEXT_LEN")
    assert max_context_len is not None, "MODEL_CONTEXT_LEN env variable must be set."
    return [2**i * MODEL_MIN_CONTEXT_LEN for i in range(int(int(max_context_len) / MODEL_MIN_CONTEXT_LEN).bit_length())]


def create_needle_insertion_depths(data: dict, context_length: int, fraction: float):
    n_turn = os.getenv("N_TURNS")
    assert n_turn is not None, "N_TURNS env variable must be set."
    actual_context_length = trim_haystack(data, context_length)
    insertion_fractions = [random.uniform(0, 0.1) for i in range(int(n_turn))]
    insertion_positions = [int(actual_context_length * (f + fraction)) for f in insertion_fractions]
    return insertion_positions


def trim_haystack(data: dict, context_length: int):
    trimmed_length = context_length - len(data["needle"])
    data["trimmed_haystack"] = data["haystack"][:trimmed_length]
    return trimmed_length


def generate_result(position, data, client, parameters):
    tokenized_haystack_with_needle = insert_needle_in_haystack(data["trimmed_haystack"], position, data["needle"])
    haystack_with_needle = client.tokenizer.convert_tokens_to_string(tokenized_haystack_with_needle)
    prompt = template.get_prompt("needle-in-haystack", {"text": haystack_with_needle, "city": data["city"]})
    output = generation.generate_with_huggingface(prompt, client, parameters)
    return output


def insert_needle_in_haystack(tokenized_haystack: list[int], insertion_position: int, tokenized_needle: list[int]):
    return tokenized_haystack[:insertion_position] + tokenized_needle + tokenized_haystack[insertion_position:]


def format_result(context_len: int, fraction: float, output, data: dict) -> Dict:
    answer = helper.clean_answer(output.text)
    return {
        "context_length": context_len,
        "fraction": f"{fraction:0.1f}-{fraction + 0.1:0.1f}",
        "model_output": output.text,
        "model_cleaned_output": answer,
        "correct_answer": data["anniversary"],
    }

def compute_scores(args, results: list) -> dict:
    for result in results:
        answer = result["model_cleaned_output"]
        anniversary = result["correct_answer"]
        if str(answer).strip() == str(anniversary):
            result["score"] = GOOD_SOLUTION
        else:
            result["score"] = BAD_SOLUTION

    logging.info("NIH benchmark results computed.")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.NIH}-{args.model_name}-eval-results.json")

    return compute_average_score(args, results)


def compute_average_score(args, results: list) -> dict:

    # aggregate results by context length and fraction
    aggregated_results = {}
    for result in results:

        context_len = str(result["context_length"])
        fraction = str(result["fraction"])

        if context_len not in aggregated_results:
            aggregated_results[context_len] = {}
        if fraction not in aggregated_results[context_len]:
            aggregated_results[context_len][fraction] = 0

        aggregated_results[context_len][fraction] += result["score"]

    # average the scores
    for fractions in aggregated_results.values():
        for fraction in fractions:
            fractions[fraction] /= int(os.getenv("N_TURNS", 5))

    logging.info("NIH benchmark aggregated results computed.")

    helper.plot_needle_in_haystack(
        aggregated_results,
        save_path=f"{config.RESULTS_DIR}/{config.NIH}-{args.model_name}-nih-plot.png",
        )

    return aggregated_results
