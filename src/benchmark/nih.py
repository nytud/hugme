from typing import Dict
import random
import logging

import config
import helper
import template
import generation


N_TURNS = 5
MAX_NEW_TOKENS = 20
MODEL_MIN_CONTEXT_LEN = 1024
MODEL_MAX_CONTEXT_LEN = 2048
GOOD_SOLUTION = 1.0
BAD_SOLUTION = 0.0


def compute_metric(args, task_name: str):

    n_turns, model_context_len = check_prerequisites(args)

    client = generation.load_model(args, task_name)
    parameters = generation.create_parameters(args, task_name)

    results = generate_results(args, client, parameters, n_turns, model_context_len)
    scores = compute_scores(args, results, n_turns)
    return scores


def check_prerequisites(args):
    if args.provider == config.OPENAI_PROVIDER_NAME:
        raise ValueError("The NIH task is not supported with OpenAI API. Please use local model instead (transformers API).")
    if config.N_TURNS is None:
        logging.warning(
            f"N_TURNS environment variables are not set. The N_TURNS={N_TURNS} default value will be used."
        )
    if config.MODEL_CONTEXT_LEN is None:
        logging.warning(
            "MODEL_CONTEXT_LEN environment variables are not set."
            f"The MODEL_CONTEXT_LEN={MODEL_MAX_CONTEXT_LEN} default value will be used"
        )
    n_turns = int(config.N_TURNS) if config.N_TURNS is not None else N_TURNS
    model_context_len = int(config.MODEL_CONTEXT_LEN) if config.MODEL_CONTEXT_LEN is not None else MODEL_MAX_CONTEXT_LEN
    return n_turns, model_context_len


def generate_results(args, client, parameters: dict, n_turns: int, model_context_len: int):

    tokenized_needle, tokenized_haystack, city, anniversary = create_needle_and_haystack(client.tokenizer.tokenize)

    results = []
    context_lengths = create_context_lengths(model_context_len, MODEL_MIN_CONTEXT_LEN)
    logging.info(f"Created context lengths for evaluation: {context_lengths}")

    for context_len in context_lengths:
        trimmed_haystack, actual_context_len = trim_haystack(tokenized_haystack, tokenized_needle, context_len)
        fractions = [i / 10 for i in range(0, 10)] # [0.0, 0.1, ..., 0.9]
        for fraction in fractions:
            # create n_turns random insertion positions between a given interval, e.g. 0.0-0.1
            # for each position, the model is evaluated, and collect results
            insertion_positions = create_needle_insertion_depths(n_turns, actual_context_len, fraction)
            for i, insertion_position in enumerate(insertion_positions):
                tokenized_haystack_with_needle = insert_needle_in_haystack(
                    trimmed_haystack, insertion_position, tokenized_needle
                )
                haystack_with_needle = client.tokenizer.convert_tokens_to_string(tokenized_haystack_with_needle)
                prompt = template.get_prompt("needle-in-haystack", {"text": haystack_with_needle, "city": city})
                output = generation.generate_with_huggingface(prompt, client, parameters)
                result = format_result(
                    context_len, actual_context_len, fraction,
                    insertion_position, anniversary, prompt, output
                )
                results.append(result)

        logging.info(f"Completed evaluation for context length: {context_len}\n")

    if args.save_results:
        generation.save_results(results, config.NIH, args.model_name, False)

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

    return tokenized_needle, tokenized_haystack, random_city, anniversary


def create_context_lengths(max_context_len: int, current_context_len: int) -> list[int]:
    # [1024, 2048, ..., max_context_len]
    return [2**i * current_context_len for i in range(int(max_context_len / current_context_len).bit_length())]


def create_needle_insertion_depths(n_turn: int, actual_context_length: int, fraction: float):
    insertion_fractions = [random.uniform(0, 0.1) for i in range(n_turn)]
    insertion_positions = [int(actual_context_length * (f + fraction)) for f in insertion_fractions]
    return insertion_positions


def trim_haystack(tokenized_haystack: list[int], tokenized_needle: list[int], context_length: int):
    trimmed_length = context_length - len(tokenized_needle)
    logging.info(f"Trimmed tokenized haystack length to '{trimmed_length}' to fit current context length of '{context_length}'.")
    return tokenized_haystack[:trimmed_length], trimmed_length


def insert_needle_in_haystack(tokenized_haystack: list[int], insertion_position: int, tokenized_needle: list[int]):
    return tokenized_haystack[:insertion_position] + tokenized_needle + tokenized_haystack[insertion_position:]


def format_result(context_len, actual_context_len, fraction, insertion_position, anniversary, prompt, output) -> Dict:
    answer = helper.clean_answer(output.text)
    return {
        "context_length": context_len,
        "actual_context_length": actual_context_len,
        "fraction": f"{fraction:0.1f}-{fraction + 0.1:0.1f}",
        "interval": f"{int((actual_context_len / 10) * (fraction * 10))}-{int((actual_context_len / 10) * ((fraction + 0.1) * 10))}",
        "insertion_position": insertion_position,
        "prompt": prompt,
        "model_output": output.text,
        "model_cleaned_output": answer,
        "correct_answer": anniversary,
    }

def compute_scores(args, results: list, n_turns: int) -> dict:
    for result in results:
        answer = result["model_cleaned_output"]
        anniversary = result["correct_answer"]
        if str(answer).strip() == str(anniversary):
            result["score"] = 1
        else:
            result["score"] = 0

    logging.info("NIH benchmark results computed.")
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.NIH}-{args.model_name}-eval-results.json")

    return compute_average_score(args, results, n_turns)


def compute_average_score(args, results: list, n_turns: int) -> dict:

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
    for context_length in aggregated_results:
        for fraction in aggregated_results[context_length]:
            aggregated_results[context_length][fraction] /= n_turns

    logging.info("NIH benchmark aggregated results computed.")

    helper.plot_needle_in_haystack(
        aggregated_results,
        save_path=f"{config.RESULTS_DIR}/{config.NIH}-{args.model_name}-nih-plot.png",
        )

    return aggregated_results
