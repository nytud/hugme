import random
import re
import time
from typing import Dict, List

import torch

import config
import helper
from answer_provider import AbstractGenerator, GenerationInput
from args import HuGMEArgs
from helper import read_file

MAX_CONTEXT_LENGTH = 1024
TURNS = 2
MAX_NEW_TOKENS = 10
HUNDREDTH = 0.01
GOOD_SOLUTION = 1.0
BAD_SOLUTION = 0.0


def select_needle(generation_pipeline: AbstractGenerator):
    cities = read_file(config.NEEDLE_FILE).split("\n")
    random_city = random.choice(cities)
    anniversary = random.randint(1, 100)
    needle_text = (
        f"Ezen a napon ünnepelte {random_city} város a {anniversary}. évfordulóját."
    )
    tokenized_needle = generation_pipeline.tokenizer.tokenize(needle_text)
    return tokenized_needle, random_city, anniversary


def cut_haystack(fraction, tokenized_haystack, tokenized_needle):
    trimmed_length = int(MAX_CONTEXT_LENGTH * (fraction * HUNDREDTH)) - len(
        tokenized_needle
    )
    return tokenized_haystack[:trimmed_length]


def insert_needle(cutted_haystack, fraction, tokenized_needle):
    lower_bound = int(len(cutted_haystack) * ((fraction - MAX_NEW_TOKENS) * HUNDREDTH))
    upper_bound = int(len(cutted_haystack) * (fraction * HUNDREDTH))
    insert_position = random.randint(lower_bound, upper_bound)
    return (
        cutted_haystack[:insert_position]
        + tokenized_needle
        + cutted_haystack[insert_position:]
    )


def clean_answer(answer):
    return re.sub(r"\D", "", str(answer))


def generate_answer_scores(
    cutted_haystack, city, anniversary, generation_pipeline, tokenized_needle
):
    system_prompt = (
        f"Kizárólag a következő szöveg alapján, "
        f"hanyadik évfordulóját ünnepelte {city} város?\n"
        "Csak egy számot adj vissza!"
    )
    new_rows = []
    for j in range(MAX_NEW_TOKENS, 100 + MAX_NEW_TOKENS, MAX_NEW_TOKENS):
        tokenized_haystack_with_needle = insert_needle(
            cutted_haystack, j, tokenized_needle
        )
        full_stack_text = generation_pipeline.tokenizer.convert_tokens_to_string(
            tokenized_haystack_with_needle
        )
        generation_input = GenerationInput(
            prompt=f"{system_prompt}\n {full_stack_text}",
            task_name="nih",
        )
        with torch.no_grad():
            actual_answer = generation_pipeline.generate_for_task(generation_input)

        torch.cuda.empty_cache()

        if str(clean_answer(actual_answer)).strip() == str(anniversary):
            goodness = 1.0
        elif len(str(actual_answer)) >= 3 and str(anniversary) in str(actual_answer):
            goodness = 0.5
        else:
            goodness = 0

        new_row = {
            "context_window": [len(tokenized_haystack_with_needle)],
            "fraction": [j],
            "answer": [clean_answer(actual_answer)],
            "score": [goodness],
            "actual_answer": [actual_answer],
        }
        new_rows.append(new_row)
    return new_rows


def evaluate_haystack_context(
    tokenized_haystack, tokenized_needle, city, anniversary, generation_pipeline
):
    results = []
    for i in range(10, 110, 10):
        cutted_haystack = cut_haystack(i, tokenized_haystack, tokenized_needle)
        results.append(
            generate_answer_scores(
                cutted_haystack,
                city,
                anniversary,
                generation_pipeline,
                tokenized_needle,
            )
        )
    return results


def compute_metric(args: HuGMEArgs, generation_pipeline: AbstractGenerator, task_name: str) -> List[Dict[str, float]]:
    tokenized_needle, city, anniversary = select_needle(generation_pipeline)
    haystack = read_file(config.HAYSTACK_DATASET)
    tokenized_haystack = generation_pipeline.tokenizer.tokenize(haystack)
    results = []
    for _ in range(TURNS):
        result = evaluate_haystack_context(
            tokenized_haystack, tokenized_needle, city, anniversary, generation_pipeline
        )
        results.append(result)

    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{task_name}-{args.model_name.replace('/', '-')}-\
                         {int(time.time())}-eval-results.json")


    return results
