from typing import Dict, List
import random
import torch

import config
import helper
import template


TURNS = 2
MAX_NEW_TOKENS = 20
MAX_CONTEXT_LENGTH = 1024
HUNDREDTH = 0.01
GOOD_SOLUTION = 1.0
BAD_SOLUTION = 0.0


def select_needle(generation_pipeline):
    cities = helper.read_file(config.NEEDLE_FILE).split("\n")
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
    lower_bound = max(0, int(len(cutted_haystack) * ((fraction-10) * HUNDREDTH)))
    upper_bound = int(len(cutted_haystack) * (fraction * HUNDREDTH))
    insert_position = random.randint(lower_bound, upper_bound)

    return (
        cutted_haystack[:insert_position]
        + tokenized_needle
        + cutted_haystack[insert_position:]
    )



def generate_answer_scores(cutted_haystack, city, anniversary, generation_pipeline, tokenized_needle):
    system_prompt = (
        f"Kizárólag a következő szöveg alapján, "
        f"hanyadik évfordulóját ünnepelte {city} város?\n"
        "Csak egy számot adj vissza!"
    )
    new_rows = []
    for j in range(10, 110, 10):
        tokenized_haystack_with_needle = insert_needle(
            cutted_haystack, j, tokenized_needle
        )
        full_stack_text = generation_pipeline.tokenizer.convert_tokens_to_string(
            tokenized_haystack_with_needle
        )
        entry = {"system_prompt": system_prompt, "full_stack_text": full_stack_text}
        prompt = template.get_prompt("needle-in-haystack", entry)

        with torch.no_grad():
            actual_answer = generation_pipeline(
                text_inputs=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                return_full_text = False,
            )

        answer = helper.clean_answer(actual_answer)

        if str(answer).strip() == str(anniversary) and len(str(actual_answer[0]["generated_text"])) <= 3:
            goodness = 1.0
        elif len(str(actual_answer[0]["generated_text"])) >= 3 and str(anniversary) in str(actual_answer[0]["generated_text"]):
            goodness = 0.5
        else:
            goodness = 0

        new_row = {
            "context_window": [len(tokenized_haystack_with_needle)],
            "fraction": [j],
            "answer": [answer],
            "score": [goodness],
            "actual_answer": [actual_answer],
        }
        torch.cuda.empty_cache()
        new_rows.append(new_row)
        answer = None
    return new_rows



def evaluate_haystack_context(tokenized_haystack, tokenized_needle, city, anniversary, generation_pipeline):
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


def compute_metric(task_name, args, generation_pipeline) -> List[Dict[str, float]]:
    tokenized_needle, city, anniversary = select_needle(generation_pipeline)
    haystack = helper.read_file(config.HAYSTACK_DATASET)
    tokenized_haystack = generation_pipeline.tokenizer.tokenize(haystack)
    results = []
    for _ in range(TURNS):
        result = evaluate_haystack_context(
            tokenized_haystack, tokenized_needle, city, anniversary, generation_pipeline
        )
        results.append(result)

    return results
