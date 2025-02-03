import random
import re
import torch
from transformers import AutoTokenizer
import config

TEST_COMTEXT_LENGTH = 8192
TURNS = 2
DATASET = config.DATASETS + "nih.txt"
NEEDLE = config.DATASETS + "nih_needle.txt"


def select_needle(tokenizer):
    with open(needle, "r", encoding="utf-8") as file:
        cities = file.readlines()
    random_city = random.choice([city.strip() for city in cities])
    anniversary = random.randint(1, 100)
    needle = (
        f"Ezen a napon ünnepelte {random_city} város a {anniversary}. évfordulóját."
    )
    tokenized_needle = tokenizer.tokenize(needle)
    return tokenized_needle, random_city, anniversary


def cut_haystack(tokenized_haystack, fraction, tokenized_needle):
    trimmed_length = int(TEST_COMTEXT_LENGTH * (fraction * 0.01)) - len(
        tokenized_needle
    )
    return tokenized_haystack[:trimmed_length]


def insert_needle(cutted_haystack, tokenized_needle, fraction):
    lower_bound = int(len(cutted_haystack) * ((fraction - 10) * 0.01))
    upper_bound = int(len(cutted_haystack) * (fraction * 0.01))
    insert_position = random.randint(lower_bound, upper_bound)
    haystack_with_needle = (
        cutted_haystack[:insert_position]
        + tokenized_needle
        + cutted_haystack[insert_position:]
    )
    return haystack_with_needle


def clean_answer(answer):
    return re.sub(r"\D", "", str(answer))


def evaluate_haystack_context(
    self, tokenized_needle, anniversary, system_prompt, answers
):
    for i in range(10, 110, 10):
        cutted_haystack = self.cut_haystack(
            self.tokenized_haystack, i, tokenized_needle
        )
        answers = generate_answer_scores(
            self, tokenized_needle, anniversary, system_prompt, answers, cutted_haystack
        )
    return answers


def generate_answer_scores(
    self, tokenized_needle, anniversary, system_prompt, answers, cutted_haystack
):
    for j in range(10, 110, 10):
        tokenized_haystack_with_needle = self.insert_needle(
            cutted_haystack, tokenized_needle, j
        )
        full_stack_text = self.detokenize(tokenized_haystack_with_needle)

        with torch.no_grad():
            actual_answer = self.llm2.generate(
                query_string=f"{system_prompt}\n {full_stack_text}"
            )

        answer = clean_answer(actual_answer)
        if str(answer).strip() == str(anniversary):
            goodness = 1
        elif len(str(actual_answer)) >= 3 and str(anniversary) in str(actual_answer):
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
        answers = answers.append(new_row)

        torch.cuda.empty_cache()
    return answers


def compute_metric(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_needle, city, anniversary = select_needle(tokenizer)
    system_prompt = f"Kizárólag a következő szöveg alapján, hanyadik évfordulóját ünnepelte {city} város?\nCsak egy számot adj vissza!"
    answers = []

    for _ in range(0, int(TURNS)):
        answers = evaluate_haystack_context(
            self, tokenized_needle, anniversary, system_prompt, answers
        )
