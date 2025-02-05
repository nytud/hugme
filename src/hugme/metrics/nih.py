import random
import re
import torch
from hugme import config
from hugme.helper import read_file

MAX_CONTEXT_LENGTH = 8192
TURNS = 2
DATASET = config.DATASETS + "nih.txt"
NEEDLE = config.DATASETS + "nih_needle.txt"
MAX_NEW_TOKENS = 10
HUNDREDTH = 0.01
GOOD_SOLUTION = 1.0
BAD_SOLUTION = 0.0


class NIHEvaluator:
    def __init__(self, args, generation_pipeline):
        self.args = args
        self.generation_pipeline = generation_pipeline
        # Initialize needle-related values and system prompt
        self.tokenized_needle, self.city, self.anniversary = self._select_needle()
        haystack = read_file(DATASET)
        self.tokenized_haystack = self.tokenizer.tokenize(haystack)
        self.answers = []

    def _select_needle(self):
        cities = read_file(NEEDLE).split("\n")
        random_city = random.choice(cities)
        anniversary = random.randint(1, 100)
        needle_text = f"Ezen a napon ünnepelte {random_city} város a {anniversary}. évfordulóját."
        tokenized_needle = self.generation_pipeline.tokenizer.tokenize(needle_text)
        return tokenized_needle, random_city, anniversary

    def _cut_haystack(self, fraction):
        trimmed_length = int(MAX_CONTEXT_LENGTH * (fraction * HUNDREDTH)) - len(self.tokenized_needle)
        return self.tokenized_haystack[:trimmed_length]

    def _insert_needle(self, cutted_haystack, fraction):
        lower_bound = int(len(cutted_haystack) * ((fraction - MAX_NEW_TOKENS) * HUNDREDTH))
        upper_bound = int(len(cutted_haystack) * (fraction * HUNDREDTH))
        insert_position = random.randint(lower_bound, upper_bound)
        return (
            cutted_haystack[:insert_position]
            + self.tokenized_needle
            + cutted_haystack[insert_position:]
        )

    @staticmethod
    def _clean_answer(answer):
        return re.sub(r"\D", "", str(answer))

    def _generate_answer_scores(self, cutted_haystack):
        system_prompt = (
            f"Kizárólag a következő szöveg alapján, "
            f"hanyadik évfordulóját ünnepelte {self.city} város?\n"
            "Csak egy számot adj vissza!"
        )
        for j in range(MAX_NEW_TOKENS, 100 + MAX_NEW_TOKENS, MAX_NEW_TOKENS):
            tokenized_haystack_with_needle = self._insert_needle(cutted_haystack, j)
            full_stack_text = self.generation_pipeline.tokenizer.convert_tokens_to_string(tokenized_haystack_with_needle)

            with torch.no_grad():
                actual_answer = self.generation_pipeline(
                    text_inputs=f"{system_prompt}\n {full_stack_text}", max_new_tokens=MAX_NEW_TOKENS
                )

            answer = self._clean_answer(actual_answer)
            if str(answer).strip() == str(self.anniversary):
                goodness = 1.0
            elif len(str(actual_answer)) >= 3 and str(self.anniversary) in str(actual_answer):
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
            self.answers.append(new_row)
            torch.cuda.empty_cache()

    def _evaluate_haystack_context(self):
        for i in range(10, 110, 10):
            cutted_haystack = self._cut_haystack(i)
            self._generate_answer_scores(cutted_haystack)

    def compute_metric(self):
        for _ in range(TURNS):
            self._evaluate_haystack_context()
        return self.answers


def compute_metric(args, generation_pipeline) -> None:
    """
    This is the externally visible function.
    Its signature is unchanged so that existing calls (e.g. from your evaluation script)
    continue to work.
    """
    evaluator = NIHEvaluator(args, generation_pipeline)
    answers = evaluator.compute_metric()
    return answers
