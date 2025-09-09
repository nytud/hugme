from typing import List, Dict

import os
import string
import random
import logging
import requests
from spellchecker import SpellChecker

import config
import helper
import generation


spell = SpellChecker(local_dictionary=config.SPELLING_DICT)


def compute_metric(args, task_name: str):
    dataset = helper.read_json(config.SPELLING_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_score(args, results)


def format_result(entry: dict, prompt: str, output: generation.ModelOutput) -> dict:
    return {
        "prompt": prompt,
        "output": output.text,
        "token_usage": output.total_tokens
    }


def compute_score(args, results: List[Dict]):
    text_lens, misspelled_count = 0, 0
    spelling_results = []
    for i, entry in enumerate(results):
        output = entry["output"]
        text_len, misspelled, correct_rate = check_spelling(output)
        spelling_results.append(
            {
                "index": i,
                "output": output,
                "misspelled": misspelled,
                "correct_rate": correct_rate
            }
        )
        text_lens += text_len
        misspelled_count += len(misspelled)

    if args.save_results:
        helper.save_json(
            spelling_results, config.RESULTS_DIR, f"{config.SPELLING}-{args.model_name}-{args.thinking}-eval-results.json"
        )
    spelling_score = (1 - misspelled_count / text_lens) * 100

    logging.info(f"Spell checking results score: {spelling_score}")
    return {"score": spelling_score}


def check_spelling(text: str):
    text = remove_punctuation(text)
    texts = text.split()
    text_len = len(texts)
    misspelled = list(spell.unknown(texts))
    correct_rate = 1 - len(misspelled) / text_len
    return text_len, misspelled, correct_rate


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def check_spelling_with_llm(words: list):
    if not words:
        return 1
    prompt = f"A következő szó vagy szavak helyesek-e magyarul? \
        Kizárólag 'yes' vagy 'no' lehet a válaszod! szó: {words}"
    result = generate_with_openai(prompt)
    if "yes" in result.lower() or "igen" in result.lower():
        return 1
    return 0


def generate_with_openai(query: str, temperature: float = 0.4, max_tokens: int = 128, **kwargs) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    endpoint = "https://api.openai.com/v1/chat/completions"
    messages = [{"role": "user", "content": query}]
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }
    response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    data = response.json()
    return data['choices'][0]['message']['content'] if data.get('choices') else ""
