import os
import string
import requests
from spellchecker import SpellChecker

import config
import helper
import metrics


spell = SpellChecker(local_dictionary=config.SPELLING_DICT)


def compute_metric(task_name, args, generation_pipeline):
    dataset = helper.read_json(config.SPELLING_DATASET)
    results = metrics.generate_results(args, generation_pipeline, dataset, "spelling")
    results = compute_score(results)
    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{task_name}-results.json")
    return results


def compute_score(results: list):
    spelling_results = []
    for i, entry in enumerate(results):
        output = entry["output"]
        spelling, bad_words, score = check_spelling(output)
        spelling_results.append(
            {
                "index": i, "output": output, "spelling": spelling,
                "bad_words": bad_words, "score": score
            }
        )
    return spelling_results


def check_spelling(text: str):
    text = remove_punctuation(text)
    texts = text.split()
    misspelled = spell.unknown(texts)
    print(f"texts: {texts}")
    print(f"misspelled: {misspelled}")
    if misspelled:
        return "BAD", list(misspelled), 0
    return "OK", [], 1


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def generate_with_openai(query: str, temperature: float = 0.4, max_tokens: int = 128, **kwargs):
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
    return data['choices'][0]['message']['content'] if data.get('choices') else None
