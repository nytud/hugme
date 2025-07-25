import os
import string
import random
import requests
from spellchecker import SpellChecker

import config
import helper
import metrics


spell = SpellChecker(local_dictionary=config.SPELLING_DICT)


def compute_metric(task_name, args, generate):
    dataset = helper.read_json(config.SPELLING_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)
    if args.use_gen_results:
        print("Using generation results from path: ", args.use_gen_results)
        results = helper.read_json(args.use_gen_results)
    else:
        results = metrics.generate_results(args, generate, dataset, config.SPELLING)
    scores = compute_score(args, results, task_name)
    return scores


def compute_score(args, results: list, task_name: str):
    text_lens, misspelled_count = 0, 0
    spelling_results = []
    for i, entry in enumerate(results):
        output = entry["output"]
        text_len, misspelled, correct_rate = check_spelling(output)
        spelling_results.append(
            {
                "index": i, "output": output,
                "misspelled": misspelled, "correct_rate": correct_rate
            }
        )
        text_lens += text_len
        misspelled_count += len(misspelled)

    if args.save_results:
        helper.save_json(spelling_results, config.RESULTS_DIR, f"{task_name}-eval-results.json")

    spelling_score = (1 - misspelled_count / text_lens) * 100

    print(f"Spell checking results score: {spelling_score}")
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
