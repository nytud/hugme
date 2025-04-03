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
    spelling_score, llm_spelling_score = 0.0, 0.0
    spelling_results = []
    for i, entry in enumerate(results):
        output = entry["output"]
        spelling, bad_words, score = check_spelling(output)
        llm_score = check_spelling_with_llm(bad_words)
        spelling_results.append(
            {
                "index": i, "output": output, "spelling": spelling,
                "bad_words": bad_words, "score": score, "llm_score": llm_score
            }
        )
        spelling_score += score
        llm_spelling_score += llm_score
    if args.save_results:
        helper.save_json(spelling_results, config.RESULTS_DIR, f"{task_name}-eval-results.json")
    spelling_score = spelling_score / len(results) * 100
    llm_spelling_score = llm_spelling_score / len(results) * 100
    print(f"Spell checking results score: {spelling_score}, llm-score: {llm_spelling_score}")
    return {"score": spelling_score, "llm-score": llm_spelling_score}


def check_spelling(text: str):
    text = remove_punctuation(text)
    texts = text.split()
    misspelled = list(spell.unknown(texts))
    if misspelled:
        return "BAD", misspelled, 0
    return "OK", misspelled, 1


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
