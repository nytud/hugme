import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

import config
import helper
import metrics
import time

MAX_LENGTH = 512
MODEL_NAME = 'SZTAKI-HLT/hubert-base-cc'

def compute_metric(args, generation_pipeline):
    dataset = helper.read_json(config.TEXT_COHERENCE_DATASET)
    results = metrics.generate_results(args, generation_pipeline, dataset, config.TEXT_COHERENCE)
    scores = compute_score(args, results, config.TEXT_COHERENCE)
    return scores


def compute_score(args, results: list, task_name: str):
    scores = []
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForNextSentencePrediction.from_pretrained(MODEL_NAME)
    for i, entry in enumerate(results):
        sentences = helper.split_sentences(entry["output"])
        avg_prob_next, avg_prob_not_next = calculate_nsp_probabilities(tokenizer, model, sentences)
        scores.append({"index": i, "avg_prob_next": avg_prob_next, "avg_prob_not_next": avg_prob_not_next})
        print(f"Avg. probability for next sentence follows previous sentences: {avg_prob_next:.5f}")
        print(f"Avg. probability for next sentence *not* follows previous sentences: {avg_prob_not_next:.5f}")
    if args.save_results:
        helper.save_json(scores, config.RESULTS_DIR, f"{task_name}-{args.model_name}-{int(time.time())}-eval-results.json")
    return scores


def calculate_nsp_probabilities(tokenizer, model, sentences):
    prob_next, prob_not_next = 0.0, 0.0
    for i in range(len(sentences)-1):
        context_sentences, current_sentence = sentences[:i+1], sentences[i + 1]
        context_text = ""
        # collect all sentences that fit into the max length
        for context_sentence in reversed(context_sentences):
            tokens = tokenizer(
                context_text, current_sentence, return_tensors='pt', max_length=MAX_LENGTH, truncation=False
            )
            if tokens["input_ids"].shape[1] <= MAX_LENGTH:
                context_text = context_sentence + ". " + context_text
            else:
                break
        inputs = tokenizer(
            context_text, current_sentence, return_tensors="pt",
            max_length=MAX_LENGTH, truncation=True, padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = logits.softmax(dim=1)
        prob_next += probabilities[0][0].item()
        prob_not_next += probabilities[0][1].item()
    return prob_next / len(sentences), prob_not_next / len(sentences)
