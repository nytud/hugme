import logging
import random
import openai
import huspacy
import spacy.util
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

import config
import helper
import generation
import template


if not spacy.util.is_package(config.HUSPACY_MODEL_NAME):
    logging.info(f"Downloading {config.HUSPACY_MODEL_NAME}...")
    huspacy.download(config.HUSPACY_MODEL_NAME)

nlp = huspacy.load()


def compute_metric(args, task_name: str) -> float:

    dataset = helper.read_json(config.COLA_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset))) # at least 1 sample
    dataset = random.sample(dataset, sample_size)

    results = generation.generate_results(args, task_name, dataset, format_result)

    model = load_classifier()
    classification_results = classify_results(args, results, model)
    return compute_scores(args, classification_results)


def format_result(entry: dict, prompt: str, output: generation.ModelOutput) -> dict:
    return {
        "input": prompt,
        "output": output.text,
        "questions": entry["questions"],
        "token_usage": output.total_tokens
    }


def classify_results(args, results, model):
    classified = []
    for i, entry in enumerate(results):
        sentences = [sent.text for sent in nlp(entry["output"]).sents]
        sentence_results = classify_sentences(sentences, model, args.judge)
        classified.append({
            "index": i,
            "input": entry["input"],
            "sentences": sentences,
            "results": sentence_results,
        })
    return classified


def load_classifier():
    if not config.BERT_MODEL:
        raise ValueError("BERT model path is not set in config.BERT_MODEL")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(config.BERT_MODEL)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device_map="auto")
    logging.info(f"Loaded BERT model from {config.BERT_MODEL} for CoLA evaluation.")
    return classifier


def classify_sentences(sentences, classifier, judge):
    results = []
    for sentence in sentences:
        bert_result = classify_sentences_with_bert(sentence, classifier)
        judge_score = None

        if bert_result["label"] == 0: # fallback to judge if BERT predicts ungrammatical
            judge_score = classify_sentences_with_openai(judge, sentence)

        results.append({
            "sentence": sentence,
            "bert-label": bert_result["label"],
            "bert-score": bert_result["score"],
            "judge-score": judge_score
        })
    return results


def classify_sentences_with_bert(sentence, classifier):
    # grammatical 1, ungrammatical 0
    result = classifier(sentence)
    return {"sentence": sentence, "label": int(result[0]['label']), "score": result[0]['score']}


def classify_sentences_with_openai(judge, sentence):
    client = openai.OpenAI()
    messages = template.get_cola_prompt_for_model(sentence)
    response = client.chat.completions.create(
        model=judge,
        messages=messages,
        max_tokens=15
    )
    label = response.choices[0].message.content.strip()
    try:
        label = int(label)
    except ValueError:
        logging.error(f"Invalid judge response: {label}. Expected an integer (0 or 1).")
        raise ValueError(f"Invalid judge response: {label}. Expected an integer (0 or 1).")
    if label not in [0, 1]:
        logging.error(f"Judge response must be 0 or 1, got {label}.")
        raise ValueError(f"Judge response must be 0 or 1, got {label}.")
    return label


def compute_scores(args, results):

    total_sentences = 0
    grammatical_sentence_count = 0

    for item in results:
        sentence_scores = [res["bert-label"] + (res["judge-score"] or 0) for res in item["results"]]
        item["grammatical-score"] = (
            sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
        )
        grammatical_sentence_count += sum(sentence_scores)
        total_sentences += len(sentence_scores)

    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.COLA}-{args.model_name}-eval-results.json")

    accuracy = ((grammatical_sentence_count / total_sentences) * 100) if total_sentences else 0.0
    logging.info(f"CoLA benchmark accuracy: {accuracy:.2f}%")
    return accuracy
