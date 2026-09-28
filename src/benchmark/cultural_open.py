from typing import Any, Tuple

import random
import textwrap
from tqdm import tqdm
from enum import Enum

import openai

import config
import helper
import generation


class AnswerType(str, Enum):
    ENTITY = "entity"
    SHORT_ANSWER = "short_answer"
    EXPLANATION = "explanation"

class Verdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.CULTURAL_OPEN_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset)))
    dataset = random.sample(dataset, sample_size)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def format_result(entry: dict, prompt: Any, output: generation.ModelOutput) -> dict:

    remove_punctuation = entry["answer_type"] in [AnswerType.ENTITY, AnswerType.SHORT_ANSWER]

    return {
        "id":                   entry["question_id"],
        "question":             entry["question"],
        "prompt":               prompt,
        "output":               output.text,
        "output_normalized":    helper.normalize_text(output.text, remove_punctuation=remove_punctuation),
        "gold_answer":          entry["gold_answer"],
        "answer_type":          entry["answer_type"],
        "category":             entry["category"],
        "total_tokens":         output.total_tokens,
        "scoring_rubric": {
            "required_elements": entry["required_elements"],
            "optional_elements": entry["optional_elements"],
            "critical_errors":   entry["critical_errors"]
        },
        "accepted_aliases":      entry["accepted_aliases"],
    }


def compute_scores(args, results: list) -> dict:
    score = 0.0
    outputs = []

    for entry in tqdm(results, desc="Calculating scores", unit="query"):

        verdict = grade_entry(entry, args)
        score += verdict[1]
        outputs.append({
            "id":               entry["question_id"],
            "question":         entry["question"],
            "category":         entry["category"],
            "output":           entry["output"],
            "output_thinking":  entry["thinking_output"],
            "output_normalized": entry["output_normalized"],

            "answer_type":      entry["answer_type"],
            "gold_answer":      entry["gold_answer"],

            "scoring_rubric": entry["scoring_rubric"],
            "accepted_aliases": entry["accepted_aliases"],
            "verdict": verdict[0],
            "score": verdict[1],
            "judgment_detail": verdict[2],
        })
    total_score = score / len(outputs)
    print(f"Cultural open benchmark score: {round(total_score * 100, 2)}%")

    uncertain_cases = [r for r in outputs if r["verdict"] == "uncertain"]
    explanation_or_short = [r for r in outputs if r["answer_type"] in ["explanation", "short_answer"]]
    print(f"Uncertain cases: {len(uncertain_cases)} / {len(explanation_or_short)}")

    if args.save_results:
        model_name = helper.cleanup_model_name(args.model_name)
        helper.save_json(outputs, config.RESULTS_DIR, f"{config.CULTURAL_OPEN}-{model_name}-eval-results.json")
        helper.save_json(uncertain_cases, config.RESULTS_DIR, f"{config.CULTURAL_OPEN}-{model_name}-uncertain-cases.json")

    return {
        "category_scores": helper.group_by_category(outputs, total_score),
        "summary_statistics": create_summary_statistics(outputs)
    }


def grade_entry(entry: dict, args) -> Tuple[str, float, str]:
    if entry["answer_type"] == AnswerType.ENTITY:
        return grade_entry_manually_with_entity_answer_type(entry)
    return grade_entry_by_judge(entry, args)


def grade_entry_manually_with_entity_answer_type(entry: dict) -> Tuple[str, float, str]:

    def grade_candidate(output_norm: str, candidate: str):
        if output_norm == candidate:
            return "correct", 1.0, "Exact match"
        if output_norm.startswith(candidate) or candidate.startswith(output_norm):
            return "partially_correct", 1.0, "Prefix match"
        if len(candidate) > 3 and candidate in output_norm:
            return "partially_correct", 1.0, "Substring match"
        return None

    output_norm = entry["output_normalized"]
    if not output_norm:
        return "incorrect", 0.0, "Empty output"

    candidates = []

    if "gold_answer" in entry:
        candidates.append(entry["gold_answer"])

    if "accepted_aliases" in entry:
        aliases = entry["accepted_aliases"]
        if isinstance(aliases, list):
            for alias in aliases:
                candidates.append(str(alias))

    for candidate in candidates:
        if not candidate:
            continue

        match = grade_candidate(output_norm, candidate)
        if match:
            return match

    return "incorrect", 0.0, "No match"


def grade_entry_by_judge(entry: dict, args) -> Tuple[str, float, str]:
    prompt = build_judge_prompt(entry)
    judge_client = load_judge_model(args)
    response = generation.generate(judge_client, prompt, args.judge, parameters={})
    verdict = parse_judge_response(response.text)
    return verdict


def build_judge_prompt(entry: dict) -> list:

    is_short = entry["answer_type"] == AnswerType.SHORT_ANSWER
    kind = "rövid választ" if is_short else "magyarázatot"
    rubric = entry["scoring_rubric"]

    prompt_text = textwrap.dedent(f"""\
        Értékeld az alábbi {kind} a rubrika alapján.

        Kérdés: {entry["question"]}
        Referencia: {entry["gold_answer"]}
        Modell válasza: {entry["output"]}

        Rubrika:
        Szükséges (required) elemek: {rubric["required_elements"]}
        Opcionális (optional) elemek: {rubric["optional_elements"]}
        Kritikus (critical) hibák: {rubric["critical_errors"]}

        Szabályok (az első illeszkedő érvényes):
        1) Ha van kritikus hiba: INCORRECT
        2) Ha minden szükséges (required) elem megvan: CORRECT
        3) Ha a szükséges (required) elemek egy része megvan: PARTIALLY_CORRECT
        4) Ha egyetlen szükséges (required) elem sincs meg: INCORRECT
        5) Ha a rubrika alapján nem tudsz dönteni: UNCERTAIN

        Csak az egyik szóval válaszolj: CORRECT, PARTIALLY_CORRECT, INCORRECT vagy UNCERTAIN.""")

    return [{"role": "user", "content": prompt_text}]


def parse_judge_response(text: str) -> Tuple[str, float, str]:
    normalized = helper.normalize_text(text)

    # order matters: "correct" is a substring of "incorrect" and "partially_correct"
    if Verdict.INCORRECT.value in normalized or "nem" in normalized.split():
        verdict, score = Verdict.INCORRECT, 0.0
    elif Verdict.PARTIALLY_CORRECT.value in normalized:
        verdict, score = Verdict.PARTIALLY_CORRECT, 0.5
    elif Verdict.UNCERTAIN.value in normalized or "bizonytalan" in normalized:
        verdict, score = Verdict.UNCERTAIN, 0.0
    elif Verdict.CORRECT.value in normalized:
        verdict, score = Verdict.CORRECT, 1.0
    else:
        raise ValueError(f"Unclear judge response: {text!r}")

    return verdict, score, f"LLM verdict: {verdict.value}"


def load_judge_model(args):
    assert args.judge is not None, "Judge model must be specified."
    assert config.PROVIDER_API_KEY is not None, "Provider API key must be specified."
    client = openai.OpenAI(api_key=config.PROVIDER_API_KEY, base_url=config.PROVIDER_URL)
    return client


def create_summary_statistics(outputs: list) -> dict:
    verdicts = [e.get("verdict") for e in outputs]
    summary_stats = {
        "total":             len(outputs),
        "correct":           verdicts.count("correct"),
        "partially_correct": verdicts.count("partially_correct"),
        "incorrect":         verdicts.count("incorrect"),
        "uncertain":         verdicts.count("uncertain"),
    }
    return summary_stats