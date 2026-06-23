from typing import Any, Dict, Optional, Tuple
import random
import logging
import re
from tqdm import tqdm
import openai

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import config
import helper
import generation

def compute_metric(args, task_name: str) -> dict:
    dataset = helper.read_json(config.CULTURAL_OPEN_DATASET)
    sample_size = max(1, int(args.sample_size * len(dataset)))
    dataset = random.sample(dataset, sample_size)
    gen_results = generation.generate_results(args, task_name, dataset, format_result)
    return compute_scores(args, gen_results)


def format_result(entry: Dict[str, Any], prompt: Any, output: generation.ModelOutput) -> Dict:
    raw_output = output.text.strip()
    return {
        "question_id": entry.get("question_id"),
        "question": entry.get("question"),
        "prompt": prompt,
        "output_raw": raw_output,
        "output_normalized": normalize_answer(raw_output, entry.get("answer_type")),
        "target": get_target_text(entry),
        "answer_type": entry.get("answer_type"),
        "category": entry.get("category"),
        "total_tokens": output.total_tokens
    }


def get_target_text(entry: Dict[str, Any]) -> Any:
    if "gold_answer" in entry:
        return entry["gold_answer"]
    return None


def normalize_text(
    text: str,
    remove_punctuation: bool = False,
) -> str:

    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text).strip()
    if remove_punctuation:
        text = re.sub(r'[.,;:!?\'"()[\]{}—–-]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_answer(text: str, answer_type: Optional[str] = None) -> str:
    if answer_type == "entity":
        return normalize_text(text, remove_punctuation=True)
    if answer_type == "short_answer":
        return normalize_text(text, remove_punctuation=True)
    return normalize_text(text, remove_punctuation=False)


def judge_wrapper(entry: Dict[str, Any], generated: str, args) -> Tuple[str, float, str]:
    answer_type = entry.get("answer_type")

    if answer_type == "entity":
        return judge_entity_item(entry, generated)

    return judge_item_with_llm(entry, generated, args)

def _judge_entity_candidate(output_norm: str, candidate: str) -> Optional[Tuple[str, float, str]]:
    if output_norm == candidate:
        return "correct", 1.0, "Exact match"
    if output_norm.startswith(candidate) or candidate.startswith(output_norm):
        return "partially_correct", 1.0, "Prefix match"
    if len(candidate) > 3 and candidate in output_norm:
        return "partially_correct", 1.0, "Substring match"
    return None


def judge_entity_item(entry: Any, output: str) -> Tuple[str, float, str]:
    output_norm = normalize_text(output, remove_punctuation=True)
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

        match = _judge_entity_candidate(output_norm, candidate)
        if match:
            return match

    return "incorrect", 0.0, "No match"


def judge_item_with_llm(entry: Dict[str, Any], generated: str, args) -> Tuple[str, float, str]:
    answer_type = entry.get("answer_type")
    if not isinstance(answer_type, str):
        answer_type = ""

    question = entry.get("question", "")
    reference_answer = entry.get("reference_answer")
    scoring_rubric = entry.get("scoring_rubric", {})

    prompt = build_judge_prompt(answer_type, question, generated, reference_answer, scoring_rubric)

    try:
        judge_model, is_openai = load_judge_client(args)

        if is_openai:
            response = generation.generate_with_openai(prompt, judge_model, args.judge, {})
        else:
            response = generation.generate_with_huggingface(prompt, judge_model, {}, {})

        return parse_judge_response(response.text)
    except (RuntimeError, ValueError, openai.OpenAIError) as e:
        logging.error(f"LLM judge failed: {e}")
        return "uncertain", 0.0, f"Judge error: {str(e)}"


def build_judge_prompt(
    answer_type: str,
    question: str,
    generated: str,
    reference_answer: Optional[str] = None,
    scoring_rubric: Optional[Dict[str, Any]] = None,
) -> Any:
    generated_text = generated.strip()

    rubric = scoring_rubric or {}
    req = rubric.get("required_elements", [])
    opt = rubric.get("optional_elements", [])
    crit = rubric.get("critical_errors", [])

    def list_to_str(items):
        if not items:
            return "(nincs megadva)"
        return "; ".join([str(x) for x in items])

    rubric_text = (
        f"Required elements: {list_to_str(req)}\n"
        f"Optional elements: {list_to_str(opt)}\n"
        f"Critical errors: {list_to_str(crit)}"
    )

    if answer_type == "short_answer":
        prompt_text = f"""Értékeld az alábbi rövid választ az alábbi rubrika alapján. Válaszd ki az egyik lehetőséget.
        Kérdés: {question}
        Referencia (vagy gold): {reference_answer}
        Modell válasza: {generated_text}
        Értékelési rubrika:
        {rubric_text}
        Szabályok:
        1) Ha a válasz tartalmazza a rubrikában megadott minden required elemet és nincs kritikus hiba, válaszolj: CORRECT
        2) Ha a válasz tartalmaz néhány required vagy több optional elemet, válaszolj: PARTIALLY_CORRECT
        3) Ha hiányoznak a required elemek vagy kritikus hibák vannak: INCORRECT
        4) Ha nem tudsz dönteni: UNCERTAIN

        Válasz formátuma: Csak az egyik szó legyen: CORRECT, PARTIALLY_CORRECT, INCORRECT vagy UNCERTAIN."""
    else:
        prompt_text = f"""Értékeld az alábbi magyarázatot az alábbi rubrika alapján. Válaszd ki az egyik lehetőséget.
        Kérdés: {question}
        Referencia magyarázat: {reference_answer}
        Modell magyarázata: {generated_text}
        Értékelési rubrika:
        {rubric_text}
        Szabályok:
        1) Ha a magyarázat tartalmazza a rubrikában megadott minden required elemet és nincs kritikus hiba: CORRECT
        2) Ha részben megfelel: PARTIALLY_CORRECT
        3) Ha hiányos vagy hibás: INCORRECT
        4) Ha bizonytalan vagy többértelmű: UNCERTAIN
        Válasz formátuma: Csak az egyik szó legyen: CORRECT, PARTIALLY_CORRECT, INCORRECT vagy UNCERTAIN."""

    return [{"role": "user", "content": prompt_text}]


def parse_judge_response(text: str) -> Tuple[str, float, str]:
    normalized = normalize_text(text)

    if "correct" in normalized and "partially" not in normalized:
        return "correct", 1.0, "LLM verdict: correct"
    if "partially_correct" in normalized or "partial" in normalized:
        return "partially_correct", 0.5, "LLM verdict: partially_correct"
    if "incorrect" in normalized or "nem" in normalized:
        return "incorrect", 0.0, "LLM verdict: incorrect"
    if "uncertain" in normalized or "bizonytalan" in normalized:
        return "uncertain", 0.0, "LLM verdict: uncertain"

    # Ha nem érthető a válasz
    logging.warning(f"Judge response unclear, marking as uncertain: {text}")
    return "uncertain", 0.0, f"Unclear LLM response: {text[:50]}"


def load_judge_client(args):
    if config.PROVIDER_API_KEY:
        return generation.initialize_openai_client(), True

    logging.info(f"Loading local judge model: {args.judge}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.judge,
        token=config.HF_TOKEN,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.judge,
        device_map="auto",
        token=config.HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=config.MODEL_DTYPE
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    if pipe.tokenizer is not None and pipe.model is not None and hasattr(pipe.tokenizer, "pad_token_id"):
        pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    return pipe, False


def compute_scores(args, results: list) -> dict:
    score = 0.0
    output = []

    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        verdict = judge_wrapper(entry, entry["output_raw"], args)

        score += verdict[1]
        output.append({
            "question_id": entry.get("question_id"),
            "question": entry.get("question"),
            "category": entry.get("category"),
            "output_raw": entry.get("output_raw"),
            "output_normalized": entry.get("output_normalized"),
            "target": entry.get("target"),
            "answer_type": entry.get("answer_type"),
            "verdict": verdict[0],
            "score": verdict[1],
            "judgment_detail": verdict[2]
        })

    total_score = score / len(output) if output else 0.0

    uncertain_cases = [r for r in output if r["verdict"] == "uncertain"]
    explanation_or_short = [
        r for r in output
        if r.get("answer_type") in ("explanation", "short_answer")
    ]

    logging.info(f"Cultural open benchmark score: {round(total_score * 100, 2)}%")
    logging.info(f"Uncertain cases: {len(uncertain_cases)} / {len(explanation_or_short)}")

    if args.save_results:
        helper.save_json(
            output,
            config.RESULTS_DIR,
            f"{config.CULTURAL_OPEN}-{args.model_name}-{args.thinking}-eval-results.json"
        )

        helper.save_json(
            uncertain_cases,
            config.RESULTS_DIR,
            f"{config.CULTURAL_OPEN}-{args.model_name}-{args.thinking}-uncertain-cases.json"
        )


    verdicts = [e.get("verdict") for e in output]
    stat_summary = {
        "total": len(output),
        "correct": verdicts.count("correct"),
        "partially_correct": verdicts.count("partially_correct"),
        "incorrect": verdicts.count("incorrect"),
        "uncertain": verdicts.count("uncertain"),
    }

    return {
        "category_scores": helper.group_by_category(output, total_score),
        "stat_summary": stat_summary
    }
