from typing import Any, Dict, List, Optional, Tuple
import random
import logging
import re
from tqdm import tqdm

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
    elif answer_type == "short_answer":
        return normalize_text(text, remove_punctuation=True)
    else:
        return normalize_text(text, remove_punctuation=False)


def build_target_candidates(target: Any) -> List[str]:
    if isinstance(target, list):
        candidates = [str(x) for x in target]
    else:
        candidates = [str(target)]

    split_pattern = r'\s*(?:\||,|;|/|vagy)\s*'
    expanded = []
    for candidate in candidates:
        if re.search(split_pattern, candidate):
            expanded.extend(re.split(split_pattern, candidate))
        else:
            expanded.append(candidate)

    return [normalize_text(item, remove_punctuation=True) for item in expanded if item and normalize_text(item, remove_punctuation=True)]


def is_entity_correct(output: str, target: Any) -> bool:
    output_norm = normalize_text(output)
    if not output_norm:
        return False

    candidates = build_target_candidates(target)
    for candidate in candidates:
        if not candidate:
            continue

        if output_norm == candidate:
            return True
        if output_norm.startswith(candidate):
            return True
        if candidate in output_norm:
            return True

        output_words = re.findall(r"[\wáéíóöőúüű]+", output_norm)
        candidate_words = re.findall(r"[\wáéíóöőúüű]+", candidate)
        if output_words and candidate_words and output_words[: len(candidate_words)] == candidate_words:
            return True

    return False


def judge_item_with_llm(entry: Dict[str, Any], generated: str, args) -> float:
    answer_type = entry.get("answer_type")
    target = get_target_text(entry)
    question = entry.get("question", "")

    if answer_type == "entity":
        return 1.0 if is_entity_correct(generated, target) else 0.0

    prompt = build_judge_prompt(answer_type, question, target, generated)
    judge_model, is_openai = load_judge_client(args)
    response = None

    if is_openai:
        response = generation.generate_with_openai(prompt, judge_model, args.judge, {})
        score_text = response.text
    else:
        response = generation.generate_with_huggingface(prompt, judge_model, {}, {})
        score_text = response.text

    return parse_judge_response(score_text)


def build_judge_prompt(answer_type: str, question: str, reference: Any, generated: str) -> Any:
    reference_text = reference if isinstance(reference, str) else " ".join(map(str, reference))
    generated_text = generated.strip()

    system_message = {
        "role": "system",
        "content": (
            "You are a Hungarian evaluation assistant. Compare the generated response to the reference answer. "
            "Answer only with 'IGEN' if the generated answer is correct and appropriate, or 'NEM' otherwise. "
            "Do not add any extra explanation."
        )
    }

    if answer_type == "short_answer":
        user_content = (
            f"Kérdés: {question}\n"
            f"Referencia válasz: {reference_text}\n"
            f"Generált válasz: {generated_text}\n"
            "A generált válasz legyen rövid és pontos. Ha a generált szöveg helyes és a kérdésre válaszol, válaszolj IGENNEL. "
            "Ha téves, hiányos vagy nem a kérdésre válaszol, válaszolj NEMMEL."
        )
    else:
        user_content = (
            f"Kérdés: {question}\n"
            f"Referencia magyarázat: {reference_text}\n"
            f"Generált magyarázat: {generated_text}\n"
            "Ellenőrizd, hogy a generált magyarázat pontos, objektív és a kérdésre válaszol-e. "
            "Ha megfelel a referenciának és tartalmazza a legfontosabb tényeket, válaszolj IGENNEL. "
            "Ha pontatlan, túl szubjektív vagy irreleváns, válaszolj NEMMEL."
        )

    return [system_message, {"role": "user", "content": user_content}]


def parse_judge_response(text: str) -> float:
    normalized = normalize_text(text)
    if "igen" in normalized or normalized.startswith("1") or "yes" in normalized:
        return 1.0
    if "nem" in normalized or normalized.startswith("0") or "no" in normalized:
        return 0.0

    logging.warning(f"Judge did not return a clear label for cultural_open: {text}")
    return 0.0


def load_judge_client(args):
    # Prefer OpenAI-based judge if a provider API key is configured.
    if config.PROVIDER_API_KEY:
        return generation.initialize_openai_client(), True

    # Fallback to a local judge model if OpenAI access is unavailable.
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    logging.info(f"Loading local judge model for cultural_open evaluation: {args.judge}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.judge,
        token=config.HF_TOKEN,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge,
        device_map="auto",
        token=config.HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=config.MODEL_DTYPE
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id
    return pipe, False


def compute_scores(args, results: list) -> dict:
    score = 0.0
    for entry in tqdm(results, desc="Calculating scores", unit="query"):
        score_val = judge_item_with_llm(entry, entry["output"], args)
        entry["score"] = score_val
        score += score_val

    total_score = score / len(results)
    logging.info(f"Cultural open benchmark score: {round(total_score * 100, 2)}%")

    if args.save_results:
        helper.save_json(
            results,
            config.RESULTS_DIR,
            f"{config.CULTURAL_OPEN}-{args.model_name}-{args.thinking}-eval-results.json"
        )

    return helper.group_by_category(results, total_score)
