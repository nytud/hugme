import time

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import hugme.benchmark as benchmark
import hugme.config as config
import hugme.helper as helper
import hugme.metrics as metrics

TASKS = {
    "answerrelevancy": metrics.answer_relevancy.compute_metric,
    "bias": metrics.bias.compute_metric,
    "coherence": metrics.coherence.compute_metric,
    "hallucination": metrics.hallucination.compute_metric,
    "faithfulness": metrics.faithfulness.compute_metric,
    "mmlu": benchmark.mmlu.benchmark,
    "needleinhaystack": metrics.nih.compute_metric,
    "spell": metrics.spell.compute_metric,
    "summarization": metrics.summerization.compute_metric,
    "toxicity": metrics.toxicity.compute_metric,
    "truthfulqa": benchmark.truthfulqa.benchmark,
}


def evaluate(args) -> None:
    print("Evaluation started...")
    score_results = {}
    eval_start_time = time.time()

    for task_name in args.tasks:

        print(f"Started evaluation on {task_name}.")

        task_start_time = time.time()

        if task_name not in TASKS:
            raise ValueError(
                f"Task {task_name} is not among tasks: {list(TASKS.keys())}."
            )

        generation_pipeline = get_generation_pipeline(args)
        print("Finished loading model and tokenizer...")

        print(f"Started evaluation on {task_name}.")
        task_start_time = time.time()

        print(
            f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}."
        )

    print(
        f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}."
    )

    if args.save_results:
        helper.save_json(score_results, config.RESULTS_DIR, "hugme-results.json")


def get_generation_pipeline(args):
    parameters = helper.read_json(args.parameters) if args.parameters else {}
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, token=args.hf_token
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", token=args.hf_token
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **parameters)
    return pipe
