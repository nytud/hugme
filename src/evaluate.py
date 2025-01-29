import time
import json

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import metrics
import benchmarks


TASKS = {
    "answer_relevancy": metrics.compute_answer_relevancy,
    "bias": metrics.compute_bias,
    "coherence": metrics.compute_coherence,
    "hallucination": metrics.compute_hallucination,
    "faithfulness": metrics.compute_faithfulness,
    "mmlu":  benchmarks.mmlu_benchmark,
    "needle_in_haystack": metrics.compute_nih,
    "spell": metrics.compute_spell,
    "summarization": metrics.compute_summarization,
    "toxicity": metrics.compute_toxicity,
    "truthful_qa": benchmarks.truthfulqa_benchmark,
}


def eval(args) -> None:

    print("Evaluation started.")

    eval_start_time = time.time()

    for task_name in args.tasks:

        print(f"Started evaluation on {task_name}.")

        task_start_time = time.time()

        if task_name not in TASKS:
            raise ValueError(f"Task {task_name} is not among tasks: {list(TASKS.keys())}.")

        generation_pipeline = get_generation_pipeline(args)

        task = TASKS[task_name]

        task(args, generation_pipeline)

        print(f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    print(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}.")


def get_generation_pipeline(args):

    parameters = json.load(args.config) if args.config else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", token=args.hf_token)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **parameters)

    return pipe
