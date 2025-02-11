import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import config
import helper
import metrics
import spelling
import coherence
import benchmark


def evaluate(args) -> None:
    print("Evaluation started...")
    score_results = {}
    eval_start_time = time.time()

    for task_name in args.tasks:

        print("Loading model and tokenizer...")
        generation_pipeline = get_generation_pipeline(args)
        print("Finished loading model and tokenizer...")

        print(f"Started evaluation on {task_name}.")
        task_start_time = time.time()

        if task_name in config.METRICS:
            results = metrics.compute_metric(task_name, args, generation_pipeline)
        elif task_name == config.MMLU:
            results = benchmark.mmlu.benchmark(args, generation_pipeline)
        elif task_name == config.TRUTHFUL_QA:
            results = benchmark.truthfulqa.benchmark(args, generation_pipeline)
        elif task_name == config.SPELLING:
            results = spelling.compute_metric(task_name, args, generation_pipeline)
        elif task_name == config.TEXT_COHERENCE:
            results = coherence.compute_metric(args, generation_pipeline)
        else:
            raise ValueError(
                f"Task {task_name} is not among tasks: {config.METRICS + [config.MMLU, config.TRUTHFUL_QA]}."
            )
        score_results[task_name] = results
        print(f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    print(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}.")
    if args.save_results:
        helper.save_json(score_results, config.RESULTS_DIR, "hugme-results.json")


def get_generation_pipeline(args):
    parameters = helper.read_json(args.parameters) if args.parameters else {}
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=config.HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=config.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", token=config.HF_TOKEN)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **parameters)
    return pipe
