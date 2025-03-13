import time

import benchmark
import coherence
import config
import helper
import metrics
import readability
import spelling
from src.cli import HuGMEArgs
from answer_provider import AbstractGenerator, LocalGenerator, OpenAIGenerator, CustomGenerator, TextGenerator

TASK_HANDLERS = {
    **{task: metrics.compute_metric for task in config.METRICS},
    config.MMLU: benchmark.mmlu.benchmark,
    config.TRUTHFUL_QA: benchmark.truthfulqa.benchmark,
    config.SPELLING: spelling.compute_metric,
    config.TEXT_COHERENCE: coherence.compute_metric,
    config.PROMPT_ALIGNMENT: benchmark.prompt_alignment.compute_metric,
    config.READABILITY: readability.compute_metric,
}

def get_generator(args: HuGMEArgs) -> AbstractGenerator:
    if args.generated_file:
        return TextGenerator(args)
    elif args.api_provider and args.api_provider == 'openai':
        return OpenAIGenerator(args)
    elif args.api_provider:
        return CustomGenerator(args)
    return LocalGenerator(args)

def evaluate(args: HuGMEArgs) -> None:
    print("Evaluation started...")
    score_results = {}
    eval_start_time = time.time()

    for task_name in args.tasks:

        print("Loading model and tokenizer...")

        generation_pipeline = AbstractGenerator(args)
        print("Finished loading model and tokenizer...")

        print(f"Started evaluation on {task_name}.")
        task_start_time = time.time()

        if task_name not in TASK_HANDLERS:
            raise ValueError(f"Task '{task_name}' is not supported. Valid tasks: {list(TASK_HANDLERS.keys())}")

        results = TASK_HANDLERS[task_name](args, generation_pipeline)
        score_results[task_name] = results

        print(f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    print(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}.")

    if args.save_results:
        helper.save_json(score_results, config.RESULTS_DIR, f"hugme-results-{args.model_name}-{int(time.time())}.json")
