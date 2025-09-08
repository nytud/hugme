import time
import logging
from datetime import datetime

import config
import helper
import benchmark
from generation import ModelHandler


TASK_HANDLERS = {
    **{task: benchmark.metrics.compute_metric for task in config.METRICS}, # deepeval-based metrics
    config.MMLU: benchmark.mmlu.benchmark,
    config.TRUTHFUL_QA: benchmark.truthfulqa.benchmark,
    config.SPELLING: benchmark.spelling.compute_metric,
    config.PROMPT_ALIGNMENT: benchmark.prompt_alignment.compute_metric,
    config.READABILITY: benchmark.readability.compute_metric,
    config.NIH: benchmark.nih.compute_metric,
    config.COLA: benchmark.cola.compute_metric,
}

def evaluate(args) -> None:
    logging.info("Evaluation started for tasks: " + ", ".join(args.tasks))
    score_results = {}
    eval_start_time = time.time()

    for task_name in args.tasks:

        model_handler = ModelHandler(task_name, args) if not args.use_gen_results else None

        logging.info(f"Started evaluation on {task_name}.")
        task_start_time = time.time()

        if task_name not in TASK_HANDLERS:
            raise ValueError(f"Task '{task_name}' is not supported. Valid tasks: {list(TASK_HANDLERS.keys())}")

        score = TASK_HANDLERS[task_name](task_name, args, model_handler)
        score_results[task_name] = score

        logging.info(f"Task {task_name} took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    logging.info(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device} for tasks: {', '.join(args.tasks)}.")

    if args.save_results:
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        helper.save_json(score_results, config.RESULTS_DIR, f"hugme-results-{current_time}.json")
