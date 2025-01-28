import time

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

    for task in args.tasks:

        print(f"Started evaluation on {task}.")

        task_start_time = time.time()

        if task not in TASKS:
            raise ValueError(f"Task {task} is not among tasks: {list(TASKS.keys())}.")

        # current_task = TASKS[task]

        print(f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    print(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}.")

    return None