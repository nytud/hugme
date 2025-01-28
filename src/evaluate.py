import logging

from src import metrics
from src import benchmarks


TASKS = {
    "answer_relevancy": metrics.answer_relevancy.compute_metric,
    "bias": metrics.bias.compute_metric,
    "coherence": metrics.coherence.compute_metric,
    "hallucination": metrics.hallucination.compute_metric,
    "faithfulness": metrics.faithfulness.compute_metric,
    "mmlu":  benchmarks.mmlu.benchmark,
    "needleinhaystack": metrics.nih.compute_metric,
    "relevancy": benchmarks.truthfulqa.benchmark,
    "spell": metrics.spell.compute_metric,
    "summarization": metrics.summerization.compute_metric,
    "toxicity": metrics.toxicity.compute_metric,
    "truthfulqa": benchmarks.truthfulqa.benchmark,
}


def evaluate(args) -> None:

    logging.info("Evaluation started.")

    for task in args.tasks:

        logging.info(f"Started evaluation on {task}.")

        if task not in TASKS:
            raise ValueError(f"Task {task} is not among tasks: {TASKS.keys()}.")




    return None