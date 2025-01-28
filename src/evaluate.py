
import metrics
import benchmarks


TASKS = {
    "answer_relevancy": metrics.compute_answer_relevancy,
    "bias": metrics.compute_bias,
    "coherence": metrics.compute_coherence,
    "hallucination": metrics.compute_hallucination,
    "faithfulness": metrics.compute_faithfulness,
    "mmlu":  benchmarks.mmlu_benchmark,
    "needleinhaystack": metrics.compute_nih,
    "spell": metrics.compute_spell,
    "summarization": metrics.compute_summarization,
    "toxicity": metrics.compute_toxicity,
    "truthfulqa": benchmarks.truthfulqa_benchmark,
}


def evaluate(args) -> None:

    print("Evaluation started.")

    for task in args.tasks:

        print(f"Started evaluation on {task}.")

        if task not in TASKS:
            raise ValueError(f"Task {task} is not among tasks: {TASKS.keys()}.")




    return None