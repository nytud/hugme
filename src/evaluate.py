

from benchmarks import mmlu
from metrics import faithfulness, nih


TASKS = {
    "bias": None,
    "coherence": None,
    "hallucination": None,
    "faithfulness": faithfulness.benchmark,
    "mmlu":  mmlu.benchmark,
    "needinhaystack": nih.benchmark,
    "relevancy": None,
    "spell": None,
    "summary": None,
    "toxicity": None,
    "truthfulqa": None,
}


def evaluate(args) -> None:

    print("Evaluation started.")

    for task in args.tasks:

        print(f"Started evaluatiion on {task}.")