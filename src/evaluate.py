
from src.benchmarks import mmlu
from src.metrics import faithfulness


TASKS = {
    "hummlu":  mmlu.benchmark,
    "faithfulness": faithfulness.benchmark,
}


def evaluate(args) -> None:

    print("Evaluation started.")
