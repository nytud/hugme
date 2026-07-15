#!/usr/bin/env python3

import json
import pathlib
import argparse
import pandas as pd
from rich.table import Table
from rich.console import Console

"""
python3 scripts/compare_cultural.py \
    --model-name puli-chat \
    --cultural-abcd-file /home/username/hugme/results/cultural-abcd-puli-chat-false-eval-results.json \
    --cultural-open-file /home/username/hugme/results/cultural-open-puli-chat-false-eval-results.json \
    --output-dir ./comparison/ \
    --save
"""


def compare_cultural(abcd_file, open_file) -> list[dict]:
    # compare cultural ABCD and open-answer results for a given model

    abcd_results = read_json(abcd_file)
    open_results = read_json(open_file)

    abcd_by_question_id = group_by_question_id(abcd_results)
    open_by_question_id = group_by_question_id(open_results)

    common_question_ids = sorted(set(abcd_by_question_id) & set(open_by_question_id))

    results = []
    for qid in common_question_ids:
        result = compare_entry(abcd_by_question_id[qid], open_by_question_id[qid])
        results.append(result)
    return results


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def group_by_question_id(results):
    return {entry["question_id"]: entry for entry in results}


def compare_entry(abcd_entry, open_entry) -> dict:

    assert abcd_entry["category"] == open_entry["category"], "Categories do not match"
    assert abcd_entry["question_id"] == open_entry["question_id"], "Question IDs do not match"

    return {
        "question_id": abcd_entry["question_id"],
        "category": abcd_entry["category"],
        "question": open_entry["question"],

        "abcd_score": abcd_entry["score"],
        "abcd_output": abcd_entry["output"],

        "open_score": open_entry["score"],
        "open_output": open_entry["output_normalized"],

        "final": compare_score(
            abcd_entry["score"], open_entry["score"]
        ),
    }


def compare_score(abcd_score, open_score):
    if abcd_score == open_score:
        return "both good"
    elif abcd_score > open_score:
        return "abcd better"
    elif abcd_score > open_score and open_score > 0:
        return "abcd better, but open above 0"
    elif open_score > abcd_score:
        return "open better"
    elif open_score == 0 and abcd_score == 0:
        return "both bad"
    else:
        raise ValueError(f"Unexpected score comparison: abcd_score={abcd_score}, open_score={open_score}")

def build_report(results, args) -> None:

    df = pd.DataFrame(results)

    abcd_accuracy = df["abcd_score"].mean()
    open_strict_accuracy = (df["open_score"] == 1).mean()
    open_partial_score = (df["open_score"] == 0.5).mean()
    diff = abcd_accuracy - open_partial_score

    console = Console(record=True)

    table = Table(title=f"Cultural eval summary for {args.model_name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Questions compared", str(len(df)))
    table.add_row("ABCD accuracy", f"{abcd_accuracy:.2%}")
    table.add_row("Open-answer strict accuracy", f"{open_strict_accuracy:.2%}")
    table.add_row("Open-answer partial-credit score", f"{open_partial_score:.2%}")
    table.add_row("ABCD - Open difference", f"{diff:+.2%}")

    console.print(table)

    # group by category
    cat_table = Table(title="By category")
    cat_table.add_column("Category")
    cat_table.add_column("N", justify="right")
    cat_table.add_column("ABCD acc", justify="right")
    cat_table.add_column("Open strict", justify="right")
    cat_table.add_column("Open partial", justify="right")
    cat_table.add_column("Diff", justify="right")

    for category, group in df.groupby("category"):
        cat_table.add_row(
            str(category),
            str(len(group)),
            f"{group['abcd_score'].mean():.2%}",
            f"{(group['open_score'] == 1).mean():.2%}",
            f"{(group['open_score'] == 0.5).mean():.2%}",
            f"{group['abcd_score'].mean() - group['open_score'].mean():+.2%}",
        )

    console.print(cat_table)

    # 'final' column distribution
    final_table = Table(title="Comparison outcomes")
    final_table.add_column("Outcome")
    final_table.add_column("Count", justify="right")
    final_table.add_column("%", justify="right")

    for outcome, count in df["final"].value_counts().items():
        final_table.add_row(str(outcome), str(count), f"{count / len(df):.2%}")

    console.print(final_table)

    if args.save:
        summary_file = pathlib.Path(args.output_dir) / f"{args.model_name}-summary.txt"
        summary_file.write_text(console.export_text(), encoding="utf-8")


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name of the model to compare")
    parser.add_argument("--cultural-abcd-file", required=True, help="Path to the cultural ABCD eval results file")
    parser.add_argument("--cultural-open-file", required=True, help="Path to the cultural open-answer eval results file")
    parser.add_argument("--output-dir", default="./comparison/", help="Output directory")
    parser.add_argument("--save", action="store_true", help="Save the results and report")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abcd_file = pathlib.Path(args.cultural_abcd_file)
    open_file = pathlib.Path(args.cultural_open_file)

    assert abcd_file.exists(), f"ABCD file does not exist: {abcd_file}"
    assert open_file.exists(), f"Open file does not exist: {open_file}"
    assert args.model_name in abcd_file.name, "Model name must be part of the ABCD file name"
    assert args.model_name in open_file.name, "Model name must be part of the open file name"

    results = compare_cultural(abcd_file, open_file)
    if args.save:
        pd.DataFrame(results).to_json(output_dir / f"{args.model_name}-cultural-comparison.json")

    build_report(results, args)


if __name__ == "__main__":
    main()

