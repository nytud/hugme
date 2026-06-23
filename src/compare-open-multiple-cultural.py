#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strict_open_correct(score):
    return score >= 1.0


def abcd_correct(score):
    return score >= 1.0


def extract_model_name(filename):
    name = filename.replace("-open", "")
    name = name.replace("False-eval-results.json", "")
    name = name.replace("True-eval-results.json", "")
    name = name.replace("cultural", "")
    return name


def pair_files(abcd_dir, open_dir):
    abcd_files = {
        file_path.name: file_path
        for file_path in Path(abcd_dir).glob("*-eval-results.json")
    }

    pairs = []

    for open_file in Path(open_dir).glob("*-open-*-eval-results.json"):
        abcd_name = open_file.name.replace("-open", "")

        if abcd_name in abcd_files:
            pairs.append((abcd_files[abcd_name], open_file))

    return pairs


def compare_pair(abcd_file, open_file):
    abcd_data = load_json(abcd_file)
    open_data = load_json(open_file)

    model_name = extract_model_name(abcd_file.name)

    abcd_by_question_id = {
        item["question_id"]: item
        for item in abcd_data
    }

    open_by_question_id = {
        item["question_id"]: item
        for item in open_data
    }

    rows = []

    common_ids = sorted(
        set(abcd_by_question_id.keys()) &
        set(open_by_question_id.keys())
    )

    for question_id in common_ids:
        abcd_item = abcd_by_question_id[question_id]
        open_item = open_by_question_id[question_id]

        abcd_is_correct = abcd_correct(abcd_item["score"])
        open_is_correct = strict_open_correct(open_item["score"])

        if abcd_is_correct and open_is_correct:
            outcome = "both_correct"
        elif abcd_is_correct and not open_is_correct:
            outcome = "abcd_only"
        elif not abcd_is_correct and open_is_correct:
            outcome = "open_only"
        else:
            outcome = "both_wrong"

        rows.append({
            "model": model_name,
            "question_id": question_id,
            "category": abcd_item["category"],
            "question": open_item["question"],

            "abcd_score": float(abcd_item["score"]),
            "abcd_correct": abcd_is_correct,

            "open_score": float(open_item["score"]),
            "open_correct": open_is_correct,
            "open_verdict": open_item.get("verdict"),

            "outcome": outcome,

            "abcd_answer": abcd_item.get("output"),
            "open_answer": open_item.get("output_raw"),
            "target": open_item.get("target"),
        })

    return rows


def build_reports(df):
    summary_rows = []

    for model, model_group in df.groupby("model"):

        abcd_accuracy = model_group["abcd_correct"].mean()

        open_accuracy = model_group["open_correct"].mean()

        open_partial = model_group["open_score"].mean()

        summary_rows.append({
            "model": model,
            "n": len(model_group),
            "abcd_accuracy": abcd_accuracy,
            "open_strict_accuracy": open_accuracy,
            "open_partial_score": open_partial,
            "accuracy_gap": abcd_accuracy - open_accuracy,
        })

    model_summary = pd.DataFrame(summary_rows)

    category_data = (
        df.groupby(["model", "category"])
        .agg(
            n=("question_id", "count"),
            abcd_accuracy=("abcd_correct", "mean"),
            open_accuracy=("open_correct", "mean"),
            open_partial=("open_score", "mean"),
        )
        .reset_index()
    )

    category_summary = category_data.pivot_table(
        index="model",
        columns="category",
        values=["n", "abcd_accuracy", "open_accuracy", "open_partial"],
        aggfunc="first"
    ).reset_index()
    category_summary.columns.name = None

    outcome_data = (
        df.groupby(["model", "outcome"])
        .size()
        .reset_index(name="count")
    )

    outcome_summary = outcome_data.pivot_table(
        index="model",
        columns="outcome",
        values="count",
        aggfunc="first"
    ).reset_index()
    outcome_summary.columns.name = None

    differences = df[
        df["outcome"].isin(
            ["abcd_only", "open_only"]
        )
    ].copy()

    return (
        model_summary,
        category_summary,
        outcome_summary,
        differences,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "abcd_dir",
        help="ABCD results directory"
    )

    parser.add_argument(
        "open_dir",
        help="Open-answer results directory"
    )

    parser.add_argument(
        "--out",
        default="comparison_output",
        help="Output directory"
    )

    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    pairs = pair_files(
        args.abcd_dir,
        args.open_dir
    )

    if not pairs:
        raise RuntimeError(
            "No matching ABCD/Open file pairs found."
        )

    for abcd_file, open_file in pairs:
        rows = compare_pair(
            abcd_file,
            open_file
        )
        all_rows.extend(rows)

    merged_df = pd.DataFrame(all_rows)

    (
        model_summary,
        category_summary,
        outcome_summary,
        differences,
    ) = build_reports(merged_df)

    # Write item-level files per model
    for model in merged_df["model"].unique():
        model_data = merged_df[merged_df["model"] == model]
        model_data.to_csv(
            output_dir / f"merged_item_level_{model}.csv",
            index=False
        )

    model_differences = differences[differences["model"].notna()]
    for model in model_differences["model"].unique():
        model_diff_data = model_differences[model_differences["model"] == model]
        model_diff_data.to_csv(
            output_dir / f"item_level_differences_{model}.csv",
            index=False
        )

    model_summary.to_csv(
        output_dir / "model_summary.csv",
        index=False
    )

    category_summary.to_csv(
        output_dir / "category_summary.csv",
        index=False
    )

    outcome_summary.to_csv(
        output_dir / "outcome_summary.csv",
        index=False
    )

    print(f"Processed {len(merged_df)} questions")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()