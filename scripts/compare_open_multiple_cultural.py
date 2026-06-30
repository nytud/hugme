import logging

import argparse
from pathlib import Path
import pandas as pd

from helper import read_json

def is_correct(score):
    return score >= 1.0


def extract_model_name(filename):
    return (
        filename.replace("-open", "")
        .replace("False-eval-results.json", "")
        .replace("True-eval-results.json", "")
        .replace("cultural", "")
    )


def determine_outcome(abcd_is_correct, open_is_correct):
    if abcd_is_correct and open_is_correct:
        return "both_correct"
    if abcd_is_correct:
        return "abcd_only"
    if open_is_correct:
        return "open_only"
    return "both_wrong"


def load_question_map(path):
    return {item["question_id"]: item for item in read_json(path)}


def make_comparison_row(model_name, question_id, abcd_item, open_item):
    abcd_is_correct = is_correct(abcd_item["score"])
    open_is_correct = is_correct(open_item["score"])

    return {
        "model": model_name,
        "question_id": question_id,
        "category": abcd_item["category"],
        "question": open_item["question"],
        "abcd_score": float(abcd_item["score"]),
        "abcd_correct": abcd_is_correct,
        "open_score": float(open_item["score"]),
        "open_correct": open_is_correct,
        "open_verdict": open_item.get("verdict"),
        "outcome": determine_outcome(abcd_is_correct, open_is_correct),
        "abcd_answer": abcd_item.get("output"),
        "open_answer": open_item.get("output_raw"),
        "target": open_item.get("target"),
    }


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
    model_name = extract_model_name(abcd_file.name)
    abcd_by_question_id = load_question_map(abcd_file)
    open_by_question_id = load_question_map(open_file)

    common_ids = sorted(set(abcd_by_question_id) & set(open_by_question_id))
    return [
        make_comparison_row(
            model_name,
            question_id,
            abcd_by_question_id[question_id],
            open_by_question_id[question_id],
        )
        for question_id in common_ids
    ]


def build_reports(df):
    model_summary = pd.DataFrame(
        [
            {
                "model": model,
                "n": len(group),
                "abcd_accuracy": group["abcd_correct"].mean(),
                "open_strict_accuracy": group["open_correct"].mean(),
                "open_partial_score": group["open_score"].mean(),
                "accuracy_gap": group["abcd_correct"].mean() - group["open_correct"].mean(),
            }
            for model, group in df.groupby("model")
        ]
    )

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
        aggfunc="first",
    ).reset_index()
    category_summary.columns.name = None

    outcome_summary = (
        df.groupby(["model", "outcome"])
        .size()
        .reset_index(name="count")
        .pivot_table(
            index="model",
            columns="outcome",
            values="count",
            aggfunc="first",
        )
        .reset_index()
    )
    outcome_summary.columns.name = None

    differences = df[df["outcome"].isin(["abcd_only", "open_only"])].copy()

    return model_summary, category_summary, outcome_summary, differences


def write_report_csvs(output_dir, merged_df, differences, summaries):
    model_summary, category_summary, outcome_summary, _ = summaries

    for model in merged_df["model"].unique():
        merged_df[merged_df["model"] == model].to_csv(
            output_dir / f"merged_item_level_{model}.csv",
            index=False,
        )

    for model in differences["model"].dropna().unique():
        differences[differences["model"] == model].to_csv(
            output_dir / f"item_level_differences_{model}.csv",
            index=False,
        )

    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    category_summary.to_csv(output_dir / "category_summary.csv", index=False)
    outcome_summary.to_csv(output_dir / "outcome_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("abcd_dir", help="ABCD results directory")
    parser.add_argument("open_dir", help="Open-answer results directory")
    parser.add_argument("--out", default="comparison_output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_files(args.abcd_dir, args.open_dir)
    if not pairs:
        raise RuntimeError("No matching ABCD/Open file pairs found.")

    all_rows = []
    for abcd_file, open_file in pairs:
        all_rows.extend(compare_pair(abcd_file, open_file))

    merged_df = pd.DataFrame(all_rows)
    summaries = build_reports(merged_df)
    _, _, _, differences = summaries
    write_report_csvs(output_dir, merged_df, differences, summaries)

    logging.info(f"Processed {len(merged_df)} questions")
    logging.info(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
