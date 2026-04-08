import os
import re
import gc
import json
import random
import logging
import pathlib
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seeds(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def set_device(args) -> None:
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(args.cuda_ids)
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    args.device = device


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def read_file(file_path, readlines: bool = False):
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return file.readlines() if readlines else file.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e


def read_json(file_path):
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e


def save_json(data, dir_path, file_name: str) -> None:
    dir_path = pathlib.Path(dir_path)
    file_path = dir_path / file_name
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, default=str)
    except OSError as e:
        raise OSError(f"Could not save file: {file_path}") from e


def split_sentences(text: str):
    return [sentence.strip() for sentence in text.split('.') if sentence]


def group_by_category(results: list, acc: float) -> dict:
    score_sums = defaultdict(float)
    counts = defaultdict(int)

    for entry in results:
        category = entry["category"]
        score_sums[category] += entry["score"]
        counts[category] += 1

    percentages = {}
    for category in score_sums:
        avg = score_sums[category] / counts[category] if counts[category] else 0
        percentages[category] = round(avg * 100, 2)

    percentages["total"] = round(acc * 100, 2)

    return percentages


def clean_answer(answer):
    return re.sub(r"\D", "", str(answer))


# pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-positional-arguments
def plot_needle_in_haystack(
    results,
    save_path: str,
    title: str = "Needle-in-a-Haystack (heatmap)",
    annotate: bool = False,
    annotate_fmt: str = ".2f",
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 200,
) -> None:
    # sort x-axis numerically (context lengths stored as strings)
    x_vals = sorted((int(k) for k in results.keys()))
    x_labels = [str(x) for x in x_vals]

    # collect all fraction bins and sort by their numeric start (e.g. "0.1-0.2" -> 0.1)
    def frac_start(s: str) -> float:
        return float(s.split("-", 1)[0])

    all_fracs = set()
    for ctx_str, frac_map in results.items():
        if not isinstance(frac_map, dict):
            raise TypeError(f"Expected dict for context '{ctx_str}', got {type(frac_map)}")
        all_fracs.update(frac_map.keys())

    y_vals = sorted(all_fracs, key=frac_start)  # fraction labels
    y_labels = y_vals

    # build grid [y, x]
    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
    y_index = {f: i for i, f in enumerate(y_vals)}
    x_index = {ctx: j for j, ctx in enumerate(x_vals)}

    for ctx_str, frac_map in results.items():
        ctx = int(ctx_str)
        j = x_index[ctx]
        for frac_label, score in frac_map.items():
            i = y_index[frac_label]
            grid[i, j] = float(score)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", origin="lower")

    ax.set_title(title)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Fraction")

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels(y_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Success rate")

    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, format(v, annotate_fmt), ha="center", va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved figure to {save_path}")


def shorten_model_name(args) -> None:
    # extract model name from full repo name or local model path, necessary for saving results later, e.g.
    # "/home/models/models/meta-llama-3.1-8B-instruct" -> meta-llama-3.1-8b-instruct
    # "meta-llama/Meta-Llama-3.1-8B-Instruct" -> meta-llama-3.1-8b-instruct
    model_name_or_path = pathlib.Path(args.model_name)
    args.model_name_short = model_name_or_path.name.lower()
