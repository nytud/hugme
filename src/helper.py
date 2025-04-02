import os
import re
import json
import logging
import random
import pathlib
from collections import defaultdict
import torch


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
