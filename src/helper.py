import gc
import json
import random
import pathlib
import torch

import config


def set_seeds(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def set_device(args) -> None:

    use_cuda = args.use_cuda and torch.cuda.is_available()

    if use_cuda:
        device = torch.device(f'cuda:{args.cuda_id}') if args.cuda_id else torch.device('cuda')
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    args.device = device


def read_file(file_path, readlines: bool = False):
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r") as file:
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
            json.dump(data, file, ensure_ascii=False, indent=4)
    except OSError as e:
        raise OSError(f"Could not save file: {file_path}") from e


def free_memory(device: torch.device) -> None:
    print("Freeing memory.")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_memory_alloc(device: str, divisor: int = 1024 ** 3) -> float:
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    f_memory, t_memory = free_memory / divisor, total_memory / divisor
    memory_allocated = t_memory - f_memory
    print(
        f"Memory statistics for {device} device: "
        f"allocated: {memory_allocated:.2f}/ {t_memory:.2f} GB, "
        f"free: {f_memory:.2f}/ {t_memory:.2f} GB"
    )
    return memory_allocated


def get_free_device() -> str: # ~ device with lowest memory usage

    devices_memory_allocated = [(device, get_memory_alloc(device)) for device in config.DEVICES]

    devices_memory_allocated = sorted(devices_memory_allocated, key=lambda t: t[1])

    return devices_memory_allocated[0][0]


def get_model_prompt(model_id: str, query: str, prompt: str = "Válaszolj a kérdésre!"):

    model_prompts = {
        "google/gemma-2-2b-it": [
            {"role": "user", "content": query}
        ],
        "meta-llama/Meta-Llama-3.1-8B-Instruct":  [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    }
    return model_prompts.get(model_id, query)

