from typing import Any, List

import logging
import random
import pathlib
import numpy
import torch

import config


def set_seeds(args) -> None:
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def set_device(args) -> None:

    use_cuda = args.use_cuda and torch.cuda.is_available()

    if use_cuda:
        device = torch.device(f'cuda{args.cuda_id}') if args.cuda_id else torch.device('cuda')
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


def free_memory(self, vars: List[Any], device: str) -> None:
    logging.warning(f"Freeing memory for {len(vars)} variables.")
    del vars
    if self.device.type == "cuda":
        torch.cuda.empty_cache()
    elif self.device.type == "mps":
        torch.mps.empty_cache()


def log_startup_diagnostics() -> None:
    print(f"Number of CPU Workers: {config.NUM_WORKERS}")
    print(f"Number of GPUs available: {config.NUM_GPUS}")
    print(f"Is CUDA available: {config.CUDA_AVAILABLE}")
    print(f"Selected device: {config.DEVICE}")
    print(f"Available devices: {config.DEVICES}")


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