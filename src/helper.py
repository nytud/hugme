import random
import numpy
import torch


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
