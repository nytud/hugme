
import random
import argparse

import numpy
import torch



__doc__ = """
This script is designed as a starting point for fine-tuning and evaluating your models using HuGME.
It includes configurable options for model loading, training arguments and parameters and model saving functionalities.
"""


def cli() -> None:

    parser = argparse.ArgumentParser(description='cli tool')

    parser.add_argument('--model-name', type=str, metavar='S', help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default=None, metavar='S', help='tokenizer name or path')
    parser.add_argument('--task-names', type=str, default='all', metavar='S', help='task name(s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
    parser.add_argument('--optimizer', type=str, default="adam", metavar='N', help='optimizer name (default: adam)')
    parser.add_argument('--n-epochs', type=int, default=5, help='-')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=4, help='the number of CPUs workers')
    parser.add_argument('--use-cuda', type=lambda x: x.lower()=='true', default=True, metavar='S', help='gpu use')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='S', help='gpu id')
    parser.add_argument('--save-model', action='store_true', default=False, help='save model')

    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if use_cuda:
        device = torch.device(f'cuda{args.cuda_id}') if args.cuda_id else torch.device('cuda')
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")



if __name__ == '__main__':
    cli()