import os
import logging
import argparse
from pathlib import Path

import helper
import eval as evaluate


__doc__ = """
This script is designed as a starting point for evaluating your models using HuGME.
"""

log_level = int(os.getenv('LOG_LEVEL', 5))

log_levels = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.DEBUG,
    3: logging.INFO,
    4: logging.WARNING,
    5: logging.NOTSET
}

logging.basicConfig(level=log_levels.get(log_level, logging.INFO), format="%(levelname)s: %(message)s")


def cli() -> None:

    parser = argparse.ArgumentParser(description='hugme cli tool')

    parser.add_argument('--model-name', type=str, metavar='S', help='model name or path')
    parser.add_argument('--tasks', type=str, nargs="+", default=[], help='task name(s)')
    parser.add_argument('--judge', type=str, default="gpt-4o", metavar='S', help='judge model name(s)')
    parser.add_argument('--use-cuda', type=lambda x: x.lower()=='true', default=True, metavar='S', help='gpu use')
    parser.add_argument('--cuda-ids', type=list, default=[0], metavar='S', help='gpu ids to use')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
    parser.add_argument("--parameters", type=str, default=None, help="path to JSON config file for model params")
    parser.add_argument("--save-results", type=lambda x: x.lower()=='true', default=True, help='save results')
    parser.add_argument("--use-gen-results", type=Path, default=None, help='use generation results from path')
    parser.add_argument("--batch-size", type=int, default=2, help="batch size for generation")
    parser.add_argument("--provider", type=str, default=None, choices=['openai'])
    parser.add_argument("--use-alpaca-prompt", type=lambda x: x.lower()=='true', default=False)
    parser.add_argument("--sample-size", type=float, default=1.0, help="sample size for evaluation")

    args = parser.parse_args()

    helper.set_seeds(args)
    helper.set_device(args)

    evaluate.evaluate(args)


if __name__ == '__main__':
    cli()
