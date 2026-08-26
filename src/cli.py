import argparse
from pathlib import Path

import eval as evaluate


def cli() -> None:

    parser = argparse.ArgumentParser(description='hugme cli tool')

    parser.add_argument('--tasks', type=str, nargs="+", default=[], help='task name(s)')
    parser.add_argument('--model-name', type=str, required=True, help='model name or path')
    parser.add_argument('--model-url',  type=str, required=True, help='model URL')
    parser.add_argument("--parameters", type=str, required=True, help="JSON config path for model params")

    parser.add_argument("--save-results", action="store_true", help='save results')
    parser.add_argument("--use-gen-results", type=Path, default=None, help='use generation results from path')
    parser.add_argument("--sample-size", type=float, default=1.0, help="sample size for evaluation")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for generation")

    parser.add_argument('--judge', type=str, default="gpt-4o", metavar='S', help='judge model name(s)')
    parser.add_argument("--provider", type=str, default=None, choices=['openai'])

    args = parser.parse_args()

    evaluate.evaluate(args)


if __name__ == '__main__':
    cli()
