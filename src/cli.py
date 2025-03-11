import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate_hugme as evaluate
import helper

__doc__ = """
This script is designed as a starting point for evaluating your models using HuGME.
"""


def cli() -> None:

    parser = argparse.ArgumentParser(description='hugme cli tool')

    parser.add_argument('--model-name', type=str, metavar='S', help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default=None, metavar='S', help='tokenizer name or path')
    parser.add_argument('--tasks', type=str, nargs="+", default=["nih","bias","toxicity","faithfulness","hallucination","summarization",
    "answer-relevancy"], help='task name(s)')
    parser.add_argument('--judge', type=str, default="gpt-3.5-turbo-1106", metavar='S', help='judge model name(s)')
    parser.add_argument('--use-cuda', type=lambda x: x.lower()=='true', default=True, metavar='S', help='gpu use')
    parser.add_argument('--cuda-ids', type=list, default=[0], metavar='S', help='gpu ids to use')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
    parser.add_argument("--parameters", type=str, default=None, help="path to JSON config file for model params")
    parser.add_argument("--save-results", type=lambda x: x.lower()=='true', default=True, help='save results')
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for generation")
    parser.add_argument('--model-type', type=str, default='local', choices=['local', 'api'],
                        help='Model type: local (default) or api for external APIs')
    parser.add_argument('--api-provider', type=str, default='openai',
                        choices=['openai', 'anthropic', 'cohere', 'custom'],
                        help='API provider name when using model-type=api')
    parser.add_argument('--api-config', type=str, default=None,
                        help='Path to JSON config file with additional API parameters')
    parser.add_argument('--generated-file', type=str, default=None, metavar='S', help='File path for already generated answers')
    args = parser.parse_args()

    helper.set_seeds(args)
    helper.set_device(args)

    evaluate.evaluate(args)


@dataclass
class HuGMEArgs:
    model_name: str
    tokenizer_name: Optional[str] = None
    tasks: List[str] = field(default_factory=list)
    judge: str = "gpt-3.5-turbo-1106"
    use_cuda: bool = True
    cuda_ids: List[int] = field(default_factory=lambda: [0])
    seed: int = 42
    parameters: Optional[str] = None
    save_results: bool = True

    model_type: str = "local"
    api_provider: Optional[str] = None
    api_key: Optional[str] = None
    api_config: Optional[str] = None

    device: Union[str, List[str]] = "cpu"

    model_params: Dict[str, Any] = field(default_factory=dict)
    api_params: Dict[str, Any] = field(default_factory=dict)


if __name__ == '__main__':
    cli()
