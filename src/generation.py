import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import openai
import requests
from tqdm import tqdm

import helper
import template


@dataclass
class ModelOutput:
    text: str
    total_tokens: Optional[int] = None


def generate_results(
        args,
        task_name: str,
        dataset: List,
        format_fn: Callable[[Dict, Any, ModelOutput], Dict]
    ) -> List[Dict[str, Any]]:

    if args.use_gen_results:
        print(f"Using generation results from path: {args.use_gen_results}")
        results = helper.read_json(args.use_gen_results)
        return results

    client = load_model(args)
    parameters = load_parameters(args)

    results = []
    with tqdm(total=len(dataset), desc="Generating responses...", unit="query") as pbar:
        for batch in batch_dataset(dataset, args.batch_size):
            batched_prompts = [template.get_prompt(task_name, entry) for entry in batch]
            outputs = generate_batch(client, batched_prompts, args.model_name, parameters)
            results.extend(format_batch(batch, batched_prompts, outputs, format_fn))
            pbar.update(len(batch))
            if args.save_results:
                helper.save_results(results, task_name, args.model_name)
    print(f"Finished saving generation results for {task_name} with {args.model_name}.")
    return results


def format_batch(batch, prompts, outputs, format_fn) -> List[Dict[str, Any]]:
    formatted = []
    for entry, prompt, output in zip(batch, prompts, outputs):
        result = format_fn(entry, prompt, output)
        formatted.append(helper.remove_reasoning_traces(result))
    return formatted


def load_model(args):
    if args.model_url == "not_given":
        base_url = openai.not_given
    else:
        base_url = args.model_url
        response = requests.get(f"{args.model_url}/models", timeout=30)
        print(f"Available models: {response.json()}")
    client = openai.OpenAI(api_key=os.getenv("MODEL_API_KEY"), base_url=base_url)
    print(f"Initialized OpenAI client with base URL {args.model_url}.")
    return client


def load_parameters(args) -> dict:
    if args.parameters:
        parameters = helper.read_json(args.parameters)
    else:
        raise ValueError("No generation parameters provided.")
    print(f"Parameters: {parameters}")
    return parameters


def tokenize(text: str, args) -> List[int]:
    return _post_to_server(args, "tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]


def detokenize(tokens: List[int], args) -> str:
    return _post_to_server(args, "detokenize", {"tokens": tokens})["prompt"]


def _post_to_server(args, endpoint: str, payload: dict) -> dict:
    api_key = os.getenv("MODEL_API_KEY")
    response = requests.post(
        f"{args.model_url.rstrip('/').removesuffix('/v1')}/{endpoint}",
        json={"model": args.model_name, **payload},
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def generate(client: openai.OpenAI, messages: list, model_name: str, parameters: dict) -> ModelOutput:
    try:
        completion = client.chat.completions.create(model=model_name, messages=messages, **parameters)
    except openai.BadRequestError as e:
        print(f"OpenAI API request failed for: \n{messages}\n with parameters: {parameters}")
        inappropriate_content_message = "Input data may contain inappropriate content."
        if e.status_code == 400 and e.code == "data_inspection_failed" and inappropriate_content_message in e.message:
            return ModelOutput(inappropriate_content_message)
        raise e
    return ModelOutput(completion.choices[0].message.content, completion.usage.total_tokens)


def generate_batch(client: openai.OpenAI, messages: List, model_name: str, parameters: dict) -> List[ModelOutput]:
    # the OpenAI chat completions API takes one prompt per request, so a "batch" here means
    # firing the requests concurrently and letting vLLM's continuous batching do the actual work
    with ThreadPoolExecutor(max_workers=len(messages)) as executor:
        return list(
            executor.map(
                lambda message: generate(client, message, model_name, parameters),
                messages
            )
        )


def batch_dataset(dataset: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]
