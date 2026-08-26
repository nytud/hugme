from typing import Any, Callable, Dict, Iterator, List, Optional

import os
import requests
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import openai
from tqdm import tqdm

import config
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
        helper.cleanup_model_name(args)
        print(f"Using generation results from path: {args.use_gen_results}")
        results = helper.read_json(args.use_gen_results)
        return results

    client = load_model(args)

    if args.parameters:
        parameters = helper.read_json(args.parameters)
    else:
        raise ValueError("No generation parameters provided.")
    print(f"Parameters: {parameters}")

    results = []
    with tqdm(total=len(dataset), desc="Generating responses...", unit="query") as pbar:
        for batch in batch_dataset(dataset, args.batch_size):

            batched_prompts = [ template.get_prompt(task_name, entry) for entry in batch ]

            outputs = generate_batch(client, batched_prompts, args.model_name, parameters)

            for entry, prompt, output in zip(batch, batched_prompts, outputs):

                formatted_result = format_fn(entry, prompt, output)

                formatted_result = helper.remove_reasoning_traces(formatted_result)

                results.append(formatted_result)

            pbar.update(len(batch))

            if args.save_results:
                print(f"Saving intermediate generation results for {task_name} at batch size {args.batch_size}.")
                save_results(results, task_name, args.model_name)

    if args.save_results:
        save_results(results, task_name, args.model_name)
    return results


def load_model(args):
    api_key = os.getenv("MODEL_API_KEY")

    client = openai.OpenAI(api_key=api_key, base_url=args.model_url)
    print(f"Initialized OpenAI client with base URL {args.model_url}.")

    # curl bolka:8001/v1/models
    response = requests.get(f"{args.model_url}/models")
    print(f"Available models: {response.json()}")

    return client


def generate(client: openai.OpenAI, messages: list, model_name: str, parameters: dict) -> ModelOutput:
    try:
        completion = client.chat.completions.create(model=model_name, messages=messages, **parameters)
    except openai.BadRequestError as e:
        print(f"OpenAI API request failed for: \n{messages}\n with parameters: {parameters}")
        print(f"OpenAI API request failed: {e}")
        inappropriate_content_message = "Input data may contain inappropriate content."
        if e.status_code == 400 and e.code == "data_inspection_failed" and inappropriate_content_message in e.message:
            return ModelOutput(inappropriate_content_message)
        raise e
    return ModelOutput(completion.choices[0].message.content, completion.usage.total_tokens)


def generate_batch(client: openai.OpenAI, messages: List, model_name: str, parameters: dict) -> List[ModelOutput]:
    # the OpenAI chat completions API takes one prompt per request, so a "batch" here means
    # firing the requests concurrently and letting vLLM's continuous batching do the actual batching
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


def save_results(results: List[Dict], task_name: str, model_name: str) -> None:
    if not results:
        print("No results to save.")
        return
    helper.save_json(
        results,
        config.RESULTS_DIR,
        f"{task_name}-{model_name.replace("/", "-").lower()}-generation-results.json"
    )
    print(f"Saved generation results to {config.RESULTS_DIR} directory.")
