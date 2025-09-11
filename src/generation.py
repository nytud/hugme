from typing import Any, Callable, Dict, Iterator, List,Optional

import logging
import pathlib
from dataclasses import dataclass
from tqdm import tqdm
import openai
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        logging.info(f"Using generation results from path: {args.use_gen_results}")
        results = helper.read_json(args.use_gen_results)
        return results

    client = load_model(args, task_name)
    parameters = create_parameters(args, task_name)

    results = []
    for idx, entry in enumerate(tqdm(dataset, desc="Generating responses...", unit="query")):

        prompt = template.get_prompt(task_name, entry, args.use_alpaca_prompt)
        output = generate(
            prompt, client, parameters, model_name=args.model_name, provider=args.provider
        )
        formatted_result = format_fn(entry, prompt, output)
        results.append(formatted_result)

        if args.save_results and (idx + 1) % 10 == 0: # save intermediate results every 10 generations
            save_results(results, task_name, args.model_name, args.thinking)

    if args.save_results:
        save_results(results, task_name, args.model_name, args.thinking)
    return results


def load_model(args, task_name):
    if task_name == config.NIH and args.provider:
        raise ValueError("The NIH task is not supported with OpenAI API. Use local model instead.")
    if args.provider:
        return initialize_openai_client()
    return initialize_huggingface_model(args)


def initialize_huggingface_model(args):
    logging.info(f"Loading HuggingFace model and tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=config.HF_TOKEN, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", token=config.HF_TOKEN, trust_remote_code=True
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    logging.info(f"Finished loading {args.model_name} model and tokenizer from HuggingFace (or from locally).")

    # extract model name from full repo name or local model path, necessary for saving results later, e.g.
    # "/home/models/models/meta-llama-3.1-8B-instruct" -> meta-llama-3.1-8b-instruct
    # "meta-llama/Meta-Llama-3.1-8B-Instruct" -> meta-llama-3.1-8b-instruct
    model_name_or_path = pathlib.Path(args.model_name)
    args.model_name = model_name_or_path.name.lower()
    return pipe


def initialize_openai_client():
    client = openai.OpenAI(api_key=config.PROVIDER_API_KEY, base_url=config.PROVIDER_URL)
    logging.info(f"Initialized OpenAI client with base URL {config.PROVIDER_URL}.")
    return client


def create_parameters(args, task_name) -> dict:
    parameters = helper.read_json(args.parameters) if args.parameters else {}

    parameters["max_new_tokens"] = config.MAX_NEW_TOKENS.get( # limit max new tokens for some tasks
        task_name, parameters.get("max_new_tokens", config.DEFAULT_MAX_NEW_TOKENS)
    )
    if not args.provider: # huggingface's transformers lib is used
        return parameters

    # openai's lib specific parameter handling
    p = ("return_full_text", "do_sample", "repetition_penalty")
    for param in p:
        if param in parameters:
            parameters.pop(param)

    if parameters.get("max_new_tokens"):
        parameters["max_tokens"] = parameters.pop("max_new_tokens")

    if args.provider and args.thinking:  # openai's thinking mode
        parameters.update({"extra_body": {"enable_thinking": True, "result_format": "message"}})

    logging.info(f"Using generation parameters: {parameters}")
    return parameters


def generate(
        prompt: Any,
        client: Any,
        parameters: dict,
        model_name: Optional[str] = None,
        provider: Optional[str] = None
    ) -> ModelOutput:
    if provider:
        assert model_name is not None, "Model name must be provided when using OpenAI API."
        return generate_with_openai(prompt, client, model_name, parameters)
    return generate_with_huggingface(prompt, client, parameters)


def generate_with_openai(prompt, client: openai.OpenAI, model_name: str, parameters: dict) -> ModelOutput:
    try:
        completion = client.chat.completions.create(model=model_name, messages=prompt, **parameters)
    except openai.BadRequestError as e:
        logging.error(f"OpenAI API request failed for: \n{prompt}\n with parameters: {parameters}")
        logging.error(f"OpenAI API request failed: {e}")
        inappropriate_content_message = "Input data may contain inappropriate content."
        if e.status_code == 400 and e.code == "data_inspection_failed" and inappropriate_content_message in e.message:
            return ModelOutput(inappropriate_content_message)
        raise e
    return ModelOutput(completion.choices[0].message.content, completion.usage.total_tokens)


def generate_with_huggingface(prompts: List[str], client, parameters: dict) -> ModelOutput: # TODO check NiH task
    # TODO implement batch generation for openai package, then reimplement here
    try:
        results = client(prompts, **parameters)
        generated_texts = results[0]["generated_text"]
    except Exception as e:
        logging.error(f"HuggingFace model generation failed for prompts: {prompts} with parameters: {parameters}")
        logging.error(f"HuggingFace model generation failed: {e}")
        raise e
    return ModelOutput(generated_texts) # TODO implement total tokens for huggingface


def generate_batches(dataset: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def save_results(results: List[Dict], task_name: str, model_name: str, thinking: bool) -> None:
    if not results:
        logging.warning("No results to save.")
        return
    helper.save_json(
        results,
        config.RESULTS_DIR,
        f"{task_name}-{model_name}-{str(thinking).lower()}-generation-results.json"
    )
    logging.info(f"Saved generation results to {config.RESULTS_DIR} directory.")
