import time
import logging
from datetime import datetime
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import config
import helper
import metrics
import spelling
import benchmark
import readability


TASK_HANDLERS = {
    **{task: metrics.compute_metric for task in config.METRICS}, # deepeval-based metrics
    config.MMLU: benchmark.mmlu.benchmark,
    config.TRUTHFUL_QA: benchmark.truthfulqa.benchmark,
    config.SPELLING: spelling.compute_metric,
    config.PROMPT_ALIGNMENT: benchmark.prompt_alignment.compute_metric,
    config.READABILITY: readability.compute_metric,
    config.NIH: benchmark.nih.compute_metric
}

def evaluate(args) -> None:
    logging.info("Evaluation started...")
    score_results = {}
    eval_start_time = time.time()

    for task_name in args.tasks:

        generate = get_generation(task_name, args) if not args.use_gen_results else None

        logging.info(f"Started evaluation on {task_name}.")
        task_start_time = time.time()

        if task_name not in TASK_HANDLERS:
            raise ValueError(f"Task '{task_name}' is not supported. Valid tasks: {list(TASK_HANDLERS.keys())}")

        results = TASK_HANDLERS[task_name](task_name, args, generate)
        score_results[task_name] = results

        logging.info(f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    logging.info(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}.")

    if args.save_results:
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        helper.save_json(score_results, config.RESULTS_DIR, f"hugme-results-{current_time}.json")


def get_generation(task_name, args):

    logging.info(f"Loading {args.model_name} model and tokenizer...")

    args.parameters = parameters = helper.read_json(args.parameters) if args.parameters else {}

    if task_name == config.NIH and args.provider:
        raise ValueError("The NIH task is not supported with OpenAI API. Use local model instead.")

    if args.model_name and not args.provider:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=config.HF_TOKEN, trust_remote_code=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", token=config.HF_TOKEN, trust_remote_code=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        if task_name == config.NIH: # needle in haystack requires special handling
            return pipe

    def generate(prompts, **kwargs):

        alpaca_prompt = None
        if "alpaca_prompt" in kwargs:
            alpaca_prompt = kwargs.pop("alpaca_prompt")

        parameters.update(kwargs)

        logging.debug(f"Generating with parameters: {parameters}")

        results = pipe(prompts, **parameters)

        if parameters.get("batch_size", 1) > 1:
            generated_texts = [r[0]["generated_text"] for r in results]
        else:
            generated_texts = results[0]["generated_text"]

        logging.debug(f"Prompt: {prompts}")
        logging.debug(f"Parameters: {parameters}")
        logging.debug(f"Result: {results}")
        logging.debug(f"Generated text: {generated_texts}")

        # if isinstance(generated_text, str):
        #     output = generated_text
        # else:
        #     output = generated_text[-1]["content"]
        # return output
        return generated_texts

    def generate_with_openai(prompt, **parameters) -> str:

        if parameters.get("max_new_tokens"):
            parameters["max_tokens"] = parameters.pop("max_new_tokens")
        if parameters.get("alpaca_prompt"):
            parameters.pop("alpaca_prompt")
        if parameters.get("do_sample"):
            parameters.pop("do_sample")

        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=prompt,
            **parameters
        )
        return completion.choices[0].message.content

    logging.info(f"Finished loading {args.model_name} model and tokenizer...")

    return generate if args.provider is None else generate_with_openai