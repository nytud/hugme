import time
import openai
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import config
import helper
import metrics
import spelling
import coherence
import benchmark
import readability


TASK_HANDLERS = {
    **{task: metrics.compute_metric for task in config.METRICS}, # deepeval-based metrics
    config.MMLU: benchmark.mmlu.benchmark,
    config.TRUTHFUL_QA: benchmark.truthfulqa.benchmark,
    config.SPELLING: spelling.compute_metric,
    config.TEXT_COHERENCE: coherence.compute_metric,
    config.PROMPT_ALIGNMENT: benchmark.prompt_alignment.compute_metric,
    config.READABILITY: readability.compute_metric,
    config.NIH: benchmark.nih.compute_metric
}

def evaluate(args) -> None:
    print("Evaluation started...")
    score_results = {}
    eval_start_time = time.time()

    for task_name in args.tasks:

        print(f"Loading {args.model_name} model and tokenizer...")
        generate = get_generation(task_name, args)
        print(f"Finished loading {args.model_name} model and tokenizer...")

        print(f"Started evaluation on {task_name}.")
        task_start_time = time.time()

        if task_name not in TASK_HANDLERS:
            raise ValueError(f"Task '{task_name}' is not supported. Valid tasks: {list(TASK_HANDLERS.keys())}")

        results = TASK_HANDLERS[task_name](task_name, args, generate)
        score_results[task_name] = results

        print(f"Task took {time.time() - task_start_time:.3f} seconds on {args.device}.")

    print(f"Evaluation took {time.time() - eval_start_time:.3f} seconds on {args.device}.")

    if args.save_results:
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        helper.save_json(score_results, config.RESULTS_DIR, f"hugme-results-{current_time}.json")


def get_generation(task_name, args):

    parameters = helper.read_json(args.parameters) if args.parameters else {}

    if task_name == config.NIH and args.provider:
        raise ValueError("The NIH task is not supported with OpenAI API. Use local model instead.")

    if args.model_name and not args.provider:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=config.HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", token=config.HF_TOKEN)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **parameters)

        if task_name == config.NIH: # needle in haystack requires special handling
            return pipe

    def generate(prompt, **parameters) -> str:

        if "alpaca_prompt" in parameters:
            alpaca_prompt = parameters.pop("alpaca_prompt")

        result: list = pipe(prompt, **parameters)
        generated_text: str = result[0]["generated_text"]

        if alpaca_prompt:
            output = generated_text.split("### Válasz:")[1]
        else:
            if isinstance(generated_text, str):
                output = generated_text[len(prompt):].strip()
            else:
                output = generated_text[-1]["content"]
        return output

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

    return generate if args.provider is None else generate_with_openai