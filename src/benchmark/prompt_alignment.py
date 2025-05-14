import random
import config
import helper

from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric


MAX_NEW_TOKENS = 256


def compute_metric(task_name, args, generate):
    
    # load Dataset  
    dataset = helper.read_json(config.PROMPT_ALIGNMENT_DATASET)

    sample_size = max(1, int(args.sample_size * len(dataset)))
    dataset = random.sample(dataset, sample_size)

    # generate
    results = []
    for entry in dataset:
        prompt_instructions = entry['prompt_instructions']
        query = entry['query']

        
        actual_output = generate(
            query,
            max_new_tokens=MAX_NEW_TOKENS,
            alpaca_prompt=args.use_alpaca_prompt
        )

        
        results.append({
            "query": str(query),
            "actual_output": str(actual_output),
            "prompt_instructions": prompt_instructions
        })

    # Mérés 
    threshold = 0.5
    total_score = 0.0
    passed = 0

    for item in results:
        test_case = LLMTestCase(
            input=item["query"],
            actual_output=item["actual_output"]
        )

        metric = PromptAlignmentMetric(
            prompt_instructions=item["prompt_instructions"],  #a prompt instruction lista legyen!
            model="gpt-4o",
            include_reason=True
        )
        metric.measure(test_case)

        total_score += metric.score
        if metric.score >= threshold:
            passed += 1

    # Eredmények kiértékelése
    avg_score = total_score / len(results)
    success_rate = passed / len(results)

    print(f"Átlagos pontszám: {avg_score:.2f}")
    print(f"Sikerráta ({threshold}+): {success_rate:.2%}")


    if args.save_results:
        helper.save_json(results, config.RESULTS_DIR, f"{config.PROMPT_ALIGNMENT}-eval-results.json")
    return results
