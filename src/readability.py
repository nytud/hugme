import re
from statistics import mean
from textstat import textstat
from tqdm import tqdm

import helper
import config


def generate_similar_text(generation_pipeline, text, args):
    try:
        prompt = f"Folytasd a szöveget azonos stílusban!\n{text}"
        if "parameters" not in args:
            args.parameters = {"max_new_tokens": 256}
        if "temperature" not in args.parameters:
            args.parameters["temperature"] = 0.4
        actual_output = generation_pipeline(text_inputs=prompt,
                                            temperature=args.parameters.temperature,
                                            max_new_tokens=args.parameters.max_new_tokens,
                                            do_sample=True)[0]['generated_text']
    except RuntimeError as e:
        print(f"Error during text generation: {e}")
        actual_output = None
    return actual_output


def calculate_scores(text):
    col_score = int(textstat.coleman_liau_index(text))
    std_score = mean(map(float, re.findall(r'\d+(?:\.\d+)?', textstat.text_standard(text))))

    return col_score, std_score


def calculate_similarity_score(original_coleman, original_std, generated_coleman, generated_std):
    weight_coleman = 0.6
    weight_std = 0.4

    coleman_diff = abs(generated_coleman - original_coleman)
    std_diff = abs(generated_std - original_std)

    coleman_score = max(0, 100 - coleman_diff * 10)
    std_score = max(0, 100 - std_diff * 10)

    similarity_score = (coleman_score * weight_coleman) + (std_score * weight_std)
    return round(similarity_score, 2)


def compute_metric(args, generation_pipeline):
    dataset = helper.read_json(config.READABILITY_DATASET)
    similarity_scores = []
    for item in tqdm(dataset):
        text = item["query"]
        original_mean_coleman, original_mean_std = calculate_scores(text)
        generated_text = generate_similar_text(generation_pipeline, text, args)
        generated_mean_coleman, generated_mean_std = calculate_scores(generated_text)

        similarity_score = calculate_similarity_score(
            original_mean_coleman, original_mean_std,
            generated_mean_coleman, generated_mean_std
        )
        similarity_scores.append(similarity_score)

    return mean(similarity_scores)
