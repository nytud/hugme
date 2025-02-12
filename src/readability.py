import re
import os
import sys
import pandas as pd
from statistics import mean
from textstat import textstat
from tqdm import tqdm


class TextAnalyzer:
    def __init__(self, model="Llumix32", tokenizer="", maxtoken=256):
        textstat.set_lang("hu")
        print(f"Start testing")

    def set_environment_variables(self, api_key):
        # Set environment variables for DeepEval
        os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./results"
        os.environ['OPENAI_API_KEY'] = api_key
    
    def initialize_base_llm(self):
        self.base_llm = Llumix32
        #self.base_llm = GPT3Client(model="GPT3.5", tokenizer="")        
        #self.base_llm = GPT4ominiClient(model="GPT4o-mini", tokenizer="")
        #self.base_llm = GPTrio(model="GPtrio", tokenizer="")
        #self.base_llm = GPT4Client(model="GPT4-turbo", tokenizer="")
        #self.base_llm = GemmaGenerator()
        #self.base_llm = ParancsPuli
        self.temp = "0.4"
        self.maxtoken = "256"	

    def import_dataset(self):
        df = pd.read_csv("./LL_eval_w_deepeval/eval_datasets/readability.csv", on_bad_lines="warn", delimiter=",", index_col=False, header=0)         
        if "query" not in df.columns:
                print(f"Hiba: A CSV fájl nem tartalmaz 'query' oszlopot.")
                
        return df    
    
    def generate_text(self, df):
        """Generates text for each theme using the given LLM model, with a progress bar."""
        generated_texts = []
        
        for ind in tqdm(df.index, desc="Szöveg generálása", unit="válasz"):
            try:
                text = df["query"][ind]
                actual_output = self.base_llm.generate(prompt=f"Folytasd a szöveget azonos stílusban!\n{text}")
            except Exception as e:
                print(f"Hiba történt a szöveg generálásakor: {e}")
                actual_output = None
            generated_texts.append(actual_output)
            with open("./results/generated_readability_texts.txt", "w", encoding="utf-8") as f:
                for item in generated_texts:
                    f.write(str(item) + "\n")
        print("Generált szöveg mentve: ./results/generated_readability_texts.txt")
        return generated_texts
    
    @staticmethod
    def calculate_scores(texts):
        """Calculates Coleman-Liau and Text Standard scores for a list of texts."""
        coleman_scores = []
        std_scores = []
        
        for text in texts:
            if text:  # Ha a szöveg nem üres
                col_score = int(textstat.coleman_liau_index(text))
                std_score = mean(map(float, re.findall(r'\d+(?:\.\d+)?', textstat.text_standard(text))))
                coleman_scores.append(col_score)
                std_scores.append(std_score)
        
        mean_coleman = mean(coleman_scores) if coleman_scores else 0
        mean_std = mean(std_scores) if std_scores else 0
        return mean_coleman, mean_std

    @staticmethod
    def calculate_similarity_score(original_coleman, original_std, generated_coleman, generated_std):
        """Calculates a similarity score (0-100) based on the differences between original and generated scores."""
        # Súlyok a két metrika között
        weight_coleman = 0.6
        weight_std = 0.4
        
        # Normalizált abszolút különbségek (0-100 skálán)
        coleman_diff = abs(generated_coleman - original_coleman)
        std_diff = abs(generated_std - original_std)
        
        # A normalizáció alapja: a különbség maximalizált hatása 10 pont (pl. ha a különbség >10, akkor 0 pont jár arra a metrikára)
        coleman_score = max(0, 100 - coleman_diff * 10)
        std_score = max(0, 100 - std_diff * 10)
        
        # Súlyozott átlag számítása
        similarity_score = (coleman_score * weight_coleman) + (std_score * weight_std)
        return round(similarity_score, 2)


    def run_evaluation(self, df, output_file):
        """Processes df, calculates statistics, and saves summary."""
        
        # Inicializáljuk a summary_data listát
        summary_data = []
        
        # Eredeti szövegek elemzése
        original_texts = df["query"].tolist()
        original_mean_coleman, original_mean_std = self.calculate_scores(original_texts)
        
        # Generált szövegek elemzése
        generated_texts = self.generate_text(df)
        generated_mean_coleman, generated_mean_std = self.calculate_scores(generated_texts)
        
        # Különbségek kiszámítása
        abs_diff_coleman = abs(generated_mean_coleman - original_mean_coleman)
        abs_diff_std = abs(generated_mean_std - original_mean_std)
        
        percent_diff_coleman = (
            (abs_diff_coleman / original_mean_coleman * 100) if original_mean_coleman != 0 else 0
        )
        percent_diff_std = (
            (abs_diff_std / original_mean_std * 100) if original_mean_std != 0 else 0
        )
        
        # Hasonlósági pontszám számítása
        similarity_score = self.calculate_similarity_score(
            original_mean_coleman, original_mean_std,
            generated_mean_coleman, generated_mean_std
        )
    
        # Adatok hozzáadása az összefoglalóhoz
        summary_data.append({
            "file_name": output_file,  # Ha nincs külön file_name változó
            "original_mean_coleman": original_mean_coleman,
            "original_mean_std": original_mean_std,
            "generated_mean_coleman": generated_mean_coleman,
            "generated_mean_std": generated_mean_std,
            "abs_diff_coleman": abs_diff_coleman,
            "abs_diff_std": abs_diff_std,
            "percent_diff_coleman": percent_diff_coleman,
            "percent_diff_std": percent_diff_std,
            "similarity_score": similarity_score,
        })
        
        # Összefoglaló mentése
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_file, index=False)
        print(f"\nÖsszefoglaló mentve ide: {output_file}")
    
        # Similarity score átlagának kiírása
        if summary_data:
            mean_similarity_score = mean([item["similarity_score"] for item in summary_data])
            print(f"\nA similarity score átlagértéke: {mean_similarity_score:.2f}")



def calculate_metric(generation_pipeline):
    pass