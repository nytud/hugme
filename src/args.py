from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class ModelInfo:
    model_name: str
    tokenizer_name: Optional[str] = None
    tasks: Optional[List[str]] = None
    judge: str = "gpt-3.5-turbo-1106"
    use_cuda: bool = True

@dataclass
class SeedBatch(ModelInfo):
    seed: int = 42
    parameters: Optional[str] = None
    save_results: bool = True
    batch_size: int = 1
    model_type: Literal["local", "api"] = "local"

@dataclass
class NearFinal(SeedBatch):
    cuda_ids: Optional[List[int]] = None
    api_provider: Literal["openai", "anthropic", "cohere", "custom"] = "custom"
    api_config: Optional[str] = None
    generated_file: Optional[str] = None
    device : Optional[str] = None

@dataclass
class HuGMEArgs(NearFinal):
    def __post_init__(self):
        if self.tasks == []:
            self.tasks = ["nih", "bias", "toxicity", "faithfulness",
                          "hallucination", "summarization", "answer-relevancy"]
        if self.cuda_ids == []:
            self.cuda_ids = [0]
