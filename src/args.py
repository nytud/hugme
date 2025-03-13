from dataclasses import dataclass
from typing import List, Optional, Literal

@dataclass
class HuGMEArgs:
    model_name: str
    tokenizer_name: Optional[str] = None
    tasks: Optional[List[str]] = None
    judge: str = "gpt-3.5-turbo-1106"
    use_cuda: bool = True
    cuda_ids: Optional[List[int]] = None
    seed: int = 42
    parameters: Optional[str] = None
    save_results: bool = True
    batch_size: int = 1
    model_type: Literal["local", "api"] = "local"
    api_provider: Literal["openai", "anthropic", "cohere", "custom"] = "openai"
    api_config: Optional[str] = None
    generated_file: Optional[str] = None
    device : Optional[str] = None
    
    def __post_init__(self):
        if self.tasks == []:
            self.tasks = ["nih", "bias", "toxicity", "faithfulness", 
                          "hallucination", "summarization", "answer-relevancy"]
        if self.cuda_ids == []:
            self.cuda_ids = [0]    