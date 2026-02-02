from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path: Path = Path("models/customer-support-model")
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2

CONFIG = Config()
