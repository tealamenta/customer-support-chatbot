import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)


@dataclass
class InferenceMetrics:
    timestamp: str
    question: str
    response_length: int
    latency_ms: float
    tokens_generated: int


@dataclass
class ModelMetrics:
    model_name: str
    adapter_path: str
    load_time_s: float
    total_inferences: int = 0
    avg_latency_ms: float = 0.0
    errors: int = 0


class MetricsTracker:
    def __init__(self, model_name: str = "customer-support-chatbot"):
        self.model_name = model_name
        self.inference_history: List[InferenceMetrics] = []
        self.model_metrics = ModelMetrics(
            model_name=model_name,
            adapter_path="",
            load_time_s=0.0
        )
        self.metrics_file = METRICS_DIR / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
    
    def log_model_load(self, adapter_path: str, load_time: float):
        self.model_metrics.adapter_path = adapter_path
        self.model_metrics.load_time_s = load_time
    
    def log_inference(self, question: str, response: str, latency: float, tokens: int = 0):
        metric = InferenceMetrics(
            timestamp=datetime.now().isoformat(),
            question=question[:100],
            response_length=len(response),
            latency_ms=latency * 1000,
            tokens_generated=tokens or len(response.split())
        )
        self.inference_history.append(metric)
        self.model_metrics.total_inferences += 1
        
        # Update average latency
        total_latency = sum(m.latency_ms for m in self.inference_history)
        self.model_metrics.avg_latency_ms = total_latency / len(self.inference_history)
    
    def log_error(self):
        self.model_metrics.errors += 1
    
    def get_summary(self) -> dict:
        return {
            "model": asdict(self.model_metrics),
            "recent_inferences": [asdict(m) for m in self.inference_history[-10:]],
            "stats": {
                "total_requests": self.model_metrics.total_inferences,
                "error_rate": self.model_metrics.errors / max(1, self.model_metrics.total_inferences),
                "avg_latency_ms": self.model_metrics.avg_latency_ms,
                "avg_response_length": sum(m.response_length for m in self.inference_history) / max(1, len(self.inference_history))
            }
        }
    
    def save(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


tracker = MetricsTracker()
