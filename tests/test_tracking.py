import pytest
from src.evaluation.tracking import MetricsTracker, InferenceMetrics


class TestMetricsTracker:
    def test_init(self):
        tracker = MetricsTracker("test-model")
        assert tracker.model_name == "test-model"
        assert tracker.model_metrics.total_inferences == 0
    
    def test_log_model_load(self):
        tracker = MetricsTracker()
        tracker.log_model_load("/path/to/model", 5.5)
        assert tracker.model_metrics.adapter_path == "/path/to/model"
        assert tracker.model_metrics.load_time_s == 5.5
    
    def test_log_inference(self):
        tracker = MetricsTracker()
        tracker.log_inference("Hello", "Hi there, how can I help?", 0.5)
        assert tracker.model_metrics.total_inferences == 1
        assert len(tracker.inference_history) == 1
        assert tracker.inference_history[0].latency_ms == 500.0
    
    def test_log_error(self):
        tracker = MetricsTracker()
        tracker.log_error()
        assert tracker.model_metrics.errors == 1
    
    def test_get_summary(self):
        tracker = MetricsTracker()
        tracker.log_inference("Q1", "R1", 0.1)
        tracker.log_inference("Q2", "R2", 0.2)
        
        summary = tracker.get_summary()
        
        assert "model" in summary
        assert "stats" in summary
        assert summary["stats"]["total_requests"] == 2


class TestInferenceMetrics:
    def test_dataclass(self):
        metric = InferenceMetrics(
            timestamp="2024-01-01",
            question="Test",
            response_length=50,
            latency_ms=100.0,
            tokens_generated=10
        )
        assert metric.question == "Test"
        assert metric.latency_ms == 100.0
