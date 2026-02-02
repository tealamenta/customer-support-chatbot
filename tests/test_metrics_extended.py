import pytest
from unittest.mock import MagicMock, patch
from src.evaluation.metrics import (
    load_test_data,
    evaluate_response_length,
    evaluate_keyword_overlap,
    evaluate_coherence,
    run_evaluation,
)


class TestLoadTestData:
    @patch("src.evaluation.metrics.load_dataset")
    def test_load_test_data(self, mock_load):
        mock_dataset = {
            "train": MagicMock()
        }
        mock_dataset["train"].select.return_value = [
            {"instruction": "test1", "response": "resp1", "intent": "cancel"},
            {"instruction": "test2", "response": "resp2", "intent": "refund"},
        ]
        mock_load.return_value = mock_dataset
        
        result = load_test_data(n_samples=2)
        
        assert len(result) == 2
        assert result[0]["instruction"] == "test1"
        assert result[0]["intent"] == "cancel"


class TestEvaluateResponseLength:
    def test_ratio_below_threshold(self):
        # ratio = 2/50 = 0.04 < 0.3 -> 0.0
        score = evaluate_response_length("Hi", "This is a very long response with many many words here")
        assert score == 0.0
    
    def test_ratio_in_middle_range(self):
        # ratio needs to be between 0.3-0.5 or 1.5-2.0 for score 0.5
        # "Hello there test" (16 chars) vs "Hello there friend how are" (26 chars)
        # ratio = 16/26 = 0.61 -> in 0.5-1.5 range -> 1.0
        # Let's use: "Hello" (5) vs "Hello there friend" (18) -> 5/18 = 0.28 -> 0.0
        # Better: "Hello test" (10) vs "Hello there friend how are you" (30) -> 10/30 = 0.33 -> 0.5
        score = evaluate_response_length("Hello test", "Hello there friend how are you")
        assert score == 0.5
    
    def test_ratio_in_good_range(self):
        score = evaluate_response_length("Hello world test", "Hello world here")
        assert score == 1.0


class TestEvaluateKeywordOverlap:
    def test_case_insensitive(self):
        score = evaluate_keyword_overlap("HELLO WORLD", "hello world")
        assert score == 1.0
    
    def test_partial_match(self):
        score = evaluate_keyword_overlap("hello world test", "hello world")
        assert score == 1.0


class TestRunEvaluation:
    def test_run_evaluation_basic(self):
        mock_bot = MagicMock()
        mock_bot.chat.return_value = "I can help you with your order cancellation."
        
        test_data = [
            {"instruction": "Cancel my order", "response": "I can help you cancel your order.", "intent": "cancel_order"},
            {"instruction": "Track package", "response": "Let me check your package status.", "intent": "track_order"},
        ]
        
        results = run_evaluation(mock_bot, test_data)
        
        assert results["total"] == 2
        assert "coherence_rate" in results
        assert "avg_length_score" in results
        assert "avg_keyword_score" in results
        assert "by_intent" in results
    
    def test_run_evaluation_by_intent(self):
        mock_bot = MagicMock()
        mock_bot.chat.return_value = "I will help you with that request right away."
        
        test_data = [
            {"instruction": "Q1", "response": "R1", "intent": "cancel_order"},
            {"instruction": "Q2", "response": "R2", "intent": "cancel_order"},
            {"instruction": "Q3", "response": "R3", "intent": "refund"},
        ]
        
        results = run_evaluation(mock_bot, test_data)
        
        assert "cancel_order" in results["by_intent"]
        assert "refund" in results["by_intent"]
        assert results["by_intent"]["cancel_order"]["count"] == 2
        assert results["by_intent"]["refund"]["count"] == 1
    
    def test_run_evaluation_incoherent_response(self):
        mock_bot = MagicMock()
        mock_bot.chat.return_value = "<<<<<<<"
        
        test_data = [
            {"instruction": "Q1", "response": "R1", "intent": "test"},
        ]
        
        results = run_evaluation(mock_bot, test_data)
        
        assert results["coherent"] == 0
