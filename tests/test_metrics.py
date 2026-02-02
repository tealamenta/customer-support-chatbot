import pytest
from src.evaluation.metrics import (
    evaluate_response_length,
    evaluate_keyword_overlap,
    evaluate_coherence,
)


class TestResponseLength:
    def test_similar_length(self):
        score = evaluate_response_length("Hello world test", "Hello world here")
        assert score == 1.0
    
    def test_very_different_length(self):
        score = evaluate_response_length("Hi", "This is a very long response with many words")
        assert score < 1.0
    
    def test_empty_expected(self):
        score = evaluate_response_length("Hello", "")
        assert score == 0.0


class TestKeywordOverlap:
    def test_full_overlap(self):
        score = evaluate_keyword_overlap("hello world", "hello world")
        assert score == 1.0
    
    def test_partial_overlap(self):
        score = evaluate_keyword_overlap("hello there", "hello world")
        assert 0 < score < 1
    
    def test_no_overlap(self):
        score = evaluate_keyword_overlap("abc def", "xyz uvw")
        assert score == 0.0
    
    def test_empty_expected(self):
        score = evaluate_keyword_overlap("hello", "")
        assert score == 0.0


class TestCoherence:
    def test_coherent_response(self):
        response = "I can help you with your order cancellation request."
        assert evaluate_coherence(response) == 1.0
    
    def test_too_short(self):
        assert evaluate_coherence("Hi") == 0.0
    
    def test_garbage_with_brackets(self):
        response = "Test < < < < < < < < < < garbage"
        assert evaluate_coherence(response) == 0.0
    
    def test_repetitive_chars(self):
        response = "aaaaaaaaaa"
        assert evaluate_coherence(response) == 0.0
