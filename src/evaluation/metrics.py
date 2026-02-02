import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset


def load_test_data(n_samples: int = 100) -> List[Dict]:
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    test_data = dataset["train"].select(range(n_samples))
    return [{"instruction": d["instruction"], "response": d["response"], "intent": d["intent"]} for d in test_data]


def evaluate_response_length(generated: str, expected: str) -> float:
    if len(expected) == 0:
        return 0.0
    ratio = len(generated) / len(expected)
    if 0.5 <= ratio <= 1.5:
        return 1.0
    elif 0.3 <= ratio <= 2.0:
        return 0.5
    return 0.0


def evaluate_keyword_overlap(generated: str, expected: str) -> float:
    gen_words = set(generated.lower().split())
    exp_words = set(expected.lower().split())
    if len(exp_words) == 0:
        return 0.0
    overlap = len(gen_words & exp_words)
    return min(overlap / len(exp_words), 1.0)


def evaluate_coherence(response: str) -> float:
    if len(response) < 10:
        return 0.0
    if response.count("<") > 2 or response.count(">") > 2:
        return 0.0
    if len(set(response)) < 10:
        return 0.0
    return 1.0


def run_evaluation(bot, test_data: List[Dict]) -> Dict:
    results = {
        "total": len(test_data),
        "coherent": 0,
        "length_score": 0.0,
        "keyword_score": 0.0,
        "by_intent": {},
    }
    
    for item in test_data:
        generated = bot.chat(item["instruction"])
        
        coherence = evaluate_coherence(generated)
        length = evaluate_response_length(generated, item["response"])
        keyword = evaluate_keyword_overlap(generated, item["response"])
        
        results["coherent"] += coherence
        results["length_score"] += length
        results["keyword_score"] += keyword
        
        intent = item["intent"]
        if intent not in results["by_intent"]:
            results["by_intent"][intent] = {"count": 0, "coherent": 0}
        results["by_intent"][intent]["count"] += 1
        results["by_intent"][intent]["coherent"] += coherence
    
    n = results["total"]
    results["coherence_rate"] = results["coherent"] / n
    results["avg_length_score"] = results["length_score"] / n
    results["avg_keyword_score"] = results["keyword_score"] / n
    
    return results
