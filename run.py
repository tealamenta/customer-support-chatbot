#!/usr/bin/env python3
"""
Customer Support Chatbot

Usage:
    python run.py demo    # Interactive demo
    python run.py eval    # Run evaluation
    python run.py api     # Launch API server
"""

import argparse


def demo():
    from src.model.inference import load_model
    
    print("=" * 60)
    print("CUSTOMER SUPPORT CHATBOT - DEMO")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    bot = load_model()
    
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        
        response = bot.chat(question)
        print(f"\nAssistant: {response}")
    
    print("\nBye!")


def evaluate():
    from src.model.inference import load_model
    from src.evaluation.metrics import load_test_data, run_evaluation
    
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    bot = load_model()
    test_data = load_test_data(n_samples=50)
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    results = run_evaluation(bot, test_data)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Coherence Rate: {results['coherence_rate']:.1%}")
    print(f"Avg Length Score: {results['avg_length_score']:.2f}")
    print(f"Avg Keyword Score: {results['avg_keyword_score']:.2f}")
    
    print("\nBy Intent:")
    for intent, data in sorted(results["by_intent"].items())[:10]:
        rate = data["coherent"] / data["count"] if data["count"] > 0 else 0
        print(f"  {intent}: {rate:.0%} ({data['count']} samples)")


def api():
    import uvicorn
    print("=" * 60)
    print("CUSTOMER SUPPORT API")
    print("=" * 60)
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Support Chatbot")
    parser.add_argument("command", choices=["demo", "eval", "api"])
    args = parser.parse_args()
    
    if args.command == "demo":
        demo()
    elif args.command == "eval":
        evaluate()
    elif args.command == "api":
        api()
