import argparse
import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_model():
    """Load the trained model and vectorizer"""
    # Load dataset to retrain the model
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")
    df['label'] = df['is_malicious'].astype(int)
    
    # Train the model
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return vectorizer, model

def test_jailbreak_prompt(prompt, vectorizer, model):
    """Test if a prompt is detected as a jailbreak attempt"""
    features = vectorizer.transform([prompt])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "prompt": prompt,
        "is_jailbreak": bool(prediction),
        "confidence": probability,
        "result": "JAILBREAK DETECTED" if prediction else "SAFE PROMPT",
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }


def print_result(result):
    print(f"\n--- Analysis Results ---")
    print(
        f"Prompt: {result['prompt'][:100]}..."
        if len(result['prompt']) > 100
        else f"Prompt: {result['prompt']}"
    )
    print(f"Result: {result['result']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Jailbreak Probability: {result['confidence']:.1%}")
    print("-" * 30 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt Injection Detector - Jailbreak Tester"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt text to analyze (use single quotes in PowerShell).",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a text file containing the prompt to analyze.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=== Prompt Injection Detector - Jailbreak Tester ===")
    print("This tool will test your prompts against the trained model.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Load the model
    vectorizer, model = load_model()
    print("Model loaded successfully!\n")

    # Single-run modes
    if args.file or args.prompt:
        if args.file:
            if not os.path.exists(args.file):
                print(f"File not found: {args.file}")
                sys.exit(1)
            with open(args.file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = args.prompt.strip()

        result = test_jailbreak_prompt(prompt, vectorizer, model)
        print_result(result)
        return

    # Interactive mode
    while True:
        try:
            prompt = input("Enter your jailbreak prompt to test: ").strip()
        except EOFError:
            # Gracefully handle Ctrl+Z in Windows
            print("\nGoodbye!")
            break

        if prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter a valid prompt.\n")
            continue

        result = test_jailbreak_prompt(prompt, vectorizer, model)
        print_result(result)

if __name__ == "__main__":
    main()