
import os, json
from datasets import load_dataset

def load_data(dataset_name):
    """Load dataset from local file or HuggingFace."""
    if dataset_name == "math_mini":
        local_path = os.path.join("data", "math_mini.jsonl")
        if os.path.exists(local_path):
            print(f"[INFO] Loading local dataset: {local_path}")
            with open(local_path, "r") as f:
                return [json.loads(line) for line in f]
    print(f"[INFO] Loading dataset from HuggingFace: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    return [{"problem": d.get("problem", ""), "solution": d.get("solution", "")} for d in dataset]
