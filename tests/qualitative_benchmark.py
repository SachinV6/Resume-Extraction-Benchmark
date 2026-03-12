import os
import json
import pandas as pd
from datetime import datetime

from resume_extractor.extractor import extract_resume_info, pdf_to_text
from tests.scoring import evaluate_resume


DEFAULT_MODELS = [
    "azure:gpt-4.1",
    "qwen3:8b",
    "llama3:latest",
    "deepseek-r1:7b",
    "mistral:7b",
]

CHECKPOINT_PATH = "outputs/qualitative_benchmark/checkpoint.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(cp):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cp, f, indent=2)


def run_benchmark(pdf_path: str, models=None, output_dir="outputs/qualitative_benchmark"):
    if models is None:
        models = DEFAULT_MODELS

    os.makedirs(output_dir, exist_ok=True)

    resume_text = pdf_to_text(pdf_path)
    checkpoint = load_checkpoint()

    all_results = []

    for model in models:
        if model in checkpoint:
            print(f"[SKIP] {model} already completed (checkpoint)")
            all_results.append(checkpoint[model])
            continue

        print(f"[RUN] {model}")

        result = extract_resume_info(resume_text, model=model)
        data = result.get("data") or {}

        score = evaluate_resume(data)
        result["score"] = score

        checkpoint[model] = result
        save_checkpoint(checkpoint)

        all_results.append(result)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"results_{ts}.json")
    csv_path = os.path.join(output_dir, f"results_{ts}.csv")
    excel_path = os.path.join(output_dir, f"results_{ts}.xlsx")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.json_normalize(all_results)
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print("\nDONE.")
    print("JSON :", json_path)
    print("CSV  :", csv_path)
    print("XLSX :", excel_path)