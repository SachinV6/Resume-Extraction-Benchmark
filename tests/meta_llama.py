import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_extractor.extractor import pdf_to_text
from tests.scoring import evaluate_resume, normalize_meta_llama
from tests.metrics import capture_usage, compute_metrics
from langchain_openai import AzureChatOpenAI


# -------------------------
# Load .env.llama
# -------------------------
load_dotenv(".env.llama")


# -------------------------
# Meta-Llama Azure Loader
# -------------------------
def load_llama():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLAMA_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_LLAMA_ENDPOINT"),
        api_key=os.getenv("AZURE_LLAMA_API_KEY"),
        api_version=os.getenv("AZURE_LLAMA_VERSION"),
        temperature=0,
        max_tokens=3000,  # Increased to avoid truncation
    )


# -------------------------
# Safe JSON Loader
# -------------------------
def safe_json_load(content):
    try:
        return json.loads(content)
    except Exception:
        content = content.strip()

        # Auto-close brackets if truncated
        if content.count("{") > content.count("}"):
            content += "}" * (content.count("{") - content.count("}"))

        if content.count("[") > content.count("]"):
            content += "]" * (content.count("[") - content.count("]"))

        try:
            return json.loads(content)
        except Exception:
            return {"error": "Invalid JSON from model", "raw_output": content}


# -------------------------
# Extraction Function
# -------------------------
def extract_with_llama(resume_text, llm):

    system_prompt = """
You are a resume information extraction system.
Extract structured information from the resume and return VALID JSON only.
Do not include explanations.
Do not include markdown. Do not include comments.
CRITICAL RULES: 
1. The "skills" array MUST NOT be empty. 
2. Extract ALL technologies mentioned anywhere in the resume. 
3. If a dedicated skills section does not exist, infer skills from: - Experience responsibilities - Project descriptions - Technologies used 
4. Skills must be atomic technologies (e.g., "Python", "Docker", "FastAPI"). 
5. Do NOT invent skills that are not mentioned. 
6. Do NOT omit skills that are clearly mentioned.
"""

    user_prompt = f"""
Resume Text:
{resume_text}
"""

    full_prompt_text = system_prompt + user_prompt

    # -------- Metrics Start --------
    usage_before = capture_usage()
    start_time = time.time()

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    end_time = time.time()
    usage_after = capture_usage()
    # -------- Metrics End --------

    print("\n[DEBUG] RAW MODEL OUTPUT:\n")
    print(response.content)

    metrics = compute_metrics(
        prompt=full_prompt_text,
        output=response.content,
        start_time=start_time,
        end_time=end_time,
        usage_before=usage_before,
        usage_after=usage_after,
    )

    parsed_json = safe_json_load(response.content)

    return parsed_json, metrics


# -------------------------
# Run Test
# -------------------------
def test_llama_resume(
    pdf_path="/home/sachv/projects/resume-extractor/tests/data/final_test.pdf",
    output_dir="outputs/test_llama_azure"
):

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Extracting PDF text...")
    resume_text = pdf_to_text(pdf_path)

    print("[INFO] Loading Meta-Llama Azure...")
    llm = load_llama()

    print("[INFO] Running extraction...")
    data, metrics = extract_with_llama(resume_text, llm)

    result = {
        "model_used": "meta-llama-azure",
        "data": data,
        "metrics": metrics
    }

    # -------- Scoring --------
    if isinstance(data, dict) and "error" not in data:
        normalized_data = normalize_meta_llama(data)
        score = evaluate_resume(normalized_data)
        result["score"] = score
    else:
        result["score"] = {"error": "Scoring skipped due to invalid extraction"}

    # -------- Save Outputs --------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"llama_result_{ts}.json")
    csv_path = os.path.join(output_dir, f"llama_result_{ts}.csv")
    excel_path = os.path.join(output_dir, f"llama_result_{ts}.xlsx")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    df = pd.json_normalize(result)
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print("\nDONE.")
    print("JSON :", json_path)
    print("CSV  :", csv_path)
    print("XLSX :", excel_path)


# -------------------------
# Run Directly
# -------------------------
if __name__ == "__main__":
    test_llama_resume()