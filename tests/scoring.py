from typing import Dict, List, Set
import unicodedata
from rapidfuzz import fuzz


# =========================
# GROUND TRUTH
# =========================

GROUND_TRUTH = {
    "name": "vikram narayanan",
    "education_keywords": {
        "m.tech",
        "artificial intelligence",
        "b.e",
        "computer science"
    },
    "companies": {
        "quantaxis analytics",
        "datanest labs",
        "infocrest technologies"
    },
    "skills": {
        "python",
        "fastapi",
        "docker",
        "kubernetes",
        "postgresql",
        "redis",
        "microservices",
        "spark",
        "nlp",
        "django",
        "embeddings",
        "transformer",
        "react",
        "azure openai",
        "llama",
        "sql"
    }
}


# =========================
# SYNONYM MAPPING
# =========================

SKILL_SYNONYMS = {
    "postgres": "postgresql",
    "postgre": "postgresql",
    "postgres db": "postgresql",
    "azure openai service": "azure openai",
    "llama2": "llama",
    "llama 2": "llama",
    "reactjs": "react",
    "transformers": "transformer",
}


# =========================
# NORMALIZATION
# =========================

def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower().strip()


def canonicalize_skill(skill: str) -> str:
    skill = normalize(skill)
    return SKILL_SYNONYMS.get(skill, skill)


def safe_str(value):
    if value is None:
        return ""
    return str(value)


# =========================
# TEXT COLLECTION
# =========================

def collect_all_text(model_output: Dict) -> str:
    texts = []

    texts.append(safe_str(model_output.get("full_name")))

    for edu in model_output.get("education", []) or []:
        texts.extend([
            safe_str(edu.get("degree")),
            safe_str(edu.get("institution")),
            safe_str(edu.get("year")),
        ])

    for exp in model_output.get("experience", []) or []:
        texts.extend([
            safe_str(exp.get("company")),
            safe_str(exp.get("role")),
            safe_str(exp.get("duration")),
        ])

        for r in exp.get("responsibilities", []) or []:
            texts.append(safe_str(r))

    for proj in model_output.get("projects", []) or []:
        texts.extend([
            safe_str(proj.get("name")),
            safe_str(proj.get("description")),
        ])

        for tech in proj.get("technologies", []) or []:
            texts.append(safe_str(tech))

    for skill in model_output.get("skills", []) or []:
        texts.append(safe_str(skill))

    return normalize(" ".join(texts))


# =========================
# MATCHING UTILITIES
# =========================

def fuzzy_match(value: str, candidates: Set[str], threshold: int = 85) -> bool:
    for c in candidates:
        if fuzz.ratio(value, c) >= threshold:
            return True
    return False


def evaluate_skill_predictions(predicted: Set[str], ground_truth: Set[str]):
    true_positives = set()
    false_positives = set()

    for skill in predicted:
        if skill in ground_truth or fuzzy_match(skill, ground_truth):
            true_positives.add(skill)
        else:
            false_positives.add(skill)

    false_negatives = {
        gt for gt in ground_truth
        if not any(
            gt == p or fuzz.ratio(gt, p) >= 85
            for p in predicted
        )
    }

    return true_positives, false_positives, false_negatives


def compute_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def normalize_meta_llama(data: dict) -> dict:
    """
    Normalizes Meta Llama Azure output to match expected scoring schema.
    """

    if not isinstance(data, dict):
        return data

    # -------------------------
    # 1. name -> full_name
    # -------------------------
    if "name" in data and "full_name" not in data:
        data["full_name"] = data.pop("name")

    # -------------------------
    # 2. experience -> work_experience
    # -------------------------
    if "experience" in data and "work_experience" not in data:
        data["work_experience"] = data.pop("experience")

    # -------------------------
    # 3. Extract company from title
    # Format: "Principal Engineer – QuantAxis Analytics"
    # -------------------------
    for exp in data.get("work_experience", []):
        title = exp.get("title", "")
        if "–" in title:
            role, company = title.split("–", 1)
            exp["role"] = role.strip()
            exp["company"] = company.strip()

    # -------------------------
    # 4. Ensure company field exists for scoring
    # -------------------------
    companies = []
    for exp in data.get("work_experience", []):
        if "company" in exp:
            companies.append(exp["company"])
    data["company"] = companies

    # -------------------------
    # 5. Ensure skills list exists
    # -------------------------
    if "skills" not in data:
        data["skills"] = []

    return data


# =========================
# MAIN EVALUATION
# =========================

def evaluate_resume(model_output: Dict) -> Dict:

    results = {}

    # ---------------------------------
    # Full text extraction
    # ---------------------------------
    full_text = collect_all_text(model_output)

    # ---------------------------------
    # Name
    # ---------------------------------
    name_correct = GROUND_TRUTH["name"] in full_text
    results["name_correct"] = name_correct

    # ---------------------------------
    # Education
    # ---------------------------------
    edu_hits = sum(
        1 for kw in GROUND_TRUTH["education_keywords"]
        if kw in full_text
    )
    results["education_hits"] = edu_hits
    results["education_total"] = len(GROUND_TRUTH["education_keywords"])

    # ---------------------------------
    # Companies
    # ---------------------------------
    company_hits = sum(
        1 for c in GROUND_TRUTH["companies"]
        if c in full_text
    )
    results["company_hits"] = company_hits
    results["company_total"] = len(GROUND_TRUTH["companies"])

    # ---------------------------------
    # Skills Evaluation
    # ---------------------------------
    predicted_skills = {
        canonicalize_skill(safe_str(s))
        for s in model_output.get("skills", []) or []
        if isinstance(s, str)
    }

    tp, fp, fn = evaluate_skill_predictions(
        predicted_skills,
        GROUND_TRUTH["skills"]
    )

    precision = compute_precision(len(tp), len(fp))
    recall = compute_recall(len(tp), len(fn))
    f1 = compute_f1(precision, recall)

    results["skills"] = {
        "predicted_count": len(predicted_skills),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hallucination_rate": round(
            len(fp) / len(predicted_skills), 4
        ) if predicted_skills else 0.0,
    }

    # ---------------------------------
    # Composite Score (0–100)
    # ---------------------------------
    composite_score = (
        (10 if name_correct else 0)
        + (edu_hits / max(results["education_total"], 1)) * 15
        + (company_hits / max(results["company_total"], 1)) * 15
        + f1 * 60
    )

    results["final_score_100"] = round(
        max(min(composite_score, 100), 0), 2
    )

    return results