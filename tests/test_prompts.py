TESTS = [
    {
        "id": "contact_extraction",
        "name": "Contact Info Accuracy",
        "prompt": "Extract full_name, email, phone, location. Ensure email and phone are correct."
    },
    {
        "id": "skills_depth",
        "name": "Skill Coverage",
        "prompt": "Extract skills. Must include both technical + soft skills if present."
    },
    {
        "id": "skills_inference_from_context",
        "name": "Skills Inference From Other Fields",
        "prompt": "If skills section is empty, infer skills only from explicit evidence in experience/projects/technologies."
    },
    {
        "id": "skills_inference_non_hallucination",
        "name": "Inference Without Hallucination",
        "prompt": "Infer missing skills only when supported by resume text. Do not add tools/skills not explicitly evidenced."
    },
    {
        "id": "education_structure",
        "name": "Education Structure",
        "prompt": "Extract education with degree, institution, year."
    },
    {
        "id": "experience_structure",
        "name": "Experience Structure",
        "prompt": "Extract experience with company, role, duration, responsibilities."
    },
    {
        "id": "project_structure",
        "name": "Project Extraction",
        "prompt": "Extract projects with name, description, technologies."
    },
    {
        "id": "hallucination_check",
        "name": "Hallucination Resistance",
        "prompt": "Return ONLY information present in resume. Do not invent anything."
    },
    {
        "id": "format_strictness",
        "name": "JSON Format Strictness",
        "prompt": "Return valid JSON ONLY. No markdown. No extra text."
    },
    {
        "id": "consistency_check",
        "name": "Consistency",
        "prompt": "Ensure extracted values match each other (same company names, same durations)."
    },
    {
        "id": "missing_field_behavior",
        "name": "Missing Field Handling",
        "prompt": "If something is missing, return null or empty list. For skills only, infer from explicit evidence in experience/projects; otherwise do not guess."
    },
    {
        "id": "multi_page_understanding",
        "name": "Multi Page Resume Understanding",
        "prompt": "Ensure you extract info across all pages."
    },
]
