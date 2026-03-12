import fitz
from dotenv import load_dotenv
import os

from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from resume_extractor.schemas import ResumeData
from resume_extractor.metrics import Timer, count_tokens, get_full_usage_snapshot

load_dotenv()

SYSTEM_PROMPT = """
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
""".strip()


def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


def load_llm(model: str):
    if model.startswith("azure:"):
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_VERSION")

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment,
            temperature=0
        )

    return ChatOllama(model=model, temperature=0)


def extract_resume_info(resume_text: str, model):

    if isinstance(model, str):
        model_name = model
        llm = load_llm(model)
    else:
        llm = model
        model_name = llm.__class__.__name__ 
    parser = JsonOutputParser(pydantic_object=ResumeData)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Resume Text:\n\n{resume_text}\n\n{format_instructions}")
        ]
    )

    format_instructions = parser.get_format_instructions()
    formatted = prompt.format_messages(
        resume_text=resume_text,
        format_instructions=format_instructions
    )

    prompt_text = "\n".join(m.content for m in formatted)
    prompt_tokens = count_tokens(prompt_text)

    timer = Timer()
    before_usage = get_full_usage_snapshot()
    timer.start()

    try:
        # 🔹 Step 1 — get raw output first
        raw_response = (prompt | llm).invoke(
            {"resume_text": resume_text, "format_instructions": format_instructions}
        )

        raw_text = raw_response.content

        # 🔹 Step 2 — try parsing normally
        try:
            data = parser.parse(raw_text)
        except Exception:
            # 🔥 Step 3 — retry with strict correction prompt
            fix_prompt = f"""
You returned invalid JSON.
Return ONLY valid JSON matching this schema.
No markdown. No explanation.

Schema:
{format_instructions}

Previous output:
{raw_text}
"""
            fixed = llm.invoke(fix_prompt)
            data = parser.parse(fixed.content)

        timer.stop()
        after_usage = get_full_usage_snapshot()

        output_tokens = count_tokens(str(data))

        return {
    "data": data,
    "metrics": {
        "model_used": model_name,
        "prompt_tokens_estimate": prompt_tokens,
        "output_tokens_estimate": output_tokens,
        "total_tokens_estimate": prompt_tokens + output_tokens,
        "time_seconds": round(timer.elapsed, 3),
        "tokens_per_second_estimate": round(output_tokens / timer.elapsed, 2)
        if timer.elapsed > 0 else 0,
        "usage_before": before_usage,
        "usage_after": after_usage,
    }
}

    except Exception as e:
        timer.stop()
        return {
            "error": "Extraction failed",
            "details": str(e),
            "metrics": {
                "model_used": model_name,
                "time_seconds": round(timer.elapsed, 3),
                "usage_before": before_usage,
            }
        }