import json
from functools import lru_cache

# ----------------------------
# SIMPLE LLM WRAPPER (replace with your backend)
# ----------------------------
import requests

def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": prompt,
            "temperature": 0,
            "stream": False
        }
    )
    data = response.json()
    print(data)
    return response.json()["response"]


# ----------------------------
# SAFE PARSER
# ----------------------------
def safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        return {
            "modality": "unknown",
            "primary_anatomy": "unknown",
            "secondary_anatomy": []
        }


# ----------------------------
# LLM EXTRACTION FUNCTION
# ----------------------------
@lru_cache(maxsize=100000)
def extract_study_metadata(study_text: str):
    prompt = f"""
Extract structured metadata from this radiology study.

Return ONLY valid JSON:

{{
  "modality": "CT | MRI | XR | US | PET | NM | unknown",
  "primary_anatomy": "string",
  "secondary_anatomy": ["string", "string"]
}}

TEXT:
\"\"\"{study_text}\"\"\"
""".strip()

    raw = call_llm(prompt)
    return safe_json_load(raw)