import json
from typing import Generator


def _chunk_text(text, chunk_size=120):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def _fallback_structure(payload):
    final = {
        "patterns_by_user": payload.get("users", []),
        "confidence_method": "Weakly-supervised logistic model probability with temporal feature engineering",
        "loopholes": [
            "Observational temporal links are not guaranteed causation.",
            "Small sample size may overstate confidence.",
            "Tag-level granularity can miss nuance from full text.",
        ],
        "improvement_plan": payload.get("suggestions", []),
    }
    return json.dumps(final, ensure_ascii=False, indent=2)


def stream_structured_output(payload, api_key=None, model="llama-3.3-70b-versatile") -> Generator[str, None, None]:
    prompt = (
        "Convert the input into strict JSON only.\n"
        "Output schema keys: patterns_by_user, confidence_method, loopholes, improvement_plan.\n"
        "Each pattern must contain relation, confidence, confidence_label, justification.\n"
        "Do not add markdown."
    )

    if not api_key:
        fallback = _fallback_structure(payload)
        for part in _chunk_text(fallback):
            yield part
        return

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        stream = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=True,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        seen = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                seen = True
                yield delta
        if not seen:
            fallback = _fallback_structure(payload)
            for part in _chunk_text(fallback):
                yield part
    except Exception as e:
        fallback = _fallback_structure(payload)
        fallback_obj = json.loads(fallback)
        fallback_obj["groq_error"] = str(e)
        text = json.dumps(fallback_obj, ensure_ascii=False, indent=2)
        for part in _chunk_text(text):
            yield part

