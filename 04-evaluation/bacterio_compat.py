"""Drop-in replacements for course helpers that assume the Responses API.

Bacterio.telcryp only fully supports chat.completions, so we re-implement
the helpers we need on top of that.
"""
import json as _json


def llm_structured(client, instructions, user_prompt, output_type, model="devstral"):
    schema = output_type.model_json_schema()
    schema_str = _json.dumps(schema, indent=2)
    system_msg = (
        instructions
        + "\n\nYou MUST respond with a single valid JSON object matching this schema. "
          "No prose, no markdown, no code fences — only the JSON object.\n\n"
          f"Schema:\n{schema_str}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content
    
    # --- FIX START: Strip out code fences if the model generated them ---
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        # Split lines, drop the first line (```json) and the last line (```)
        lines = cleaned_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()
    # --- FIX END ---

    parsed = output_type.model_validate_json(cleaned_text)
    return parsed, response.usage


def llm_structured_retry(client, instructions, user_prompt, output_type, model="devstral", retries=3):
    """Same as llm_structured, but retries on transient failures."""
    import time
    last_err = None
    for attempt in range(retries):
        try:
            return llm_structured(client, instructions, user_prompt, output_type, model=model)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)   # 1s, 2s, 4s
    raise last_err
