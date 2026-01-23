import argparse
import json
import re
from typing import Optional
from transformers import pipeline

# DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MODEL = "chillies/IELTS-writing-task-2-evaluation"

END_TOKEN = "</JSON>"

RUBRIC = """
You are an IELTS Writing Task 2 examiner.

Score these 4 criteria from 0 to 9 (half bands allowed):
- Task Response (TR)
- Coherence and Cohesion (CC)
- Lexical Resource (LR)
- Grammatical Range and Accuracy (GRA)

Then compute Overall as the average rounded to the nearest 0.5 band.

Return STRICT JSON with keys:
TR, CC, LR, GRA, Overall, Feedback, Improvements

Where:
- Feedback is an object with keys TR, CC, LR, GRA (each: short paragraph).
- Improvements is a list of 6 specific actionable edits.
Output must be a single JSON object only (no Markdown fences, no extra text), then append the literal token </JSON>.
"""

def build_prompt(task_prompt: str, essay: str) -> str:
    return f"Task 2 Question:\n{task_prompt}\n\nEssay:\n{essay}\n\n{RUBRIC}\nJSON (end with </JSON>):"

def extract_json(text: str) -> dict:
    """
    Try to pull the first JSON object from the model output.
    Handles ```json fenced blocks and trims surrounding chatter.
    """
    # Trim everything after the explicit end token if present
    end_idx = text.find(END_TOKEN)
    if end_idx != -1:
        text = text[:end_idx]

    def _first_balanced_json(s: str) -> Optional[str]:
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for idx, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
        return None

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidate = fenced.group(1)
    else:
        candidate = _first_balanced_json(text)
        if not candidate:
            raise RuntimeError("Model did not output JSON.\n---RAW OUTPUT---\n" + text[-1200:])

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Model output was not valid JSON: {exc}\n---RAW OUTPUT---\n{text[-1200:]}"
        ) from exc

def main():
    ap = argparse.ArgumentParser(description="IELTS Writing Task 2 checker (LLM-based).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default: Mistral 7B Instruct)")
    ap.add_argument("--prompt", required=True, help="Task 2 prompt/question")
    ap.add_argument("--essay_file", required=True, help="Path to a .txt file containing the essay")
    ap.add_argument("--max_new_tokens", type=int, default=650)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    with open(args.essay_file, "r", encoding="utf-8") as f:
        essay = f.read().strip()

    gen = pipeline(
        "text-generation",
        model=args.model,
        device_map="auto",
    )

    full_prompt = build_prompt(args.prompt, essay)
    out = gen(
        full_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        return_full_text=False,
    )[0]["generated_text"]

    result = extract_json(out)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
