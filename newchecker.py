
import argparse
import json
import re
from typing import Optional, Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "chillies/IELTS-writing-task-2-evaluation"
END_TOKEN = "</JSON>"

JSON_TEMPLATE = """{
  "TR": 0,
  "CC": 0,
  "LR": 0,
  "GRA": 0,
  "Overall": 0,
  "Evidence": {
    "TR": "",
    "CC": "",
    "LR": "",
    "GRA": ""
  },
  "Feedback": {
    "TR": "",
    "CC": "",
    "LR": "",
    "GRA": ""
  },
  "Improvements": [
    "",
    "",
    "",
    "",
    "",
    ""
  ]
}"""

RUBRIC = """
You are a strict IELTS Writing Task 2 examiner using the official public band descriptors (2019).

Scoring rules (0–9, half bands allowed):
- Task Response (TR): clear position; addresses all parts; well-developed ideas. If position is unclear, ideas are thinly developed, or the essay is <250 words, cap TR at 5.5.
- Coherence and Cohesion (CC): logical progression, clear paragraphing, varied cohesive devices. If cohesion is mostly additive (and/also) or topic sentences are weak, cap CC at 6.
- Lexical Resource (LR): range and precision; accurate collocations; minimal repetition. If vocabulary is repetitive or imprecise, cap LR at 6.
- Grammatical Range and Accuracy (GRA): variety of complex forms with good control; errors rarely impede meaning. If there are several noticeable errors per paragraph, cap GRA at 6.5; if many basic S/V or tense errors, cap at 6.

Additional limits:
- If no counter-argument is acknowledged in an agree/disagree essay, TR ≤ 7.
- If >3 clear grammar errors per ~150 words, GRA ≤ 6.5.

Output STRICT JSON with keys (numbers, not strings):
TR, CC, LR, GRA, Overall, Evidence, Feedback, Improvements
Where:
- Evidence is an object with keys TR, CC, LR, GRA containing short quoted snippets from the essay that justify the band.
- Feedback is an object with keys TR, CC, LR, GRA (each a short paragraph).
- Improvements is a list of exactly 6 specific, actionable edits tied to the essay.
Overall = average of TR, CC, LR, GRA, rounded to nearest 0.5.
Fill and return ONLY this JSON template (keys/order must stay the same): 
{template}
All band values must be numbers (not strings). Overall = average of TR, CC, LR, GRA rounded to nearest 0.5.
Output must be a single JSON object only (no Markdown fences, no extra text), then append the literal token </JSON>.
"""

def build_prompt(task_prompt: str, essay: str) -> str:
    return (
        f"Task 2 Question:\n{task_prompt}\n\n"
        f"Essay:\n{essay}\n\n"
        f"{RUBRIC.format(template=JSON_TEMPLATE)}\n"
        "JSON (end with </JSON>):"
    )

def maybe_chat_wrap(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt

def _first_balanced_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None

# def extract_json(text: str) -> Dict[str, Any]:
#     end_idx = text.find(END_TOKEN)
#     if end_idx != -1:
#         text = text[:end_idx]

#     fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
#     candidate = fenced.group(1) if fenced else _first_balanced_json(text)

#     if not candidate:
#         raise RuntimeError("Model did not output JSON.\n---RAW OUTPUT---\n" + text[-1200:])

#     try:
#         return json.loads(candidate)
#     except json.JSONDecodeError as exc:
#         raise RuntimeError(
#             f"Model output was not valid JSON: {exc}\n---RAW OUTPUT---\n{text[-1200:]}"
#         ) from exc
def extract_json(text: str) -> Dict[str, Any]:
    # 1) cut at END_TOKEN if exists
    end_idx = text.find(END_TOKEN)
    if end_idx != -1:
        text = text[:end_idx]

    # 2) try fenced ```json ... ```
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidate = fenced.group(1)
        return json.loads(candidate)

    # 3) normal: find first balanced {...}
    candidate = _first_balanced_json(text)
    if candidate:
        return json.loads(candidate)

    # 4) REPAIR FALLBACK:
    # If model omitted the opening "{", but output contains JSON-like keys, we rebuild.
    # We try to start at the first quote of a known key.
    key_match = re.search(r'"\s*(TR|CC|LR|GRA|Overall|Evidence|Feedback|Improvements)\s*"\s*:', text)
    if key_match:
        fragment = text[key_match.start():].strip()

        # Ensure it starts as an object
        rebuilt = "{\n" + fragment

        # Now balance braces/brackets/strings to close object safely
        def _balance(s: str) -> str:
            stack = []
            in_str = False
            esc = False
            out = []
            for ch in s:
                out.append(ch)
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        stack.append("}")
                    elif ch == "[":
                        stack.append("]")
                    elif ch in "}]" and stack:
                        stack.pop()

            if in_str:
                out.append('"')
            out.extend(reversed(stack))
            return "".join(out)

        rebuilt = _balance(rebuilt)

        # Sometimes model cuts mid-item in Improvements.
        # If JSON still invalid, try to trim to last complete ] or }.
        try:
            return json.loads(rebuilt)
        except json.JSONDecodeError:
            # Trim after the last closing brace we have (if any)
            last_brace = rebuilt.rfind("}")
            if last_brace != -1:
                try:
                    return json.loads(rebuilt[: last_brace + 1])
                except json.JSONDecodeError:
                    pass

            raise RuntimeError(
                "Model output looked like JSON but could not be repaired.\n---RAW OUTPUT---\n" + text[-2000:]
            )

    # 5) total failure
    raise RuntimeError("Model did not output JSON.\n---RAW OUTPUT---\n" + text[-1200:])


def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",   # faqat shu yerda!
        # torch_dtype="auto",
          dtype="auto",
    )
    model.eval()
    return tok, model

@torch.inference_mode()
def generate_text(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tok(prompt, return_tensors="pt")

    # inputsni modelning birinchi device'iga yuborish (sharded bo'lsa ham ishlaydi)
    if hasattr(model, "hf_device_map"):
        # device_map qiymatlari: 'cuda:0', 'cpu', va hok.
        first = next(iter(model.hf_device_map.values()))
        if isinstance(first, str) and first != "cpu":
            inputs = {k: v.to(first) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    out_ids = model.generate(**inputs, **gen_kwargs)
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=False)

def main():
    ap = argparse.ArgumentParser(description="IELTS Writing Task 2 checker (LLM-based).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    ap.add_argument("--prompt", required=True, help="Task 2 prompt/question")
    ap.add_argument("--essay_file", required=True, help="Path to a .txt file containing the essay")
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.essay_file, "r", encoding="utf-8") as f:
        essay = f.read().strip()

    tok, model = load_model(args.model)

    full_prompt = build_prompt(args.prompt, essay)
    full_prompt = maybe_chat_wrap(tok, full_prompt)

    raw = generate_text(tok, model, full_prompt, args.max_new_tokens, args.temperature)

    # </JSON> bo'yicha kesish
    cut = raw.split(END_TOKEN, 1)[0]
    result = extract_json(cut)

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
