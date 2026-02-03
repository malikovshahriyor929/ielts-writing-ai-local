#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

END_TOKEN = "</JSON>"

# -----------------------------
# 1) JSON templates (Task1/Task2)
# -----------------------------
TASK2_JSON_TEMPLATE = """{
  "TR": 0,
  "CC": 0,
  "LR": 0,
  "GRA": 0,
  "Overall": 0,
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

TASK1_JSON_TEMPLATE = """{
  "TA": 0,
  "CC": 0,
  "LR": 0,
  "GRA": 0,
  "Overall": 0,
  "Feedback": {
    "TA": "",
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

# -----------------------------
# 2) Rubrics (based on official "public version" band descriptors)
#    We DO NOT paste the whole descriptors table; we summarize rules & criteria names.
# -----------------------------
SYSTEM_PROMPT = "You are an IELTS Writing examiner. Be accurate and consistent with band descriptors."


RUBRIC_COMMON = f"""
You are a STRICT IELTS Writing examiner being real examiner. Use the official IELTS Writing band descriptors (public/public-facing version).

Rules:
- Scores are 0–9 (half bands allowed: 0, 0.5, 1, 1.5 ... 9).
- Band 0 MUST be used ONLY if the response is blank OR has no assessable language.
  If there is a meaningful attempt in English, the minimum is 1.0.
- Return ONE single JSON object only (no Markdown, no extra text).
- Start output with '{{' as the first character. No text before it.
- End output with '}}' then append the literal token {END_TOKEN}.
- Numbers MUST be numbers (not strings).
- "Improvements" MUST be exactly 6 items (strings).
- Feedback must reference THIS essay (quote at least 1 short phrase per criterion).

Important:
- Do NOT give the same score for TR, CC, LR, and GRA unless you are absolutely sure they are equal.
  If you still give identical scores, you must justify it clearly in Feedback.
"""


RUBRIC_TASK2 = """
Task type: IELTS Writing Task 2.

Criteria (4):
- Task Response (TR): address ALL parts; clear position; well-developed ideas; relevant examples.
- Coherence & Cohesion (CC): clear paragraphing; logical progression; appropriate cohesive devices.
- Lexical Resource (LR): range + precision; natural collocations; minimal repetition; spelling/word formation control.
- Grammatical Range & Accuracy (GRA): mix of simple/complex; good control; errors rarely reduce communication.

Hard constraints:
- If word count < 250, TR should be capped at 5.5.
- If position is unclear or ideas are thin, TR should be low.
- If grammar errors are frequent and noticeable, cap GRA accordingly.

Compute Overall = average(TR, CC, LR, GRA) rounded to nearest 0.5.

Fill and return ONLY this JSON template (keys/order should stay the same):
{template}
"""

RUBRIC_TASK1_ACADEMIC = """
Task type: IELTS Writing Task 1 (Academic report).

Criteria (4):
- Task Achievement (TA): covers requirements; selects key features; provides a clear overview; accurate comparisons.
- Coherence & Cohesion (CC): logical organization; paragraphing; referencing; linking devices not over/under-used.
- Lexical Resource (LR): appropriate academic/report vocabulary; precise trend language; minimal repetition.
- Grammatical Range & Accuracy (GRA): correct tense for overview/trends; variety of structures; control of errors.

Hard constraints:
- If word count < 150, TA should be capped at 5.5.
- If there is NO clear overview of main trends, TA should be reduced.

Compute Overall = average(TA, CC, LR, GRA) rounded to nearest 0.5.

Fill and return ONLY this JSON template (keys/order should stay the same):
{template}
"""

RUBRIC_TASK1_GT_LETTER = """
Task type: IELTS Writing Task 1 (General Training letter).

Criteria (4):
- Task Achievement (TA): clear purpose; all bullet points addressed; appropriate tone; fully developed response.
- Coherence & Cohesion (CC): logical paragraphing; clear sequencing; effective cohesive devices.
- Lexical Resource (LR): appropriate tone; natural phrases for letters; precise vocabulary; minimal repetition.
- Grammatical Range & Accuracy (GRA): variety + control; errors rarely reduce communication.

Hard constraints:
- If word count < 150, TA should be capped at 5.5.
- If purpose/tone is unclear or inconsistent, TA should be reduced.

Compute Overall = average(TA, CC, LR, GRA) rounded to nearest 0.5.

Fill and return ONLY this JSON template (keys/order should stay the same):
{template}
"""

# -----------------------------
# 3) Model auto-selection
# -----------------------------
def candidate_models(task: int, quality: str) -> List[str]:
    """
    You can always override with --model.
    quality: fast | balanced | best | myfast
    """
    quality = quality.lower().strip()
    if quality not in {"fast", "balanced", "best", "myfast"}:
        quality = "balanced"

    # NOTE:
    # - Qwen2.5 Instruct family exists in multiple sizes. :contentReference[oaicite:3]{index=3}
    # - A specialized IELTS evaluator model exists but may be heavy. :contentReference[oaicite:4]{index=4}
    if task == 2:
        if quality == "best":
            return [
                "JacobLinCool/mistral-7b-ielts-evaluator-safetensors",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ]
        if quality == "balanced":
            return [
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ]
        if quality == "myfast":
            return [
                 "chillies/mistral-7b-ielts-evaluator-q4",
            ]
        return ["Qwen/Qwen2.5-1.5B-Instruct"]
    else:
        # Task 1
        if quality == "best":
            return [
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ]
        if quality == "balanced":
            return [
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ]
        return ["Qwen/Qwen2.5-1.5B-Instruct"]

def maybe_chat_wrap(tokenizer: Any, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # fallback (no chat template)
    return f"SYSTEM: {SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}"

def scores_are_uniform(d: Dict[str, Any], keys: List[str]) -> bool:
    vals = []
    for k in keys:
        try:
            vals.append(float(d.get(k, 0)))
        except Exception:
            vals.append(0.0)
    vals = [round_half(v) for v in vals]
    return len(set(vals)) == 1

def build_rescore_prompt(original_prompt: str) -> str:
    return (
        original_prompt
        + "\n\nIMPORTANT: Your previous scoring gave the same band for all criteria (e.g., TR=CC=LR=GRA). "
          "This is usually incorrect. Re-evaluate each criterion independently. "
          "TR, CC, LR, and GRA may differ. Quote at least one short phrase from the essay in each Feedback field.\n"
          f"Return JSON only and end with {END_TOKEN}.\n"
    )


# -----------------------------
# 4) Prompt building
# -----------------------------
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def build_prompt(task: int, task1_mode: str, task_prompt: str, essay: str) -> Tuple[str, str]:
    """
    Returns (full_prompt, json_template_used)
    """
    if task == 2:
        template = TASK2_JSON_TEMPLATE
        rubric = RUBRIC_TASK2.format(template=template)
    else:
        template = TASK1_JSON_TEMPLATE
        if task1_mode == "gt":
            rubric = RUBRIC_TASK1_GT_LETTER.format(template=template)
        else:
            rubric = RUBRIC_TASK1_ACADEMIC.format(template=template)

    wc = word_count(essay)

    # Force '{' right after "JSON:" to reduce "missing {" issues
    full = (
    f"Task Question:\n{task_prompt.strip()}\n\n"
    f"Candidate Answer (word_count={wc}):\n{essay.strip()}\n\n"
    f"{RUBRIC_COMMON}\n"
    f"{rubric}\n"
    f"JSON (end with {END_TOKEN}):\n"
)
    return full, template


# def maybe_chat_wrap(tokenizer: Any, user_prompt: str) -> str:
#     """
#     If tokenizer has chat_template (Qwen/Llama/etc.), use it.
#     Otherwise return plain prompt.
#     """
#     if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
#         messages = [{"role": "user", "content": user_prompt}]
#         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     return user_prompt


# -----------------------------
# 5) Generation (NO pipeline -> avoids device_map -> generate kwargs bug)
# -----------------------------
def pick_runtime_device(requested: str) -> str:
    requested = requested.lower().strip()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

GGUF_MAP = {
    "chillies/mistral-7b-ielts-evaluator-q4": "GUFF-ielts-fighter-unsloth.Q4_K_M.gguf",
}

def load_model(model_id: str, device: str):
    gguf_file = GGUF_MAP.get(model_id)

    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        gguf_file=gguf_file,   # <- MUHIM
    )

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            gguf_file=gguf_file,  # <- MUHIM
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            gguf_file=gguf_file,
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            gguf_file=gguf_file,
        ).to("cpu")

    model.eval()
    return tok, model

# def load_model(model_id: str, device: str) -> Tuple[Any, Any]:
#     """
#     Load tokenizer + model safely for cpu/mps/cuda.
#     dtype uses new parameter name "dtype".
#     """
#     tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

#     if device == "cuda":
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map="auto",
#             dtype="auto",
#         )
#     elif device == "mps":
#         # MPS usually prefers float16
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map=None,
#             dtype=torch.float16,
#         )
#         model.to("mps")
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             device_map=None,
#             dtype="auto",
#         )
#         model.to("cpu")

#     model.eval()
#     return tok, model


def infer_input_device(model: Any) -> torch.device:
    """
    If model is sharded with hf_device_map, pick first non-cpu device.
    Otherwise use model.device.
    """
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev not in {"cpu", "disk", "meta"}:
                return torch.device(dev)
        return torch.device("cpu")
    # Fallback
    try:
        return model.device
    except Exception:
        return torch.device("cpu")


@torch.inference_mode()
def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    do_sample = temperature and temperature > 0

    inputs = tokenizer(prompt, return_tensors="pt")
    dev = infer_input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    out_ids = model.generate(**inputs, **gen_kwargs)

    # only generated continuation
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


def load_best_available(task: int, quality: str, device: str, user_model: Optional[str]) -> Tuple[str, Any, Any]:
    """
    Returns (model_id_used, tokenizer, model).
    """
    tried: List[str] = []
    models = [user_model] if user_model else candidate_models(task, quality)

    last_err: Optional[BaseException] = None
    for mid in models:
        if not mid:
            continue
        try:
            tok, model = load_model(mid, device=device)
            return mid, tok, model
        except BaseException as e:
            tried.append(mid)
            last_err = e

    msg = "Could not load any model.\nTried:\n" + "\n".join(f"- {m}" for m in tried)
    if last_err:
        msg += f"\nLast error: {type(last_err).__name__}: {last_err}"
    raise RuntimeError(msg)


# -----------------------------
# 6) JSON extraction + repair
# -----------------------------
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

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _remove_trailing_commas(s: str) -> str:
    # Fix: {"a": 1,} or [1,2,]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def extract_json(raw: str) -> Dict[str, Any]:
    # Cut at END_TOKEN if exists
    end_idx = raw.find(END_TOKEN)
    if end_idx != -1:
        raw = raw[:end_idx]

    # 1) fenced ```json ... ```
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
    if fenced:
        candidate = fenced.group(1)
        candidate = _remove_trailing_commas(candidate)
        return json.loads(candidate)

    # 2) balanced {...}
    candidate = _first_balanced_json(raw)
    if candidate:
        candidate = _remove_trailing_commas(candidate)
        return json.loads(candidate)

    # 3) Repair fallback (missing '{' but has JSON-like keys)
    key_match = re.search(r'"\s*(TR|TA|CC|LR|GRA|Overall|Feedback|Improvements)\s*"\s*:', raw)
    if key_match:
        fragment = raw[key_match.start() :].strip()
        rebuilt = "{\n" + fragment

        # Balance braces/brackets
        stack: List[str] = []
        in_str = False
        esc = False
        out_chars: List[str] = []
        for ch in rebuilt:
            out_chars.append(ch)
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
                    stack.append("}")
                elif ch == "[":
                    stack.append("]")
                elif ch in "}]" and stack:
                    stack.pop()

        if in_str:
            out_chars.append('"')
        out_chars.extend(reversed(stack))
        rebuilt = "".join(out_chars)
        rebuilt = _remove_trailing_commas(rebuilt)

        try:
            return json.loads(rebuilt)
        except json.JSONDecodeError:
            pass

    raise RuntimeError("Model did not output JSON.\n---RAW OUTPUT---\n" + raw[-1600:])


# -----------------------------
# 7) Validate / normalize output
# -----------------------------
def round_half(x: float) -> float:
    return math.floor(x * 2 + 0.5) / 2.0


def clamp_band(x: float) -> float:
    x = float(x)
    if x < 0:
        x = 0.0
    if x > 9:
        x = 9.0
    return round_half(x)


def ensure_str_list_len6(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        xs = []
    xs = [str(x) for x in xs]
    if len(xs) > 6:
        xs = xs[:6]
    while len(xs) < 6:
        xs.append("")
    return xs


def normalize_result(task: int, task1_mode: str, essay: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Coerce numeric fields
    - Enforce word-count caps (TR/TA cap at 5.5 if under required words)
    - Ensure feedback structure
    - Ensure improvements length 6
    - Recompute Overall reliably
    """
    wc = word_count(essay)
    if task == 2:
        main = "TR"
        required_words = 250
        template = json.loads(TASK2_JSON_TEMPLATE)
        fb_keys = ["TR", "CC", "LR", "GRA"]
    else:
        main = "TA"
        required_words = 150
        template = json.loads(TASK1_JSON_TEMPLATE)
        fb_keys = ["TA", "CC", "LR", "GRA"]

    # Start from template, overlay model output
    out = dict(template)
    out.update(data or {})

    # Scores
    for k in [main, "CC", "LR", "GRA"]:
        v = out.get(k, 0)
        # accept strings like "6.5"
        try:
            v = float(v)
        except Exception:
            v = 0.0
        out[k] = clamp_band(v)

    # Word-count cap (official minimum word counts)
    if wc < required_words:
        out[main] = min(out[main], 5.5)

    # Feedback
    fb = out.get("Feedback", {})
    if not isinstance(fb, dict):
        fb = {}
    norm_fb = {}
    for k in fb_keys:
        norm_fb[k] = str(fb.get(k, "")).strip()
    out["Feedback"] = norm_fb

    # Improvements
    out["Improvements"] = ensure_str_list_len6(out.get("Improvements", []))

    # Overall: compute ourselves (do not trust model math)
    overall = (out[main] + out["CC"] + out["LR"] + out["GRA"]) / 4.0
    out["Overall"] = clamp_band(overall)

    # Ordered output
    if task == 2:
        ordered = OrderedDict(
            [
                ("TR", out["TR"]),
                ("CC", out["CC"]),
                ("LR", out["LR"]),
                ("GRA", out["GRA"]),
                ("Overall", out["Overall"]),
                ("Feedback", out["Feedback"]),
                ("Improvements", out["Improvements"]),
            ]
        )
    else:
        ordered = OrderedDict(
            [
                ("TA", out["TA"]),
                ("CC", out["CC"]),
                ("LR", out["LR"]),
                ("GRA", out["GRA"]),
                ("Overall", out["Overall"]),
                ("Feedback", out["Feedback"]),
                ("Improvements", out["Improvements"]),
            ]
        )
    return ordered


# -----------------------------
# 8) CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="IELTS Writing checker (Task 1/2) with strict JSON output.")
    ap.add_argument("--task", type=int, choices=[1, 2], required=True, help="1 for Writing Task 1, 2 for Writing Task 2")
    ap.add_argument(
        "--task1_mode",
        choices=["academic", "gt"],
        default="academic",
        help="Task 1 mode: academic (default) or gt (General Training letter)",
    )
    ap.add_argument("--prompt", required=True, help="Task question/prompt text")
    ap.add_argument("--essay_file", required=True, help="Path to .txt file with candidate response")
    ap.add_argument("--max_new_tokens", type=int, default=650)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--quality",
        choices=["fast", "balanced", "best", "myfast"],
        default="balanced",
        help="Auto-model quality tier (used only if --model not provided)",
    )
    ap.add_argument("--model", default=None, help="HF model id to override auto model selection")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Runtime device")
    ap.add_argument("--debug_raw", action="store_true", help="Print raw model output to stderr (debug)")

    args = ap.parse_args()

    with open(args.essay_file, "r", encoding="utf-8") as f:
        essay = f.read().strip()

    device = pick_runtime_device(args.device)

    model_id_used, tok, model = load_best_available(
        task=args.task,
        quality=args.quality,
        device=device,
        user_model=args.model,
    )

    full_prompt, _template = build_prompt(
        task=args.task,
        task1_mode=args.task1_mode,
        task_prompt=args.prompt,
        essay=essay,
    )
    formatted_prompt = maybe_chat_wrap(tok, full_prompt)

    raw = generate_text(
        tokenizer=tok,
        model=model,
        prompt=formatted_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.debug_raw:
        print(f"\n[DEBUG] model={model_id_used} device={device}\n---RAW START---\n{raw}\n---RAW END---\n", file=sys.stderr)

    # Cut at END_TOKEN, but keep repair robustness
    # cut = raw.split(END_TOKEN, 1)[0]
    # parsed = extract_json(cut)
    # normalized = normalize_result(args.task, args.task1_mode, essay, parsed)

    # print(json.dumps(normalized, indent=2, ensure_ascii=False))
    cut = raw.split(END_TOKEN, 1)[0]
    parsed = extract_json(cut)
    normalized = normalize_result(args.task, args.task1_mode, essay, parsed)

    # If model lazily outputs same score for all criteria, do a second pass
    if args.task == 2 and scores_are_uniform(normalized, ["TR", "CC", "LR", "GRA"]):
        rescore_prompt = build_rescore_prompt(full_prompt)
        formatted_prompt2 = maybe_chat_wrap(tok, rescore_prompt)
        raw2 = generate_text(
            tokenizer=tok,
            model=model,
            prompt=formatted_prompt2,
            max_new_tokens=min(450, args.max_new_tokens),
            temperature=args.temperature,
        )
        cut2 = raw2.split(END_TOKEN, 1)[0]
        parsed2 = extract_json(cut2)
        normalized = normalize_result(args.task, args.task1_mode, essay, parsed2)

    print(json.dumps(normalized, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()
