import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
import requests

# ---------------- Configuration ----------------
MODEL_NAME = "llama3:8b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

DATA_DIR = "data"
OUTPUT_DIR = "output"

SLOKAS_FILE = os.path.join(DATA_DIR, "Valmiki_Ramayan_Shlokas.json")
PARTS_FILE = os.path.join(DATA_DIR, "sub_parts.json")

REQUEST_TIMEOUT = 120
TEMPERATURE = 0.25
MAX_RETRIES = 3

# ---------------- Utilities ----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_int(x, default=None):
    try:
        return int(x)
    except:
        return default

def normalize_str(x) -> str:
    if x is None:
        return ""
    return str(x)

def is_sundara_kanda(val: str) -> bool:
    if not val:
        return False
    norm = val.strip().lower()
    # Accept common names/spellings
    return any(k in norm for k in [
        "sundara", "sunder", "sundara kanda", "sundarakanda", "sunderkand"
    ])

def parse_sarga_range(rng: str) -> Tuple[Optional[int], Optional[int]]:
    if not rng or not isinstance(rng, str):
        return None, None
    txt = rng.strip().lower()
    txt = txt.replace("–", "-").replace("—", "-")
    txt = re.sub(r"\s+to\s+", "-", txt)
    txt = re.sub(r"\s+", "", txt)
    if "-" in txt:
        a, b = txt.split("-", 1)
        return safe_int(a), safe_int(b)
    n = safe_int(txt)
    return (n, n)

def filter_sundara_slokas_for_range(slokas: List[Dict[str, Any]], s_start: int, s_end: int) -> List[Dict[str, Any]]:
    out = []
    for s in slokas:
        kanda = s.get("kanda") or s.get("Kanda") or ""
        if not is_sundara_kanda(str(kanda)):
            continue
        sarga = safe_int(s.get("sarga") or s.get("Sarga"))
        if sarga is None:
            continue
        if s_start <= sarga <= s_end:
            # keep only those with explanation and shloka_text
            if s.get("explanation") and s.get("shloka_text"):
                out.append(s)
    out.sort(key=lambda it: (safe_int(it.get("sarga"), 10**9), safe_int(it.get("shloka"), 10**9)))
    return out

def pick_begin_mid_end(slokas: List[Dict[str, Any]]):
    if not slokas:
        return None, None, None
    first = slokas[0]
    last = slokas[-1]
    mid = slokas[len(slokas)//2]
    return first, mid, last

def build_system_instruction() -> str:
    return """You are an expert Sanskrit epic summarizer.
You MUST read ONLY the 'explanation' fields of the provided slokas (do NOT use translations or external knowledge).
Return strictly a single JSON object with this schema and constraints:

{
  "overview": "2 to 4 lines. Summarize the subpart thematically according to the Part title.",
  "shloka_1": {
    "identifier": "Sundara Kanda <sarga>.<shloka>",
    "text": "Exact shloka_text copied verbatim from the provided dataset",
    "meaning": "1 to 3 lines. Brief meaning in the context of the Part title, derived only from explanations."
  },
  "story_1": ["exactly 15 lines, concise sentences derived only from explanations"],
  "shloka_2": {
    "identifier": "Sundara Kanda <sarga>.<shloka>",
    "text": "Exact shloka_text copied verbatim from the provided dataset",
    "meaning": "1 to 3 lines, derived only from explanations"
  },
  "story_2": ["exactly 15 lines derived only from explanations"],
  "shloka_3": {
    "identifier": "Sundara Kanda <sarga>.<shloka>",
    "text": "Exact shloka_text copied verbatim from the provided dataset",
    "meaning": "1 to 3 lines, derived only from explanations"
  },
  "story_3": ["exactly 15 lines derived only from explanations"],
  "conclusion": "2 to 4 lines concluding the subpart"
}

Selection rules:
- Choose exactly 3 slokas that best mark major events within the provided sarga range: one near the beginning, one around the middle, one near the end.
- Select only from 'available_slokas' and copy 'shloka_text' exactly as given.
- Keep chronology consistent.

Content rules:
- Use ONLY the 'explanation' text for narrative and meanings. Do NOT use 'translation' or outside lore.
- Do NOT leave any field blank.
- Do NOT add extra keys or commentary.
- Return valid JSON only.

Example (values are illustrative; keep your output factual and concise):

{
  "overview": "Line 1. Line 2.",
  "shloka_1": {"identifier":"Sundara Kanda 9.1","text":"…","meaning":"Line 1."},
  "story_1": ["Line 1.","Line 2.","Line 3.","Line 4.","Line 5.","Line 6.","Line 7.","Line 8.","Line 9.","Line 10.","Line 11.","Line 12.","Line 13.","Line 14.","Line 15."],
  "shloka_2": {"identifier":"Sundara Kanda 10.27","text":"…","meaning":"Line 1."},
  "story_2": ["Line 1.","Line 2.","Line 3.","Line 4.","Line 5.","Line 6.","Line 7.","Line 8.","Line 9.","Line 10.","Line 11.","Line 12.","Line 13.","Line 14.","Line 15."],
  "shloka_3": {"identifier":"Sundara Kanda 12.25","text":"…","meaning":"Line 1."},
  "story_3": ["Line 1.","Line 2.","Line 3.","Line 4.","Line 5.","Line 6.","Line 7.","Line 8.","Line 9.","Line 10.","Line 11.","Line 12.","Line 13.","Line 14.","Line 15."],
  "conclusion": "Line 1. Line 2."
}
"""

def build_user_prompt(part_title: str,
                      sarga_range_str: str,
                      slokas_subset: List[Dict[str, Any]],
                      fallback_triplet: Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]) -> str:
    def sloka_view(s):
        return {
            "identifier": f"{s.get('kanda','')} {s.get('sarga','')}.{s.get('shloka','')}",
            "sarga": s.get("sarga"),
            "shloka": s.get("shloka"),
            "shloka_text": s.get("shloka_text",""),
            "explanation": s.get("explanation","")
        }

    available = [sloka_view(s) for s in slokas_subset]
    fb1, fb2, fb3 = fallback_triplet
    fallback_ids = []
    for fb in (fb1, fb2, fb3):
        if fb:
            fallback_ids.append(f"{fb.get('kanda','')} {fb.get('sarga','')}.{fb.get('shloka','')}")

    instruction = {
        "part_title": part_title,
        "sarga_range": sarga_range_str,
        "guidance": {
            "read_only": "Use only 'explanation' fields for narrative; use shloka_text only for quoting.",
            "pick_three": "Pick start, middle, end slokas that mark major events.",
            "fallback_identifiers": fallback_ids
        },
        "available_slokas": available
    }

    return build_system_instruction() + "\n\n" + json.dumps(instruction, ensure_ascii=False, indent=2)

def call_ollama(prompt: str, max_retries: int = MAX_RETRIES, temperature: float = TEMPERATURE) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0 + attempt)
                continue
            return ""

def extract_json_block(text: str) -> dict:
    # Try to find a JSON object in the text
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        pass
    return {}

def ensure_15_lines(arr, max_chars_per_line=180):
    if not isinstance(arr, list):
        arr = []
    out = []
    for x in arr:
        s = (str(x) if x is not None else "").strip()
        if s:
            if len(s) > max_chars_per_line:
                s = s[:max_chars_per_line].rstrip() + "…"
            out.append(s)
    while len(out) < 15:
        out.append("")
    return out[:15]

def make_15_lines_from_explanations(exps: List[str], max_chars_per_line=180) -> List[str]:
    out = []
    for e in exps:
        if not e:
            continue
        parts = re.split(r"[।.!?]\s+", e.strip())
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) > max_chars_per_line:
                p = p[:max_chars_per_line].rstrip() + "…"
            out.append(p)
            if len(out) >= 15:
                return out
    while len(out) < 15:
        out.append("")
    return out[:15]

def find_by_identifier(subset: List[Dict[str, Any]], identifier: str) -> Optional[Dict[str, Any]]:
    if not identifier or not isinstance(identifier, str):
        return None
    m = re.search(r"(\d+)\.(\d+)", identifier)
    if not m:
        return None
    sarga_i = safe_int(m.group(1))
    shloka_i = safe_int(m.group(2))
    if sarga_i is None or shloka_i is None:
        return None
    for s in subset:
        if safe_int(s.get("sarga")) == sarga_i and safe_int(s.get("shloka")) == shloka_i:
            return s
    return None

def build_shloka_entry(llm_piece: Any,
                       subset: List[Dict[str, Any]],
                       fb: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out = {"identifier": "", "text": "", "meaning": ""}
    identifier = ""
    meaning = ""
    if isinstance(llm_piece, dict):
        identifier = normalize_str(llm_piece.get("identifier"))
        meaning = normalize_str(llm_piece.get("meaning"))

    ds = find_by_identifier(subset, identifier) if identifier else None
    if ds is None:
        ds = fb

    if ds:
        out["identifier"] = f"{ds.get('kanda','')} {ds.get('sarga','')}.{ds.get('shloka','')}"
        out["text"] = ds.get("shloka_text", "")
    out["meaning"] = meaning.strip()
    return out

def coerce_to_schema(llm_obj: dict,
                     subset: List[Dict[str, Any]],
                     fallback_triplet: Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]],
                     part_title: str) -> dict:
    fb1, fb2, fb3 = fallback_triplet
    overview = normalize_str(llm_obj.get("overview")).strip()
    conclusion = normalize_str(llm_obj.get("conclusion")).strip()

    sh1 = build_shloka_entry(llm_obj.get("shloka_1", {}), subset, fb1)
    sh2 = build_shloka_entry(llm_obj.get("shloka_2", {}), subset, fb2)
    sh3 = build_shloka_entry(llm_obj.get("shloka_3", {}), subset, fb3)

    story_1 = ensure_15_lines(llm_obj.get("story_1", []))
    story_2 = ensure_15_lines(llm_obj.get("story_2", []))
    story_3 = ensure_15_lines(llm_obj.get("story_3", []))

    return {
        "part_title": part_title,
        "overview": overview,
        "shloka_1": sh1,
        "story_1": story_1,
        "shloka_2": sh2,
        "story_2": story_2,
        "shloka_3": sh3,
        "story_3": story_3,
        "conclusion": conclusion
    }

def fill_if_empty(s: str, fallback: str) -> str:
    s = (s or "").strip()
    return s if s else fallback

def enforce_schema_nonempty(final_obj: dict,
                            subset: List[Dict[str, Any]],
                            fallback_triplet: Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]],
                            part_title: str) -> dict:
    # Overview / Conclusion fallbacks from explanations
    exps_all = [s.get("explanation","") for s in subset if s.get("explanation")]
    first_exp = exps_all[0] if exps_all else f"This part aligns with its theme: {part_title}."
    last_exp = exps_all[-1] if exps_all else f"It culminates in a resolution coherent with: {part_title}."

    final_obj["overview"] = fill_if_empty(final_obj.get("overview"),
                                          (first_exp[:220]).rstrip())
    final_obj["conclusion"] = fill_if_empty(final_obj.get("conclusion"),
                                            (last_exp[:220]).rstrip())

    # Fix shloka entries: ensure text present; meaning fallback from explanation first sentence
    def fix_shloka(slot_key: str, fb: Optional[Dict[str, Any]]):
        entry = final_obj.get(slot_key, {})
        if not isinstance(entry, dict):
            entry = {}
        if not entry.get("text") and fb:
            entry["identifier"] = f"{fb.get('kanda','')} {fb.get('sarga','')}.{fb.get('shloka','')}"
            entry["text"] = fb.get("shloka_text","")
        if not entry.get("meaning"):
            # derive from matched explanation or fb
            match = None
            ident = entry.get("identifier","")
            m = re.search(r"(\d+)\.(\d+)", ident)
            if m:
                sarg = safe_int(m.group(1)); sh = safe_int(m.group(2))
                for s in subset:
                    if safe_int(s.get("sarga")) == sarg and safe_int(s.get("shloka")) == sh:
                        match = s; break
            exp = (match or fb or {}).get("explanation", "")
            # take the first sentence-like chunk
            sentence = ""
            if "।" in exp:
                sentence = exp.split("।")[0].strip() if isinstance(exp, str) else exp
            elif "." in exp:
                sentence = exp.split(".")[0].strip() if isinstance(exp, str) else exp
            entry["meaning"] = sentence if sentence else f"Key line within the theme: {part_title}."
        final_obj[slot_key] = entry

    fb1, fb2, fb3 = fallback_triplet
    fix_shloka("shloka_1", fb1)
    fix_shloka("shloka_2", fb2)
    fix_shloka("shloka_3", fb3)

    # Build/normalize stories to 15 lines using explanations around corresponding sarga
    def fix_story(key: str, around_key: str):
        arr = final_obj.get(key, [])
        has_content = isinstance(arr, list) and any((str(x).strip() if x is not None else "") for x in arr)
        if not has_content:
            ident = final_obj.get(around_key, {}).get("identifier", "")
            m = re.search(r"(\d+)\.(\d+)", ident) if isinstance(ident, str) else None
            sarg = safe_int(m.group(1)) if m else None
            if sarg is not None:
                exps = [s.get("explanation","") for s in subset if safe_int(s.get("sarga")) == sarg and s.get("explanation")]
            else:
                exps = [s.get("explanation","") for s in subset if s.get("explanation")]
            final_obj[key] = make_15_lines_from_explanations(exps)
        else:
            final_obj[key] = ensure_15_lines(arr)

    fix_story("story_1", "shloka_1")
    fix_story("story_2", "shloka_2")
    fix_story("story_3", "shloka_3")

    return final_obj

def write_output(part_name: str, obj: dict):
    ensure_dir(OUTPUT_DIR)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", part_name.strip())
    out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")

# ---------------- Main Pipeline ----------------

def main():
    ensure_dir(OUTPUT_DIR)

    # Load inputs
    slokas_all = load_json(SLOKAS_FILE)
    parts_map = load_json(PARTS_FILE)

    # Expecting parts_map like: { "Sunderkand": { "Part 1": {"Title":"...", "Sargas":"1 to 1", "Total_Sargas":1}, ... } }
    sunder_map = (
        parts_map.get("Sunderkand")
        or parts_map.get("Sundarkand")
        or parts_map.get("SundaraKanda")
        or parts_map.get("Sundara Kanda")
        or {}
    )
    if not isinstance(sunder_map, dict) or not sunder_map:
        print("No 'Sunderkand' mapping found in sub_parts.json")
        return

    for part_name, meta in sunder_map.items():
        title = normalize_str(meta.get("Title") or part_name).strip()
        s_arg = normalize_str(meta.get("Sargas")).strip()
        s_start, s_end = parse_sarga_range(s_arg)
        if s_start is None or s_end is None:
            print(f"Skipping {part_name}: invalid Sargas '{s_arg}'")
            continue
        if s_end < s_start:
            s_start, s_end = s_end, s_start

        subset = filter_sundara_slokas_for_range(slokas_all, s_start, s_end)
        if not subset:
            print(f"No Sundara Kanda slokas for {part_name}: Sargas {s_start}-{s_end}")
            continue

        fallback_triplet = pick_begin_mid_end(subset)

        # Construct prompt and query model
        prompt = build_user_prompt(title, f"{s_start} to {s_end}", subset, fallback_triplet)
        llm_text = call_ollama(prompt)

        # Parse model output and coerce to schema
        llm_obj = extract_json_block(llm_text)
        final_obj = coerce_to_schema(llm_obj, subset, fallback_triplet, title)

        # Enforce non-empty fields with explanation-based fallbacks
        final_obj = enforce_schema_nonempty(final_obj, subset, fallback_triplet, title)

        # As an additional guard: if any shloka text is still empty, fill from fallback
        def fill_from_fb(sh_entry: dict, fb: Optional[Dict[str, Any]]):
            if fb and not sh_entry.get("text"):
                sh_entry["identifier"] = f"{fb.get('kanda','')} {fb.get('sarga','')}.{fb.get('shloka','')}"
                sh_entry["text"] = fb.get("shloka_text","")

        fb1, fb2, fb3 = fallback_triplet
        fill_from_fb(final_obj["shloka_1"], fb1)
        fill_from_fb(final_obj["shloka_2"], fb2)
        fill_from_fb(final_obj["shloka_3"], fb3)

        write_output(part_name, final_obj)

if __name__ == "__main__":
    main()
