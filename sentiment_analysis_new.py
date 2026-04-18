import csv
import json
import os
import re
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()
print("API KEY LOADED:", os.getenv("ANTHROPIC_API_KEY")[:20], "...")

# ── CONFIGURATION ─────────────────────────────────────────────
INPUT_FOLDER = "mediation_files"
OUTPUT_FILE = "coded_results_new.csv"
CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds to wait before retrying on rate limit / overload
api_key = os.getenv("ANTHROPIC_API_KEY")

# ── SYSTEM PROMPT ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert political scientist specializing in conflict mediation and peace studies. Your task is to code official state statements using a binary variable derived from Johan Galtung's distinction between positive and negative peace, reoperationalized for international mediation contexts.
 
You will be given the full text of an official statement. Some statements include metadata such as a date, speaker, or conflict label at the top — treat that as context only and base your coding exclusively on the language of the statement itself.
 
---
 
THE VARIABLE
"In this statement, did the weight of the language concerning the country's mediation's aims and goals lean significantly towards positive or negative peace?"

Codebook: This study operationalizes mediator strategic orientation as a binary variable derived from Johan Galtung's (1969) distinction between positive and negative peace, reoperationalized for the context of international mediation. 

Meaning of positive peace orientation: Does the statement attach political, governance, or ideological conditions to the country's mediation goals or aims: electoral outcomes, transitional government arrangements, governance reform, ideological prerequisites, justice requirements, power-sharing demands? Is the mediator seeking to transform the structural conditions of the conflict, not merely to end violence?
 
CODE 0 — Positive peace logic (ideology-oriented mediation)
Assign 0 when the dominant language of the statement:
  - Advances a specific governance model in its mediation aim (democratic institutions, civilian-led government, liberal political order)
  - Describes the goal of the country's mediation as outcomes aligned with the country's own political values or foreign policy identity
  - Framing mediation outcomes in terms of the mediator's own values (human rights, self-determination, normalization)
  - Explicitly supporting the value of a conflict party's ideology in a mediation outcome, rather than being willing to engage with all involved parties
 
Meaning of negative peace orientation: This statement describes the mediator's goals in security-only terms: ceasefire, cessation of hostilities, humanitarian access, stability, de-escalation, prisoner exchange, safe passage. The mediator is seeking to end violence without requiring political transformation.

CODE 1 — Negative peace logic (stability-oriented mediation)
Assign 1 when the dominant mediation rationale focuses on stopping violence without conditioning engagement on ideological transformation. Look for:
  - Ceasefire, cessation of hostilities, or de-escalation as the primary stated mediation goal
  - Overall focus on security or humanitarian objectives that do not prescribe a specific political outcome
  - Non-interference in the internal governance or political identity of a conflict party
  - Framing mediation as a procedural tool to end violence rather than a vehicle for political transformation
 
---
 
DECISION RULES
 
Mixed texts: When both logics appear, assign the code reflecting the DOMINANT rationale:
  1. Weight of language across the full text — which logic appears more frequently or emphatically?
  2. Whether ideological conditions are attached to the mediation offer or goal itself, or appear only as background context
 
Metadata: If the text contains headers, speaker labels, or dates before the statement body, ignore them for coding purposes. Code only the statement language.
 
Multi-conflict statements: If the text addresses multiple conflicts, code only the language describing mediation philosophy. If the conflicts are so intertwined you cannot isolate a dominant logic, set confidence below 0.70 and explain in your rationale.
 
Short or vague statements: If the text is too brief or non-specific to code reliably, make your best-effort assignment based on whatever language is present, set confidence below 0.70, and note the limitation in your rationale.
 
---
 
OUTPUT FORMAT
Respond ONLY with a JSON object in exactly this structure. No preamble, no explanation outside the JSON, no markdown code fences:
{
  "code": 0 or 1,
  "confidence": a number between 0.0 and 1.0,
  "rationale": "2-3 sentences identifying the dominant indicators and explaining your assignment. If the text was vague, multi-conflict, or genuinely mixed, note that here."
}"""

# ── SETUP ──────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=api_key)

files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
print(f"Found {len(files)} files in '{INPUT_FOLDER}'\n")


# ── HELPERS ────────────────────────────────────────────────────
def parse_filename(filename):
    """
    Expected format: "29_Iran_Israel ceasefire_US.txt"
    First segment (before first underscore) is the index number.
    Last segment (after last underscore) is the mediator.
    Everything in between is the conflict name.
    """
    base = filename.replace(".txt", "")
    parts = base.split("_")
    index_num = parts[0]
    mediator = parts[-1]
    conflict = "_".join(parts[1:-1])
    return index_num, conflict, mediator


def read_file(filepath):
    """Read file with UTF-8, falling back to latin1 with a warning."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip(), None
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin1") as f:
            text = f.read().strip()
        return text, "WARNING: file decoded as latin1 — check for encoding artifacts"


def call_claude_with_retry(text):
    """Call the Claude API with retry logic for rate limits and overload errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                temperature=0.1,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            return response
        except anthropic.RateLimitError as e:
            if attempt < MAX_RETRIES:
                print(
                    f"    Rate limit hit — waiting {RETRY_DELAY}s before retry {attempt}/{MAX_RETRIES - 1}..."
                )
                time.sleep(RETRY_DELAY)
            else:
                raise e
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < MAX_RETRIES:
                print(
                    f"    API overloaded — waiting {RETRY_DELAY}s before retry {attempt}/{MAX_RETRIES - 1}..."
                )
                time.sleep(RETRY_DELAY)
            else:
                raise e


def parse_claude_response(raw):
    """Strip markdown fences if present, then parse JSON."""
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE
    ).strip()
    return json.loads(cleaned)


# ── MAIN LOOP ──────────────────────────────────────────────────
results = []

for i, filename in enumerate(files):  # remove [:4] slice — runs all files
    index_num, conflict, mediator = parse_filename(filename)

    label = f"[{i + 1}/{len(files)}] {conflict} | {mediator}"
    print(label)

    filepath = os.path.join(INPUT_FOLDER, filename)
    text, encoding_warning = read_file(filepath)

    if encoding_warning:
        print(f"    {encoding_warning}")

    text = " ".join(text.split())

    if not text:
        print("    SKIPPED — file is empty\n")
        results.append(
            {
                "filename": filename,
                "index_num": index_num,
                "conflict": conflict,
                "mediator": mediator,
                "claude_code": "",
                "claude_confidence": "",
                "claude_flag": "NO_TEXT",
                "claude_rationale": "File was empty",
            }
        )
        continue

    try:
        time.sleep(1)
        response = call_claude_with_retry(text)

        raw = response.content[0].text.strip()
        result = parse_claude_response(raw)

        code = result["code"]
        confidence = result["confidence"]
        rationale = result["rationale"]
        flag = "REVIEW" if confidence < CONFIDENCE_THRESHOLD else ""

        results.append(
            {
                "filename": filename,
                "index_num": index_num,
                "conflict": conflict,
                "mediator": mediator,
                "claude_code": code,
                "claude_confidence": confidence,
                "claude_flag": flag,
                "claude_rationale": rationale,
            }
        )

        print(
            f"    Code: {code}  |  Confidence: {confidence}  {'[FLAGGED FOR REVIEW]' if flag else ''}"
        )
        print(f"    {rationale}\n")

    except json.JSONDecodeError as e:
        print(f"    ERROR: Claude returned non-JSON for {filename}: {e}\n")
        results.append(
            {
                "filename": filename,
                "index_num": index_num,
                "conflict": conflict,
                "mediator": mediator,
                "claude_code": "",
                "claude_confidence": "",
                "claude_flag": "ERROR",
                "claude_rationale": f"JSON parse error: {e}",
            }
        )

    except Exception as e:
        import traceback

        print(f"    ERROR on {filename}: {e}\n")
        traceback.print_exc()
        results.append(
            {
                "filename": filename,
                "index_num": index_num,
                "conflict": conflict,
                "mediator": mediator,
                "claude_code": "",
                "claude_confidence": "",
                "claude_flag": "ERROR",
                "claude_rationale": str(e),
            }
        )

# ── SAVE ───────────────────────────────────────────────────────
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "filename",
        "index_num",
        "conflict",
        "mediator",
        "claude_code",
        "claude_confidence",
        "claude_flag",
        "claude_rationale",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Done. Results saved to {OUTPUT_FILE}")

# ── QUICK SUMMARY ──────────────────────────────────────────────
total = len(results)
coded = sum(1 for r in results if r["claude_code"] != "")
code0 = sum(1 for r in results if r["claude_code"] == 0)
code1 = sum(1 for r in results if r["claude_code"] == 1)
reviews = sum(1 for r in results if r["claude_flag"] == "REVIEW")
errors = sum(1 for r in results if r["claude_flag"] == "ERROR")

print("\n── Summary ──────────────────")
print(f"  Total files:             {total}")
print(f"  Successfully coded:      {coded}")
print(f"  Code 0 (positive peace): {code0}")
print(f"  Code 1 (negative peace): {code1}")
print(f"  Flagged for review:      {reviews}")
print(f"  Errors:                  {errors}")
