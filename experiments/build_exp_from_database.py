# =============================================================================
#  STAGE 1 / 6 — PROMPT DATABASE  ->  EXPERIMENT LIST
#  file: experiments/build_exp_from_database.py
# -----------------------------------------------------------------------------
#  WHAT IT DOES : Flattens database.json (approach -> 7 tasks) into the flat list
#                 the pipeline reads, giving each prompt a unique stable id
#                 "<approach_slug>__<task>" (e.g. role_play__explosives) so audio
#                 caches never collide. Only appends genuinely new prompts.
#  INPUT        : experiments/database.json   (5 approaches x 7 tasks = 35)
#  OUTPUT       : experiments/exp1.json        (backup written to exp1.json.bak)
#  CALLED BY    : you, manually, whenever database.json changes.
#  RUN          : python experiments/build_exp_from_database.py [--dry-run]
#  NEEDS        : nothing (no API calls).
# -----------------------------------------------------------------------------
#  WORKFLOW (one prompt, end to end):
#
#   [1] database.json ──build_exp_from_database.py──> exp1.json        (prompt database)
#   [2] English prompt ──translate_prompts.py──> hi / gu / te / bn text (Gemini)
#   [3] text ──generate_tts.py──> audio_standard/<id>_<lang>.wav     (Sarvam TTS)
#   [4] .wav + English text ──gemma4_target.py──> model responses     (local Gemma 4)
#   [5] responses ──judge_responses.py──> outcome + approach labels   (Gemini judge)
#   [6] everything ──run_full_pipeline.py──> ultimate.json            (orchestrator)
#        languages.py = shared per-language config used by [2] [3] [5] [6]
#        batch_runner.py = the same steps, run step-by-step in resumable batches
#        api_keys.py = several keys per platform, switched automatically
#
# =============================================================================

"""Flatten database.json (approaches -> prompts) into the flat list that
run_full_pipeline.py reads (experiments/exp1.json).

Why this exists
---------------
In database.json the prompt ids 1-7 repeat inside every approach, but the
pipeline caches audio as audio/<id>_<lang>.wav. Using the raw ids would make
"Role-play #1" reuse the audio of "Instructional #1". This script gives every
prompt a unique, stable id: <approach_slug>__<task>, e.g. role_play__explosives.

Credit safety
-------------
* Items already in the output file are never touched or renamed, so their
  existing audio and translations stay valid.
* A prompt whose English text already exists in the output file keeps that
  item's old id instead of getting a new one (no accidental re-generation).
* Only prompts that are genuinely new are appended. Re-running the script is
  a no-op unless you add prompts to database.json.

Usage:
    python build_exp_from_database.py
    python build_exp_from_database.py --database database.json --out exp1.json --dry-run
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def flatten(database: dict) -> list:
    items = []
    for approach in database.get("approaches", []):
        a_name = approach.get("name", "unknown")
        a_slug = slugify(a_name)
        for p in approach.get("prompts", []):
            task = p.get("task", f"prompt_{p.get('id')}")
            items.append({
                "id": f"{a_slug}__{slugify(task)}",
                "approach": a_name,
                "task": task,
                "english_prompt": p["prompt"],
                "text_prompt": p["prompt"],
                "hindi_prompt": "",
                "gujarati_prompt": "",
                "telugu_prompt": "",
                "bengali_prompt": "",
            })
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database", default=str(CURRENT_DIR / "database.json"))
    ap.add_argument("--out", default=str(CURRENT_DIR / "exp1.json"))
    ap.add_argument("--dry-run", action="store_true", help="Show what would change, write nothing.")
    args = ap.parse_args()

    with open(args.database, "r", encoding="utf-8") as f:
        database = json.load(f)
    new_items = flatten(database)

    out_path = Path(args.out)
    existing = []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            print(f"Error: {out_path} is not a list.", file=sys.stderr)
            sys.exit(1)

    by_text = {}
    for it in existing:
        text = it.get("english_prompt") or it.get("text_prompt") or it.get("english_audio_prompt") or ""
        by_text[norm(text)] = it
    used_ids = {it.get("id") for it in existing}

    added, matched = [], []
    for it in new_items:
        hit = by_text.get(norm(it["english_prompt"]))
        if hit:
            # Same prompt already present: keep its id, just fill in metadata.
            matched.append(hit["id"])
            hit.setdefault("approach", it["approach"])
            hit.setdefault("task", it["task"])
            continue
        if it["id"] in used_ids:
            print(f"Warning: id '{it['id']}' already used by a different prompt; skipping.", file=sys.stderr)
            continue
        used_ids.add(it["id"])
        added.append(it)

    print(f"Prompts in database : {len(new_items)}")
    print(f"Already in {out_path.name:<10}: {len(matched)} (ids and audio reused)")
    print(f"New (need translation + audio): {len(added)}")
    for it in added:
        print(f"  + {it['id']}")

    if args.dry_run:
        print("\nDry run, nothing written.")
        return

    if out_path.exists():
        shutil.copyfile(out_path, out_path.with_suffix(".json.bak"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing + added, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(existing) + len(added)} items to {out_path}")


if __name__ == "__main__":
    main()