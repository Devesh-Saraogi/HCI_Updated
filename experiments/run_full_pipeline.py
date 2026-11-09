# =============================================================================
#  STAGE 6 / 6 — ORCHESTRATOR  (START HERE)
#  file: experiments/run_full_pipeline.py
# -----------------------------------------------------------------------------
#  WHAT IT DOES : Runs every stage for one prompt (--id), all prompts (--all) or a
#                 custom string (--text). For each prompt:
#                   Step 1  translate (stage 2) + TTS (stage 3) for each language
#                   Step 2  each .wav  -> Gemma 4 target (stage 4)
#                   Step 3  English text -> same target (text baseline)
#                   Step 4  judge all responses (stage 5)
#                   Step 5  save the combined record
#                 Cached audio, responses and judgements are reused when still
#                 valid, so re-runs and added languages don't repeat paid calls.
#  INPUT        : experiments/exp1.json (built by stage 1)
#  OUTPUT       : experiments/ultimate.json        (all records, merged by id)
#                 experiments/latest_ultimate.json (last record only)
#                 experiments/audio_standard/      (generated audio)
#  MODES        : no flags      -> MENU: pick / create a named RUN, then run one
#                                   step for the next N prompts, resumable
#                                   (see batch_runner.py; runs live in experiments/runs/)
#                 --run R --step X -> same, without the menu (for scripts)
#                 --id/--all/--text -> the original end-to-end run per prompt
#                                   (add --run R to store it in that run's folder)
#  RUN          : python experiments/run_full_pipeline.py                 (menu)
#                 python experiments/run_full_pipeline.py --run pilot --from-json experiments/exp2.json --step translate --count 50
#                 python experiments/run_full_pipeline.py --id role_play__explosives
#                 python experiments/run_full_pipeline.py --all --languages en,hi
#                 key flags: --gemma4-size E4B, --force-tts, --rerun-responses,
#                            --skip-judge, --skip-text-baseline, --sync-exp-json
#  NEEDS        : Gemini + Sarvam keys in .env (one or several each, see api_keys.py),
#                 Gemma 4 requirements.
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

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# Ensure console supports UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure experiments folder is on sys.path for local module imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "experiments" else CURRENT_DIR
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import sibling modules
try:
    import translate_prompts
    import generate_tts
    import judge_responses
    import languages
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Default Pipeline Configuration
# ---------------------------------------------------------------------------
DEFAULT_EXP_JSON = str(CURRENT_DIR / "exp1.json")
DEFAULT_ULTIMATE_JSON = str(CURRENT_DIR / "ultimate.json")
DEFAULT_LATEST_JSON = str(CURRENT_DIR / "latest_ultimate.json")
# New folder on purpose: the old "audio" folder holds audio made from the previous
# slang-style translations, which must not be reused.
DEFAULT_AUDIO_DIR = str(CURRENT_DIR / "audio_standard")

TRANSLATION_MODEL = "gemini-3.5-flash-lite"
TRANSLATION_FALLBACK = "gemini-3.5-flash-lite"
TTS_MODEL = "canopylabs/orpheus-arabic-saudi"
DEFAULT_TTS_VOICE = "fahad"
DEFAULT_TTS_PROVIDER = "sarvam"
DEFAULT_SARVAM_MODEL = "bulbul:v4-flash"
TARGET_SIZE = "E2B"
TARGET_MODEL = "google/gemma-4-E2B-it"
# ---------------------------------------------------------------------------


def load_environment():
    """Loads environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def post_with_retry(url: str, payload: dict, max_attempts: int = 6, timeout: int = 180):
    """
    POST that retries temporary Gemini failures (429 / 500 / 503 / 504) with
    exponential backoff, so 'model is busy' errors don't end up as results.
    """
    import time
    delay = 5
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            print(f"   network error ({e}); attempt {attempt}/{max_attempts}")
            response = None
        else:
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
            print(f"   HTTP {response.status_code} (temporary); attempt {attempt}/{max_attempts}")
        if attempt < max_attempts:
            print(f"   waiting {delay}s before retrying...")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    if response is None:
        raise RuntimeError("Network error: could not reach the Gemini API after several attempts.")
    return response


def query_target_audio_model(
    wav_file_path: str,
    model_name: str = TARGET_MODEL
) -> str:
    """
    Sends the audio file directly to the locally loaded Gemma 4 audio model.
    """
    import gemma4_target
    size = next((key for key, value in gemma4_target.MODEL_IDS.items() if value == model_name), None)
    if size is None:
        raise ValueError(f"Unsupported Gemma 4 model: {model_name}")
    print(f"-> Sending audio ({Path(wav_file_path).name}) directly to target model '{model_name}'...")
    return gemma4_target.query_audio(wav_file_path, size=size)


def _deep_merge(old: dict, new: dict) -> dict:
    out = dict(old)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_to_ultimate_json(record: dict, ultimate_json_path: str = DEFAULT_ULTIMATE_JSON):
    """Saves or updates the record in ultimate.json and latest_ultimate.json."""
    target_path = Path(ultimate_json_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    records = content
                elif isinstance(content, dict):
                    records = [content]
        except Exception:
            records = []

    # Update if ID exists, otherwise append
    rec_id = record.get("id")
    updated = False
    if rec_id:
        for idx, item in enumerate(records):
            if item.get("id") == rec_id:
                if item.get("translation_style") == record.get("translation_style"):
                    # Merge instead of replace, so running one language (--languages gu)
                    # does not erase the other languages' results already saved.
                    records[idx] = _deep_merge(item, record)
                    record = records[idx]
                else:
                    # Older record from the slang-style translations: replace it entirely.
                    records[idx] = record
                updated = True
                break
    if not updated:
        records.append(record)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"-> Saved into ultimate JSON: {target_path} (Total entries: {len(records)})")

    # Update latest_ultimate.json
    latest_path = Path(DEFAULT_LATEST_JSON)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


INVALID_RESPONSE_PREFIXES = ("Error ", "Failed to parse response", "Text baseline error")


def _is_valid_response(text) -> bool:
    """A stored response counts only if it is real model output (not an API error string)."""
    t = (text or "").strip()
    return bool(t) and not t.startswith(INVALID_RESPONSE_PREFIXES)


def _load_existing_record(ultimate_json_path: str, prompt_id: str) -> dict:
    """Previous record for this prompt, only if it was made with the current translation style."""
    p = Path(ultimate_json_path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            content = json.load(f)
    except Exception:
        return {}
    if isinstance(content, dict):
        content = [content]
    for rec in content if isinstance(content, list) else []:
        if rec.get("id") == prompt_id and rec.get("translation_style") == languages.TRANSLATION_STYLE:
            return rec
    return {}


def _speaker_for(cfg: dict, sarvam_model: str) -> str:
    if "v4" in (sarvam_model or ""):
        return cfg.get("sarvam_speaker_v4")
    return cfg["sarvam_speaker_v3"]


def run_pipeline_for_prompt(
    item: dict,
    voice: str = DEFAULT_TTS_VOICE,
    force_translate: bool = False,
    update_exp_json: bool = True,
    exp_json_path: str = DEFAULT_EXP_JSON,
    ultimate_json_path: str = DEFAULT_ULTIMATE_JSON,
    skip_text_baseline: bool = False,
    skip_judge: bool = False,
    language_codes: list = None,
    tts_provider: str = DEFAULT_TTS_PROVIDER,
    sarvam_model: str = DEFAULT_SARVAM_MODEL,
    force_tts: bool = False,
    rerun_responses: bool = False,
) -> dict:
    """
    End-to-end pipeline for one prompt. Test conditions:
      english text (baseline), english audio, hindi audio, gujarati audio, telugu audio, bengali audio

      1. For each language: get the text (English is used as is; the others are translated
         into standard language) and make the audio with Sarvam. Audio that already exists
         is reused, together with the text that was spoken in it (no credits spent).
      2. Send each audio file to the target Gemma 4 model
      3. Send the English text to the same Gemma 4 target (text baseline)
      4. Judge all responses
      5. Save the combined record to ultimate.json

    Earlier valid model responses and judgements are reused unless rerun_responses=True,
    so adding a language later does not repeat the others.
    """
    prompt_id = item.get("id", "custom_id")
    english_prompt = item.get("english_prompt") or item.get("text_prompt") or item.get("english_audio_prompt") or ""
    language_codes = language_codes or list(languages.PIPELINE_LANGUAGE_CODES)
    prior = _load_existing_record(ultimate_json_path, prompt_id)

    print("\n" + "=" * 70)
    print(f"PIPELINE: Processing Prompt [{prompt_id}]")
    print("=" * 70)
    print(f"Languages: {', '.join(language_codes)}")
    print(f"English Input:\n{english_prompt.strip()}\n")

    # ------------------------------------------------------------------
    # Step 1: text + audio for each language
    # ------------------------------------------------------------------
    audio_dir = Path(DEFAULT_AUDIO_DIR)
    audio_dir.mkdir(parents=True, exist_ok=True)

    translations, translation_models, audio_meta, json_updates = {}, {}, {}, {}
    language_errors = []
    active = []   # languages that got both text and audio

    for lang_code in language_codes:
        cfg = languages.get_language(lang_code)
        specific_wav = generate_tts.cached_audio_path(prompt_id, lang_code, str(audio_dir))
        cached_wav = None if force_tts else generate_tts.find_existing_audio(prompt_id, lang_code, str(audio_dir))
        translate = cfg.get("translate", True)

        try:
            # ---- text ----
            if cached_wav:
                text = generate_tts.read_spoken_text(cached_wav)
                if not text and translate:
                    text = languages.existing_translation(item, lang_code)
                if not translate:
                    text = english_prompt.strip()
                if not text:
                    print(f"Notice: cached {cfg['name']} audio has no saved text file; the text is not recorded.")
                if force_translate and translate:
                    print(f"Notice: {cfg['name']} audio is cached, so its text is kept. Use --force-tts to redo both.")
                model_used = "cached with audio"
            elif not translate:
                text, model_used = english_prompt.strip(), "none (English is spoken as is)"
            else:
                existing = None if force_translate else languages.existing_translation(item, lang_code)
                if existing:
                    print(f"-> [1/5] Using saved {cfg['name']} text from JSON.")
                    text, model_used = existing, "pre-existing"
                else:
                    print(f"-> [1/5] Translating to {cfg['name']} using Gemini ({TRANSLATION_MODEL})...")
                    text, model_used = translate_prompts.convert_to_language(
                        input_text=english_prompt, lang_code=lang_code, model_name=TRANSLATION_MODEL,
                    )
                    json_updates[cfg["json_field"]] = text

            # ---- audio ----
            if cached_wav:
                print(f"-> [1/5] Cached {cfg['name']} audio found, not calling TTS: {cached_wav.name}")
                wav_path, cached = cached_wav, True
            else:
                print(f"-> [1/5] Generating {cfg['name']} audio via {tts_provider}...")
                tts_model = sarvam_model if tts_provider == "sarvam" else TTS_MODEL
                wav_path = Path(generate_tts.text_to_speech(
                    text=text,
                    output_path=str(specific_wav),
                    voice=voice,
                    model=tts_model,
                    provider=tts_provider,
                    lang_code=lang_code,
                    skip_if_exists=not force_tts,
                    force=force_tts,
                ))
                generate_tts.write_spoken_text(wav_path, text)
                cached = False
        except Exception as e:
            msg = f"{lang_code}: {e}"
            language_errors.append(msg)
            print(f"ERROR for {cfg['name']}, skipping this language for this prompt: {e}", file=sys.stderr)
            continue

        translations[lang_code] = text
        translation_models[lang_code] = model_used
        audio_meta[lang_code] = {
            "file": wav_path.name,
            "path": str(wav_path),
            "cached": cached,
            "tts_provider": tts_provider,
            "tts_model": sarvam_model if tts_provider == "sarvam" else TTS_MODEL,
            "tts_language_code": cfg["sarvam_language_code"],
            "tts_voice": _speaker_for(cfg, sarvam_model) if tts_provider == "sarvam" else voice,
            "tts_size_bytes": wav_path.stat().st_size if wav_path.exists() else 0,
        }
        active.append(lang_code)

    if update_exp_json and json_updates and os.path.exists(exp_json_path):
        translate_prompts.update_json_language_prompts(exp_json_path, prompt_id, json_updates)

    # ------------------------------------------------------------------
    # Step 2: each audio file -> target Gemma 4 model
    # ------------------------------------------------------------------
    prior_responses = prior.get("audio_responses") or {}
    prior_texts = prior.get("translations") or {}
    audio_responses = {}
    reused_keys = set()
    for lang_code in active:
        cfg = languages.get_language(lang_code)
        can_reuse = (
            not rerun_responses
            and prior.get("target_model") == TARGET_MODEL
            and audio_meta[lang_code]["cached"]
            and _is_valid_response(prior_responses.get(lang_code))
            and prior_texts.get(lang_code) == translations[lang_code]
        )
        if can_reuse:
            print(f"-> [2/5] Reusing earlier {cfg['name']} audio response (use --rerun-responses to redo).")
            audio_responses[lang_code] = prior_responses[lang_code]
            reused_keys.add(f"{judge_responses.LANG_KEY[lang_code]}_audio")
            continue
        print(f"-> [2/5] Sending {cfg['name']} audio to target model '{TARGET_MODEL}'...")
        response_text = query_target_audio_model(audio_meta[lang_code]["path"], model_name=TARGET_MODEL)
        audio_responses[lang_code] = response_text
        print("\n" + "-" * 70)
        print(f"[{cfg['name'].upper()} AUDIO RESPONSE] {TARGET_MODEL}:")
        print("-" * 70)
        print(response_text)
        print("-" * 70 + "\n")

    # ------------------------------------------------------------------
    # Step 3: English text baseline -> the same Gemma 4 target.
    # ------------------------------------------------------------------
    text_baseline_response = ""
    baseline_reused = False
    if skip_text_baseline:
        print("-> [3/5] Skipping text baseline (--skip-text-baseline flag set).")
    elif (
        not rerun_responses
        and prior.get("target_model") == TARGET_MODEL
        and _is_valid_response(prior.get("text_baseline_response"))
    ):
        print("-> [3/5] Reusing earlier English text response (use --rerun-responses to redo).")
        text_baseline_response = prior["text_baseline_response"]
        baseline_reused = True
        reused_keys.add("english_text")
    else:
        print(f"-> [3/5] Sending English text to target model '{TARGET_MODEL}' (text baseline)...")
        import gemma4_target
        text_baseline_response = gemma4_target.query_text(english_prompt, size=TARGET_SIZE)
        print("\n" + "-" * 70)
        print(f"[TEXT BASELINE RESPONSE] {TARGET_MODEL}:")
        print("-" * 70)
        print(text_baseline_response)
        print("-" * 70 + "\n")

    # ------------------------------------------------------------------
    # Step 4: judge all responses
    # ------------------------------------------------------------------
    judgement = {}
    if skip_judge:
        print("-> [4/5] Skipping response judgement (--skip-judge flag set).")
    else:
        print(f"-> [4/5] Judging responses with judge model '{judge_responses.JUDGE_MODEL}'...")
        try:
            # Only earlier judgements whose response was reused may be reused.
            prior_mods = ((prior.get("judgement") or {}).get("modalities")) or {}
            interim = {
                "id": prompt_id,
                "english_prompt": english_prompt,
                "audio_responses": audio_responses,
                "text_baseline_response": text_baseline_response,
                "judgement": {"modalities": {k: v for k, v in prior_mods.items() if k in reused_keys}},
            }
            judgement = judge_responses.judge_record(interim, language_codes=active)
        except Exception as e:
            judgement = {"error": str(e)}
            print(f"Warning: Judgement step failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 5: build combined record and save to ultimate.json
    # ------------------------------------------------------------------
    print("-> [5/5] Saving combined record to ultimate.json...")
    record = {
        "id": prompt_id,
        "timestamp": datetime.now().isoformat(),
        "english_prompt": english_prompt,
        "translation_style": languages.TRANSLATION_STYLE,
        "translations": translations,
        "translation_models": translation_models,
        "audio": audio_meta,
        "target_model": TARGET_MODEL,
        "target_provider": "google-gemma-local",
        "audio_responses": audio_responses,
        "text_baseline_model": TARGET_MODEL,
        "judgement": judgement,
        "language_errors": language_errors,
        "status": "completed" if not language_errors else "partial",
    }
    for key in ("approach", "task"):
        if item.get(key):
            record[key] = item[key]
    for lang_code, text in translations.items():
        cfg = languages.get_language(lang_code)
        if cfg.get("translate", True):
            record[cfg["record_field"]] = text
    if not skip_text_baseline:
        record["text_baseline_response"] = text_baseline_response
        if not baseline_reused:
            record["text_baseline_timestamp"] = datetime.now().isoformat()

    save_to_ultimate_json(record, ultimate_json_path=ultimate_json_path)
    if language_errors:
        print("\nSome languages failed for this prompt: " + "; ".join(language_errors), file=sys.stderr)
    return record


def main():
    global TARGET_SIZE, TARGET_MODEL, DEFAULT_AUDIO_DIR, DEFAULT_LATEST_JSON
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline (5 steps per prompt):\n"
            "  1. English -> standard Hindi / Gujarati / Telugu / Bengali (Gemini); English is kept as is\n"
            "     then each language (incl. English) -> TTS audio (Sarvam; skips if WAV exists)\n"
            "  2. Each audio -> Target Gemma 4 model (local)\n"
            "  3. English text -> Target Gemma 4 model  [text baseline]\n"
            "  4. Judge all responses -> jailbreak approach + outcome category\n"
            "  5. Save combined record to ultimate.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from-json",
        type=str,
        default=DEFAULT_EXP_JSON,
        help=f"Path to experiment JSON file (default: {DEFAULT_EXP_JSON})"
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Prompt ID to process from JSON (e.g. w4hngc, uyfg91). If omitted with --all, all prompts are run."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL prompts in the specified JSON file sequentially."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom English text prompt to run through the entire pipeline (creates a standalone record)."
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=DEFAULT_TTS_VOICE,
        choices=["abdullah", "fahad", "sultan", "lulwa", "aisha", "noura"],
        help="Only used by the legacy Groq/Orpheus path. Sarvam voices are set per language in languages.py."
    )
    parser.add_argument(
        "--from-ultimate",
        action="store_true",
        help="Run pipeline using the prompts already configured in ultimate.json."
    )
    parser.add_argument(
        "--sync-exp-json",
        action="store_true",
        help="Also write the generated translations back into exp1.json (default is False to keep exp1.json clean)."
    )
    parser.add_argument(
        "--force-translate",
        action="store_true",
        help="Force re-translation even if a saved translation exists (audio that is already cached keeps its text)."
    )
    parser.add_argument(
        "--ultimate-json", "-u",
        type=str,
        default=DEFAULT_ULTIMATE_JSON,
        help=f"Path to output ultimate master JSON (default: {DEFAULT_ULTIMATE_JSON})"
    )
    parser.add_argument(
        "--skip-text-baseline",
        action="store_true",
        help="Skip the English text baseline."
    )
    parser.add_argument(
        "--gemma4-size",
        type=str,
        choices=["E2B", "E4B", "12B"],
        default="E2B",
        help="Local Gemma 4 target size (default: E2B, the smallest option).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the response judgement step (Step 5). No jailbreak category or outcome will be assigned."
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,hi,gu,te,bn",
        help="Comma-separated language codes to run: en,hi,gu,te,bn (default: all five)."
    )
    parser.add_argument(
        "--tts-provider",
        type=str,
        default=DEFAULT_TTS_PROVIDER,
        choices=["sarvam", "groq", "orpheus"],
        help="TTS provider (default: sarvam). groq/orpheus is the legacy Hindi-only path."
    )
    parser.add_argument(
        "--sarvam-model",
        type=str,
        default=DEFAULT_SARVAM_MODEL,
        help=f"Sarvam TTS model (default: {DEFAULT_SARVAM_MODEL}). Use bulbul:v3 for the older model (needs fresh audio)."
    )
    parser.add_argument(
        "--force-tts",
        action="store_true",
        help="Regenerate audio even if experiments/audio_standard/<id>_<lang>.wav already exists. Uses Sarvam credits."
    )

    batch = parser.add_argument_group(
        "batch mode (step by step, resumable; this is the default when no --id/--all/--text is given)")
    batch.add_argument("--run", type=str, default=None,
                       help="Run name: everything is stored in experiments/runs/<name>_run/. "
                            "Created from --from-json if it does not exist. Also usable with --id/--all.")
    batch.add_argument("--user", type=str, default=None,
                       help="Your name: recorded as creator of a new run and in each batch's history (default: OS user name).")
    batch.add_argument("--step", choices=["translate", "audio", "target", "judge"], default=None,
                       help="Run ONE step for the next --count ready prompts, without the menu.")
    batch.add_argument("--count", type=int, default=50,
                       help="Prompts per batch (default: 50).")
    batch.add_argument("--retry-failed", action="store_true",
                       help="With --step: retry prompts that failed this step earlier.")
    batch.add_argument("--gap", type=float, default=4.0,
                       help="Minimum seconds between API calls (default: 4).")
    batch.add_argument("--status", action="store_true",
                       help="With --run: print progress, done ranges, failed prompts and history, then exit.")
    batch.add_argument("--yes", action="store_true",
                       help="With --step audio: skip the Sarvam cost confirmation.")
    parser.add_argument(
        "--rerun-responses",
        action="store_true",
        help="Ask the target model and the judge again even if an earlier valid response exists."
    )

    args = parser.parse_args()
    import api_keys
    api_keys.install()          # several keys per platform from .env, switched on 429/quota
    import gemma4_target
    TARGET_SIZE = args.gemma4_size
    TARGET_MODEL = gemma4_target.MODEL_IDS[TARGET_SIZE]
    language_codes = languages.parse_language_list(args.languages)

    # Runs: every experiment lives in experiments/runs/<name>_run/ (see batch_runner.py).
    run_dir = None
    if args.user:
        import batch_runner
        batch_runner.USER_OVERRIDE = args.user
    if args.run:
        import batch_runner
        run_dir = batch_runner.run_dir_for(args.run)
        if not run_dir.exists():
            src = Path(args.from_json)
            if not src.is_absolute():
                src = src.resolve() if src.exists() else PROJECT_ROOT / src
            run_dir = batch_runner.create_run(
                args.run, src, created_by=args.user, langs=language_codes,
                sarvam_model=args.sarvam_model, gemma_size=args.gemma4_size)
            print(f"Created run {run_dir.name} from {src.name}")

    # Mode 0: batch / menu (default). One step at a time over the next N prompts.
    if args.step or args.status or not (args.text or args.id or args.all or args.from_ultimate):
        import batch_runner
        if args.status:
            if run_dir is None:
                print("Error: --status needs --run NAME.", file=sys.stderr)
                sys.exit(1)
            session = batch_runner.Session(run_dir, count=args.count, gap=args.gap)
            batch_runner.show_status(session)
            batch_runner.show_details(session)
            batch_runner.show_history(session)
            return
        if args.step and run_dir is None:
            print("Error: --step needs --run NAME (add --from-json to create a new run).", file=sys.stderr)
            sys.exit(1)
        session = batch_runner.Session(run_dir, count=args.count, gap=args.gap) if run_dir else None
        if args.step:
            batch_runner.run_batch(session, args.step, args.count,
                                   retry_failed=args.retry_failed, confirm=not args.yes)
        else:
            batch_runner.menu(session, count=args.count, gap=args.gap)
        return

    # --id / --all / --text inside a run: use the run's folder and its fixed settings.
    if run_dir is not None:
        import batch_runner
        info = batch_runner._read_json(run_dir / "run_info.json")
        st = info["settings"]
        DEFAULT_AUDIO_DIR = str(run_dir / "audio")
        DEFAULT_LATEST_JSON = str(run_dir / "latest.json")
        args.ultimate_json = str(run_dir / "results.json")
        if args.from_json == DEFAULT_EXP_JSON:                 # default -> the run's own prompts
            args.from_json = str(run_dir / info.get("dataset_file", "dataset.json"))
        args.sarvam_model = st["tts_model"]
        TARGET_SIZE, TARGET_MODEL = st["gemma_size"], st["target_model"]
        language_codes = [c for c in language_codes if c in st["languages"]] or list(st["languages"])
        print(f"Using run {run_dir.name}: results, audio and settings come from the run folder.")

    # Mode 1: Custom text
    if args.text:
        import hashlib
        custom_id = f"custom_{hashlib.md5(args.text.strip().encode('utf-8')).hexdigest()[:6]}"
        custom_item = {
            "id": custom_id,
            "text_prompt": args.text.strip(),
            "hindi_prompt": ""
        }
        run_pipeline_for_prompt(
            item=custom_item,
            voice=args.voice,
            force_translate=True,
            update_exp_json=False,
            ultimate_json_path=args.ultimate_json,
            skip_text_baseline=args.skip_text_baseline,
            skip_judge=args.skip_judge,
            language_codes=language_codes,
            tts_provider=args.tts_provider,
            sarvam_model=args.sarvam_model,
            force_tts=args.force_tts,
            rerun_responses=args.rerun_responses,
        )
        return

    # Mode 2: From ultimate.json or exp1.json
    if args.from_ultimate:
        source_path = Path(args.ultimate_json)
    else:
        source_path = Path(args.from_json)

    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    if not source_path.exists():
        print(f"Error: JSON file not found at {source_path}", file=sys.stderr)
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Error: Expected a list of prompts in {source_path}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        print(f"Starting full pipeline for ALL {len(data)} prompts from {source_path}...")
        for item in data:
            try:
                run_pipeline_for_prompt(
                    item=item,
                    voice=args.voice,
                    force_translate=args.force_translate,
                    update_exp_json=args.sync_exp_json,
                    exp_json_path=str(source_path),
                    ultimate_json_path=args.ultimate_json,
                    skip_text_baseline=args.skip_text_baseline,
                    skip_judge=args.skip_judge,
                    language_codes=language_codes,
                    tts_provider=args.tts_provider,
                    sarvam_model=args.sarvam_model,
                    force_tts=args.force_tts,
                    rerun_responses=args.rerun_responses,
                )
            except Exception as e:
                print(f"Error processing prompt {item.get('id')}: {e}", file=sys.stderr)
        print("\nFull pipeline completed! Results saved to:", args.ultimate_json)

    elif args.id:
        target_item = None
        for item in data:
            if item.get("id") == args.id:
                target_item = item
                break
        if not target_item:
            print(f"Error: Prompt ID '{args.id}' not found in {source_path}", file=sys.stderr)
            sys.exit(1)

        run_pipeline_for_prompt(
            item=target_item,
            voice=args.voice,
            force_translate=args.force_translate,
            update_exp_json=args.sync_exp_json,
            exp_json_path=str(source_path),
            ultimate_json_path=args.ultimate_json,
            skip_text_baseline=args.skip_text_baseline,
            skip_judge=args.skip_judge,
            language_codes=language_codes,
            tts_provider=args.tts_provider,
            sarvam_model=args.sarvam_model,
            force_tts=args.force_tts,
            rerun_responses=args.rerun_responses,
        )


if __name__ == "__main__":
    main()
