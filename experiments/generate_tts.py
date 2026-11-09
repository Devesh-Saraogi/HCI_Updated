# =============================================================================
#  STAGE 3 / 6 — TEXT-TO-SPEECH  (text -> .wav)
#  file: experiments/generate_tts.py
# -----------------------------------------------------------------------------
#  WHAT IT DOES : Turns each language's text into a WAV with Sarvam Bulbul (v4-flash
#                 by default; long text is split and stitched). Saves the spoken text
#                 next to the audio as <id>_<lang>.txt. Existing WAVs are REUSED,
#                 so no Sarvam credits are spent twice (override: --force-tts).
#                 The Groq/Orpheus provider is a legacy Hindi-only path.
#  INPUT        : text + lang code (from translate_prompts.py / exp1.json).
#  OUTPUT       : experiments/audio_standard/<id>_<lang>.wav + .txt
#  CALLED BY    : run_full_pipeline.py, Step 1 (text_to_speech, cached_audio_path,
#                 find_existing_audio, read/write_spoken_text).
#  RUN ALONE    : python experiments/generate_tts.py --help
#  NEEDS        : SARVAM_API_KEY in .env  (GROQ key only for the legacy path)
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
import io
import json
import wave
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv
import requests

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from languages import get_language

# Reconfigure console encoding for Windows terminals to display Hindi characters without crashing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration Block
# ---------------------------------------------------------------------------
# Edit this line with your desired "Chalu Hindi" (colloquial street Hindi) text:
DEFAULT_CHALU_HINDI_LINE = (
    "ये एक कंट्रोल सिक्योरिटी टेस्ट है।"
)

# Groq / Orpheus (legacy Hindi path)
MODEL_NAME = "canopylabs/orpheus-arabic-saudi"
DEFAULT_VOICE = "abdullah"

# Sarvam (default for every pipeline language). Free-tier credits are limited — never
# regenerate if a non-empty WAV already exists unless force=True.
DEFAULT_TTS_PROVIDER = "sarvam"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_MODEL = "bulbul:v4-flash"
SARVAM_MAX_CHARS = 2400

# Output audio file path
DEFAULT_OUTPUT_FILE = "experiments/tts.wav"
# ---------------------------------------------------------------------------


def get_project_root() -> Path:
    """Finds the root directory containing .env or experiments folder."""
    current = Path(__file__).resolve().parent
    if (current / ".env").exists() or (current / "experiments").exists():
        return current
    if (current.parent / ".env").exists():
        return current.parent
    return current


def load_environment():
    """Loads .env file from project root or parent directories."""
    root = get_project_root()
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def cached_audio_path(prompt_id: str, lang_code: str, audio_dir: str) -> Path:
    """Canonical on-disk path: experiments/audio/<id>_<lang>.wav"""
    return Path(audio_dir) / f"{prompt_id}_{lang_code}.wav"


def find_existing_audio(prompt_id: str, lang_code: str, audio_dir: str):
    """
    Return an existing non-empty audio file for this prompt+language, or None.
    Only the exact <id>_<lang> name counts. (Older Hindi names like <id>.wav or tts.wav
    came from the previous slang-style translations and must not be reused.)
    """
    audio_dir = Path(audio_dir)
    for name in (f"{prompt_id}_{lang_code}.wav", f"{prompt_id}_{lang_code}.mp3"):
        path = audio_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def spoken_text_path(wav_path) -> Path:
    """Sidecar file holding the exact text that was spoken in this audio file."""
    return Path(wav_path).with_suffix(".txt")


def write_spoken_text(wav_path, text: str) -> None:
    spoken_text_path(wav_path).write_text((text or "").strip(), encoding="utf-8")


def read_spoken_text(wav_path) -> str:
    p = spoken_text_path(wav_path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _split_text_for_sarvam(text: str, max_chars: int = SARVAM_MAX_CHARS) -> list:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    remaining = text
    separators = ["।", "॥", ".", "?", "!", "\n", ",", " "]
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break
        window = remaining[:max_chars]
        cut = -1
        for sep in separators:
            idx = window.rfind(sep)
            if idx > cut:
                cut = idx
        if cut < 20:
            cut = max_chars
        else:
            cut = cut + 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [c for c in chunks if c]


def _concat_wav_bytes(wav_blobs: list, output_path: Path) -> None:
    """Concatenate one or more WAV blobs into a single file."""
    if len(wav_blobs) == 1:
        output_path.write_bytes(wav_blobs[0])
        return
    frames = []
    params = None
    for blob in wav_blobs:
        with wave.open(io.BytesIO(blob), "rb") as src:
            if params is None:
                params = src.getparams()
            frames.append(src.readframes(src.getnframes()))
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for block in frames:
            out.writeframes(block)


def text_to_speech_sarvam(
    text: str,
    output_path: str,
    lang_code: str = "hi",
    model: str = SARVAM_MODEL,
    speaker: str = None,
) -> str:
    """Convert text to WAV via Sarvam Bulbul. Requires SARVAM_API_KEY."""
    load_environment()
    api_key = os.environ.get("SARVAM_API_KEY") or os.environ.get("SARVAM_SUBSCRIPTION_KEY")
    if not api_key:
        raise ValueError(
            "SARVAM_API_KEY not found in environment or .env file. "
            "Add SARVAM_API_KEY=... to .env (Sarvam dashboard subscription key)."
        )

    cfg = get_language(lang_code)
    language_bcp47 = cfg["sarvam_language_code"]
    if not speaker:
        if "v4" in (model or ""):
            speaker = cfg.get("sarvam_speaker_v4")
            if not speaker:
                # v4 rejects v3 short names, so never fall back to them.
                raise ValueError(f"No bulbul v4 speaker set for {cfg['name']} in languages.py.")
        else:
            speaker = cfg["sarvam_speaker_v3"]

    root = get_project_root()
    target_path = Path(output_path)
    if not target_path.is_absolute():
        if Path.cwd().name == "experiments" and target_path.parts[0] == "experiments":
            target_path = Path.cwd() / target_path.name
        else:
            target_path = root / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = _split_text_for_sarvam(text)
    if not chunks:
        raise ValueError("TTS text is empty.")

    print("=" * 60)
    print("Sarvam TTS")
    print("=" * 60)
    print(f"Model:       {model}")
    print(f"Language:    {language_bcp47} ({cfg['name']})")
    print(f"Speaker:     {speaker}")
    print(f"Characters:  {len(text)}  chunks={len(chunks)}")
    print(f"Output File: {target_path}")
    print("-" * 60)

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    wav_blobs = []
    for i, chunk in enumerate(chunks, start=1):
        payload = {
            "text": chunk,
            "language_code": language_bcp47,
            "speaker": speaker,
            "model": model,
            "output_audio_codec": "wav",
            "speech_sample_rate": 22050,
        }
        if "v4" in (model or ""):
            payload["enable_preprocessing"] = True   # recommended by Sarvam for v4
        print(f"Sending chunk {i}/{len(chunks)} ({len(chunk)} chars) to Sarvam...")
        # Retry only rate limits / temporary server errors (a failed request is not billed
        # audio). A 4xx such as a bad speaker name or exhausted credits stops right away.
        response = None
        for attempt in range(1, 4):
            response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=120)
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            wait = 5 * attempt
            print(f"   Sarvam HTTP {response.status_code} (temporary), retrying in {wait}s ({attempt}/3)...")
            import time
            time.sleep(wait)
        if response.status_code != 200:
            raise RuntimeError(
                f"Sarvam TTS failed (HTTP {response.status_code}): {response.text}"
            )
        data = response.json()
        audios = data.get("audios") or []
        if not audios:
            raise RuntimeError(f"Sarvam TTS returned no audio: {data}")
        wav_blobs.append(base64.b64decode(audios[0]))

    try:
        _concat_wav_bytes(wav_blobs, target_path)
    except Exception:
        # If WAV concat fails (e.g. API returned mp3), write the first blob as-is.
        target_path.write_bytes(wav_blobs[0])

    file_size_bytes = target_path.stat().st_size
    print(f"Successfully generated audio file: {target_path}")
    print(f"File Size:   {file_size_bytes / 1024.0:.2f} KB ({file_size_bytes:,} bytes)")
    print("=" * 60)
    return str(target_path)


def _text_to_speech_groq(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    model: str = MODEL_NAME,
) -> str:
    load_environment()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment or .env file. "
            "Please ensure GROQ_API_KEY is defined in your .env file."
        )

    root = get_project_root()
    target_path = Path(output_path)
    if not target_path.is_absolute():
        if Path.cwd().name == "experiments" and target_path.parts[0] == "experiments":
            target_path = Path.cwd() / target_path.name
        else:
            target_path = root / target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Orpheus TTS Configuration")
    print("=" * 60)
    print(f"Model:       {model}")
    print(f"Voice:       {voice}")
    print(f"Characters:  {len(text)}")
    print(f"Input Text:  {text}")
    print(f"Output File: {target_path}")
    print("-" * 60)
    print("Sending request to Groq Speech API...")

    from groq import Groq  # only needed for the legacy Groq path
    client = Groq(api_key=api_key)
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice.lower(),
            input=text,
            response_format="wav"
        )
        response.write_to_file(str(target_path))
        file_size_bytes = target_path.stat().st_size
        print(f"Successfully generated audio file: {target_path}")
        print(f"File Size:   {file_size_bytes / 1024.0:.2f} KB ({file_size_bytes:,} bytes)")
        print("=" * 60)
        return str(target_path)
    except Exception as e:
        print(f"Error generating TTS audio: {e}", file=sys.stderr)
        raise


def text_to_speech(
    text: str,
    output_path: str = DEFAULT_OUTPUT_FILE,
    voice: str = DEFAULT_VOICE,
    model: str = None,
    provider: str = DEFAULT_TTS_PROVIDER,
    lang_code: str = "hi",
    skip_if_exists: bool = True,
    force: bool = False,
) -> str:
    """
    Convert text to speech. Default provider is Sarvam.

    If skip_if_exists is True (default) and output_path already has audio,
    the existing file is reused and no API credits are spent.
    """
    root = get_project_root()
    target_path = Path(output_path)
    if not target_path.is_absolute():
        if Path.cwd().name == "experiments" and target_path.parts[0] == "experiments":
            target_path = Path.cwd() / target_path.name
        else:
            target_path = root / target_path

    if skip_if_exists and not force and target_path.exists() and target_path.stat().st_size > 0:
        print(f"-> Reusing cached audio (skipping TTS API): {target_path} ({target_path.stat().st_size:,} bytes)")
        return str(target_path)

    provider = (provider or DEFAULT_TTS_PROVIDER).lower()
    if provider == "sarvam":
        return text_to_speech_sarvam(
            text=text,
            output_path=str(target_path),
            lang_code=lang_code,
            model=model or SARVAM_MODEL,
            speaker=voice if voice and voice not in {
                "abdullah", "fahad", "sultan", "lulwa", "aisha", "noura"
            } else None,
        )
    if provider in ("groq", "orpheus"):
        return _text_to_speech_groq(
            text=text,
            output_path=str(target_path),
            voice=voice or DEFAULT_VOICE,
            model=model or MODEL_NAME,
        )
    raise ValueError(f"Unknown TTS provider '{provider}'. Use 'sarvam' or 'groq'.")


def load_text_from_json(json_file_path: str, prompt_id: str = None, prompt_key: str = "hindi_prompt") -> str:
    """Helper to extract a Hindi prompt from a JSON experiment file (e.g., exp1.json)."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if prompt_id:
            for item in data:
                if item.get("id") == prompt_id:
                    return item.get(prompt_key, "")
            raise ValueError(f"Prompt with id '{prompt_id}' not found in {json_file_path}")
        else:
            # Pick first non-empty item
            for item in data:
                if item.get(prompt_key):
                    return item.get(prompt_key)
            raise ValueError(f"No entry with non-empty '{prompt_key}' found in {json_file_path}")
    elif isinstance(data, dict):
        if prompt_id and prompt_id in data:
            return data[prompt_id].get(prompt_key, "")
        return data.get(prompt_key, "")

    raise ValueError(f"Unable to parse format of {json_file_path}")


def update_ultimate_json_with_tts(audio_path: str, voice: str, model: str, text: str):
    """Updates ultimate.json with the generated audio details."""
    root = get_project_root()
    ultimate_path = root / "experiments" / "ultimate.json"
    latest_path = root / "experiments" / "latest_ultimate.json"
    
    file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else None

    # Update ultimate.json
    if ultimate_path.exists():
        try:
            with open(ultimate_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, list) and records:
                # Find matching record or take latest
                target_rec = records[-1]
                for rec in reversed(records):
                    if rec.get("chalu_hindi") == text:
                        target_rec = rec
                        break
                target_rec["audio_file"] = str(audio_path)
                target_rec["tts_model"] = model
                target_rec["tts_voice"] = voice
                target_rec["tts_generated"] = True
                target_rec["tts_size_bytes"] = file_size
                target_rec["status"] = "tts_completed"
                with open(ultimate_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Notice: Could not update ultimate.json: {e}")

    # Update latest_ultimate.json
    if latest_path.exists():
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            rec["audio_file"] = str(audio_path)
            rec["tts_model"] = model
            rec["tts_voice"] = voice
            rec["tts_generated"] = True
            rec["tts_size_bytes"] = file_size
            rec["status"] = "tts_completed"
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def main():
    import api_keys
    api_keys.install()          # several keys per platform, switched on 429/quota
    parser = argparse.ArgumentParser(
        description="Convert translated text to WAV. Default provider is Sarvam (cached if the WAV already exists)."
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Textual Chalu Hindi line to convert"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Destination WAV file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=DEFAULT_VOICE,
        choices=["abdullah", "fahad", "sultan", "lulwa", "aisha", "noura"],
        help=f"Voice persona to use (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "--from-file", "-f",
        type=str,
        default=None,
        help="Path to text file containing Chalu Hindi line (e.g. experiments/chalu_hindi.txt)"
    )
    parser.add_argument(
        "--from-ultimate", "-u",
        action="store_true",
        help="Load the latest Chalu Hindi prompt directly from experiments/ultimate.json"
    )
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="Path to experiment JSON file (e.g. experiments/exp1.json) to read the prompt from"
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Prompt ID to read when using --from-json (e.g. w4hngc)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_TTS_PROVIDER,
        choices=["sarvam", "groq", "orpheus"],
        help="TTS provider (default: sarvam). groq/orpheus is the legacy Hindi path."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=["en", "hi", "gu", "te", "bn"],
        help="Language of the input text (en, hi, gu, te, bn). Used for Sarvam language_code."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate audio even if the WAV already exists (uses Sarvam credits)."
    )
    parser.add_argument(
        "--sarvam-model",
        type=str,
        default=SARVAM_MODEL,
        help=f"Sarvam model id (default: {SARVAM_MODEL}). Use bulbul:v3 for the older model."
    )

    args = parser.parse_args()

    # Determine input text
    text = None
    if args.from_json:
        text = load_text_from_json(args.from_json, prompt_id=args.id)
    elif args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.from_ultimate:
        root = get_project_root()
        latest_path = root / "experiments" / "latest_ultimate.json"
        if latest_path.exists():
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("chalu_hindi", "")
    elif args.text:
        text = args.text
    else:
        # Check if ultimate.json has a prompt ready
        root = get_project_root()
        latest_path = root / "experiments" / "latest_ultimate.json"
        if latest_path.exists():
            try:
                with open(latest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("chalu_hindi"):
                    text = data.get("chalu_hindi")
                    print("Notice: Automatically loaded latest Chalu Hindi from experiments/latest_ultimate.json")
            except Exception:
                pass

        if not text:
            text = DEFAULT_CHALU_HINDI_LINE

    if not text.strip():
        print("Error: Input text is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        tts_model = args.sarvam_model if args.provider == "sarvam" else MODEL_NAME
        out_wav = text_to_speech(
            text=text,
            output_path=args.output,
            voice=args.voice,
            model=tts_model,
            provider=args.provider,
            lang_code=args.lang,
            skip_if_exists=not args.force,
            force=args.force,
        )
        update_ultimate_json_with_tts(out_wav, args.voice, MODEL_NAME, text)
    except Exception as exc:
        print(f"Failed to create speech file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()