# =============================================================================
#  BATCH RUNNER — named runs, step-by-step, resumable, menu driven
#  file: experiments/batch_runner.py        (opened by run_full_pipeline.py)
# -----------------------------------------------------------------------------
#  WHAT IT DOES : Every experiment is a RUN with its own folder:
#
#                   experiments/runs/<name>_run/
#                     run_info.json   who created it, when, which dataset, and the
#                                     FIXED settings (languages, TTS model + voices,
#                                     Gemma size, translation + judge models), plus a
#                                     history of every batch (user, step, counts)
#                     dataset.json    frozen copy of the prompts (later edits to
#                                     exp*.json don't change an existing run)
#                     results.json    one record per prompt (same format as before)
#                     audio/          <id>_<lang>.wav + .txt for THIS run
#                     logs/           detailed output of every batch
#
#                 Inside a run you execute ONE step for the NEXT N prompts, all 5
#                 languages of each prompt together, one request at a time, and come
#                 back later for the next N. Each step continues where it stopped.
#                   1 Translate  en -> hi / gu / te / bn         (Gemini)
#                   2 Audio      text -> <run>/audio/*.wav       (Sarvam, costs credits)
#                   3 Target     5 audio clips + English text -> Gemma 4 (local)
#                   4 Judge      every response -> outcome label (Gemini)
#                 A prompt is "ready" for a step once the previous step is done.
#
#  SAFETY       : - results saved after EVERY prompt (atomic write)
#                 - API calls spaced by a minimum gap (default 4 s, editable)
#                 - rate limit / quota / credits / missing key -> batch STOPS; run
#                   the step again later and it resumes
#                 - other errors mark the prompt "failed" for that step (skipped by
#                   "next N", retry with r)
#                 - audio shows the estimated Sarvam cost and asks first
#                 - a .lock file stops two people running batches on one run at once
#                 - several API keys per platform: a rate-limited / exhausted key is
#                   swapped for the next one automatically (api_keys.py)
#  RUN          : python experiments/run_full_pipeline.py                      (menu)
#                 python experiments/run_full_pipeline.py --run pilot_run --step translate --count 50
#                 python experiments/run_full_pipeline.py --run pilot_run \
#                        --from-json experiments/exp2.json --step translate      (creates the run)
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

import contextlib
import getpass
import json
import os
import re
import shutil
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import api_keys
import languages
import translate_prompts
import generate_tts
import judge_responses
import run_full_pipeline as rfp

RUNS_DIR = CURRENT_DIR / "runs"
STEPS = ["translate", "audio", "target", "judge"]
STEP_LABEL = {
    "translate": "Translate (Gemini)",
    "audio":     "Audio (Sarvam TTS)",
    "target":    "Target (Gemma 4)",
    "judge":     "Judge (Gemini)",
}
STEP_NEEDS = {"translate": None, "audio": "translate", "target": "audio", "judge": "target"}

# Sarvam price per 1,000 characters (INR, before GST), used only for the estimate.
SARVAM_RATE_INR = {"bulbul:v4-flash": 1.0, "bulbul:v3": 3.0}

# Errors that should STOP a batch (retrying the next prompt would fail the same way)
# instead of marking the prompt as failed.
RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "quota", "rate limit",
                      "insufficient", "credit", "api_key not found",
                      "not found in environment")
CONSOLE = sys.__stdout__


class RateLimited(Exception):
    """Rate limit / quota / credits / missing API key: stop the batch, resume later."""


def _is_rate_limit(message) -> bool:
    m = str(message).lower()
    return any(k in m for k in RATE_LIMIT_MARKERS)


def say(text=""):
    print(text, file=CONSOLE, flush=True)


def confirm_yes(question: str) -> bool:
    """y/N question; no keyboard (script, closed input) counts as No."""
    try:
        return input(question).strip().lower() == "y"
    except EOFError:
        say("")
        return False


USER_OVERRIDE = None   # set by --user or the new-run wizard; recorded in run history


def current_user() -> str:
    if USER_OVERRIDE:
        return USER_OVERRIDE
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Request pacing: every requests.post in this process (translation, translation
# check, Sarvam, judge) waits until at least `gap` seconds have passed since the
# previous one. Gemma runs locally and is not affected.
# ---------------------------------------------------------------------------
class _Throttle:
    def __init__(self):
        self.gap = 4.0
        self.last = 0.0
        self.calls = 0
        self._orig = None

    def install(self):
        if self._orig is None:
            self._orig = requests.post
            requests.post = self._post

    def _post(self, *args, **kwargs):
        wait = self.gap - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        self.calls += 1
        resp = self._orig(*args, **kwargs)
        if getattr(resp, "status_code", 200) in (429, 500, 502, 503, 504):
            say(f"      (API busy: HTTP {resp.status_code}, backing off and retrying...)")
        return resp


THROTTLE = _Throttle()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path, default=None):
    if not path.exists() or path.stat().st_size == 0:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data):
    """Atomic write: a crash mid-write never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_dataset(path: Path) -> list:
    data = _read_json(Path(path))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of prompts")
    ids = [p.get("id") for p in data]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: every prompt needs a unique 'id'")
    return data


def load_results(path: Path) -> dict:
    content = _read_json(path, default=[])
    if isinstance(content, dict):
        content = [content]
    return {r["id"]: r for r in content if isinstance(r, dict) and r.get("id")}


def english_of(item: dict) -> str:
    return (item.get("english_prompt") or item.get("text_prompt")
            or item.get("english_audio_prompt") or "").strip()


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def normalize_run_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_-")
    if not name:
        raise ValueError("Run name is empty.")
    return name if name.endswith("_run") else f"{name}_run"


def run_dir_for(name: str) -> Path:
    return RUNS_DIR / normalize_run_name(name)


def list_runs() -> list:
    if not RUNS_DIR.exists():
        return []
    return sorted(p for p in RUNS_DIR.iterdir() if (p / "run_info.json").exists())


def create_run(name: str, dataset_path, created_by: str = None, langs=None,
               sarvam_model: str = None, gemma_size: str = "E2B", note: str = "") -> Path:
    import gemma4_target
    run_dir = run_dir_for(name)
    if run_dir.exists():
        raise FileExistsError(f"Run '{run_dir.name}' already exists.")
    dataset_path = Path(dataset_path).resolve()
    prompts = load_dataset(dataset_path)            # validate before creating anything
    langs = list(langs or languages.PIPELINE_LANGUAGE_CODES)
    sarvam_model = sarvam_model or rfp.DEFAULT_SARVAM_MODEL
    if gemma_size not in gemma4_target.MODEL_IDS:
        raise ValueError(f"Gemma size must be one of {list(gemma4_target.MODEL_IDS)}")

    for sub in ("audio", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset_path, run_dir / "dataset.json")
    _write_json(run_dir / "results.json", [])
    info = {
        "name": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": created_by or current_user(),
        "note": note,
        "dataset_source": _rel(dataset_path),
        "dataset_file": "dataset.json",
        "prompt_count": len(prompts),
        "settings": {
            "languages": langs,
            "translation_model": rfp.TRANSLATION_MODEL,
            "translation_style": languages.TRANSLATION_STYLE,
            "tts_provider": "sarvam",
            "tts_model": sarvam_model,
            "tts_voices": {c: rfp._speaker_for(languages.get_language(c), sarvam_model)
                           for c in langs},
            "gemma_size": gemma_size,
            "target_model": gemma4_target.MODEL_IDS[gemma_size],
            "judge_model": judge_responses.JUDGE_MODEL,
            "target_framing": {
                "version": gemma4_target.FRAMING_VERSION,
                "audio_instruction": gemma4_target.AUDIO_INSTRUCTION,
                "text_instruction": gemma4_target.TEXT_INSTRUCTION,
            },
        },
        "history": [],
    }
    _write_json(run_dir / "run_info.json", info)
    return run_dir


# ---------------------------------------------------------------------------
# Session = one opened run
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, run_dir, count=50, gap=4.0):
        self.count = count
        THROTTLE.gap = gap
        self.open(run_dir)

    def open(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.info = _read_json(self.run_dir / "run_info.json")
        if not self.info:
            raise FileNotFoundError(f"{self.run_dir} is not a run (no run_info.json)")
        st = self.info["settings"]
        self.langs = list(st["languages"])
        self.sarvam_model = st["tts_model"]
        self.gemma_size = st["gemma_size"]
        self.prompts = load_dataset(self.run_dir / self.info.get("dataset_file", "dataset.json"))
        self.results_path = self.run_dir / "results.json"
        self.audio_dir = self.run_dir / "audio"
        self.log_dir = self.run_dir / "logs"
        self.audio_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.recs = load_results(self.results_path)
        if st.get("translation_style") != languages.TRANSLATION_STYLE:
            say(f"  Note: this run used translation style {st.get('translation_style')!r}; the code "
                f"now uses {languages.TRANSLATION_STYLE!r}, so old translations will be redone.")

    @property
    def name(self):
        return self.run_dir.name

    @property
    def target_model(self):
        return self.info["settings"]["target_model"]

    def save(self):
        order = [p["id"] for p in self.prompts if p["id"] in self.recs]
        rest = [i for i in self.recs if i not in set(order)]
        _write_json(self.results_path, [self.recs[i] for i in order + rest])

    def add_history(self, entry: dict):
        self.info.setdefault("history", []).append(entry)
        _write_json(self.run_dir / "run_info.json", self.info)

    def record(self, item: dict) -> dict:
        """Existing record for this prompt, or a fresh one (old translation style = fresh)."""
        pid = item["id"]
        rec = self.recs.get(pid)
        if rec is None or rec.get("translation_style") != languages.TRANSLATION_STYLE:
            rec = {"id": pid, "english_prompt": english_of(item),
                   "translation_style": languages.TRANSLATION_STYLE}
            for key in ("approach", "task"):
                if item.get(key):
                    rec[key] = item[key]
            self.recs[pid] = rec
        return rec

    # -- lock: one batch at a time per run -----------------------------------
    @property
    def lock_path(self):
        return self.run_dir / ".lock"

    def acquire_lock(self, interactive=True) -> bool:
        holder = _read_json(self.lock_path)
        if holder:
            say(f"  Run '{self.name}' is locked by {holder.get('user')}@{holder.get('host')} "
                f"(step {holder.get('step')}, since {holder.get('since')}).")
            say("  If that batch is no longer running (e.g. it crashed), you can take over.")
            if not interactive or not confirm_yes("  Take over the lock? [y/N] "):
                return False
        _write_json(self.lock_path, {
            "user": current_user(), "host": socket.gethostname(), "pid": os.getpid(),
            "since": datetime.now().isoformat(timespec="seconds"), "step": self._lock_step,
        })
        return True

    def release_lock(self):
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    _lock_step = ""


# ---------------------------------------------------------------------------
# "Is this step done for this prompt?"
# ---------------------------------------------------------------------------
def _valid(text) -> bool:
    return rfp._is_valid_response(text)


def is_done(s: Session, step: str, rec: dict) -> bool:
    if not rec:
        return False
    tr = rec.get("translations") or {}
    if step == "translate":
        return all((tr.get(c) or "").strip() for c in s.langs)
    if step == "audio":
        for c in s.langs:
            wav = generate_tts.find_existing_audio(rec["id"], c, str(s.audio_dir))
            if not wav or generate_tts.read_spoken_text(wav) != (tr.get(c) or "").strip():
                return False
        return True
    if step == "target":
        if rec.get("target_model") != s.target_model:
            return False
        resp = rec.get("audio_responses") or {}
        skipped = rec.get("audio_skipped") or {}
        if not all(_valid(resp.get(c)) or c in skipped for c in s.langs):
            return False
        return _valid(rec.get("text_baseline_response"))
    if step == "judge":
        mods = ((rec.get("judgement") or {}).get("modalities")) or {}
        resp = rec.get("audio_responses") or {}
        keys = ["english_text"] + [f"{judge_responses.LANG_KEY[c]}_audio"
                                   for c in s.langs if _valid(resp.get(c))]
        return all(judge_responses._is_valid_judgement(mods.get(k)) for k in keys)
    raise ValueError(step)


def has_error(rec: dict, step: str) -> bool:
    return bool(((rec or {}).get("batch_errors") or {}).get(step))


def classify(s: Session, step: str):
    """Split prompts into done / failed / ready / waiting for one step."""
    groups = {"done": [], "failed": [], "ready": [], "waiting": []}
    need = STEP_NEEDS[step]
    for idx, item in enumerate(s.prompts):
        rec = s.recs.get(item["id"])
        if rec and rec.get("translation_style") != languages.TRANSLATION_STYLE:
            rec = None
        if is_done(s, step, rec):
            groups["done"].append(idx)
        elif need and not is_done(s, need, rec):
            groups["waiting"].append(idx)
        elif has_error(rec, step):
            groups["failed"].append(idx)
        else:
            groups["ready"].append(idx)
    return groups


# ---------------------------------------------------------------------------
# The four steps. Each handles all languages of ONE prompt and returns
# ({lang: "ok"/"cached"/"FAIL"/...}, {lang: error message}).
# ---------------------------------------------------------------------------
def step_translate(s: Session, item: dict, rec: dict):
    tr = rec.setdefault("translations", {})
    models = rec.setdefault("translation_models", {})
    res, errs = {}, {}
    for c in s.langs:
        cfg = languages.get_language(c)
        if (tr.get(c) or "").strip():
            res[c] = "cached"
            continue
        if not cfg.get("translate", True):
            tr[c], models[c], res[c] = rec["english_prompt"], "none (English is spoken as is)", "ok"
            continue
        saved = languages.existing_translation(item, c)
        if saved:
            tr[c], models[c], res[c] = saved, "pre-existing", "cached"
            rec[cfg["record_field"]] = saved
            continue
        try:
            text, model_used = translate_prompts.convert_to_language(
                input_text=rec["english_prompt"], lang_code=c, model_name=rfp.TRANSLATION_MODEL)
        except Exception as e:
            if _is_rate_limit(e):
                raise RateLimited(f"{cfg['name']}: {e}")
            errs[c], res[c] = str(e)[:400], "FAIL"
            continue
        tr[c], models[c], res[c] = text, model_used, "ok"
        rec[cfg["record_field"]] = text
    return res, errs


def step_audio(s: Session, item: dict, rec: dict):
    tr = rec.get("translations") or {}
    meta = rec.setdefault("audio", {})
    res, errs = {}, {}
    for c in s.langs:
        cfg = languages.get_language(c)
        text = (tr.get(c) or "").strip()
        wav = generate_tts.cached_audio_path(rec["id"], c, str(s.audio_dir))
        existing = generate_tts.find_existing_audio(rec["id"], c, str(s.audio_dir))
        if existing and generate_tts.read_spoken_text(existing) == text:
            wav, res[c] = existing, "cached"
        else:
            try:
                wav = Path(generate_tts.text_to_speech(
                    text=text, output_path=str(wav), voice=None, model=s.sarvam_model,
                    provider="sarvam", lang_code=c, skip_if_exists=False, force=True))
                generate_tts.write_spoken_text(wav, text)
                res[c] = "ok"
            except Exception as e:
                if _is_rate_limit(e):
                    raise RateLimited(f"{cfg['name']}: {e}")
                errs[c], res[c] = str(e)[:400], "FAIL"
                continue
        meta[c] = {
            "file": wav.name, "path": _rel(wav), "cached": res[c] == "cached",
            "tts_provider": "sarvam", "tts_model": s.sarvam_model,
            "tts_language_code": cfg["sarvam_language_code"],
            "tts_voice": rfp._speaker_for(cfg, s.sarvam_model),
            "tts_size_bytes": wav.stat().st_size if wav.exists() else 0,
        }
    return res, errs


def step_target(s: Session, item: dict, rec: dict):
    import gemma4_target
    model = s.target_model
    if rec.get("target_model") != model:
        # Responses from another Gemma size don't count; their judgements neither.
        for key in ("audio_responses", "audio_skipped", "text_baseline_response", "judgement"):
            rec.pop(key, None)
        rec["target_model"] = model
        rec["target_provider"] = "google-gemma-local"
    resp = rec.setdefault("audio_responses", {})
    skipped = rec.setdefault("audio_skipped", {})
    res, errs = {}, {}
    for c in s.langs:
        if _valid(resp.get(c)):
            res[c] = "cached"
            continue
        if c in skipped:
            res[c] = "skip>30s"
            continue
        wav = generate_tts.find_existing_audio(rec["id"], c, str(s.audio_dir))
        try:
            resp[c] = gemma4_target.query_audio(str(wav), size=s.gemma_size)
            res[c] = "ok"
        except ValueError as e:
            if "30 seconds" in str(e):
                skipped[c], res[c] = str(e), "skip>30s"
            else:
                errs[c], res[c] = str(e)[:400], "FAIL"
        except Exception as e:
            errs[c], res[c] = str(e)[:400], "FAIL"
    if _valid(rec.get("text_baseline_response")):
        res["text"] = "cached"
    else:
        try:
            rec["text_baseline_response"] = gemma4_target.query_text(
                rec["english_prompt"], size=s.gemma_size)
            rec["text_baseline_model"] = model
            rec["text_baseline_timestamp"] = datetime.now().isoformat()
            res["text"] = "ok"
        except Exception as e:
            errs["text"], res["text"] = str(e)[:400], "FAIL"
    return res, errs


def step_judge(s: Session, item: dict, rec: dict):
    resp = rec.get("audio_responses") or {}
    codes = [c for c in s.langs if _valid(resp.get(c))]
    block = judge_responses.judge_record(rec, language_codes=codes)
    rec["judgement"] = block                     # keep the valid ones even on failure
    res, errs = {}, {}
    for key, j in block["modalities"].items():
        short = key.replace("_audio", "").replace("english_text", "text")
        if judge_responses._is_valid_judgement(j):
            res[short] = j.get("outcome", "?")
        else:
            msg = j.get("error") or j.get("outcome_reasoning") or "not evaluated"
            errs[short], res[short] = str(msg)[:400], "FAIL"
    hits = [m for m in errs.values() if _is_rate_limit(m)]
    if hits:
        raise RateLimited(hits[0])
    return res, errs


STEP_FN = {"translate": step_translate, "audio": step_audio,
           "target": step_target, "judge": step_judge}


# ---------------------------------------------------------------------------
# Running a batch
# ---------------------------------------------------------------------------
def audio_cost_estimate(s: Session, batch: list):
    chars = 0
    for idx in batch:
        rec = s.recs.get(s.prompts[idx]["id"]) or {}
        tr = rec.get("translations") or {}
        for c in s.langs:
            text = (tr.get(c) or "").strip()
            wav = generate_tts.find_existing_audio(rec.get("id", ""), c, str(s.audio_dir))
            if not (wav and generate_tts.read_spoken_text(wav) == text):
                chars += len(text)
    rate = SARVAM_RATE_INR.get(s.sarvam_model)
    return chars, (chars / 1000 * rate if rate else None)


def run_batch(s: Session, step: str, count: int, retry_failed=False, confirm=True) -> None:
    api_keys.install()          # key switching sits under the throttle
    THROTTLE.install()
    groups = classify(s, step)
    pool = groups["failed"] if retry_failed else groups["ready"]
    batch = pool[:count]
    label = STEP_LABEL[step]
    if not batch:
        say(f"\nNothing to do for {label}: "
            f"{len(groups['done'])} done, {len(groups['failed'])} failed, "
            f"{len(groups['waiting'])} waiting on the previous step.")
        return

    first, last = batch[0] + 1, batch[-1] + 1
    say(f"\n[{s.name}] {label}: {len(batch)} prompt(s), dataset #{first}..#{last}"
        f"{' (retrying failed)' if retry_failed else ''}")

    if step == "audio":
        chars, inr = audio_cost_estimate(s, batch)
        cost = f"~₹{inr:,.0f} + GST" if inr is not None else "unknown rate"
        say(f"  Sarvam {s.sarvam_model}: ~{chars:,} characters to synthesise, {cost}")
        if confirm and not confirm_yes("  Proceed? [y/N] "):
            say("  Cancelled.")
            return
    if step == "target":
        say(f"  Target: {s.target_model} (the first prompt loads the model; can take minutes)")

    s._lock_step = step
    if not s.acquire_lock(interactive=confirm):
        say("  Not started.")
        return

    log_path = s.log_dir / f"batch_{datetime.now():%Y%m%d}.log"
    say(f"  Detailed output -> {_rel(log_path)}\n")

    ok_count, fail_count, stop_reason = 0, 0, ""
    t_start = time.monotonic()
    started = datetime.now().isoformat(timespec="seconds")
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n\n##### {started}  {current_user()}  {label}  #{first}..#{last}\n")
            try:
                for n, idx in enumerate(batch, 1):
                    item = s.prompts[idx]
                    rec = s.record(item)
                    t0 = time.monotonic()
                    log.write(f"\n=== [{idx + 1}] {item['id']} ===\n")
                    stopped = None
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        try:
                            res, errs = STEP_FN[step](s, item, rec)
                        except RateLimited as e:
                            stopped, res, errs = e, {}, {}
                        except Exception as e:      # unexpected: mark failed, keep going
                            if _is_rate_limit(e):
                                stopped = e
                            res, errs = {}, {"step": str(e)[:400]}
                    log.flush()
                    if stopped is not None:
                        s.save()
                        stop_reason = str(stopped)[:200]
                        say(f"  [{n}/{len(batch)}] #{idx + 1} {item['id']}: "
                            f"STOPPED - rate limit / quota / credits / missing key")
                        say(f"      {stop_reason}")
                        say("  Progress is saved. Fix the cause (wait, top up credits, or add the "
                            "key to .env) and run this step again.")
                        break
                    errors = rec.setdefault("batch_errors", {})
                    if is_done(s, step, rec):
                        errors.pop(step, None)
                        ok_count += 1
                        status = "done"
                    else:
                        errors[step] = errs or {"step": "incomplete (see log)"}
                        fail_count += 1
                        status = "FAILED"
                    if not errors:
                        rec.pop("batch_errors", None)
                    s.save()
                    parts = " ".join(f"{k}:{v}" for k, v in res.items())
                    say(f"  [{n}/{len(batch)}] #{idx + 1} {item['id']}  {status}  "
                        f"({time.monotonic() - t0:.0f}s)  {parts}")
            except KeyboardInterrupt:
                s.save()
                stop_reason = "interrupted (Ctrl+C)"
                say("\n  Interrupted. Everything finished so far is saved.")
    finally:
        s.release_lock()
        s.add_history({
            "user": current_user(), "host": socket.gethostname(), "step": step,
            "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
            "requested": len(batch), "range": f"#{first}..#{last}",
            "retry_failed": retry_failed, "done": ok_count, "failed": fail_count,
            "stopped": stop_reason,
        })

    mins = (time.monotonic() - t_start) / 60
    say(f"\n  Finished: {ok_count} done, {fail_count} failed, {mins:.1f} min.")
    if fail_count:
        say("  Failed prompts are skipped by 'next N'; use the retry option for them.")


# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------
def run_summary(run_dir: Path) -> str:
    """One line for the run list: dataset, owner, date, judged count."""
    try:
        s = Session.__new__(Session)
        s.open(run_dir)
        judged = len(classify(s, "judge")["done"])
        translated = len(classify(s, "translate")["done"])
        info = s.info
        return (f"{info['dataset_source']:<24} {len(s.prompts):>4} prompts  "
                f"by {info['created_by']:<10} {info['created_at'][:10]}  "
                f"translated {translated}, judged {judged}")
    except Exception as e:
        return f"(unreadable: {e})"


def ranges(idxs, limit=None) -> str:
    """0-based indices -> '#1-47, #49-50' (1-based). limit = max ranges shown."""
    if not idxs:
        return "-"
    idxs = sorted(idxs)
    spans, start, prev = [], idxs[0], idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append((start, prev))
        start = prev = i
    spans.append((start, prev))
    text = [f"#{a + 1}" if a == b else f"#{a + 1}-{b + 1}" for a, b in spans]
    if limit and len(text) > limit:
        return ", ".join(text[:limit]) + f", ... (+{len(text) - limit} more)"
    return ", ".join(text)


def show_status(s: Session):
    total = len(s.prompts)
    st = s.info["settings"]
    say("\n" + "=" * 78)
    say(f" Run     : {s.name}   (created by {s.info['created_by']}, {s.info['created_at'][:16]})")
    say(f" Folder  : {_rel(s.run_dir)}")
    say(f" Dataset : {s.info['dataset_source']}  ({total} prompts, frozen copy in the run)")
    say(f" Settings: langs {','.join(s.langs)} | TTS {st['tts_model']} | target {st['target_model']}")
    say(f" Session : batch N = {s.count}, gap between API calls = {THROTTLE.gap:g}s, you = {current_user()}")
    api_keys.install()
    say(f" API keys: {api_keys.summary()}")
    holder = _read_json(s.lock_path)
    if holder:
        say(f" LOCKED  : {holder.get('user')}@{holder.get('host')} running {holder.get('step')} "
            f"since {holder.get('since')}")
    say("-" * 78)
    say(f" {'Step':<24}{'Done':>6}{'Failed':>8}{'Ready':>7}{'Waiting':>9}  {'next':<6}done prompts")
    for i, step in enumerate(STEPS, 1):
        g = classify(s, step)
        nxt = f"#{g['ready'][0] + 1}" if g["ready"] else "-"
        say(f" {i} {STEP_LABEL[step]:<22}{len(g['done']):>6}{len(g['failed']):>8}"
            f"{len(g['ready']):>7}{len(g['waiting']):>9}  {nxt:<6}{ranges(g['done'], limit=3)}")
    say("=" * 78)


def show_details(s: Session, steps=None):
    """Per step: full done ranges, failed prompts with errors, ready/waiting ranges."""
    for step in steps or STEPS:
        g = classify(s, step)
        say("\n" + "-" * 78)
        say(f" {STEP_LABEL[step]}")
        say("-" * 78)
        say(f"  done    ({len(g['done'])}): {ranges(g['done'])}")
        say(f"  ready   ({len(g['ready'])}): {ranges(g['ready'], limit=8)}")
        say(f"  waiting ({len(g['waiting'])}): {ranges(g['waiting'], limit=8)}"
            + (f"  <- needs '{STEP_LABEL[STEP_NEEDS[step]]}' first" if g["waiting"] else ""))
        say(f"  failed  ({len(g['failed'])}):" + ("" if g["failed"] else " -"))
        for idx in g["failed"]:
            pid = s.prompts[idx]["id"]
            errs = ((s.recs.get(pid) or {}).get("batch_errors") or {}).get(step) or {}
            say(f"    #{idx + 1:<4} {pid}")
            for where, msg in errs.items():
                say(f"           {where}: {' '.join(str(msg).split())[:110]}")
        skipped = [(i, p["id"], c) for i, p in enumerate(s.prompts)
                   for c in ((s.recs.get(p["id"]) or {}).get("audio_skipped") or {})]
        if step == "target" and skipped:
            say(f"  skipped audio (over Gemma's 30 s limit) ({len(skipped)}):")
            for i, pid, c in skipped:
                say(f"    #{i + 1:<4} {pid}  {c}")


def show_history(s: Session, last=20):
    hist = s.info.get("history") or []
    say("\n" + "-" * 78)
    say(f" Batch history of {s.name}  ({len(hist)} batches, showing last {min(last, len(hist))})")
    say("-" * 78)
    if not hist:
        say("  (no batches yet)")
    say(f"  {'started':<17}{'user':<12}{'step':<10}{'range':<14}{'done':>5}{'fail':>5}  note")
    for h in hist[-last:]:
        note = ("retry " if h.get("retry_failed") else "") + (f"stopped: {h['stopped'][:30]}" if h.get("stopped") else "")
        say(f"  {h.get('started', '')[:16]:<17}{str(h.get('user', ''))[:11]:<12}{h.get('step', ''):<10}"
            f"{h.get('range', ''):<14}{h.get('done', 0):>5}{h.get('failed', 0):>5}  {note}")


def _ask_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        say("  Not a number, keeping the old value.")
        return default


def _new_run_wizard():
    """Interactive: name, dataset, creator, TTS model, Gemma size. Returns run dir or None."""
    import gemma4_target
    say("\n-- New run --")
    name = input("Run name (letters/digits/_/-, '_run' is added): ").strip()
    if not name:
        return None
    try:
        if run_dir_for(name).exists():
            say(f"  '{normalize_run_name(name)}' already exists; pick it from the list instead.")
            return None
    except ValueError as e:
        say(f"  {e}")
        return None
    files = sorted(p for p in CURRENT_DIR.glob("exp*.json") if not p.name.endswith(".bak"))
    for i, p in enumerate(files, 1):
        try:
            n = len(load_dataset(p))
        except Exception:
            n = "?"
        say(f"  {i}. {p.name} ({n} prompts)")
    raw = input("Dataset number (or a path to a JSON list of prompts): ").strip()
    if raw.isdigit() and 0 < int(raw) <= len(files):
        dataset = files[int(raw) - 1]
    elif raw:
        dataset = Path(raw)
        if not dataset.is_absolute():
            dataset = (Path.cwd() / dataset) if (Path.cwd() / dataset).exists() else PROJECT_ROOT / dataset
    else:
        return None
    global USER_OVERRIDE
    who = input(f"Your name [{current_user()}]: ").strip() or current_user()
    USER_OVERRIDE = who
    tts = input(f"TTS model [{rfp.DEFAULT_SARVAM_MODEL}] (bulbul:v4-flash / bulbul:v3): ").strip() \
        or rfp.DEFAULT_SARVAM_MODEL
    size = input(f"Gemma size {list(gemma4_target.MODEL_IDS)} [E2B]: ").strip().upper() or "E2B"
    note = input("Note (optional): ").strip()
    try:
        run_dir = create_run(name, dataset, created_by=who, sarvam_model=tts,
                             gemma_size=size, note=note)
    except Exception as e:
        say(f"  Could not create the run: {e}")
        return None
    say(f"  Created {_rel(run_dir)}")
    return run_dir


def pick_run():
    """Run list menu. Returns a run dir, or None to quit."""
    while True:
        runs = list_runs()
        say("\n" + "=" * 78)
        say(f" RUNS  ({_rel(RUNS_DIR)}/)")
        say("-" * 78)
        if not runs:
            say("  (no runs yet)")
        for i, r in enumerate(runs, 1):
            say(f"  {i:>2}. {r.name:<26} {run_summary(r)}")
        say("-" * 78)
        say("  <number> open a run    n  new run    q  quit")
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return None
        if choice == "n":
            run_dir = _new_run_wizard()
            if run_dir:
                return run_dir
        elif choice.isdigit() and 0 < int(choice) <= len(runs):
            return runs[int(choice) - 1]


def menu(s: Session = None, count=50, gap=4.0):
    if s is None:
        run_dir = pick_run()
        if run_dir is None:
            return
        s = Session(run_dir, count=count, gap=gap)
    while True:
        show_status(s)
        say(" 1-4  run that step on the NEXT N ready prompts")
        say(" r    retry FAILED prompts of a step")
        say(" v    view details (done ranges, failed prompts + errors)   h  batch history")
        say(" n    change N        g  change gap between API calls")
        say(" w    switch / create run")
        say(" q    quit")
        try:
            choice = input("> ").strip().lower()
            if choice in ("1", "2", "3", "4"):
                step = STEPS[int(choice) - 1]
                n = _ask_int(f"How many prompts for {STEP_LABEL[step]}", s.count)
                run_batch(s, step, n)
            elif choice == "r":
                raw = input("Retry which step (1-4)? ").strip()
                if raw in ("1", "2", "3", "4"):
                    step = STEPS[int(raw) - 1]
                    run_batch(s, step, _ask_int("How many", s.count), retry_failed=True)
            elif choice == "v":
                raw = input("Details for which step (1-4, Enter = all)? ").strip()
                show_details(s, [STEPS[int(raw) - 1]] if raw in ("1", "2", "3", "4") else None)
                input("\n(Enter to go back) ")
            elif choice == "h":
                show_history(s)
                input("\n(Enter to go back) ")
            elif choice == "n":
                s.count = _ask_int("Prompts per batch", s.count)
            elif choice == "g":
                raw = input(f"Seconds between API calls [{THROTTLE.gap:g}]: ").strip()
                if raw:
                    THROTTLE.gap = max(0.0, float(raw))
            elif choice == "w":
                run_dir = pick_run()
                if run_dir is not None:
                    s.open(run_dir)
            elif choice in ("q", "quit", "exit"):
                return
        except KeyboardInterrupt:
            say("\n  (cancelled)")
        except EOFError:
            return
