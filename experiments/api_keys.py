# =============================================================================
#  SHARED — MULTIPLE API KEYS PER PLATFORM, WITH AUTOMATIC SWITCHING
#  file: experiments/api_keys.py
# -----------------------------------------------------------------------------
#  WHAT IT DOES : Lets you put several keys per platform in .env and switches to
#                 the next key whenever one is rate-limited, out of quota/credits
#                 or invalid. No other module needs to know about it: every
#                 request already goes through requests.post, so install() wraps
#                 requests.post and puts the active key into each request:
#                   Gemini : the ?key=... part of the URL
#                   Sarvam : the api-subscription-key header
#
#  .env FORMAT  : any number of keys per platform (at least 3 recommended):
#                   GOOGLE_API_KEY_1=...   GOOGLE_API_KEY_2=...   GOOGLE_API_KEY_3=...
#                   SARVAM_API_KEY_1=...   SARVAM_API_KEY_2=...   SARVAM_API_KEY_3=...
#                 The plain GOOGLE_API_KEY / SARVAM_API_KEY (one key) still work, and
#                 count as an extra key. Numbering may have gaps (up to _20).
#
#  SWITCHING    : HTTP 429 (rate limit)  -> key rests (Gemini's retryDelay if given,
#                                           else 60 s; daily quota -> 1 h), next key
#                 invalid key / no access -> key disabled for this session, next key
#                 Sarvam out of credits   -> key disabled for this session, next key
#                 5xx / network errors   -> not the key's fault; the module retries
#                 All keys resting       -> waits if the soonest is free within 90 s,
#                                           otherwise hands the 429 back, and the
#                                           batch runner STOPS (resume later).
#
#  NOTE         : Gemini quotas are per Google Cloud PROJECT, not per key: keys from
#                 the same project share one quota. Use keys from different
#                 projects/accounts (e.g. one per team member).
#  CALLED BY    : run_full_pipeline.py (all modes), batch_runner.py, and the
#                 standalone mains of translate_prompts.py / judge_responses.py.
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

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_NUMBERED_KEYS = 20
MAX_WAIT_FOR_COOLDOWN = 90      # seconds: longer than this -> give up, let the batch stop
DEFAULT_RATE_COOLDOWN = 60      # seconds a key rests after a 429 without retryDelay
DAILY_QUOTA_COOLDOWN = 3600     # seconds a key rests after a daily-quota 429
STARVED_LIMIT = 3               # after this many requests in a row found no usable key,
                                # stop waiting so the batch runner can stop quickly

PLATFORMS = {
    "gemini": {
        "label": "Gemini",
        "env": "GOOGLE_API_KEY",
        "aliases": [],
        "host": "generativelanguage.googleapis.com",
    },
    "sarvam": {
        "label": "Sarvam",
        "env": "SARVAM_API_KEY",
        "aliases": ["SARVAM_SUBSCRIPTION_KEY"],
        "host": "api.sarvam.ai",
    },
}

CONSOLE = sys.__stdout__


def _say(text):
    print(text, file=CONSOLE, flush=True)


def mask(key: str) -> str:
    return f"...{key[-4:]}" if key and len(key) > 6 else "..."


class KeyPool:
    def __init__(self, platform: str, keys: list):
        self.platform = platform
        self.label = PLATFORMS[platform]["label"]
        self.keys = keys
        self.idx = 0
        self.rest_until = [0.0] * len(keys)
        self.disabled = [""] * len(keys)       # reason, empty = usable
        self.uses = [0] * len(keys)
        self.switches = 0
        self.starved = 0                        # consecutive requests with no usable key
        self._last_msg = {}                     # (key index, reason) -> time printed

    def alive(self):
        return [i for i in range(len(self.keys)) if not self.disabled[i]]

    def pick(self):
        """Index of the key to use now: the current one if usable, else the next usable,
        else the alive key that becomes free soonest. None if every key is disabled."""
        alive = self.alive()
        if not alive:
            return None
        now = time.time()
        n = len(self.keys)
        for step in range(n):
            i = (self.idx + step) % n
            if i in alive and self.rest_until[i] <= now:
                self.idx = i
                return i
        return min(alive, key=lambda i: self.rest_until[i])

    def rest(self, i, seconds, reason):
        self.rest_until[i] = time.time() + seconds
        self._move_on(i, f"{reason}, resting {seconds:.0f}s")

    def disable(self, i, reason):
        self.disabled[i] = reason
        self._move_on(i, f"{reason}, disabled for this session")

    def _move_on(self, i, why):
        self.switches += 1
        self.idx = (i + 1) % len(self.keys)
        nxt = self.pick()
        free = nxt is not None and nxt != i and self.rest_until[nxt] <= time.time()
        target = f"switching to key #{nxt + 1} ({mask(self.keys[nxt])})" if free else "no other key free"
        self.note(f"{self.label} key #{i + 1} {mask(self.keys[i])}: {why}; {target}", (i, why.split(",")[0]))

    def note(self, text, tag):
        """Print a status line, but the same message at most once per 30 s."""
        now = time.time()
        if now - self._last_msg.get(tag, 0) >= 30:
            self._last_msg[tag] = now
            _say(f"      ({text})")

    def summary(self) -> str:
        now = time.time()
        parts = []
        for i in range(len(self.keys)):
            if self.disabled[i]:
                state = "off"
            elif self.rest_until[i] > now:
                state = f"rest {self.rest_until[i] - now:.0f}s"
            else:
                state = "ok"
            star = "*" if i == self.idx and not self.disabled[i] else ""
            parts.append(f"#{i + 1}{star} {state}")
        return f"{self.label} {len(self.keys)} key(s): " + ", ".join(parts)


POOLS = {}
_ORIG_POST = None


def _load_env():
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=env_path if env_path.exists() else None)
    except ImportError:
        pass


def collect_keys(platform: str) -> list:
    cfg = PLATFORMS[platform]
    names = [cfg["env"]] + cfg["aliases"]
    for base in [cfg["env"]] + cfg["aliases"]:
        names += [f"{base}_{n}" for n in range(1, MAX_NUMBERED_KEYS + 1)]
    keys = []
    for name in names:
        value = (os.environ.get(name) or "").strip().strip('"').strip("'")
        if value and value not in keys:
            keys.append(value)
    return keys


def _platform_for(url) -> str:
    url = str(url or "")
    for name, cfg in PLATFORMS.items():
        if cfg["host"] in url:
            return name
    return ""


def _with_key(platform, args, kwargs, key):
    args, kwargs = list(args), dict(kwargs)
    if platform == "gemini":
        url = args[0] if args else kwargs.get("url", "")
        url = re.sub(r"([?&]key=)[^&]*", lambda m: m.group(1) + key, url)
        if args:
            args[0] = url
        else:
            kwargs["url"] = url
    elif platform == "sarvam":
        headers = dict(kwargs.get("headers") or {})
        headers["api-subscription-key"] = key
        kwargs["headers"] = headers
    return tuple(args), kwargs


def _retry_delay_seconds(body: str):
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body or "")
    return float(m.group(1)) + 1 if m else None


def _verdict(platform, resp):
    """None = keep the response; ('rest', seconds, why) or ('off', why) = try another key."""
    code = getattr(resp, "status_code", 200)
    body = (getattr(resp, "text", "") or "")[:4000]
    low = body.lower()
    if code == 429:
        if platform == "gemini" and ("perday" in low or "per day" in low or "daily" in low):
            return ("rest", DAILY_QUOTA_COOLDOWN, "daily quota used up")
        if platform == "sarvam" and ("credit" in low or "insufficient" in low):
            return ("off", "out of credits")
        return ("rest", _retry_delay_seconds(body) or DEFAULT_RATE_COOLDOWN, "rate limited (429)")
    if platform == "gemini" and code in (400, 401, 403) and (
            "api_key_invalid" in low or "api key not valid" in low or "api key expired" in low
            or "permission_denied" in low or "billing" in low):
        return ("off", f"key rejected (HTTP {code})")
    if platform == "sarvam":
        if code == 402 or (code in (400, 403) and ("credit" in low or "insufficient" in low
                                                   or "quota" in low)):
            return ("off", "out of credits / quota")
        if code in (401, 403):
            return ("off", f"key rejected (HTTP {code})")
    return None


def _post(*args, **kwargs):
    url = args[0] if args else kwargs.get("url")
    platform = _platform_for(url)
    pool = POOLS.get(platform)
    if not pool:
        return _ORIG_POST(*args, **kwargs)

    last, waited = None, False
    for _ in range(len(pool.keys) + 1):
        i = pool.pick()
        if i is None:                                   # every key disabled
            pool.starved += 1
            return last if last is not None else _ORIG_POST(*args, **kwargs)
        wait = pool.rest_until[i] - time.time()
        if wait > 0:
            # Every key is resting. Wait at most once per request, only for a short rest,
            # and not at all once several requests in a row have hit this wall.
            if waited or wait > MAX_WAIT_FOR_COOLDOWN or pool.starved >= STARVED_LIMIT:
                pool.starved += 1
                if last is not None:
                    return last                         # caller sees the 429 -> batch stops
                a, k = _with_key(platform, args, kwargs, pool.keys[i])
                return _ORIG_POST(*a, **k)              # let the API itself answer
            pool.note(f"all {pool.label} keys resting; waiting {wait:.0f}s for key #{i + 1}", ("wait",))
            time.sleep(wait)
            waited = True
        a, k = _with_key(platform, args, kwargs, pool.keys[i])
        resp = _ORIG_POST(*a, **k)
        pool.uses[i] += 1
        verdict = _verdict(platform, resp)
        if verdict is None:
            pool.starved = 0
            return resp
        last = resp
        if verdict[0] == "rest":
            pool.rest(i, verdict[1], verdict[2])
        else:
            pool.disable(i, verdict[1])
    pool.starved += 1
    return last


def install(quiet=True) -> dict:
    """Load keys from .env and route requests through the key switcher. Safe to call twice."""
    global _ORIG_POST
    _load_env()
    for platform, cfg in PLATFORMS.items():
        keys = collect_keys(platform)
        old = POOLS.get(platform)
        if old and old.keys == keys:
            continue
        if keys:
            POOLS[platform] = KeyPool(platform, keys)
            # Modules check that the single variable exists before calling the API.
            if not os.environ.get(cfg["env"]):
                os.environ[cfg["env"]] = keys[0]
        else:
            POOLS.pop(platform, None)
    if _ORIG_POST is None:
        _ORIG_POST = requests.post
        requests.post = _post
    if not quiet:
        _say("  " + summary())
    return POOLS


def summary() -> str:
    parts = []
    for platform, cfg in PLATFORMS.items():
        pool = POOLS.get(platform)
        parts.append(pool.summary() if pool else f"{cfg['label']} 0 keys")
    return " | ".join(parts)
