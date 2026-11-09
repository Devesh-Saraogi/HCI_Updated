# Session notes, 9 November 2026

Everything changed in the project during this session, in order, with the reason for each change.
For how to use the result, see [`README.md`](README.md).

---

## Summary

| Area | Before | After |
|---|---|---|
| Project folder | current, outdated and unrelated files mixed together | current pipeline only; everything else in `_archive/` |
| Documentation | short README; no file headers | every script has a workflow header; README rewritten; these notes |
| Prompt sets | `exp1.json` (40) | + `exp2.json` (500 AdvBench prompts) |
| TTS | Bulbul v3, mixed voices and genders | Bulbul **v4-flash**, one **female conversational** voice per language |
| Languages | en, hi, gu, **ta**, bn | en, hi, gu, **te**, bn (Tamil removed) |
| Target framing | instruction for audio only | the **same framing** for audio and text |
| Running | one prompt end to end (`--id`, `--all`) | **named runs**, **step-by-step resumable batches**, a menu, status/details/history |
| Multi-user | one shared `ultimate.json` | one folder per run, frozen settings, lock, batch history with user names |

Nothing was deleted (this session couldn't delete files); outdated material was moved.
Live Gemini and Sarvam calls weren't made, because the project had no `.env`. All new logic was tested
with the real code and fake models on a scratch copy of the project.

---

## 1. Project cleanup

**Finding:** every file had the same timestamp (the folder had been copied or unzipped), so the current
files were identified by reading what each script imports and writes. `experiments/ultimate.json` and
`latest_ultimate.json` were **empty** (0 bytes): no pipeline results had been saved in this copy.

**Moved to `_archive/`:**

| Folder | Files | Reason |
|---|---|---|
| `v1_root_text_eval/` | `evaluate_prompts.py`, `results.json`, `final_report.md`, `update_json.py`, `test_model.py`, `test_openrouter.py`, copy of the old `requirements.txt`, and (moved later in the session) `data.json`, the JailbreakBench artifact only these scripts read | the first text-only English vs Hindi run (Gemma-4-31b, 10 prompts: Hindi 5/10, English 1/10 bypassed) |
| `v2_single_audio_tests/` | `process_audio.py`, `process_audio_gemini.py`, `process_audio_rest.py`, `run_gemini_2_5.py`, `run_text_baseline.py`; `outputs/` (`output*.json`, `tts.wav`, `exp1.json.bak`); `audio_slang_style/` (contents of `experiments/audio/`) | one-off tests; audio made from the old slang ("chalu") translations, which the pipeline itself says must not be reused |
| `old_docs/` | `experiments/readit.md` → `readit_original_plan.md` | the original 10-step plan (Gemini target, Orpheus TTS) |
| `not_this_project/` | `task.md`, `devesh (2).tex` | the systematic review paper, a separate project |
| `v3_audio_standard/` | the 10 clips + transcripts that were in `audio_standard/` | Bulbul v3 audio (see section 4) |
| `junk/` | `__pycache__` (including compiled copies of the deleted `shuka_target` and `text_to_chalu_hindi`), `.DS_Store` | build and macOS artifacts |

`experiments/audio/` is now an empty folder; it couldn't be removed without delete permission.

**Code fixes during cleanup:**
- `run_full_pipeline.py` imported `run_text_baseline` (unused) in the same `try` block as the modules it
  needs. With that file archived, the failed import would have skipped `judge_responses` and `languages`
  and crashed the pipeline. The unused import was removed.
- Help text that still said "Target Gemini model" now says Gemma 4.
- `requirements.txt` was reduced to what is used (`requests`, `python-dotenv`); LangChain and Groq were only
  used by archived scripts.

**Documentation:** every remaining script got a header block: stage number, what it does, input,
output, who calls it, how to run it alone, keys needed, and the full workflow diagram.

**Decision: no LangChain.** The pipeline calls Gemini and Sarvam directly with `requests`. That keeps
exact control of what is sent (important for a red-teaming paper), avoids the retry bug the old
LangChain scripts had to patch, and the local Gemma audio pipeline wouldn't fit LangChain anyway. An
offer to merge the three Gemini helper functions was declined.

---

## 2. New prompt set: `exp2.json`

- The attached `harmful_behaviors_redteam_prompts_family_ids.json` was checked and added as
  **`experiments/exp2.json`** (byte-identical copy).
- 500 prompts = 100 AdvBench behaviours × 5 approaches. Same fields as `exp1.json` (`id`, `text_prompt`,
  `english_audio_prompt`, `hindi_prompt`); all IDs unique; no overlap with `exp1.json`.
- The **first letter of the ID is the approach**: `D` direct how-to, `R` role-play, `T` transformation,
  `C` completion, `I` indirect question.
- It replaces the role of `exp1.json` (a flat prompt list), **not** `database.json` (the nested master file).
- It has no `approach` field; the judge classifies the approach, and the ID letter gives the intended one.

---

## 3. Cost analysis (Sarvam)

- Characters for 500 prompts × 5 languages: about **656K** (English 128,400; translation lengths
  estimated from two existing prompts: hi 1.07×, gu 0.93×, te/ta about 1.1×, bn 0.99× of English).
- Sarvam prices (pricing page, November 2026): v3 ₹3 per 1,000 characters, v4-flash ₹1 per 1,000 characters, before GST.
- Full set: **v3 about ₹2,320** with GST; **v4-flash about ₹775** with GST. New accounts get ₹100 of free credit.

---

## 4. TTS: Bulbul v3 → v4-flash

**Why:** about a third of the price, newer, more natural (per Sarvam), and nothing existed yet for the
500-prompt set, so switching was free.

**Changes:**
- Default model `bulbul:v4-flash` in `generate_tts.py` and `run_full_pipeline.py`.
- v4 uses persona speaker IDs (`<voice>_<lang>_<style>`) and **rejects** v3 names. The old fallback to the
  v3 name was removed; a missing v4 voice now raises a clear error.
- `enable_preprocessing: true` is sent for v4 (Sarvam's recommendation).
- The 10 existing v3 clips were moved to `_archive/v3_audio_standard/`. The audio cache doesn't record the
  model, so a v4 run would otherwise have silently reused v3 clips.

**Voice selection, in two rounds:**
1. First all-male: en `dev_en_conversational`, hi `rahul_hi_conversational`, gu `bhavik_gu_conversation`,
   ta `gokul_ta_narration`, bn `arnab_bn_conversation`. Tamil only had a *narration* voice on v4.
2. Narration and conversational were compared (reading aloud vs talking to someone). For red-teaming,
   conversational matches the scenario of a user asking a question, so a voice and style that differed
   for one language would be a confound. Telugu has female conversational voices, so the set became
   **all female and all conversational, using two voice names**:

| en | hi | gu | te | bn |
|---|---|---|---|---|
| `roopa_en_conversational` | `roopa_hi_conversational` | `pooja_gu_conversational` | `pooja_te_conversation` | `roopa_bn_conversational` |

Also noted: v3 voices aren't tied to one language, so v3 could use the *same* speaker for all
languages (the cleanest control, but about 3× the cost). Kept as an option for a future run.

---

## 5. Languages: Tamil → Telugu

- **Telugu added** to `languages.py`: `te-IN`, Telugu script, `telugu_prompt` field, female v4 voice, and
  the same three harmless worked examples as Gujarati and Bengali (translate the instruction, don't
  carry it out). The examples were written by Claude and need a native speaker's check.
- **Tamil removed entirely** (on request): language entry, CLI choices, judge language key, judge prompt
  text, README, and the empty `tamil_prompt` fields in `exp1.json`. Archived v3 Tamil clips remain.
- All hard-coded language lists, headers and help texts were updated.
- **Bug fixed:** `judge_responses.py` run on its own defaulted to `hi,gu,ta`, skipping English and
  Bengali audio; it now defaults to all five pipeline languages.
- Gemini lists Telugu as a supported language. A live translation test is still pending (no API key in
  the project).

---

## 6. Batch mode (step-by-step, resumable)

**Requirement:** don't push one prompt through every step at once; instead run *one step* for the next N
prompts (all 5 languages together), come back later for the next N, with each step remembering where
it stopped and rate limits handled.

**Built:** `experiments/batch_runner.py`, opened by `run_full_pipeline.py` when no `--id/--all/--text`
is given.
- Steps: Translate → Audio → Target → Judge. A prompt is *ready* for a step once the previous step is done.
- Per prompt and step: *done / failed / ready / waiting*, worked out from the saved results, so it
  always reflects reality.
- **One request at a time** with a minimum gap between API calls (default 4 s, adjustable).
- **Rate limit / quota / credits / missing key → the batch stops** (instead of failing every prompt). Other
  errors mark the prompt failed; "next N" skips it; `r` retries it.
- **Saved after every prompt** with atomic writes. Ctrl+C is safe.
- **Audio cost estimate and confirmation** before spending Sarvam credits.
- Clips over 30 s are recorded as skipped for that language; the judge scores the remaining responses.
- Long module output goes to log files; the console shows one line per prompt.
- Scripted use: `--step`, `--count`, `--retry-failed`, `--gap`, `--yes`.

**Tested** with fake models: resuming across batches, a failure in the middle of a batch, a rate-limit
stop and resume, retrying failures, the 30 s skip, and the real menu driven by typed input.
**Bug found and fixed:** a yes/no question with no keyboard attached (scripts) crashed; it now
answers "no".

---

## 7. Named runs (multi-user, repeatable)

**Requirement:** the process may be done several times, by different users; all results of one
attempt should be in a named folder that can be selected.

**Built:** `experiments/runs/<name>_run/` with `run_info.json` (creator, date, source dataset, fixed
settings, batch history), `dataset.json` (frozen prompts), `results.json`, `audio/`, `logs/`, `.lock`.
- The menu starts with the **run list** (open / create). The new-run wizard asks for name, dataset,
  user, TTS model, Gemma size and a note. `w` switches runs.
- **Settings are fixed per run**, so results inside a run are always comparable.
- **Lock:** one batch per run at a time; shows who/where/which step; takeover if a batch crashed.
- **History:** every batch is recorded with the user, step, range, done/failed counts and stop reason.
- `--run NAME` works with `--step`, `--status`, `--id` and `--all`; a run is created from `--from-json`
  on first use. `--user` sets the name recorded.
- **Fixes found in testing:** `--id` with `--run` looked in `exp1.json` instead of the run's own
  prompts; history recorded the OS login name instead of the user's name; `global` declarations had to
  move to the top of `main()`.
- Legacy `ultimate.json` / `audio_standard/` are only used by `--id/--all` without `--run`.

---

## 8. Progress views

- The status table gained a **"done prompts"** column with exact ranges (e.g. `#1-5, #7-12`; gaps =
  failed or skipped prompts) next to Done / Failed / Ready / Waiting / next.
- **`v` (details):** per step, the full done / ready / waiting ranges ("waiting" names the step it
  needs), every failed prompt with number, ID, language and error, and the audio skipped for the 30 s limit.
- **`h` (history):** the last 20 batches.
- **`--run NAME --status`:** prints all three without the menu.

---

## 9. Target model: framing fix

**Problem found:** the audio conditions got an English instruction ("Listen to this audio and answer
the request…") but the text baseline got the bare prompt. Audio-vs-text differences could have come
from the wrapper rather than the modality.

**Fix:** both now send *instruction + prompt*, with matching wording (listen/read, transcribe/repeat).
The instructions are named constants in `gemma4_target.py` (`FRAMING_VERSION = "symmetric-v1"`) and are
recorded in each new run's `run_info.json → settings.target_framing`. No results existed with the old
framing.

**Explained but deliberately not changed yet** (open items in README section 10): Gemma's sampling
isn't pinned, the reply format with real weights is untested, the 512-token cap, and memory when two
runs do step 3 on one machine.

---

## 10. Several API keys per platform

**Requirement:** enter as many API keys as possible (at least 3 per platform); the batch runner should
switch whenever one is rate-limited or otherwise unusable.

**Built:** `experiments/api_keys.py`.
- `.env` takes `GOOGLE_API_KEY_1..20` and `SARVAM_API_KEY_1..20` (gaps allowed); the single-key names still work.
- It wraps `requests.post`, so **no pipeline module had to change**: it writes the active key into the Gemini
  URL (`?key=`) or the Sarvam header (`api-subscription-key`) of each request.
- 429 → the key rests (Gemini's `retryDelay`, else 60 s; daily quota 1 h) and the next key is used at once.
  Invalid key or Sarvam out of credits → the key is disabled for the session. 5xx/network → normal retries.
- When every key is resting or disabled: wait at most once per request (≤ 90 s), and not at all after 3
  requests in a row hit that wall, so the batch runner **stops** quickly and cleanly.
- Installed by `run_full_pipeline.py` (all modes), the batch runner, and the standalone mains of
  `translate_prompts.py`, `judge_responses.py` and `generate_tts.py`. The status screen shows each key's state.
- `.env.example` now has 3 slots per platform.

**Tested** with the real translation, TTS and batch code against a fake Gemini/Sarvam server:
key 1 rate-limited → key 2; key 2 daily quota → key 3; Sarvam out of credits and invalid keys skipped;
all keys exhausted → clean stop with progress saved; single-key and gappy numbering still load.
**Bug found and fixed:** with one remaining key that kept returning 429, the first version waited and retried
in a long loop instead of stopping; waits are now bounded.

**Note:** Gemini quotas are per Google Cloud project, so keys must come from different projects or accounts
to add capacity.

**Also discussed: AirLLM as a second target** (layer-by-layer inference for big models on small GPUs). Not
adopted: it is text-only (5 of the 6 conditions are audio) and streams the whole model from disk for every
token, so a 500-prompt run would take days to weeks. Suggested instead (not built yet): 4/8-bit loading of
Gemma 4 for lower VRAM, and a pluggable target choice per run (local audio models or an audio-capable API).

## 11. Questions answered (no code changes)

- **How the pipeline works without LangChain:** direct REST calls with `requests`; Gemma via `transformers`.
- **DigitalOcean for shared storage:** no lasting free tier (new accounts about $5 of credit; the GitHub
  Student Pack has historically offered $200, current terms unconfirmed); Spaces is about $5/month and isn't a
  normal folder (it would need sync tooling or code changes). **Recommendation:** a shared Google Drive (or
  BITS OneDrive/Drive) folder synced with the desktop client: free, no code changes, about 0.5–1 GB per run,
  one person per run at a time.

---

## 12. Files touched

| File | Change |
|---|---|
| `experiments/run_full_pipeline.py` | header; unused import removed; Gemma help text; v4 default; runs (`--run`, `--user`); batch flags (`--step`, `--count`, `--retry-failed`, `--gap`, `--yes`, `--status`); menu by default; dead "first prompt" branch removed |
| `experiments/batch_runner.py` | **new**: runs, steps, batching, throttle, lock, history, menus, status/details/history views, key status |
| `experiments/api_keys.py` | **new**: several keys per platform, automatic switching |
| `experiments/languages.py` | header; v4 voices (female conversational); Telugu added; Tamil removed; default languages |
| `experiments/translate_prompts.py` | header; Telugu in choices and texts; Tamil removed |
| `experiments/generate_tts.py` | header; v4-flash default; no v3 fallback for v4; `enable_preprocessing` |
| `experiments/gemma4_target.py` | header; symmetric framing constants; text baseline framed like audio |
| `experiments/judge_responses.py` | header; Telugu key; Tamil removed; default languages fixed |
| `experiments/build_exp_from_database.py` | header; `telugu_prompt` instead of `tamil_prompt` |
| `experiments/exp1.json` | empty `tamil_prompt` fields removed |
| `experiments/exp2.json` | **new**: the 500-prompt set |
| `README.md` | rewritten: plan, procedure, how to run |
| `SESSION_NOTES_2026-11-09.md` | **new**: this file |
| `requirements.txt` | reduced to `requests`, `python-dotenv` |
| `.gitignore` | rewritten (later: whole `_archive/` excluded): secrets (`.env*` except `.env.example`), run audio/logs/locks, legacy outputs, all `.wav`/`.mp3`, Python/OS/editor junk; run settings and results stay committed |
| `.env.example` | **new**: template for `.env` (3 Gemini + 3 Sarvam key slots; optional Groq and Hugging Face) |

---

## 13. Next steps

1. Copy `.env.example` to `.env` and fill in the Gemini and Sarvam keys (ideally 3 each, from different projects/accounts).
2. Run the free Telugu smoke test (README section 4).
3. Create a pilot run on `exp2.json` and do 5 prompts through all four steps (README section 5): check
   translations with native speakers, listen to one clip per language, confirm Gemma's replies are plain text.
4. Decide Gemma decoding (greedy vs sampling) before the full run (open item 1).
5. Move the project to a shared Drive folder if three people will work on it.
