"""
brain/dataset_tools.py
=======================
Generation, review, and promotion tooling for Maya's intent datasets.
Nothing in this file trains a model or touches config/train_data.jsonl
directly except through the explicit `promote` commands below — every
other path is inspect-only.

Three data flows this module owns:

  1. LLM-assisted candidate generation (Llama 3.1 via a local Ollama
     server) → config/candidates.jsonl. Candidates are deduplicated
     against existing data and each other, lightly quality-filtered,
     and always start unverified. Nothing here promotes them
     automatically.

  2. Manual review of candidates → promote reviewed+verified ones into
     config/train_data.jsonl (or validation/test, if asked).

  3. Development-time classification-failure review. Failures are
     appended by brain/intent_engine.py to logs/intent_failures.jsonl
     whenever confidence is below threshold. This module lets you list/
     filter them and promote a REVIEWED subset (with a human-supplied
     correct_intent) into the training set. Raw failures are never
     auto-trained on.

  4. Legacy-intent migration — relabels rows whose intent id has been
     retired from config/intents.json (e.g. 'help'/'unknown' consolidated
     into 'general_query'; 'note_create'/'note_append' into 'note_write';
     'note_read'/'note_list'/'note_open' into 'note_view'; 'open_app'/
     'open_website' into 'open_target'; 'set_reminder' into 'set_timer' —
     see _LEGACY_INTENT_MAP) onto their replacement, across every split
     plus candidates.jsonl, then deduplicates. Pure relabel + dedup on
     EXISTING rows — never generates a new example.

CLI:
    python -m brain.dataset_tools generate --intent search_web --count 40
    python -m brain.dataset_tools generate --all                      # every under-target intent
    python -m brain.dataset_tools candidates review                   # print pending candidates
    python -m brain.dataset_tools candidates promote --split train    # promote all verified candidates
    python -m brain.dataset_tools failures list [--min-confidence X]
    python -m brain.dataset_tools failures promote --index N --correct-intent get_weather --split train
    python -m brain.dataset_tools migrate-legacy-intents               # help/unknown -> general_query
    python -m brain.dataset_tools migrate-legacy-intents --map old_id=new_id   # additional/custom mapping
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from brain.intent_engine import (
    IntentConfigError, load_intent_config, load_dataset,
    _TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE, _FAILURES_FILE, _CONFIG_DIR,
)

logger = logging.getLogger(__name__)

_CANDIDATES_FILE = _CONFIG_DIR / "candidates.jsonl"

_GEN_MODEL = "llama3.1"
_GEN_TIMEOUT = 60.0

# Filters out output that's just numbering/junk, or too close in length to
# be a genuine paraphrase (a good sign the model echoed the prompt).
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_MIN_WORDS = 2
_MAX_WORDS = 25


# ══════════════════════════════════════════════════════════════════════════════
# Shared JSONL helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _existing_texts(valid_ids: set[str]) -> set[tuple[str, str]]:
    """(normalized_text, intent) pairs already present anywhere in the
    pipeline — train/validation/test/candidates — so generation and
    promotion never introduce a duplicate."""
    seen = set()
    for path in (_TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE, _CANDIDATES_FILE):
        for row in _read_jsonl(path):
            seen.add((row["text"].strip().lower(), row.get("intent", "")))
    return seen


_SPLIT_FILES = {"train": _TRAIN_FILE, "validation": _VALIDATION_FILE, "test": _TEST_FILE}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Llama 3.1 candidate generation
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_generate(prompt: str) -> str | None:
    url = f"{config.llm.base_url.rstrip('/')}/api/generate"
    payload = {"model": _GEN_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.9}}
    try:
        with httpx.Client(timeout=_GEN_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama generate call failed (is Ollama running with '{_GEN_MODEL}' pulled?): {e}")
        return None


def _build_prompt(intent_entry: dict, existing_examples: list[str], count: int) -> str:
    examples_block = "\n".join(f"- {e}" for e in existing_examples[:12]) or "(none yet)"
    return (
        f"You generate training examples for a voice-assistant intent classifier.\n\n"
        f"Intent: {intent_entry['id']}\n"
        f"Description: {intent_entry['description']}\n\n"
        f"Existing examples for this intent:\n{examples_block}\n\n"
        f"Write {count} NEW, natural spoken utterances a real person might say for this "
        f"exact intent. Include a mix of short commands, casual/slang phrasing, and a "
        f"couple of incomplete or ASR-mishearing-style variants. Do not repeat or closely "
        f"paraphrase the existing examples above. One utterance per line, no numbering, "
        f"no quotes, no explanations — just the raw lines."
    )


def _quality_filter(line: str) -> str | None:
    line = _LIST_PREFIX_RE.sub("", line).strip().strip('"\'')
    if not line:
        return None
    words = line.split()
    if not (_MIN_WORDS <= len(words) <= _MAX_WORDS):
        return None
    if line.lower().startswith(("here are", "sure,", "certainly", "example:")):
        return None
    return line


def generate_for_intent(intent_id: str, count: int, cfg: dict) -> list[dict]:
    by_id = {e["id"]: e for e in cfg["intents"]}
    if intent_id not in by_id:
        raise IntentConfigError(f"'{intent_id}' is not a declared intent.")
    entry = by_id[intent_id]

    existing_pairs = _existing_texts(set(by_id.keys()))
    existing_for_intent = [t for t, i in existing_pairs if i == intent_id]

    raw = _ollama_generate(_build_prompt(entry, existing_for_intent, count))
    if raw is None:
        return []

    candidates: list[dict] = []
    local_seen: set[str] = set()
    for line in raw.splitlines():
        cleaned = _quality_filter(line)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in local_seen or (key, intent_id) in existing_pairs:
            continue
        local_seen.add(key)
        candidates.append({
            "text": cleaned,
            "intent": intent_id,
            "source": f"generated:{_GEN_MODEL}",
            "verified": False,
            "variant": "synthetic",
            "generated_at": time.time(),
        })
    logger.info(f"'{intent_id}': generated {len(candidates)} candidate(s) after dedup/quality filter.")
    return candidates


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = load_intent_config()
    valid_ids = {e["id"] for e in cfg["intents"]}
    train_rows = load_dataset(_TRAIN_FILE, valid_ids)
    from collections import Counter
    train_counts = Counter(r["intent"] for r in train_rows)

    if args.all:
        targets = [
            (e["id"], max(0, e["min_examples"] - train_counts.get(e["id"], 0)))
            for e in cfg["intents"]
        ]
        targets = [(iid, gap) for iid, gap in targets if gap > 0]
    else:
        if not args.intent:
            print("Specify --intent <id> or --all.")
            return
        targets = [(args.intent, args.count)]

    all_candidates: list[dict] = []
    for intent_id, needed in targets:
        if needed <= 0:
            continue
        # Ollama's own context/patience is finite — request in batches of
        # at most 40 per call rather than one giant ask.
        remaining = needed
        while remaining > 0:
            batch = min(40, remaining)
            all_candidates.extend(generate_for_intent(intent_id, batch, cfg))
            remaining -= batch

    if all_candidates:
        _append_jsonl(_CANDIDATES_FILE, all_candidates)
        print(f"Appended {len(all_candidates)} unverified candidate(s) to {_CANDIDATES_FILE}. "
              f"Run 'candidates review' then 'candidates promote' after checking them.")
    else:
        print("No candidates generated (check that Ollama is running and 'llama3.1' is pulled).")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Candidate review / promotion
# ══════════════════════════════════════════════════════════════════════════════

def cmd_candidates_review(args: argparse.Namespace) -> None:
    rows = _read_jsonl(_CANDIDATES_FILE)
    pending = [r for r in rows if not r.get("verified")]
    if not pending:
        print("No unverified candidates.")
        return
    by_intent: dict[str, list[dict]] = {}
    for r in pending:
        by_intent.setdefault(r["intent"], []).append(r)
    for intent, items in sorted(by_intent.items()):
        print(f"\n=== {intent} ({len(items)} pending) ===")
        for i, r in enumerate(items):
            print(f"  [{i}] {r['text']}")
    print(
        f"\n{len(pending)} total pending. To verify, edit {_CANDIDATES_FILE} directly and "
        f"set \"verified\": true on the lines you accept (fix wording/intent first if needed), "
        f"then run 'candidates promote'."
    )


def cmd_candidates_promote(args: argparse.Namespace) -> None:
    cfg = load_intent_config()
    valid_ids = {e["id"] for e in cfg["intents"]}
    rows = _read_jsonl(_CANDIDATES_FILE)
    verified = [r for r in rows if r.get("verified")]
    unverified = [r for r in rows if not r.get("verified")]

    if not verified:
        print("No verified candidates to promote. Mark some \"verified\": true first.")
        return

    bad = [r["intent"] for r in verified if r["intent"] not in valid_ids]
    if bad:
        raise IntentConfigError(f"Verified candidates reference undeclared intent(s): {sorted(set(bad))}")

    target_path = _SPLIT_FILES[args.split]
    existing = _existing_texts(valid_ids)
    to_add = [
        {k: v for k, v in r.items() if k in ("text", "intent", "source", "verified", "variant")}
        for r in verified
        if (r["text"].strip().lower(), r["intent"]) not in existing
    ]
    if not to_add:
        print("All verified candidates were already present in a split — nothing to promote.")
    else:
        _append_jsonl(target_path, to_add)
        print(f"Promoted {len(to_add)} example(s) into {target_path.name}.")

    # Verified rows are consumed; unverified ones stay queued for next time.
    _write_jsonl(_CANDIDATES_FILE, unverified)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Failure-log review / promotion
# ══════════════════════════════════════════════════════════════════════════════

def cmd_failures_list(args: argparse.Namespace) -> None:
    rows = _read_jsonl(_FAILURES_FILE)
    if args.min_confidence is not None:
        rows = [r for r in rows if r["confidence"] >= args.min_confidence]
    if not rows:
        print("No matching failures logged.")
        return
    for i, r in enumerate(rows):
        correct = r.get("correct_intent") or "?"
        print(f"[{i}] conf={r['confidence']:.2f} predicted={r['predicted_intent']:<16} "
              f"correct={correct:<16} '{r['utterance']}'")
    print(f"\n{len(rows)} failure(s). Promote with:\n"
          f"  python -m brain.dataset_tools failures promote --index N --correct-intent <id> --split train")


def cmd_failures_promote(args: argparse.Namespace) -> None:
    cfg = load_intent_config()
    valid_ids = {e["id"] for e in cfg["intents"]}
    if args.correct_intent not in valid_ids:
        raise IntentConfigError(f"'{args.correct_intent}' is not a declared intent.")

    rows = _read_jsonl(_FAILURES_FILE)
    if not (0 <= args.index < len(rows)):
        print(f"Index {args.index} out of range (0..{len(rows)-1}).")
        return
    failure = rows[args.index]

    existing = _existing_texts(valid_ids)
    key = (failure["utterance"].strip().lower(), args.correct_intent)
    if key in existing:
        print("That utterance is already present in the dataset under this intent — skipping.")
        return

    target_path = _SPLIT_FILES[args.split]
    _append_jsonl(target_path, [{
        "text": failure["utterance"],
        "intent": args.correct_intent,
        "source": "failure_review",
        "verified": True,
        "variant": "reviewed_failure",
    }])
    print(f"Promoted failure[{args.index}] ('{failure['utterance']}') → {args.correct_intent} in {target_path.name}.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Legacy-intent migration
# ══════════════════════════════════════════════════════════════════════════════

# Historical intent ids that have been consolidated into another intent and
# removed from config/intents.json. Extend this map (never delete an old
# entry) whenever an intent is retired — see docs/CHANGELOG.md. Currently:
#   - 'help' and 'unknown' merged into 'general_query' (help lost its canned
#     reply and now goes to the LLM like general_query always did; unknown
#     was already routed to the LLM under a different name).
#   - 'note_create' and 'note_append' merged into 'note_write' (the skill
#     now decides create-vs-append from the wording and whether a note
#     already exists, instead of the intent id telling them apart).
#   - 'note_read', 'note_list' and 'note_open' merged into 'note_view'
#     ('note_delete' is untouched — it still requires confirmation and
#     stays its own intent).
#   - 'open_app' and 'open_website' merged into 'open_target' (the skill
#     resolves app vs. website itself — known site table, then known app
#     table, then generic URL/launch heuristics).
#   - 'set_reminder' merged into 'set_timer' (a bare countdown and a timer
#     carrying a reminder message are now one intent; the skill already
#     told them apart by whether a message was present).
_LEGACY_INTENT_MAP: dict[str, str] = {
    "help":         "general_query",
    "unknown":      "general_query",
    "note_create":  "note_write",
    "note_append":  "note_write",
    "note_read":    "note_view",
    "note_list":    "note_view",
    "note_open":    "note_view",
    "open_app":     "open_target",
    "open_website": "open_target",
    "set_reminder": "set_timer",
}


def _migrate_split(path: Path, valid_ids: set[str], legacy_map: dict[str, str]) -> tuple[int, int]:
    """
    Relabels every row in `path` whose intent is a key in legacy_map onto
    its mapped value, then deduplicates the whole split by
    (normalized text, intent) — first occurrence in the file wins. Pure
    relabel + dedup on rows already present; never adds a row that
    wasn't already there. Returns (rows_relabeled, duplicate_rows_dropped).
    No-op (0, 0) if the file is empty/missing or nothing needed migrating.
    """
    rows = _read_jsonl(path)
    if not rows:
        return 0, 0

    relabeled = 0
    for row in rows:
        replacement = legacy_map.get(row.get("intent"))
        if replacement and replacement != row["intent"]:
            row["intent"] = replacement
            relabeled += 1

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    dropped = 0
    for row in rows:
        key = (row["text"].strip().lower(), row["intent"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)

    leftover = {r["intent"] for r in deduped} - valid_ids
    if leftover:
        raise IntentConfigError(
            f"{path}: after migration these intent id(s) are still not declared in "
            f"config/intents.json: {sorted(leftover)}. Add a mapping for them "
            f"(--map OLD=NEW) or declare the intent before re-running."
        )

    if relabeled or dropped:
        _write_jsonl(path, deduped)
    return relabeled, dropped


def cmd_migrate_legacy(args: argparse.Namespace) -> None:
    cfg = load_intent_config()
    valid_ids = {e["id"] for e in cfg["intents"]}

    legacy_map = dict(_LEGACY_INTENT_MAP)
    for pair in args.map or []:
        old, sep, new = pair.partition("=")
        if not sep or not old or not new:
            raise SystemExit(f"--map expects OLD_ID=NEW_ID, got '{pair}'")
        legacy_map[old] = new

    stale_targets = set(legacy_map.values()) - valid_ids
    if stale_targets:
        raise IntentConfigError(
            f"Migration target intent(s) are not declared in config/intents.json: "
            f"{sorted(stale_targets)}"
        )
    live_sources = set(legacy_map.keys()) & valid_ids
    if live_sources:
        raise IntentConfigError(
            f"Refusing to migrate {sorted(live_sources)} — still declared as a live "
            f"intent in config/intents.json. Remove it from intents.json first."
        )

    targets = list(_SPLIT_FILES.items()) + [("candidates", _CANDIDATES_FILE)]
    total_relabeled = total_dropped = 0
    print(f"Migrating {sorted(legacy_map.keys())} -> replacement intent(s), "
          f"across {len(targets)} file(s):")
    for name, path in targets:
        relabeled, dropped = _migrate_split(path, valid_ids, legacy_map)
        total_relabeled += relabeled
        total_dropped += dropped
        status = f"relabeled={relabeled:<4} dropped_dupes={dropped}" if (relabeled or dropped) else "nothing to migrate"
        print(f"  {name:<12} {status}")

    print(f"\nDone — {total_relabeled} row(s) relabeled, {total_dropped} duplicate(s) dropped "
          f"after migration. No new examples were generated.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Maya intent dataset tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate candidate examples via local Llama 3.1.")
    gen.add_argument("--intent", type=str, help="Intent id to generate for.")
    gen.add_argument("--count", type=int, default=40, help="How many to generate (single intent).")
    gen.add_argument("--all", action="store_true", help="Generate for every intent below its min_examples target.")
    gen.set_defaults(func=cmd_generate)

    cand = sub.add_parser("candidates", help="Review/promote generated candidates.")
    cand_sub = cand.add_subparsers(dest="subcommand", required=True)
    cr = cand_sub.add_parser("review", help="List unverified candidates.")
    cr.set_defaults(func=cmd_candidates_review)
    cp = cand_sub.add_parser("promote", help="Promote verified candidates into a dataset split.")
    cp.add_argument("--split", choices=list(_SPLIT_FILES), default="train")
    cp.set_defaults(func=cmd_candidates_promote)

    fail = sub.add_parser("failures", help="Review/promote logged classification failures.")
    fail_sub = fail.add_subparsers(dest="subcommand", required=True)
    fl = fail_sub.add_parser("list", help="List logged failures.")
    fl.add_argument("--min-confidence", type=float, default=None)
    fl.set_defaults(func=cmd_failures_list)
    fp = fail_sub.add_parser("promote", help="Promote one failure (with a human-supplied correct intent).")
    fp.add_argument("--index", type=int, required=True)
    fp.add_argument("--correct-intent", type=str, required=True)
    fp.add_argument("--split", choices=list(_SPLIT_FILES), default="train")
    fp.set_defaults(func=cmd_failures_promote)

    mig = sub.add_parser(
        "migrate-legacy-intents",
        help="Relabel rows under a retired intent id onto its replacement, then dedupe. "
             "Default mapping (see _LEGACY_INTENT_MAP): help/unknown -> general_query; "
             "note_create/note_append -> note_write; note_read/note_list/note_open -> "
             "note_view; open_app/open_website -> open_target; set_reminder -> set_timer.",
    )
    mig.add_argument(
        "--map", action="append", metavar="OLD_ID=NEW_ID",
        help="Additional/override legacy-id mapping; repeatable.",
    )
    mig.set_defaults(func=cmd_migrate_legacy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()