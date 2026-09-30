"""
brain/dataset_tools.py
=======================
Generation, review, and promotion tooling for Maya's intent datasets.
Nothing in this file trains a model or touches datasets/training/train_data.jsonl
directly except through the explicit `promote` commands below — every
other path is inspect-only.

Four data flows this module owns:

  1. LLM-assisted candidate generation (Llama 3.1 via a local Ollama
     server) → datasets/training/candidates.jsonl. Candidates are deduplicated
     against existing data and each other, lightly quality-filtered,
     and always start unverified. Nothing here promotes them
     automatically.

  1B. Automated candidate audit (NEW) — runs deterministically (plus an
     optional, capped LLM-assisted pass for genuinely ambiguous cases)
     over every unverified candidate BEFORE a human ever looks at it.
     `cmd_generate` runs this automatically right after appending new
     candidates; `candidates audit` re-runs it on demand (e.g. after
     hand-editing candidates.jsonl, or in CI). See "1B" below for the
     full list of checks. Auditing never promotes anything and never
     marks a candidate `"verified"` — a human still owns that step —
     it only removes/annotates candidates.jsonl so `candidates review`
     shows a cleaner, pre-triaged queue, and writes a full report
     (`audit_report.json` + printed summary) a human can act on.

  2. Manual review of candidates → promote reviewed+verified ones into
     datasets/training/train_data.jsonl (or validation/test, if asked).

  3. Development-time classification-failure review. Failures are
     appended by brain/intent_engine.py to logs/intent_failures.jsonl
     whenever confidence is below threshold. This module lets you list/
     filter them and promote a REVIEWED subset (with a human-supplied
     correct_intent) into the training set. Raw failures are never
     auto-trained on.

  4. Legacy-intent migration — relabels rows whose intent id has been
     retired from datasets/intents.json (e.g. 'note_create'/'note_append' into 'note_write';
     'note_read'/'note_list'/'note_open' into 'note_view'; 'open_app'/
     'open_website' into 'open_target'; 'set_reminder' into 'set_timer' —
     see _LEGACY_INTENT_MAP) onto their replacement, across every split
     plus candidates.jsonl, then deduplicates. Pure relabel + dedup on
     EXISTING rows — never generates a new example.

CLI:
    python -m brain.dataset_tools generate --intent search_web --count 40
    python -m brain.dataset_tools generate --all                      # every under-target intent
    python -m brain.dataset_tools generate --all --use-llm             # also LLM-audit ambiguous candidates
    python -m brain.dataset_tools generate --all --no-audit            # skip the automatic post-gen audit
    python -m brain.dataset_tools candidates audit                     # re-run the audit on demand
    python -m brain.dataset_tools candidates audit --use-llm --max-llm-calls 30
    python -m brain.dataset_tools candidates review                   # print pending candidates (+ audit flags)
    python -m brain.dataset_tools candidates promote --split train    # promote all verified candidates
    python -m brain.dataset_tools failures list [--min-confidence X]
    python -m brain.dataset_tools failures promote --index N --correct-intent get_weather --split train
    python -m brain.dataset_tools migrate-legacy-intents               # help/unknown -> general_query
    python -m brain.dataset_tools migrate-legacy-intents --map old_id=new_id   # additional/custom mapping
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import re
import sys
import time
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from brain.intent_engine import (
    IntentConfigError, load_intent_config, load_dataset, build_keyword_patterns,
    _TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE, _FAILURES_FILE, _DATASETS_DIR,
)

logger = logging.getLogger(__name__)

_CANDIDATES_FILE          = _DATASETS_DIR/ "training" / "candidates.jsonl"
_CANDIDATES_REJECTED_FILE = _DATASETS_DIR/ "training" / "candidates_rejected.jsonl"
_AUDIT_REPORT_FILE        = _DATASETS_DIR/ "training" / "audit_report.json"

_GEN_MODEL = "llama3.1"
_GEN_TIMEOUT = 60.0

# Filters out output that's just numbering/junk, or too close in length to
# be a genuine paraphrase (a good sign the model echoed the prompt).
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_MIN_WORDS = 2
_MAX_WORDS = 25


def _candidate_count_for_training_examples(training_examples: int) -> int:
    """Return the candidate batch size needed for an 80% training slice."""
    return math.ceil(training_examples / 0.8)


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
    pipeline — train/validation/test/candidates — used by GENERATION so
    it never regenerates something that's already sitting in
    candidates.jsonl (verified or not). Deliberately includes
    candidates.jsonl for that reason.

    Do NOT use this for promotion's own dedup check — see
    `_trusted_split_texts()` below. A row being promoted always lives in
    candidates.jsonl too, so checking "already exists" against a set that
    includes candidates.jsonl would make every verified candidate match
    itself."""
    seen = set()
    for path in (_TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE, _CANDIDATES_FILE):
        for row in _read_jsonl(path):
            seen.add((row["text"].strip().lower(), row.get("intent", "")))
    return seen


def _trusted_split_texts(valid_ids: set[str]) -> set[tuple[str, str]]:
    """(normalized_text, intent) pairs already PROMOTED into train/
    validation/test — the trusted dataset. candidates.jsonl is untrusted
    staging (see module docstring's flow 1B) and is deliberately excluded
    here: the whole point of `candidates promote` is moving a row OUT of
    staging and INTO one of these files, and the row being promoted is
    still sitting in candidates.jsonl at the moment this runs, so
    including it would make it look like it "already exists" as itself.
    Use `_existing_texts()` instead when candidates.jsonl genuinely
    should count (e.g. generation avoiding regenerating a pending
    candidate)."""
    seen = set()
    for path in (_TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE):
        for row in _read_jsonl(path):
            seen.add((row["text"].strip().lower(), row.get("intent", "")))
    return seen


_SPLIT_FILES = {"train": _TRAIN_FILE, "validation": _VALIDATION_FILE, "test": _TEST_FILE}


# ══════════════════════════════════════════════════════════════════════════════
# 1B. Automated candidate audit
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs BEFORE a human ever reviews a candidate. Deterministic/local checks
# do all the heavy lifting (no network, no model, fully unit-testable);
# an LLM-assisted pass is layered on top ONLY for candidates a deterministic
# check has already flagged as ambiguous (confusable vocabulary/keywords),
# and is capped (`max_llm_calls`) and opt-in (`use_llm=False` by default)
# so a plain "audit what I already generated" run never silently needs a
# live Ollama server.
#
# Every check operates on candidate rows still carrying `{"text", "intent"}`
# — the same shape `generate_for_intent` produces — so this also audits
# hand-added or hand-edited candidates.jsonl entries, not just generated
# ones. Nothing here ever sets `"verified"`; that stays a human decision.

# ── Tuning ───────────────────────────────────────────────────────────────────
_NEAR_DUP_RATIO                        = 0.90   # difflib.SequenceMatcher ratio treated as "near-duplicate"
_CONFUSION_MARGIN                      = 0.05   # a neighboring intent must beat the claimed one by at least this much
_MIN_CONFUSABLE_SCORE                  = 0.15   # below this, vocabulary overlap is noise, not a real signal
_TEMPLATE_CLUSTER_MIN                  = 4      # cluster size (per intent) that counts as "over-represented"
_TEMPLATE_CLUSTER_KEEP                 = 2      # canonical examples of a cluster left unflagged
_DIVERSITY_MIN_UNIQUE_UNIGRAM_RATIO    = 0.35   # below this (and count>=_DIVERSITY_MIN_COUNT) -> low-diversity warning
_DIVERSITY_MIN_COUNT                   = 5
_GENERIC_MIN_INTENT_AVG_LEN            = 2.5    # only intents whose own examples average at least this many
                                                 # content words trigger the overly-generic check — a naturally
                                                 # terse intent (e.g. "mute") never trips it
_MAX_LLM_AUDIT_CALLS                   = 15     # hard cap on Ollama calls per audit run, cost/latency control

# Independent of intent_engine._tokenize (which is private to that module) —
# this one keeps apostrophes (so "don't"/"what's" tokenize as one word) and
# is only ever used for audit heuristics, never for training/classification.
_TOKEN_RE = re.compile(r"[a-z0-9']+")

_AUDIT_STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "but", "it",
    "is", "are", "was", "were", "be", "please", "can", "you", "your", "me",
    "my", "mine", "i", "im", "that", "this", "do", "does", "did", "what",
    "how", "just", "really", "so", "up", "down", "at", "with", "if", "then",
    "there", "here", "not", "no", "yes", "am", "will", "would", "should",
    "could", "want", "like", "get", "got", "right", "now",
}

_SLOT_NUM_RE    = re.compile(r"\b\d+(?:\.\d+)?\b")
_SLOT_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]*\b")

_VALID_VERDICT_RE   = re.compile(r"^\s*valid\s*$", re.IGNORECASE)
_INVALID_VERDICT_RE = re.compile(r"invalid\s*:\s*([a-zA-Z0-9_]+)", re.IGNORECASE)

# ── Restored-intent semantic checks (help / unknown) ────────────────────────
# 'help' requires an actual, explicit request for assistance — not just any
# utterance that happens to be short or vague.
_HELP_REQUEST_RE = re.compile(
    r"\b(help|assist(?:ance)?|support me|i'?m stuck|i'?m confused|"
    r"what can you do|how do i|how does this work|walk me through|"
    r"show me how|can you help|i need help|i want help)\b",
    re.IGNORECASE,
)

# ── Closely-related skill-intent boundary phrases ───────────────────────────
# Curated, deterministic phrase patterns for intent pairs that vocabulary-
# overlap (Jaccard) alone can't reliably separate — the two intents often
# share almost no vocabulary in common but a handful of *specific phrasings*
# sit right on the boundary between them. A hit here doesn't say who's
# right; it just means the pair needs a human semantic call.
_SKILL_AMBIGUOUS_PAIRS: list[tuple[frozenset[str], "re.Pattern[str]"]] = [
    (frozenset({"play_music", "volume_up"}),
     re.compile(r"\bturn\s+(?:it|that|the volume)?\s*up\b", re.IGNORECASE)),
    (frozenset({"next_track", "open_target"}),
     re.compile(r"\b(?:next up|move on|next one|skip (?:ahead|to next))\b", re.IGNORECASE)),
    (frozenset({"get_time", "get_date"}),
     re.compile(r"\b(?:my birthday|date of birth|what day is it|what'?s?\s*(?:is\s*)?today)\b",
                re.IGNORECASE)),
    (frozenset({"cancel_timer", "set_timer"}),
     re.compile(r"\b(?:stop|cancel|end)\b.*\btimer\b|\btimer\b.*\b(?:off|cancel|stop|end)\b",
                re.IGNORECASE)),
    (frozenset({"mute", "volume_down"}),
     re.compile(r"\bturn\s+(?:it|that|the volume)?\s*down\b|\bquiet(?:er)?\b", re.IGNORECASE)),
    (frozenset({"search_web", "general_query"}),
     re.compile(r"\bsearch\b|\blook up\b|\bgoogle\b", re.IGNORECASE)),
]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> set[str]:
    """Tokens with stopwords and single characters removed — the
    vocabulary that actually carries an utterance's meaning."""
    return {t for t in _tokens(text) if t not in _AUDIT_STOPWORDS and len(t) > 1}


def _normalize_text(text: str) -> str:
    """Whitespace/punctuation/case-insensitive form used for exact and
    near-duplicate comparison — NOT used for training."""
    return " ".join(_tokens(text))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def _template_signature(text: str) -> str:
    """Collapses an utterance's variable slots (numbers, capitalized
    names) into placeholders so near-identical paraphrase templates
    ('set a timer for 5 minutes' / 'set a timer for 20 minutes') group
    together regardless of which number/name was substituted in."""
    # Proper-noun substitution runs FIRST, on the original text — running
    # it after the numeric substitution would re-match the placeholder
    # itself (e.g. "<NUM>" starts with an uppercase letter and would be
    # turned into "<NAME>", producing a mangled "<<NAME>>").
    t = _SLOT_PROPER_RE.sub("<NAME>", text)
    t = _SLOT_NUM_RE.sub("<NUM>", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9<> ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class AuditIssue:
    index: int
    code: str        # e.g. "malformed", "duplicate_batch", "vocab_confusable", ...
    severity: str    # "reject" | "flag" | "info"
    # "info" is a non-blocking annotation: it never rejects, never counts
    # toward "flagged", and never lands in the human review_queue — it
    # exists purely so a genuinely valid ML-only paraphrase (e.g. one that
    # simply lacks a deterministic keyword) still leaves a visible trail
    # in the row's audit issues and in `report.reasons`, without demanding
    # a human look at it.
    detail: str


def _issue_dict(issue: AuditIssue) -> dict:
    return {"code": issue.code, "severity": issue.severity, "detail": issue.detail}


@dataclass
class AuditReport:
    """Everything the task asks for: totals, per-intent stats, rejection
    reasons, suspected intent-confusion pairs, template/diversity
    clusters, and the subset that still needs a human look."""
    total: int = 0
    accepted: int = 0
    flagged: int = 0
    rejected: int = 0
    per_intent: dict = field(default_factory=dict)
    reasons: Counter = field(default_factory=Counter)
    confusion_pairs: Counter = field(default_factory=Counter)          # (claimed, suspected) -> count
    template_clusters: list = field(default_factory=list)              # [{"intent","signature","size","examples"}]
    diversity_warnings: list = field(default_factory=list)             # [{"intent","count","unique_unigram_ratio",...}]
    review_queue: list = field(default_factory=list)                   # [{"text","intent","issues":[codes]}]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "flagged": self.flagged,
            "rejected": self.rejected,
            "per_intent": self.per_intent,
            "reasons": dict(self.reasons),
            "confusion_pairs": [
                {"claimed": claimed, "suspected": suspected, "count": n}
                for (claimed, suspected), n in self.confusion_pairs.most_common()
            ],
            "template_clusters": self.template_clusters,
            "diversity_warnings": self.diversity_warnings,
            "review_queue": self.review_queue,
        }

    def to_text(self) -> str:
        lines = [
            "=== Candidate Audit Report ===",
            f"total={self.total}  accepted={self.accepted}  "
            f"flagged={self.flagged}  rejected={self.rejected}",
        ]
        if self.reasons:
            lines.append("\nReasons (rejections + flags):")
            for code, n in self.reasons.most_common():
                lines.append(f"  {code:<26} {n}")
        if self.per_intent:
            lines.append("\nPer-intent:")
            for intent, s in sorted(self.per_intent.items()):
                lines.append(
                    f"  {intent:<20} total={s['total']:<4} accepted={s['accepted']:<4} "
                    f"flagged={s['flagged']:<4} rejected={s['rejected']}"
                )
        if self.confusion_pairs:
            lines.append("\nSuspected intent-confusion pairs (claimed -> suspected: count):")
            for (claimed, suspected), n in self.confusion_pairs.most_common(15):
                lines.append(f"  {claimed} -> {suspected}: {n}")
        if self.template_clusters:
            lines.append("\nTemplate/paraphrase clusters:")
            for c in self.template_clusters:
                lines.append(f"  [{c['intent']}] size={c['size']}  signature='{c['signature']}'")
                for ex in c["examples"]:
                    lines.append(f"      - {ex}")
        if self.diversity_warnings:
            lines.append("\nLow-diversity intents (this batch reads as repetitive):")
            for w in self.diversity_warnings:
                lines.append(
                    f"  {w['intent']:<20} unique_unigram_ratio={w['unique_unigram_ratio']} "
                    f"unique_first_word_ratio={w['unique_first_word_ratio']}  (n={w['count']})"
                )
        if self.review_queue:
            shown = self.review_queue[:30]
            lines.append(f"\n{len(self.review_queue)} candidate(s) flagged for human review:")
            for r in shown:
                lines.append(f"  [{r['intent']}] '{r['text']}'  — {', '.join(r['issues'])}")
            if len(self.review_queue) > len(shown):
                lines.append(f"  … and {len(self.review_queue) - len(shown)} more (see {_AUDIT_REPORT_FILE.name}).")
        return "\n".join(lines)


class CandidateAuditor:
    """Holds the (cheap, precomputed) per-intent vocabulary and keyword
    patterns used by every deterministic check, so a single audit run
    builds them once and reuses them across every candidate row."""

    def __init__(self, cfg: dict, train_rows: list[dict]):
        self.cfg = cfg
        self.by_id = {e["id"]: e for e in cfg["intents"]}
        self.valid_ids = set(self.by_id.keys())
        self.keyword_patterns = build_keyword_patterns(cfg)

        self._examples_by_intent: dict[str, list[str]] = defaultdict(list)
        for row in train_rows:
            if row.get("intent") in self.valid_ids:
                self._examples_by_intent[row["intent"]].append(row["text"])
        for entry in cfg["intents"]:
            # An intent with zero training rows still gets *something* to
            # compare against, so a brand-new intent isn't automatically
            # "confusable with everything" for lack of vocabulary.
            self._examples_by_intent[entry["id"]].append(entry["description"])
            self._examples_by_intent[entry["id"]].extend(entry.get("keywords", []))

        self._vocab_cache: dict[str, set[str]] = {}
        self._avg_len_cache: dict[str, float] = {}

    def _intent_vocab(self, intent_id: str) -> set[str]:
        if intent_id not in self._vocab_cache:
            vocab: set[str] = set()
            for text in self._examples_by_intent.get(intent_id, []):
                vocab |= _content_tokens(text)
            self._vocab_cache[intent_id] = vocab
        return self._vocab_cache[intent_id]

    def _intent_avg_content_len(self, intent_id: str) -> float:
        if intent_id not in self._avg_len_cache:
            lens = [len(_content_tokens(t)) for t in self._examples_by_intent.get(intent_id, [])]
            lens = [n for n in lens if n > 0]
            self._avg_len_cache[intent_id] = (sum(lens) / len(lens)) if lens else 0.0
        return self._avg_len_cache[intent_id]

    def keyword_matches(self, norm_text: str) -> list[str]:
        return [iid for iid, pattern in self.keyword_patterns if pattern.search(norm_text)]

    def best_alternate_intent(self, tokens: set[str], claimed: str) -> tuple[str | None, float, float]:
        """(best_other_intent_id_or_None, best_other_score, claimed_score) —
        vocabulary-overlap nearest-neighbor among every OTHER declared
        intent, vs. the claimed one's own overlap."""
        claimed_score = _jaccard(tokens, self._intent_vocab(claimed))
        best_id, best_score = None, 0.0
        for iid in self.valid_ids:
            if iid == claimed:
                continue
            score = _jaccard(tokens, self._intent_vocab(iid))
            if score > best_score:
                best_id, best_score = iid, score
        return best_id, best_score, claimed_score

    def is_overly_generic(self, text: str, intent_id: str) -> bool:
        """Flags a candidate whose content is thinner than the intent's
        own examples typically need — adaptive per intent (so a naturally
        terse intent like 'mute'/'thanks' never trips this)."""
        avg = self._intent_avg_content_len(intent_id)
        return avg >= _GENERIC_MIN_INTENT_AVG_LEN and len(_content_tokens(text)) <= 1

    def template_signature(self, text: str) -> str:
        return _template_signature(text)


def _official_text_index(valid_ids: set[str], include_verified_candidates: bool = True) -> dict[str, str]:
    """normalized_text -> intent, first occurrence wins, built from the
    already-TRUSTED data: train/validation/test, plus any candidate a
    human has already marked verified=true (trusted, just not yet
    promoted). Used to catch a new candidate that's an exact duplicate
    of — or directly contradicts the label of — something already
    accepted, independent of whatever else is in the current batch."""
    index: dict[str, str] = {}
    for path in (_TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE):
        for row in _read_jsonl(path):
            if row.get("intent") in valid_ids:
                index.setdefault(_normalize_text(row["text"]), row["intent"])
    if include_verified_candidates:
        for row in _read_jsonl(_CANDIDATES_FILE):
            if row.get("verified") and row.get("intent") in valid_ids:
                index.setdefault(_normalize_text(row["text"]), row["intent"])
    return index


def _ollama_audit_judgment(text: str, claimed_entry: dict, alt_entries: list[dict]) -> str | None:
    """Asks the LLM whether `text` genuinely/unambiguously fits
    claimed_entry's intent. Returns 'VALID', a suggested intent id,
    'none', or None on any failure/unparsable response — fail-open,
    since a flaky LLM call must never block or corrupt the audit."""
    alt_lines = "\n".join(f"- {e['id']}: {e['description']}" for e in alt_entries)
    prompt = (
        "You are auditing training data for a voice-assistant intent classifier.\n"
        f"Utterance: \"{text}\"\n"
        f"Claimed intent: {claimed_entry['id']} — {claimed_entry['description']}\n"
        + (f"Other intents that may fit better:\n{alt_lines}\n" if alt_lines else "")
        + "Does the utterance genuinely and unambiguously express the claimed intent? "
          "Reply with EXACTLY one line: 'VALID' if the claimed intent is correct, or "
          "'INVALID: <intent_id>' naming a better-fitting intent id (or 'INVALID: none' "
          "if none of the intents fit well). No other text."
    )
    raw = _ollama_generate(prompt)
    if not raw or not raw.strip():
        return None
    line = raw.strip().splitlines()[0].strip()
    if _VALID_VERDICT_RE.match(line):
        return "VALID"
    m = _INVALID_VERDICT_RE.search(line)
    if m:
        return m.group(1).lower()
    return None


def audit_candidate_rows(
    pending: list[dict],
    cfg: dict,
    train_rows: list[dict],
    official_index: dict[str, str],
    use_llm: bool = False,
    max_llm_calls: int = _MAX_LLM_AUDIT_CALLS,
) -> tuple[list[dict], list[dict], AuditReport]:
    """
    Audits `pending` (unverified candidate rows: at least {"text","intent"})
    and returns (kept_rows, rejected_rows, report):
      - kept_rows     — same rows, each annotated with row["audit"] =
                        {"status": "passed"|"flagged", "issues":[...]}.
                        Order-preserving; nothing here is ever promoted or
                        marked verified.
      - rejected_rows — rows annotated {"status": "rejected", "issues":[...]},
                        meant for a rejected-candidates file, never fed
                        back into generation/review.
      - report        — full AuditReport (see dataclass above).

    Deterministic checks (always run, no network):
      malformed, duplicate (within-batch and vs. the trusted dataset),
      contradiction (same/near-identical text under conflicting intents),
      overly-generic, keyword/vocabulary-based intent confusability split
      into a genuine "keyword_boundary_conflict" vs. a merely coincidental
      "keyword_gap" (info-only, never blocks a valid ML-only paraphrase),
      restored-intent semantic checks for 'help' (must contain an actual
      request for assistance) and 'unknown' (must not cleanly match any
      intent's keyword rule), curated boundary-phrase checks for closely
      related skill-intent pairs (play_music/volume_up, next_track/
      open_target, get_time/get_date, cancel_timer/set_timer, mute/
      volume_down, search_web/general_query), near-duplicate, template/
      paraphrase clustering, per-intent linguistic diversity.

    Optional LLM-assisted check (`use_llm=True`, capped at
    `max_llm_calls`): only for candidates a deterministic check already
    flagged as vocabulary/keyword-confusable — semantic judgment is used
    where it's actually needed, not on every row.
    """
    valid_ids = {e["id"] for e in cfg["intents"]}
    auditor = CandidateAuditor(cfg, train_rows)
    report = AuditReport(total=len(pending))

    issues_by_index: dict[int, list[AuditIssue]] = defaultdict(list)
    seen_batch: dict[tuple[str, str], int] = {}
    batch_texts_by_intent: dict[str, list[tuple[int, str]]] = defaultdict(list)
    template_groups: dict[tuple[str, str], list[int]] = defaultdict(list)

    def _add(idx: int, code: str, severity: str, detail: str) -> None:
        issues_by_index[idx].append(AuditIssue(idx, code, severity, detail))

    for idx, row in enumerate(pending):
        text = row.get("text")
        intent = row.get("intent")

        if not isinstance(text, str) or not text.strip() or not isinstance(intent, str) or intent not in valid_ids:
            _add(idx, "malformed", "reject",
                 "missing/empty text, or intent not declared in intents.json")
            continue

        norm = _normalize_text(text)
        if not norm:
            _add(idx, "malformed", "reject", "text has no usable content after normalization")
            continue

        batch_key = (norm, intent)
        if batch_key in seen_batch:
            _add(idx, "duplicate_batch", "reject",
                 f"exact duplicate of candidate #{seen_batch[batch_key]} in this batch")
            continue
        seen_batch[batch_key] = idx

        official_intent = official_index.get(norm)
        if official_intent is not None:
            if official_intent == intent:
                _add(idx, "duplicate_existing", "reject",
                     "already present in train/validation/test (or previously verified)")
            else:
                _add(idx, "contradictory_existing", "reject",
                     f"identical text is already labeled '{official_intent}' in the trusted dataset")
                report.confusion_pairs[(intent, official_intent)] += 1
            continue

        # From here the row is provisionally KEPT — remaining checks only flag.
        tokens = _content_tokens(text)

        if auditor.is_overly_generic(text, intent):
            _add(idx, "overly_generic", "flag",
                 "too few content words for an intent whose examples are usually more specific")

        # ── keyword signal: split into "genuine boundary conflict" vs.
        # "coincidental overlap on an ML-only paraphrase" (see module
        # docstring item 1B). Both are always "flag"/"info" severity —
        # a keyword-level disagreement alone never auto-rejects a candidate.
        kw_matches = auditor.keyword_matches(norm)
        own_match = intent in kw_matches
        other_matches = [i for i in kw_matches if i != intent]
        best_other, best_score, claimed_score = auditor.best_alternate_intent(tokens, intent)
        vocab_favors_other = (
            best_other is not None
            and best_score >= _MIN_CONFUSABLE_SCORE
            and best_score >= claimed_score + _CONFUSION_MARGIN
        )
        keyword_conflict_used_vocab = False

        if own_match and other_matches:
            _add(idx, "keyword_overlap", "flag",
                 f"also matches keyword rule(s) for {other_matches}")
            for other in other_matches:
                report.confusion_pairs[(intent, other)] += 1
        elif len(other_matches) > 1:
            _add(idx, "keyword_boundary_conflict", "flag",
                 f"matches multiple other intents' keyword rules {other_matches}, not '{intent}''s own")
            for other in other_matches:
                report.confusion_pairs[(intent, other)] += 1
        elif len(other_matches) == 1:
            other = other_matches[0]
            if vocab_favors_other and best_other == other:
                # The keyword hit AND the vocabulary both point away from
                # the claimed intent — a genuine conflict with that
                # intent's keyword/rule boundary, not a coincidence.
                _add(idx, "keyword_boundary_conflict", "flag",
                     f"wording conflicts with '{other}''s keyword/rule boundary — vocabulary "
                     f"overlap also favors '{other}' ({best_score:.2f}) over '{intent}' "
                     f"({claimed_score:.2f})")
                report.confusion_pairs[(intent, other)] += 1
                keyword_conflict_used_vocab = True
            else:
                # Only the keyword pattern fired, not the vocabulary —
                # this is the "simply lacks a deterministic keyword"
                # case: a semantically valid, ML-only paraphrase that
                # happens to brush against another intent's surface
                # pattern. Never auto-rejected, never pushed to the
                # human review queue.
                _add(idx, "keyword_gap", "info",
                     f"lacks a deterministic keyword for '{intent}' but only superficially "
                     f"touches '{other}''s keyword pattern (vocabulary still favors '{intent}'); "
                     f"likely a valid ML-only paraphrase")
        # else: no keyword pattern matched anywhere — a plain ML-only
        # candidate with no keyword signal at all. Nothing to flag.

        if vocab_favors_other and not keyword_conflict_used_vocab:
            _add(idx, "vocab_confusable", "flag",
                 f"vocabulary overlap fits '{best_other}' (score={best_score:.2f}) better than "
                 f"'{intent}' (score={claimed_score:.2f})")
            report.confusion_pairs[(intent, best_other)] += 1

        # ── restored-intent semantic checks: 'help' and 'unknown' ───────────
        if intent == "help" and not _HELP_REQUEST_RE.search(text):
            if other_matches:
                _add(idx, "help_looks_stale", "flag",
                     f"no explicit request for assistance, and wording cleanly matches "
                     f"concrete intent {other_matches} — check whether this predates 'help' "
                     f"being restored (see _LEGACY_INTENT_MAP)")
            else:
                _add(idx, "help_missing_request", "flag",
                     "labeled 'help' but contains no actual request for assistance")

        if intent == "unknown" and kw_matches:
            _add(idx, "unknown_looks_classifiable", "flag",
                 f"labeled 'unknown' but cleanly matches keyword rule(s) for {kw_matches} — "
                 f"genuine 'unknown' examples shouldn't match any intent's rule; check for "
                 f"stale pre-restoration mislabeling")

        # ── closely-related skill-intent boundary phrases ───────────────────
        for pair, pattern in _SKILL_AMBIGUOUS_PAIRS:
            if intent in pair and pattern.search(text):
                counterpart = next(iter(pair - {intent}))
                _add(idx, "skill_pair_ambiguous", "flag",
                     f"wording sits on the '{intent}'/'{counterpart}' boundary — needs semantic "
                     f"review to confirm '{intent}' over '{counterpart}'")
                report.confusion_pairs[(intent, counterpart)] += 1
                break

        # Near-duplicate vs. an earlier candidate of the SAME intent already kept this batch.
        for other_idx, other_norm in batch_texts_by_intent[intent]:
            if difflib.SequenceMatcher(None, norm, other_norm).ratio() >= _NEAR_DUP_RATIO:
                _add(idx, "near_duplicate", "flag",
                     f"near-duplicate of candidate #{other_idx} ('{pending[other_idx]['text']}')")
                break

        # Near-identical text under a DIFFERENT intent anywhere in the batch —
        # a much stronger signal than plain low diversity.
        for other_intent, texts in batch_texts_by_intent.items():
            if other_intent == intent:
                continue
            hit = False
            for other_idx, other_norm in texts:
                if difflib.SequenceMatcher(None, norm, other_norm).ratio() >= _NEAR_DUP_RATIO:
                    _add(idx, "contradictory", "flag",
                         f"near-identical to candidate #{other_idx} labeled '{other_intent}'")
                    report.confusion_pairs[(intent, other_intent)] += 1
                    hit = True
                    break
            if hit:
                break

        batch_texts_by_intent[intent].append((idx, norm))
        signature = auditor.template_signature(text)
        template_groups[(intent, signature)].append(idx)

    # ── Template/paraphrase clusters ────────────────────────────────────────
    # Flag everything beyond the first _TEMPLATE_CLUSTER_KEEP canonical
    # examples of an over-represented pattern (rejected rows never count
    # toward a cluster — they're already gone).
    for (intent, signature), indices in template_groups.items():
        surviving = [i for i in indices if not any(iss.severity == "reject" for iss in issues_by_index.get(i, []))]
        if len(surviving) >= _TEMPLATE_CLUSTER_MIN:
            report.template_clusters.append({
                "intent": intent, "signature": signature, "size": len(surviving),
                "examples": [pending[i]["text"] for i in surviving[:5]],
            })
            for i in surviving[_TEMPLATE_CLUSTER_KEEP:]:
                _add(i, "template_cluster", "flag",
                     f"one of {len(surviving)} near-identical templates for '{intent}' "
                     f"(pattern: '{signature}')")

    # ── Per-intent linguistic/syntactic diversity ───────────────────────────
    # Report-level (a diversity problem is a property of the whole batch,
    # not any single row) — reviewers see WHY a batch reads as repetitive.
    for intent, texts in batch_texts_by_intent.items():
        if len(texts) < _DIVERSITY_MIN_COUNT:
            continue
        all_tokens: list[str] = []
        first_words: list[str] = []
        for _, norm in texts:
            toks = norm.split()
            all_tokens.extend(toks)
            if toks:
                first_words.append(toks[0])
        unique_ratio = (len(set(all_tokens)) / len(all_tokens)) if all_tokens else 1.0
        first_word_ratio = (len(set(first_words)) / len(first_words)) if first_words else 1.0
        if unique_ratio < _DIVERSITY_MIN_UNIQUE_UNIGRAM_RATIO:
            report.diversity_warnings.append({
                "intent": intent, "count": len(texts),
                "unique_unigram_ratio": round(unique_ratio, 3),
                "unique_first_word_ratio": round(first_word_ratio, 3),
            })

    # ── Optional LLM-assisted semantic pass ─────────────────────────────────
    # Only for rows already flagged confusable by a deterministic check, and
    # only up to max_llm_calls — semantic judgment where it's actually
    # needed, never as a blanket per-row classifier call.
    if use_llm and max_llm_calls > 0:
        ambiguous = [
            idx for idx, issues in issues_by_index.items()
            if any(i.code in ("vocab_confusable", "keyword_mismatch") for i in issues)
            and not any(i.severity == "reject" for i in issues)
        ]
        for idx in ambiguous[:max_llm_calls]:
            row = pending[idx]
            claimed_entry = auditor.by_id[row["intent"]]
            alt_ids = sorted({
                suspected for (claimed, suspected), _ in report.confusion_pairs.items()
                if claimed == row["intent"]
            })
            alt_entries = [auditor.by_id[a] for a in alt_ids if a in auditor.by_id]
            verdict = _ollama_audit_judgment(row["text"], claimed_entry, alt_entries)
            if verdict is None or verdict == "VALID":
                continue
            if verdict == "none":
                _add(idx, "llm_flagged_mislabel", "flag",
                     "LLM judged this doesn't clearly fit the claimed intent")
            else:
                _add(idx, "llm_flagged_mislabel", "flag",
                     f"LLM judged this a better fit for '{verdict}'")
                if verdict in valid_ids:
                    report.confusion_pairs[(row["intent"], verdict)] += 1

    # ── Assemble final rows + report ─────────────────────────────────────────
    kept: list[dict] = []
    rejected: list[dict] = []
    for idx, row in enumerate(pending):
        issues = issues_by_index.get(idx, [])
        intent = row.get("intent") if isinstance(row.get("intent"), str) else "?"
        stat = report.per_intent.setdefault(intent, {"total": 0, "accepted": 0, "flagged": 0, "rejected": 0})
        stat["total"] += 1
        for issue in issues:
            report.reasons[issue.code] += 1

        out_row = dict(row)
        if any(i.severity == "reject" for i in issues):
            out_row["audit"] = {"status": "rejected", "issues": [_issue_dict(i) for i in issues]}
            rejected.append(out_row)
            report.rejected += 1
            stat["rejected"] += 1
        elif any(i.severity == "flag" for i in issues):
            # "info"-severity issues ride along in the same issues list for
            # transparency, but only a real "flag" pushes a row into the
            # human review queue / the "flagged" count.
            out_row["audit"] = {"status": "flagged", "issues": [_issue_dict(i) for i in issues]}
            kept.append(out_row)
            report.flagged += 1
            stat["flagged"] += 1
            report.review_queue.append({
                "text": row.get("text", ""), "intent": intent,
                "issues": [i.code for i in issues if i.severity == "flag"],
            })
        else:
            # No issues, or "info"-only (e.g. keyword_gap) — a valid
            # ML-only candidate is accepted, not held up for review.
            out_row["audit"] = {"status": "passed", "issues": [_issue_dict(i) for i in issues]}
            kept.append(out_row)
            report.accepted += 1
            stat["accepted"] += 1

    return kept, rejected, report


def _run_and_report_audit(cfg: dict, use_llm: bool, max_llm_calls: int) -> AuditReport:
    """Shared by `cmd_generate`'s automatic post-generation audit and the
    standalone `candidates audit` command: reads candidates.jsonl, audits
    every unverified row, rewrites candidates.jsonl (verified rows
    untouched, kept rows annotated), appends anything rejected to
    candidates_rejected.jsonl, writes the full JSON report, and prints
    the human-readable summary."""
    valid_ids = {e["id"] for e in cfg["intents"]}
    train_rows = load_dataset(_TRAIN_FILE, valid_ids)
    rows = _read_jsonl(_CANDIDATES_FILE)
    already_verified = [r for r in rows if r.get("verified")]
    pending = [r for r in rows if not r.get("verified")]

    if not pending:
        print("No unverified candidates to audit.")
        return AuditReport()

    official_index = _official_text_index(valid_ids)
    kept, rejected, report = audit_candidate_rows(
        pending, cfg, train_rows, official_index,
        use_llm=use_llm, max_llm_calls=max_llm_calls,
    )

    _write_jsonl(_CANDIDATES_FILE, already_verified + kept)
    if rejected:
        _append_jsonl(_CANDIDATES_REJECTED_FILE, rejected)
    _AUDIT_REPORT_FILE.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(report.to_text())
    print(f"\nFull audit report written to {_AUDIT_REPORT_FILE}.")
    if rejected:
        print(f"{len(rejected)} candidate(s) auto-rejected — see {_CANDIDATES_REJECTED_FILE.name} for detail.")
    print(f"{len(kept)} candidate(s) remain in {_CANDIDATES_FILE.name} for 'candidates review'.")
    return report


def cmd_candidates_audit(args: argparse.Namespace) -> None:
    cfg = load_intent_config()
    _run_and_report_audit(cfg, use_llm=args.use_llm, max_llm_calls=args.max_llm_calls)


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
        f"paraphrase the existing examples above. Write every utterance in lowercase and "
        f"use minimal punctuation, keeping apostrophes only when needed for contractions. "
        f"One utterance per line, no numbering, no quotes, no explanations — just the raw lines."
    )


def _quality_filter(line: str) -> str | None:
    line = _LIST_PREFIX_RE.sub("", line).strip().strip("\"'").lower()
    line = re.sub(r"[^\w\s']", " ", line)
    line = re.sub(r"\s+", " ", line).strip(" '")
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
    train_counts = Counter(r["intent"] for r in train_rows)

    if args.all:
        targets = []
        for entry in cfg["intents"]:
            gap = entry["min_examples"] - train_counts.get(entry["id"], 0)
            if gap > 0:
                targets.append((entry["id"], _candidate_count_for_training_examples(gap)))
    else:
        if not args.intent:
            print("Specify --intent <id> or --all.")
            return
        by_id = {e["id"]: e for e in cfg["intents"]}
        entry = by_id.get(args.intent)
        if entry is None:
            raise IntentConfigError(f"'{args.intent}' is not a declared intent.")
        gap = max(0, entry["min_examples"] - train_counts.get(args.intent, 0))
        if args.count is not None:
            targets = [(args.intent, args.count)]
        elif gap:
            targets = [(args.intent, _candidate_count_for_training_examples(gap))]
        else:
            targets = []

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

    if not all_candidates:
        print("No candidates generated (check that Ollama is running and 'llama3.1' is pulled).")
        return

    _append_jsonl(_CANDIDATES_FILE, all_candidates)
    print(f"Appended {len(all_candidates)} unverified candidate(s) to {_CANDIDATES_FILE}.")

    if args.no_audit:
        print("Skipping automatic audit (--no-audit). Run 'candidates audit' before 'candidates review'.")
        return

    print("Running automatic candidate audit before human review…")
    _run_and_report_audit(cfg, use_llm=args.use_llm, max_llm_calls=args.max_llm_calls)
    print("Then run 'candidates review' followed by 'candidates promote' once you've checked them.")


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
            audit = r.get("audit") or {}
            issue_codes = [iss["code"] for iss in (audit.get("issues") or [])]
            flag_str = f"   ⚠ {', '.join(issue_codes)}" if issue_codes else ""
            print(f"  [{i}] {r['text']}{flag_str}")
    flagged_n = sum(1 for r in pending if (r.get("audit") or {}).get("status") == "flagged")
    unaudited_n = sum(1 for r in pending if "audit" not in r)
    print(
        f"\n{len(pending)} total pending ({flagged_n} carrying audit flags"
        + (f", {unaudited_n} never audited — run 'candidates audit'" if unaudited_n else "")
        + f"). To verify, edit {_CANDIDATES_FILE} directly and "
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
        print('No verified candidates to promote. Mark some "verified": true first.')
        return

    bad = [r["intent"] for r in verified if r["intent"] not in valid_ids]
    if bad:
        raise IntentConfigError(
            f"Verified candidates reference undeclared intent(s): "
            f"{sorted(set(bad))}"
        )

    # candidates.jsonl is untrusted staging. Only train/validation/test
    # participate in promotion deduplication.
    existing = _trusted_split_texts(valid_ids)

    # Group verified candidates by intent so each intent receives its
    # own approximately 80/10/10 distribution.
    by_intent: dict[str, list[dict]] = {}
    for row in verified:
        by_intent.setdefault(row["intent"], []).append(row)

    split_rows = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # Track everything assigned during this promotion so an example can
    # never appear in more than one split.
    seen_this_batch: set[tuple[str, str]] = set()

    rng = random.SystemRandom()

    for intent in sorted(by_intent):
        candidates = by_intent[intent]

        # Remove duplicates against existing trusted data and within
        # this promotion batch.
        eligible = []
        seen_intent: set[tuple[str, str]] = set()

        for row in candidates:
            text = row["text"].strip()
            key = (text.lower(), intent)

            if not text:
                continue

            if key in existing or key in seen_intent or key in seen_this_batch:
                continue

            seen_intent.add(key)
            eligible.append(row)

        if not eligible:
            print(f"{intent}: no new candidates to promote")
            continue

        # Randomize before assigning splits.
        rng.shuffle(eligible)

        count = len(eligible)

        # Allocate approximately 80/10/10 while ensuring every candidate is
        # assigned to exactly one split.
        if count < 3:
            train_count = count
            validation_count = 0
        else:
            train_count = max(1, round(count * 0.80))
            validation_count = round(count * 0.10)

        test_count = count - train_count - validation_count

        train_candidates = eligible[:train_count]
        validation_candidates = eligible[
            train_count:train_count + validation_count
        ]
        test_candidates = eligible[
            train_count + validation_count:
        ]

        assignments = {
            "train": train_candidates,
            "validation": validation_candidates,
            "test": test_candidates,
        }

        for split, candidates_for_split in assignments.items():
            for row in candidates_for_split:
                text = row["text"].strip()
                key = (text.lower(), intent)

                # Defensive check: never allow a cross-split duplicate.
                if key in seen_this_batch:
                    continue

                clean = {
                    k: v
                    for k, v in row.items()
                    if k in (
                        "text",
                        "intent",
                        "source",
                        "verified",
                        "variant",
                    )
                }

                split_rows[split].append(clean)
                seen_this_batch.add(key)

        print(
            f"{intent}: {count} → "
            f"train={len(train_candidates)}, "
            f"validation={len(validation_candidates)}, "
            f"test={len(test_candidates)}"
        )

    total_promoted = 0

    for split in ("train", "validation", "test"):
        candidates_for_split = split_rows[split]

        if not candidates_for_split:
            continue

        _append_jsonl(
            _SPLIT_FILES[split],
            candidates_for_split,
        )

        count = len(candidates_for_split)
        total_promoted += count

        print(
            f"Promoted {count} example(s) into "
            f"{_SPLIT_FILES[split].name}."
        )

    # Verified candidates are consumed from staging.
    # Unverified candidates remain available for later review.
    _write_jsonl(_CANDIDATES_FILE, unverified)

    print(
        f"Promotion complete: {total_promoted} new example(s) "
        f"distributed across train/validation/test."
    )
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
# removed from datasets/intents.json. Extend this map (never delete an old
# entry) whenever an intent is retired — see docs/CHANGELOG.md. Currently:
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
            f"datasets/intents.json: {sorted(leftover)}. Add a mapping for them "
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
            f"Migration target intent(s) are not declared in datasets/intents.json: "
            f"{sorted(stale_targets)}"
        )
    live_sources = set(legacy_map.keys()) & valid_ids
    if live_sources:
        raise IntentConfigError(
            f"Refusing to migrate {sorted(live_sources)} — still declared as a live "
            f"intent in datasets/intents.json. Remove it from intents.json first."
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
    gen.add_argument(
        "--count", type=int, default=None,
        help="How many candidates to generate (defaults to the amount needed to reach min_examples in training).",
    )
    gen.add_argument("--all", action="store_true", help="Generate for every intent below its min_examples target.")
    gen.add_argument("--no-audit", action="store_true",
                      help="Skip the automatic post-generation candidate audit.")
    gen.add_argument("--use-llm", action="store_true",
                      help="Let the automatic audit also use the LLM to judge candidates a "
                           "deterministic check already flagged as confusable.")
    gen.add_argument("--max-llm-calls", type=int, default=_MAX_LLM_AUDIT_CALLS,
                      help="Cap on LLM-assisted audit judgment calls per run.")
    gen.set_defaults(func=cmd_generate)

    cand = sub.add_parser("candidates", help="Audit/review/promote generated candidates.")
    cand_sub = cand.add_subparsers(dest="subcommand", required=True)
    ca = cand_sub.add_parser(
        "audit",
        help="Run the automated audit over every unverified candidate (malformed/duplicate/"
             "contradictory/overly-generic/confusable/templated/low-diversity checks).",
    )
    ca.add_argument("--use-llm", action="store_true",
                     help="Also use the LLM to judge candidates a deterministic check already "
                          "flagged as confusable (needs a running Ollama).")
    ca.add_argument("--max-llm-calls", type=int, default=_MAX_LLM_AUDIT_CALLS,
                     help="Cap on LLM-assisted audit judgment calls per run.")
    ca.set_defaults(func=cmd_candidates_audit)
    cr = cand_sub.add_parser("review", help="List unverified candidates (with audit flags).")
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
             "Default mapping (see _LEGACY_INTENT_MAP): note_create/note_append -> note_write; note_read/note_list/note_open -> "
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