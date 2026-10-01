"""
brain/intent_engine.py
======================
Intent classification and target extraction for Maya's command pipeline.

The engine loads and validates the intent taxonomy from
``datasets/intents.json`` and the labelled training split from
``datasets/training/train_data.jsonl``. It compiles the configured keyword
rules and response modes, applies deterministic dismissal, presence, and
action guards, then combines predictions from a PyTorch BiLSTM with attention
and a TensorFlow/Keras 1-D CNN. Keyword matches take precedence; otherwise a
high-confidence ensemble result is used, followed by a chance-relative
low-confidence floor and the ``unknown`` fallback.

On startup, saved models, vocabulary, and labels are reused when their
fingerprint matches the current intent configuration and training data.
Missing or stale artifacts trigger training and update
``datasets/intent_model/training_hash.txt``. ``brain.train_intent`` remains the explicit
retraining and evaluation entry point. Adding an intent therefore requires a
definition in ``intents.json`` and labelled training examples, followed by a
restart or manual retrain.

The public contract is ``IntentEngine.classify(text)`` returning
``intent``, ``target``, ``confidence``, ``raw``, ``model``, and
``response_mode`` fields, PLUS (see PATCH below) ``second_intent``,
``second_confidence``, and ``margin``. The router consumes the result to
select a skill or the LLM; target extraction removes command phrases for
intents such as ``open_target``, ``search_web``, and ``set_timer``. Dataset
validation is strict so malformed taxonomy or training records fail at
startup instead of silently changing classification behavior.

PATCH (brain/router hybrid-router stabilization pass): the hybrid router's
CommandUnderstander needs a top-2/margin signal to decide whether the
classifier's own result is trustworthy enough to dispatch without escalating
to semantic retrieval or the LLM fallback. Before this patch, classify()'s
return dict had no such field at all — only intent/target/confidence/raw/
model/response_mode — so the understander's `res.get("margin")` always read
None and every ML-sourced result was judged by the stricter no-margin rule.

This patch is purely additive:
  - `_predict()` (return shape, precedence order: dismissal/presence/action
    guards -> deterministic keyword match on a short input -> ensemble ->
    keyword fallback below threshold -> low-confidence floor -> unknown) is
    UNCHANGED and still the single source of truth `classify()` calls.
  - A new `_predict_detailed()` wraps `_predict()`'s exact logic (same
    branches, same precedence, same guard/keyword short-circuits) but also
    captures the second-best class and its probability from the SAME
    averaged ensemble output (`avg_probs`) `_predict()` already computes
    internally when the ensemble path is actually reached. For any result
    that comes from a guard, a keyword rule, or a keyword-fallback path
    (i.e. the ensemble was never consulted, or its output was overridden by
    a stronger deterministic signal), `second_intent`/`second_confidence`/
    `margin` are `None` — there is no meaningful "second place" for a
    guard/keyword decision, and reporting a fabricated one would be worse
    than reporting nothing.
  - `classify()` now calls `_predict_detailed()` instead of `_predict()`
    and adds the three new keys to its returned dict. Every existing key,
    every existing caller reading the pre-patch keys (services/llm/
    llm_service.py, core/processor.py, brain/router/adapter.py, brain/
    router/guards.py, brain/router/hybrid_engine.py's `_legacy_shaped`) is
    unaffected — they simply ignore the three new keys.
  - The PyTorch BiLSTM and TensorFlow CNN model architectures, training
    loop, and saved-artifact format are completely untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import config

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATASETS_DIR = Path(__file__).parent.parent / "datasets"
_MODEL_DIR    = _DATASETS_DIR / "intent_model"
_CONFIG_DIR   = Path(__file__).parent.parent / "config"
_LOG_DIR      = Path(__file__).parent.parent / "logs"

_PT_MODEL     = _MODEL_DIR / "pytorch_intent.pt"
_TF_MODEL     = _MODEL_DIR / "tf_intent.keras"
_VOCAB_FILE   = _MODEL_DIR / "vocab.json"
_LABELS_FILE  = _MODEL_DIR / "labels.json"
_HASH_FILE    = _MODEL_DIR / "training_hash.txt"

_INTENTS_FILE     = _DATASETS_DIR / "intents.json"
_TRAIN_FILE       = _DATASETS_DIR / "training" / "train_data.jsonl"
_VALIDATION_FILE  = _DATASETS_DIR / "training" / "validation_data.jsonl"
_TEST_FILE        = _DATASETS_DIR / "training" / "test_data.jsonl"
_FAILURES_FILE    = _LOG_DIR / "intent_failures.jsonl"

_DATASETS_DIR.mkdir(exist_ok=True)
_MODEL_DIR.mkdir(exist_ok=True)
_LOG_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_EMBED_DIM    = 64
_HIDDEN_DIM   = 128
_MAX_LEN      = 30        # tokens per utterance (pad / truncate)
_EPOCHS       = 40
_BATCH        = 16
_LR           = 1e-3
_CONF_THRESH  = 0.65      # at/above this, the ensemble's own label is final
_SEED = int(os.environ.get("MAYA_SEED", "1337"))      # fixed training seed; also part of the dataset fingerprint

# Below _CONF_THRESH the ensemble's label is no longer "final" on its own,
# but it is still evidence. _LOW_CONF_FLOOR is the point below which that
# evidence is too weak to trust at all — set relative to chance level
# (1 / number of classes) rather than as an independent guess, and marked
# for retuning once real validation/test metrics exist post-retrain (see
# module docstring and docs/CONTRIBUTING.md's "Verification Basis").
_LOW_CONF_FLOOR_MULTIPLIER = 8.0   # "at least 8x better than a random guess"
_LOW_CONF_FLOOR_MIN        = 0.20  # absolute floor regardless of class count
_LOW_CONF_FLOOR_MAX        = 0.45  # absolute ceiling regardless of class count

# ══════════════════════════════════════════════════════════════════════════════
# datasets/intents.json — schema, loading, validation
# ══════════════════════════════════════════════════════════════════════════════


class IntentConfigError(ValueError):
    """Raised for a malformed, missing, or internally inconsistent intents.json."""


_REQUIRED_INTENT_FIELDS = {"id", "category", "description", "min_examples", "keywords", "response_mode"}

# The only two valid handlers a classified intent can be routed to at
# runtime: a deterministic skill (brain/router/dispatch.py's skill map) or the
# LLM (services/llm/llm_service.py). This is the single source of truth
# routing reads from — see build_response_modes() below. No intent-name
# list is ever hardcoded in the router; it looks up response_mode here.
_VALID_RESPONSE_MODES = {"skill", "llm"}


def load_intent_config(path: Path = _INTENTS_FILE) -> dict:
    """
    Load and strictly validate datasets/intents.json. Raises IntentConfigError
    (never silently degrades) on:
      - missing/unreadable file, malformed JSON
      - a top-level "intents" list missing or not a list
      - any intent missing a required field, or a field with the wrong type
      - duplicate intent ids
      - an intent whose "category" isn't in meta.valid_categories
      - a "keyword_rules_order" entry that doesn't name a declared intent
    Returns the parsed dict (validated, not transformed) on success.
    """
    if not path.exists():
        raise IntentConfigError(f"{path} does not exist — this is the single source of truth for intents.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise IntentConfigError(f"{path} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise IntentConfigError(f"{path}: top level must be a JSON object.")

    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        raise IntentConfigError(f"{path}: 'intents' must be a non-empty list.")

    valid_categories = set(data.get("meta", {}).get("valid_categories", []))
    if not valid_categories:
        raise IntentConfigError(f"{path}: meta.valid_categories must be a non-empty list.")

    seen_ids: set[str] = set()
    for i, entry in enumerate(intents):
        if not isinstance(entry, dict):
            raise IntentConfigError(f"{path}: intents[{i}] must be an object.")
        missing = _REQUIRED_INTENT_FIELDS - entry.keys()
        if missing:
            raise IntentConfigError(
                f"{path}: intents[{i}] (id={entry.get('id')!r}) missing required field(s): {sorted(missing)}"
            )
        iid = entry["id"]
        if not isinstance(iid, str) or not iid.strip():
            raise IntentConfigError(f"{path}: intents[{i}].id must be a non-empty string.")
        if iid in seen_ids:
            raise IntentConfigError(f"{path}: duplicate intent id '{iid}'.")
        seen_ids.add(iid)

        if not isinstance(entry["category"], str) or entry["category"] not in valid_categories:
            raise IntentConfigError(
                f"{path}: intent '{iid}' has invalid category {entry.get('category')!r}; "
                f"must be one of {sorted(valid_categories)}."
            )
        if not isinstance(entry["description"], str) or not entry["description"].strip():
            raise IntentConfigError(f"{path}: intent '{iid}' must have a non-empty description.")
        if not isinstance(entry["min_examples"], int) or entry["min_examples"] < 0:
            raise IntentConfigError(f"{path}: intent '{iid}'.min_examples must be a non-negative int.")
        if not isinstance(entry["keywords"], list) or not all(isinstance(k, str) for k in entry["keywords"]):
            raise IntentConfigError(f"{path}: intent '{iid}'.keywords must be a list of strings.")
        if entry["response_mode"] not in _VALID_RESPONSE_MODES:
            raise IntentConfigError(
                f"{path}: intent '{iid}'.response_mode is {entry.get('response_mode')!r}; "
                f"must be one of {sorted(_VALID_RESPONSE_MODES)}."
            )

    order = data.get("keyword_rules_order", [])
    if not isinstance(order, list):
        raise IntentConfigError(f"{path}: keyword_rules_order must be a list.")
    unknown_refs = set(order) - seen_ids
    if unknown_refs:
        raise IntentConfigError(
            f"{path}: keyword_rules_order references intent(s) not declared in 'intents': {sorted(unknown_refs)}"
        )

    logger.info(f"intents.json loaded and validated — {len(intents)} intents.")
    return data


def _intent_by_id(cfg: dict) -> dict[str, dict]:
    return {e["id"]: e for e in cfg["intents"]}


def build_keyword_patterns(cfg: dict) -> list[tuple[str, re.Pattern]]:
    """
    Compile configured keyword rules into ordered word-boundary regexes.

    Entries in ``keyword_rules_order`` are compiled first; other intents with
    keywords follow in their declaration order. The optional suffix supports
    the inflections handled by the classifier's fallback path.
    """
    by_id = _intent_by_id(cfg)
    ordered_ids = list(cfg.get("keyword_rules_order", []))
    ordered_ids += [e["id"] for e in cfg["intents"] if e["id"] not in ordered_ids and e["keywords"]]

    suffix = r"(?:s|es|d|ed|ing)?"
    patterns: list[tuple[str, re.Pattern]] = []
    for iid in ordered_ids:
        triggers = by_id[iid]["keywords"]
        if not triggers:
            continue
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(t.strip()) for t in triggers) + r")" + suffix + r"\b"
        )
        patterns.append((iid, pattern))
    return patterns


def build_response_modes(cfg: dict) -> dict[str, str]:
    """
    id -> "skill" | "llm", straight from the validated config. This is
    the ONLY place response_mode is derived — callers (IntentEngine.
    classify(), and ultimately brain/router/dispatch.py's dispatch) must read
    it from here rather than hardcoding a skill-vs-LLM intent-name list.
    """
    return {e["id"]: e["response_mode"] for e in cfg["intents"]}


# ══════════════════════════════════════════════════════════════════════════════
# JSONL dataset loading — TRAINING_DATA / VALIDATION_DATA / TEST_DATA
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_EXAMPLE_FIELDS = {"text", "intent"}


def load_dataset(path: Path, valid_intent_ids: set[str]) -> list[dict]:
    """
    Load one JSONL split. Each line is a lightweight record:
      {"text": "...", "intent": "...", "source": "...", "verified": true, "variant": "..."}
    Only "text"/"intent" are required; other fields are informational
    (provenance for dataset_tools.py) and are ignored by training itself.
    Raises IntentConfigError on a record whose intent isn't declared in
    intents.json (prevents silent drift between data and config) or on
    malformed JSON. Missing file returns [] (a split is allowed to be
    empty during migration, but train must not be — checked by the caller).
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise IntentConfigError(f"{path}:{lineno}: invalid JSON: {e}") from e
        missing = _REQUIRED_EXAMPLE_FIELDS - rec.keys()
        if missing:
            raise IntentConfigError(f"{path}:{lineno}: missing field(s) {sorted(missing)}")
        if rec["intent"] not in valid_intent_ids:
            raise IntentConfigError(
                f"{path}:{lineno}: intent '{rec['intent']}' is not declared in {_INTENTS_FILE.name}"
            )
        rows.append(rec)
    return rows


def _dataset_fingerprint(train_rows: list[dict], intents_cfg: dict) -> str:
    payload = json.dumps(
        {"intents": intents_cfg, "seed": _SEED,
         "train": [(r["text"], r["intent"]) for r in train_rows]},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Development-time failure logging (see brain/dataset_tools.py for review/promotion)
# ══════════════════════════════════════════════════════════════════════════════

def log_classification_failure(utterance: str, predicted_intent: str, confidence: float,
                                correct_intent: str | None = None) -> None:
    """
    Append one failure record to logs/intent_failures.jsonl. Best-effort —
    a logging failure must never break classification. Failures are NEVER
    auto-trained on; brain/dataset_tools.py's review/promote workflow is
    the only path from here into datasets/training/train_data.jsonl.
    """
    record = {
        "utterance": utterance,
        "predicted_intent": predicted_intent,
        "confidence": round(float(confidence), 4),
        "correct_intent": correct_intent,
        "timestamp": time.time(),
    }
    try:
        with open(_FAILURES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug(f"Could not write intent failure log (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared tokenizer / vocabulary
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Normalize an utterance into the lowercase tokens used by both models."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _Vocab:
    PAD = 0
    UNK = 1

    def __init__(self):
        self._w2i: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def build(self, corpus: list[str]) -> None:
        for text in corpus:
            for tok in _tokenize(text):
                if tok not in self._w2i:
                    self._w2i[tok] = len(self._w2i)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self._w2i.get(t, self.UNK) for t in _tokenize(text)]
        ids = ids[:max_len] + [self.PAD] * max(0, max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self._w2i)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._w2i))

    @classmethod
    def load(cls, path: Path) -> "_Vocab":
        v = cls()
        v._w2i = json.loads(path.read_text())
        return v


# ══════════════════════════════════════════════════════════════════════════════
# Model A — PyTorch BiLSTM + Attention
# ══════════════════════════════════════════════════════════════════════════════

def _build_pytorch_model(vocab_size: int, n_classes: int):
    import torch
    import torch.nn as nn

    class _Attention(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.attn = nn.Linear(hidden * 2, 1)

        def forward(self, h):                           # h: (B, T, 2H)
            scores = self.attn(h).squeeze(-1)           # (B, T)
            weights = torch.softmax(scores, dim=-1)     # (B, T)
            ctx = (weights.unsqueeze(-1) * h).sum(1)    # (B, 2H)
            return ctx

    class BiLSTMIntent(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(vocab_size, _EMBED_DIM, padding_idx=0)
            self.lstm = nn.LSTM(_EMBED_DIM, _HIDDEN_DIM, batch_first=True,
                                bidirectional=True, num_layers=2, dropout=0.3)
            self.attn = _Attention(_HIDDEN_DIM)
            self.drop = nn.Dropout(0.4)
            self.fc   = nn.Linear(_HIDDEN_DIM * 2, n_classes)

        def forward(self, x):
            e = self.emb(x)                    # (B, T, E)
            h, _ = self.lstm(e)                # (B, T, 2H)
            ctx = self.attn(h)                 # (B, 2H)
            return self.fc(self.drop(ctx))     # (B, C)

    return BiLSTMIntent()


def _train_pytorch(X, y, vocab_size, n_classes, save_path):
    import random
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build_pytorch_model(vocab_size, n_classes).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=_LR)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(_SEED))

    model.train()
    for epoch in range(_EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.debug(f"[PyTorch] epoch {epoch+1}/{_EPOCHS}  loss={total_loss/len(loader):.4f}")

    torch.save({"state": model.state_dict(),
                "vocab_size": vocab_size,
                "n_classes": n_classes}, save_path)
    logger.info(f"PyTorch model saved → {save_path}")
    return model


def _predict_pytorch(model, X_single: list[int], device) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([X_single], dtype=torch.long).to(device)
        logits = model(inp)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]


# ══════════════════════════════════════════════════════════════════════════════
# Model B — TensorFlow/Keras 1-D CNN
# ══════════════════════════════════════════════════════════════════════════════

def _build_tf_model(vocab_size: int, n_classes: int):
    import tensorflow as tf
    from tensorflow import keras  # type: ignore

    inp  = keras.Input(shape=(_MAX_LEN,), dtype="int32")
    x    = keras.layers.Embedding(vocab_size, _EMBED_DIM, mask_zero=True)(inp)
    x    = keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x    = keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x    = keras.layers.GlobalMaxPooling1D()(x)
    x    = keras.layers.Dense(128, activation="relu")(x)
    x    = keras.layers.Dropout(0.4)(x)
    out  = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(_LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _train_tf(X, y, vocab_size, n_classes, save_path):
    import numpy as np_local
    import tensorflow as tf
    tf.keras.utils.set_random_seed(_SEED)   # seeds python, numpy and TF; must precede model build
    model = _build_tf_model(vocab_size, n_classes)
    model.fit(np_local.array(X), np_local.array(y), epochs=_EPOCHS, batch_size=_BATCH, verbose=0)
    model.save(str(save_path))
    logger.info(f"TensorFlow model saved → {save_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Intent Engine — public class
# ══════════════════════════════════════════════════════════════════════════════

class IntentEngine:
    """
    Dual-model ML classifier with keyword-rule fallback.

    Usage:
        engine = IntentEngine()
        result = engine.classify("open youtube")
        # → {"intent": "open_target", "target": "youtube",
        #    "confidence": 0.93, "raw": "open youtube",
        #    "model": "ensemble", "response_mode": "skill",
        #    "second_intent": ..., "second_confidence": ..., "margin": ...}
    """

    def __init__(self):
        self._vocab:    Optional[_Vocab] = None
        self._labels:   list[str]        = []
        self._pt_model  = None
        self._tf_model  = None
        self._pt_device = None
        self._ready     = False

        # Load + validate datasets/intents.json before anything else — a
        # malformed config must fail loudly at startup, not degrade
        # classification silently.
        self._intents_cfg = load_intent_config()
        self._intents_by_id = _intent_by_id(self._intents_cfg)
        self._valid_intent_ids = set(self._intents_by_id.keys())
        self._keyword_patterns = build_keyword_patterns(self._intents_cfg)
        self._response_modes = build_response_modes(self._intents_cfg)

        dismissal_cfg = self._intents_by_id.get("dismissal", {})
        self._dismissal_phrases: frozenset = frozenset(dismissal_cfg.get("exact_phrases", []))

        action_cfg = self._intents_by_id.get("perform_action", {})
        self._action_words: dict[str, str] = dict(action_cfg.get("action_words", {}))
        if self._action_words:
            self._ACTION_WORD_RE = re.compile(
                r"\b(?:" + "|".join(re.escape(w) for w in self._action_words) + r")\w*\b",
                re.IGNORECASE,
            )
        else:
            # No action words configured — guard never fires rather than
            # crashing; perform_action then relies purely on ML/keywords.
            self._ACTION_WORD_RE = re.compile(r"(?!x)x")

        self._load_or_train()

    # ── Public ────────────────────────────────────────────────────────────────

    def classify(self, text: str) -> dict:
        """Classify one utterance and return the router-facing result record."""
        text = text.strip()
        detail = self._predict_detailed(text)
        intent, confidence, source = detail["intent"], detail["confidence"], detail["source"]
        target = self._extract_target(text.lower(), intent)

        result = {
            "intent":        intent,
            "target":        target,
            "confidence":    round(float(confidence), 3),
            "raw":           text,
            "model":         source,
            # Config-driven — never a hardcoded skill/LLM intent list.
            # Defaults to "llm" only as a last-resort guard; every intent
            # actually reachable from _predict() is validated to have one
            # in load_intent_config().
            "response_mode": self._response_modes.get(intent, "llm"),
            # Additive — see module docstring's PATCH note. None when the
            # decision came from a guard/keyword path with no meaningful
            # "second place" (the ensemble was never consulted, or was
            # overridden by stronger deterministic evidence).
            "second_intent":     detail["second_intent"],
            "second_confidence": None if detail["second_confidence"] is None
                                  else round(float(detail["second_confidence"]), 3),
            "margin":            None if detail["margin"] is None
                                  else round(float(detail["margin"]), 3),
        }
        logger.debug(
            f"Intent '{intent}' ({confidence:.2f} via {source}, "
            f"response_mode={result['response_mode']}, margin={result['margin']}): '{text}'"
        )
        if confidence < _CONF_THRESH:
            log_classification_failure(text, intent, confidence)
        return result

    # ── Initialisation ────────────────────────────────────────────────────────

    def _low_conf_floor(self) -> float:
        """
        Chance-relative floor below which an unconfirmed ensemble
        prediction is treated as too weak to trust (see module docstring).
        Derived from the current number of classes so it tracks the
        taxonomy automatically instead of being a fixed number picked in
        isolation; clamped to a sane absolute range either way.
        """
        n = max(len(self._labels), 1)
        floor = (1.0 / n) * _LOW_CONF_FLOOR_MULTIPLIER
        return min(_LOW_CONF_FLOOR_MAX, max(_LOW_CONF_FLOOR_MIN, floor))

    def _load_or_train(self) -> None:
        """Load compatible artifacts or train and persist a current model set."""
        models_exist = (
            _PT_MODEL.exists() and
            _TF_MODEL.exists() and
            _VOCAB_FILE.exists() and
            _LABELS_FILE.exists()
        )

        train_rows = load_dataset(_TRAIN_FILE, self._valid_intent_ids)
        if not train_rows:
            raise IntentConfigError(
                f"{_TRAIN_FILE} is empty or missing — cannot train/load without training data."
            )
        current_hash = _dataset_fingerprint(train_rows, self._intents_cfg)
        saved_hash    = _read_saved_hash()
        stale         = models_exist and saved_hash != current_hash

        if models_exist and not stale:
            logger.info("Loading saved ML intent models…")
            self._load_saved()
        else:
            if stale:
                logger.info(
                    "intents.json or train_data.jsonl changed since the saved models "
                    "were trained — retraining automatically (no manual delete needed; "
                    "brain/train_intent.py still works for a full manual retrain + "
                    "evaluation report)."
                )
            else:
                logger.info("Training ML intent models (first run — please wait)…")
            self._train_and_save(train_rows)
            _write_saved_hash(current_hash)

        self._ready = True
        logger.info(
            f"IntentEngine ready — {len(self._labels)} intents, "
            f"vocab={len(self._vocab)} tokens"
        )

    def _train_and_save(self, train_rows: list[dict]) -> None:
        texts  = [r["text"] for r in train_rows]
        labels = [r["intent"] for r in train_rows]

        # Build vocabulary
        self._vocab = _Vocab()
        self._vocab.build(texts)
        self._vocab.save(_VOCAB_FILE)

        # Build label map — every declared intent gets a slot even if a
        # class currently has zero training rows (keeps the label space
        # stable as the dataset grows); a class with 0 rows just never
        # gets predicted until examples are added.
        self._labels = sorted(self._valid_intent_ids | set(labels))
        _LABELS_FILE.write_text(json.dumps(self._labels))
        label2idx = {l: i for i, l in enumerate(self._labels)}

        X = [self._vocab.encode(t, _MAX_LEN) for t in texts]
        y = [label2idx[l] for l in labels]
        n = len(self._labels)
        v = len(self._vocab)

        # Train PyTorch
        try:
            self._pt_model = _train_pytorch(X, y, v, n, _PT_MODEL)
            import torch
            self._pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("✅ PyTorch BiLSTM trained.")
        except Exception as e:
            logger.error(f"PyTorch training failed: {e}")

        # Train TensorFlow
        try:
            import os as _os
            _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            self._tf_model = _train_tf(X, y, v, n, _TF_MODEL)
            logger.info("✅ TensorFlow CNN trained.")
        except Exception as e:
            logger.error(f"TensorFlow training failed: {e}")

    def _load_saved(self) -> None:
        self._vocab  = _Vocab.load(_VOCAB_FILE)
        self._labels = json.loads(_LABELS_FILE.read_text())

        # Load PyTorch
        try:
            import torch
            ckpt = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
            self._pt_model = _build_pytorch_model(
                ckpt["vocab_size"], ckpt["n_classes"])
            self._pt_model.load_state_dict(ckpt["state"])
            self._pt_device = torch.device("cpu")
            self._pt_model.to(self._pt_device)
            logger.debug("PyTorch model loaded.")
        except Exception as e:
            logger.warning(f"PyTorch model load failed: {e}")

        # Load TensorFlow
        try:
            import os as _os
            _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
            self._tf_model = tf.keras.models.load_model(str(_TF_MODEL))
            logger.debug("TensorFlow model loaded.")
        except Exception as e:
            logger.warning(f"TensorFlow model load failed: {e}")

    # ── Prediction ────────────────────────────────────────────────────────────

    _DISMISSAL_TRIM_RE = re.compile(
        r"^(?:(?:ok|okay|um|uh|hmm|well|actually)\s+)*(?P<core>.+?)"
        rf"(?:\s+(?:please|{re.escape(config.name.lower())}|{re.escape(config.user_name.lower())}))*$"
    )

    def _is_dismissal(self, t: str) -> bool:
        norm = " ".join(re.sub(r"[^a-z0-9' ]+", " ", t.replace("’", "'")).split())
        m = self._DISMISSAL_TRIM_RE.match(norm)
        return bool(m) and m.group("core") in self._dismissal_phrases

    # Patterns that indicate presence/arrival — these should ALWAYS go to
    # smalltalk regardless of what words like "now", "here", "back" might
    # spuriously activate in the datetime-trained ML models. Cross-cutting
    # linguistic logic, not an intent definition — stays in code.
    _PRESENCE_RE = re.compile(
        r"^(ok\s+but\s+)?"
        r"(now\s+)?"
        r"(i'?m|i\s+am|i\s+just(\s+got)?|hey\s+i'?m|here\s+i)\s+"
        r"(here|back|home|ready|online|arrived|awake|at\s+my\s+desk)"
        r"|^(just\s+got\s+(back|home|here)|i\s+got\s+back|here\s+i\s+am)"
        r"|^(now\s+what|so\s+now\s+what|what\s+now|what\s+do\s+(we|i)\s+(do\s+)?now)"
        r"|(ok\s+)?(now\s+)?i'?m\s+here(\s+maya)?$",
        re.IGNORECASE,
    )

    # Questions ABOUT an action word ("what does nod mean", "why did you
    # sigh", "do you giggle") are conversation, not a request to perform
    # it — this guard keeps them out of perform_action.
    _ACTION_QUESTION_RE = re.compile(
        r"\b(what|why|how|when|where|who|which)\b"
        r"|^\s*(does|do|did|is|are|was)\b"
        r"|\b(mean|means|meaning|define|definition)\b",
        re.IGNORECASE,
    )

    _COMPARISON_WORDS = ("which", "compare", " vs ", "difference between",
                         "pros and cons", "better than", "or hdd", "or ssd")

    def _predict(self, text: str) -> tuple[str, float, str]:
        """Returns (intent, confidence, source_label). UNCHANGED — see
        module docstring's PATCH note; _predict_detailed() below wraps
        this exact logic without altering any branch or precedence."""
        detail = self._predict_detailed(text)
        return detail["intent"], detail["confidence"], detail["source"]

    def _predict_detailed(self, text: str) -> dict:
        """
        Same branches/precedence as the original _predict(), plus, ONLY
        when the ensemble path is actually reached and produces the final
        label, the second-best class and probability from that same
        averaged distribution. Returns:
            {"intent", "confidence", "source",
             "second_intent", "second_confidence", "margin"}
        The three "second_*"/"margin" fields are None whenever the
        decision came from a guard, a keyword rule, or a keyword-fallback
        override — those paths never consult the ensemble at all, or
        consult it only to discard its label, so there is no genuine
        "runner-up" to report.
        """
        _none = {"second_intent": None, "second_confidence": None, "margin": None}

        if not self._ready or self._vocab is None:
            kw_intent, kw_conf, _ = self._keyword_fallback(text)
            if kw_conf > 0:
                return {"intent": kw_intent, "confidence": kw_conf, "source": "keyword", **_none}
            return {"intent": "unknown", "confidence": 0.0, "source": "keyword", **_none}

        t = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+", t)

        # ── Negation / dismissal guard ────────────────────────────────────────
        if self._is_dismissal(t):
            return {"intent": "dismissal", "confidence": 1.0, "source": "negation_guard", **_none}

        # ── Presence / arrival guard ──────────────────────────────────────────
        if self._PRESENCE_RE.search(t):
            logger.debug(f"Presence guard fired for '{text}' → smalltalk")
            return {"intent": "smalltalk", "confidence": 1.0, "source": "presence_guard", **_none}

        # ── Action-request guard ──────────────────────────────────────────────
        if self._ACTION_WORD_RE.search(t) and not self._ACTION_QUESTION_RE.search(t):
            logger.debug(f"Action-request guard fired for '{text}' → perform_action")
            return {"intent": "perform_action", "confidence": 1.0, "source": "action_guard", **_none}

        # ── Deterministic keyword check ─────────────────────────────────────
        # Computed once and reused by both the short-input path and the
        # low-confidence path below — a keyword match always wins outright
        # over the ensemble, at any utterance length or confidence level.
        kw_intent, kw_conf, _ = self._keyword_fallback(text)

        is_comparison = any(w in t for w in self._COMPARISON_WORDS)
        is_short = not is_comparison and len(tokens) <= 3

        if is_short and kw_conf > 0:
            return {"intent": kw_intent, "confidence": kw_conf, "source": "keyword_short_input", **_none}

        enc = self._vocab.encode(text, _MAX_LEN)
        probs_list = []

        if self._pt_model is not None:
            try:
                p = _predict_pytorch(self._pt_model, enc, self._pt_device)
                probs_list.append(("pytorch", p))
            except Exception as e:
                logger.debug(f"PyTorch inference error: {e}")

        if self._tf_model is not None:
            try:
                import numpy as np_local
                inp = np_local.array([enc])
                p   = self._tf_model.predict(inp, verbose=0)[0]
                probs_list.append(("tensorflow", p))
            except Exception as e:
                logger.debug(f"TensorFlow inference error: {e}")

        if not probs_list:
            # Neither model is available/loaded — the only remaining
            # signal is the keyword check already computed above.
            if kw_conf > 0:
                return {"intent": kw_intent, "confidence": kw_conf, "source": "keyword_fallback", **_none}
            return {"intent": "unknown", "confidence": 0.0, "source": "keyword_fallback", **_none}

        avg_probs = np.mean([p for _, p in probs_list], axis=0)
        order     = np.argsort(avg_probs)[::-1]   # best..worst class indices
        idx       = int(order[0])
        conf      = float(avg_probs[idx])
        intent    = self._labels[idx]
        source    = "+".join(name for name, _ in probs_list)

        second_intent = second_conf = margin = None
        if len(order) > 1:
            sidx = int(order[1])
            second_intent = self._labels[sidx]
            second_conf = float(avg_probs[sidx])
            margin = conf - second_conf
        second_fields = {"second_intent": second_intent, "second_confidence": second_conf, "margin": margin}

        if conf >= _CONF_THRESH:
            return {"intent": intent, "confidence": conf, "source": source, **second_fields}

        # A deterministic keyword match still wins below the high-confidence
        # bar because it is stronger evidence than the ensemble estimate.
        # The ensemble's own top-2/margin is no longer what's being acted
        # on here, so it is NOT reported — the keyword decision has no
        # genuine "second place" of its own.
        if kw_conf > 0:
            logger.debug(
                f"ML confidence {conf:.2f} < threshold; "
                f"using keyword fallback → '{kw_intent}'"
            )
            return {"intent": kw_intent, "confidence": kw_conf, "source": "keyword_fallback", **_none}

        # Without keyword corroboration, retain an ensemble label only when
        # its probability is meaningfully above chance for this taxonomy.
        # This IS still the ensemble's own decision, so its margin is real
        # and worth reporting (a caller may want to distinguish a trusted
        # low-confidence label from a totally unknown one by margin too).
        if conf >= self._low_conf_floor():
            return {"intent": intent, "confidence": conf, "source": "low_confidence_trusted", **second_fields}

        return {"intent": "unknown", "confidence": conf, "source": "low_confidence_fallback", **second_fields}

    def _keyword_fallback(self, text: str) -> tuple[str, float, str]:
        """Return the first matching configured keyword rule, if any."""
        t = text.lower()
        for intent, pattern in self._keyword_patterns:
            if pattern.search(t):
                return intent, 1.0, "keyword"
        # The zero-confidence intent is a sentinel; callers use it only when
        # the paired confidence is positive.
        return "unknown", 0.0, "keyword"

    # ── Target extraction ─────────────────────────────────────────────────────

    def _extract_target(self, text: str, intent: str) -> str:
        """Remove a known command phrase and return the remaining target text."""
        trigger_map = {
            # Specific phrases precede generic fallbacks so multi-word
            # commands such as "go to" and "launch" are stripped correctly.
            "open_target": ["go to", "navigate to", "open website",
                             "launch", "start", "open"],
            "search_web":  ["search for", "google", "look up", "search"],
            # A bare timer request has no trigger and keeps the complete
            # utterance as the target for the timer parser.
            "set_timer":   ["remind me to", "remind me", "set a timer for",
                             "set a reminder for", "timer for", "alarm for"],
        }
        trigger_hit = False
        for trigger in trigger_map.get(intent, []):
            if trigger in text:
                trigger_hit = True
                after = text.split(trigger, 1)[-1].strip()
                after = re.sub(r"^(?:for|to|the|a|an|me)(?:\s+|$)", "", after)
                if after:
                    return after
        return "" if trigger_hit else text


# ══════════════════════════════════════════════════════════════════════════════
# Retrain-fingerprint persistence
# ══════════════════════════════════════════════════════════════════════════════

def _read_saved_hash() -> Optional[str]:
    if not _HASH_FILE.exists():
        return None
    try:
        return _HASH_FILE.read_text().strip() or None
    except OSError:
        return None


def _write_saved_hash(digest: str) -> None:
    _HASH_FILE.write_text(digest)
