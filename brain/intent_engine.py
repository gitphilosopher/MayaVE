"""
brain/intent_engine.py
======================
Dual-model ML intent classification for Maya.

Architecture
------------
Two independent models vote on every utterance:

  Model A — PyTorch  : BiLSTM + Attention (bag-of-words token embeddings)
  Model B — TensorFlow/Keras : 1-D CNN over character n-grams

Final prediction = argmax of averaged softmax probabilities from both.

On first run, both models are TRAINED on TRAINING_DATA (loaded from
config/train_data.jsonl — see "Data sources" below) and saved to
models/pytorch_intent.pt and models/tf_intent.keras. Subsequent runs
load the saved weights — inference is instant.

Data sources (Maya VE11 — externalized, see config/intents.json)
------------------------------------------------------------------
This module used to hardcode intent names, descriptions, and keyword
trigger lists in Python. As of VE11:

  config/intents.json          — every intent's id/category/description/
                                  min_examples/keywords, the dismissal
                                  guard's exact-phrase list, and the
                                  perform_action guard's action-word map.
                                  brain/dataset_tools.py validates this
                                  file (duplicate ids, missing fields,
                                  invalid categories/references).
  config/train_data.jsonl      — TRAINING_DATA (model training only)
  config/validation_data.jsonl — VALIDATION_DATA (dev-time tuning)
  config/test_data.jsonl       — TEST_DATA (final unbiased evaluation)

Python remains responsible only for classification *logic* (tokenizing,
model architecture, guards, ensemble/threshold behaviour, keyword
regex compilation). Loading/validating the above files happens once,
at import time, via load_intent_config() / load_dataset().

Automatic retrain detection
----------------------------
A sha256 fingerprint of (intents.json + train_data.jsonl) is saved
alongside the models in models/training_hash.txt. On every startup,
_load_or_train() compares the current fingerprint against the saved
one:

  - Models missing            → train from scratch (first run)
  - Fingerprint matches       → load saved weights, instant startup
  - Fingerprint doesn't match → data changed since the models were
                                 trained — retrain automatically, no
                                 manual file deletion required

To retrain from scratch, delete the model files and restart Maya — the
missing-files path above still exists for a full manual reset.
brain/train_intent.py remains the preferred entry point for a full
manual retrain: it wipes every saved file (including the hash) up
front and prints train/validation/test metrics at the end.

Adding new intents
------------------
1. Add the intent's definition to config/intents.json ("intents" list).
2. Add labelled examples to config/train_data.jsonl (and, ideally, a
   few held-out ones to validation/test).
3. Restart Maya — the fingerprint no longer matches the saved hash, so
   models retrain automatically on that startup.
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
_MODEL_DIR    = Path(__file__).parent.parent / "models"
_CONFIG_DIR   = Path(__file__).parent.parent / "config"
_LOG_DIR      = Path(__file__).parent.parent / "logs"

_PT_MODEL     = _MODEL_DIR / "pytorch_intent.pt"
_TF_MODEL     = _MODEL_DIR / "tf_intent.keras"
_VOCAB_FILE   = _MODEL_DIR / "vocab.json"
_LABELS_FILE  = _MODEL_DIR / "labels.json"
_HASH_FILE    = _MODEL_DIR / "training_hash.txt"

_INTENTS_FILE     = _CONFIG_DIR / "intents.json"
_TRAIN_FILE       = _CONFIG_DIR / "train_data.jsonl"
_VALIDATION_FILE  = _CONFIG_DIR / "validation_data.jsonl"
_TEST_FILE        = _CONFIG_DIR / "test_data.jsonl"
_FAILURES_FILE    = _LOG_DIR / "intent_failures.jsonl"

_MODEL_DIR.mkdir(exist_ok=True)
_LOG_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_EMBED_DIM    = 64
_HIDDEN_DIM   = 128
_MAX_LEN      = 30        # tokens per utterance (pad / truncate)
_EPOCHS       = 40
_BATCH        = 16
_LR           = 1e-3
_CONF_THRESH  = 0.65      # below this → fall back to keyword rules

# ══════════════════════════════════════════════════════════════════════════════
# config/intents.json — schema, loading, validation
# ══════════════════════════════════════════════════════════════════════════════


class IntentConfigError(ValueError):
    """Raised for a malformed, missing, or internally inconsistent intents.json."""


_REQUIRED_INTENT_FIELDS = {"id", "category", "description", "min_examples", "keywords", "response_mode"}

# The only two valid handlers a classified intent can be routed to at
# runtime: a deterministic skill (brain/router.py's skill map) or the
# LLM (services/llm/llm_service.py). This is the single source of truth
# routing reads from — see build_response_modes() below. No intent-name
# list is ever hardcoded in the router; it looks up response_mode here.
_VALID_RESPONSE_MODES = {"skill", "llm"}


def load_intent_config(path: Path = _INTENTS_FILE) -> dict:
    """
    Load and strictly validate config/intents.json. Raises IntentConfigError
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
    Compile the ordered keyword-fallback rules from intents.json into the
    same word-boundary + inflection-suffix regex shape the engine always
    used, just sourced from data instead of a hardcoded Python list.
    Intents in keyword_rules_order come first (in that order, mirroring
    the old _KEYWORD_RULES priority — multi-word/specific before broad
    single-word); any other intent with a non-empty keywords list is
    appended afterward in intents.json declaration order.
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
    classify(), and ultimately brain/router.py's dispatch) must read it
    from here rather than hardcoding a skill-vs-LLM intent-name list.
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
        {"intents": intents_cfg, "train": [(r["text"], r["intent"]) for r in train_rows]},
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
    the only path from here into config/train_data.jsonl.
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


def _train_pytorch(X: list[list[int]], y: list[int],
                   vocab_size: int, n_classes: int,
                   save_path: Path) -> object:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build_pytorch_model(vocab_size, n_classes).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=_LR)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=_BATCH, shuffle=True)

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
    from tensorflow import keras # type: ignore

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


def _train_tf(X: list[list[int]], y: list[int],
              vocab_size: int, n_classes: int,
              save_path: Path) -> object:
    import numpy as np_local
    model = _build_tf_model(vocab_size, n_classes)
    Xa = np_local.array(X)
    ya = np_local.array(y)
    model.fit(Xa, ya, epochs=_EPOCHS, batch_size=_BATCH, verbose=0)
    model.save(str(save_path))
    logger.info(f"TensorFlow model saved → {save_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Intent Engine — public class
# ══════════════════════════════════════════════════════════════════════════════

class IntentEngine:
    """
    Dual-model ML classifier with keyword-rule fallback.

    Usage (same public API as before VE11):
        engine = IntentEngine()
        result = engine.classify("open youtube")
        # → {"intent": "open_website", "target": "youtube",
        #    "confidence": 0.93, "raw": "open youtube",
        #    "model": "ensemble"}
    """

    def __init__(self):
        self._vocab:    Optional[_Vocab] = None
        self._labels:   list[str]        = []
        self._pt_model  = None
        self._tf_model  = None
        self._pt_device = None
        self._ready     = False

        # Load + validate config/intents.json before anything else — a
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
        text = text.strip()
        intent, confidence, source = self._predict(text)
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
        }
        logger.debug(
            f"Intent '{intent}' ({confidence:.2f} via {source}, "
            f"response_mode={result['response_mode']}): '{text}'"
        )
        if confidence < _CONF_THRESH:
            log_classification_failure(text, intent, confidence)
        return result

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_or_train(self) -> None:
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

    def _predict(self, text: str) -> tuple[str, float, str]:
        """Returns (intent, confidence, source_label)."""
        if not self._ready or self._vocab is None:
            return self._keyword_fallback(text)

        t = text.lower().strip()

        # ── Negation / dismissal guard ────────────────────────────────────────
        tokens = re.findall(r"[a-z0-9]+", t)
        if self._is_dismissal(t):
            return "dismissal", 1.0, "negation_guard"

        # ── Presence / arrival guard ──────────────────────────────────────────
        if self._PRESENCE_RE.search(t):
            logger.debug(f"Presence guard fired for '{text}' → smalltalk")
            return "smalltalk", 1.0, "presence_guard"

        # ── Action-request guard ──────────────────────────────────────────────
        if self._ACTION_WORD_RE.search(t) and not self._ACTION_QUESTION_RE.search(t):
            logger.debug(f"Action-request guard fired for '{text}' → perform_action")
            return "perform_action", 1.0, "action_guard"

        # ── Keyword-first guard ───────────────────────────────────────────────
        _COMPARISON_WORDS  = ("which", "compare", " vs ", "difference between",
                               "pros and cons", "better than", "or hdd", "or ssd")
        is_comparison = any(w in t for w in _COMPARISON_WORDS)
        use_keywords = not is_comparison and len(tokens) <= 3
        if use_keywords:
            kw_intent, kw_conf, _ = self._keyword_fallback(text)
            if kw_conf > 0:
                return kw_intent, kw_conf, "keyword_short_input"
            # No keyword rule matched a short input — general_query is the
            # catch-all conversational intent (absorbs the old 'unknown').
            return "general_query", 1.0, "short_input_fallback"

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
            return self._keyword_fallback(text)

        avg_probs = np.mean([p for _, p in probs_list], axis=0)
        idx       = int(np.argmax(avg_probs))
        conf      = float(avg_probs[idx])
        intent    = self._labels[idx]
        source    = "+".join(name for name, _ in probs_list)

        if conf < _CONF_THRESH:
            kw_intent, kw_conf, _ = self._keyword_fallback(text)
            if kw_conf > 0:
                logger.debug(
                    f"ML confidence {conf:.2f} < threshold; "
                    f"using keyword fallback → '{kw_intent}'"
                )
                return kw_intent, kw_conf, "keyword_fallback"
            # No keyword rule matched either — don't force a low-confidence
            # label onto a specific intent. general_query is the catch-all
            # conversational intent (absorbs the old 'unknown') and is
            # always response_mode="llm", so an uncertain utterance still
            # gets a natural, context-aware reply instead of a wrong skill.
            return "general_query", conf, "low_confidence_fallback"

        return intent, conf, source

    def _keyword_fallback(self, text: str) -> tuple[str, float, str]:
        t = text.lower()
        for intent, pattern in self._keyword_patterns:
            if pattern.search(t):
                return intent, 1.0, "keyword"
        # No keyword rule matched at all — general_query (absorbs the old
        # 'unknown') is the catch-all, always routed to the LLM.
        return "general_query", 0.0, "keyword"

    # ── Target extraction ─────────────────────────────────────────────────────

    def _extract_target(self, text: str, intent: str) -> str:
        """Strip the intent trigger to leave the target entity."""
        trigger_map = {
            "open_app":     ["open", "launch", "start"],
            "search_web":   ["search for", "google", "look up", "search"],
            "open_website": ["open website", "go to", "navigate to", "open"],
            "set_reminder": ["remind me to", "remind me", "set a timer for",
                             "set a reminder for", "timer for", "alarm for"],
        }
        trigger_hit = False
        for trigger in trigger_map.get(intent, []):
            if trigger in text:
                trigger_hit = True
                after = text.split(trigger, 1)[-1].strip()
                after = re.sub(r"^(for|to|the|a|an|me)\s+", "", after)
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