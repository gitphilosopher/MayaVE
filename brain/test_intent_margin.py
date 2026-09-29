"""
brain/test_intent_margin.py
Focused test for the classifier-integration patch in brain/intent_engine.py:
classify() must return second_intent/second_confidence/margin computed from
the SAME averaged ensemble probabilities the original _predict() already
produced internally, and must return None for all three when the decision
came from a guard/keyword path that never consulted the ensemble.

Deliberately does NOT construct a real IntentEngine() — that would require
datasets/training/train_data.jsonl (not available to this test in isolation)
and would try to import torch/tensorflow inside _train_and_save/_load_saved.
Instead this builds a bare instance via IntentEngine.__new__ and sets only
the attributes _predict_detailed()/classify() actually touch, with a fake
TensorFlow-shaped model (a plain object with .predict()) standing in for
the real one — no torch, no tensorflow, no Ollama, per the stabilization
pass's testing brief.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from brain.intent_engine import IntentEngine


class _FakeVocab:
    def encode(self, text, max_len):
        return [0] * max_len   # content doesn't matter — the fake model ignores input


class _FakeTFModel:
    """`.predict(inp, verbose=0)` shaped like tf.keras.Model.predict —
    returns a batch of one softmax row over `probs`."""
    def __init__(self, probs: list[float]):
        self._probs = np.array(probs, dtype=np.float32)

    def predict(self, inp, verbose=0):
        return np.array([self._probs])


def _bare_engine(labels, tf_probs=None, keyword_patterns=None) -> IntentEngine:
    """Construct an IntentEngine without running __init__ (no dataset load,
    no training, no torch/tensorflow import) — only the attributes
    classify()/_predict_detailed() read are set."""
    eng = IntentEngine.__new__(IntentEngine)
    eng._ready = True
    eng._vocab = _FakeVocab()
    eng._labels = labels
    eng._pt_model = None
    eng._tf_model = _FakeTFModel(tf_probs) if tf_probs is not None else None
    eng._keyword_patterns = keyword_patterns or []
    eng._response_modes = {l: "skill" for l in labels}
    eng._dismissal_phrases = frozenset()
    eng._action_words = {}
    eng._ACTION_WORD_RE = re.compile(r"(?!x)x")   # never matches — no action words configured
    return eng


LABELS = ["get_time", "get_date", "set_timer", "unknown"]


def test_classify_high_confidence_reports_margin():
    # get_time clearly wins: 0.90 vs runner-up get_date at 0.05.
    eng = _bare_engine(LABELS, tf_probs=[0.90, 0.05, 0.03, 0.02])
    result = eng.classify("what time is it")
    assert result["intent"] == "get_time"
    assert result["confidence"] == pytest.approx(0.90, abs=1e-3)
    assert result["second_intent"] == "get_date"
    assert result["second_confidence"] == pytest.approx(0.05, abs=1e-3)
    assert result["margin"] == pytest.approx(0.85, abs=1e-3)


def test_classify_ambiguous_reports_thin_margin():
    # Still above _CONF_THRESH (0.65) so the ensemble label is final, but the
    # margin over the runner-up is thin — the understander's margin gate is
    # what's meant to catch this case, not classify() withholding the field.
    eng = _bare_engine(LABELS, tf_probs=[0.68, 0.66, 0.03, 0.03])
    result = eng.classify("timer thing")
    assert result["intent"] == "get_time"
    assert result["second_intent"] == "get_date"
    assert result["margin"] == pytest.approx(0.02, abs=1e-3)


def test_classify_low_confidence_trusted_still_reports_margin():
    # Below _CONF_THRESH, no keyword corroboration, but above the
    # chance-relative low-confidence floor -> "low_confidence_trusted".
    # This path IS still the ensemble's own decision, so its margin is real.
    eng = _bare_engine(LABELS, tf_probs=[0.45, 0.40, 0.10, 0.05])
    result = eng.classify("some ambiguous phrase")
    assert result["intent"] == "get_time"
    assert result["model"] == "low_confidence_trusted"
    assert result["margin"] == pytest.approx(0.05, abs=1e-3)


def test_classify_keyword_match_has_no_margin():
    # A deterministic keyword hit never consults the ensemble at all for a
    # short utterance — second_intent/margin must be None, not fabricated.
    pattern = re.compile(r"\btime\b")
    eng = _bare_engine(LABELS, tf_probs=[0.10, 0.10, 0.10, 0.70],
                       keyword_patterns=[("get_time", pattern)])
    result = eng.classify("time")
    assert result["intent"] == "get_time"
    assert result["model"] in ("keyword", "keyword_short_input")
    assert result["second_intent"] is None
    assert result["second_confidence"] is None
    assert result["margin"] is None


def test_classify_keyword_fallback_below_threshold_has_no_margin():
    # Ensemble confidence is below _CONF_THRESH, but a keyword rule still
    # wins outright over it ("keyword_fallback") — the ensemble's own
    # top-2 is no longer what's being acted on, so it must not be reported
    # as if it were.
    pattern = re.compile(r"\bwhen\b")
    eng = _bare_engine(LABELS, tf_probs=[0.40, 0.35, 0.15, 0.10],
                       keyword_patterns=[("get_time", pattern)])
    result = eng.classify("when will this happen exactly")
    assert result["intent"] == "get_time"
    assert result["model"] == "keyword_fallback"
    assert result["second_intent"] is None
    assert result["margin"] is None


def test_classify_unknown_low_confidence_still_reports_margin():
    # Genuinely unknown: top class is "unknown" but below BOTH
    # _CONF_THRESH and the chance-relative low-confidence floor (0.45 for
    # this 4-label taxonomy) — reaches "low_confidence_fallback", and the
    # field is still populated: a caller may want to distinguish "the
    # ensemble leaned somewhere, just not enough" from "no signal at all".
    eng = _bare_engine(LABELS, tf_probs=[0.25, 0.20, 0.15, 0.40])
    result = eng.classify("completely unrelated gibberish text")
    assert result["intent"] == "unknown"
    assert result["model"] == "low_confidence_fallback"
    assert result["margin"] is not None
