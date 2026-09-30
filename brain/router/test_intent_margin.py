"""
brain/test_intent_margin.py
classify() must return second_intent/second_confidence/margin from the SAME
averaged ensemble probabilities _predict() already produced, and None for all
three when the decision came from a guard/keyword path.

Builds a bare IntentEngine via __new__ with a fake TF-shaped model — no torch,
tensorflow, datasets, or Ollama.

BATCH 1: the low_confidence_trusted case used 0.45 as the top probability, which
is exactly _LOW_CONF_FLOOR_MAX; as float32 it is 0.44999999 and fell below the
floor. It now uses 0.50/0.40 so the test does not depend on float rounding.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from brain.intent_engine import IntentEngine


class _FakeVocab:
    def encode(self, text, max_len):
        return [0] * max_len


class _FakeTFModel:
    def __init__(self, probs):
        self._probs = np.array(probs, dtype=np.float32)

    def predict(self, inp, verbose=0):
        return np.array([self._probs])


def _bare_engine(labels, tf_probs=None, keyword_patterns=None) -> IntentEngine:
    eng = IntentEngine.__new__(IntentEngine)
    eng._ready = True
    eng._vocab = _FakeVocab()
    eng._labels = labels
    eng._pt_model = None
    eng._pt_device = None
    eng._tf_model = _FakeTFModel(tf_probs) if tf_probs is not None else None
    eng._keyword_patterns = keyword_patterns or []
    eng._response_modes = {l: "skill" for l in labels}
    eng._dismissal_phrases = frozenset()
    eng._action_words = {}
    eng._ACTION_WORD_RE = re.compile(r"(?!x)x")
    return eng


LABELS = ["get_time", "get_date", "set_timer", "unknown"]


def test_classify_high_confidence_reports_margin():
    eng = _bare_engine(LABELS, tf_probs=[0.90, 0.05, 0.03, 0.02])
    result = eng.classify("what time is it")
    assert result["intent"] == "get_time"
    assert result["confidence"] == pytest.approx(0.90, abs=1e-3)
    assert result["second_intent"] == "get_date"
    assert result["second_confidence"] == pytest.approx(0.05, abs=1e-3)
    assert result["margin"] == pytest.approx(0.85, abs=1e-3)


def test_classify_ambiguous_reports_thin_margin():
    eng = _bare_engine(LABELS, tf_probs=[0.68, 0.66, 0.03, 0.03])
    result = eng.classify("timer thing")
    assert result["intent"] == "get_time"
    assert result["second_intent"] == "get_date"
    assert result["margin"] == pytest.approx(0.02, abs=1e-3)


def test_classify_low_confidence_trusted_still_reports_margin():
    eng = _bare_engine(LABELS, tf_probs=[0.50, 0.40, 0.06, 0.04])
    result = eng.classify("some ambiguous phrase")
    assert result["intent"] == "get_time"
    assert result["model"] == "low_confidence_trusted"
    assert result["margin"] == pytest.approx(0.10, abs=1e-3)


def test_classify_keyword_match_has_no_margin():
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
    pattern = re.compile(r"\bwhen\b")
    eng = _bare_engine(LABELS, tf_probs=[0.40, 0.35, 0.15, 0.10],
                       keyword_patterns=[("get_time", pattern)])
    result = eng.classify("when will this happen exactly")
    assert result["intent"] == "get_time"
    assert result["model"] == "keyword_fallback"
    assert result["second_intent"] is None
    assert result["margin"] is None


def test_classify_unknown_low_confidence_still_reports_margin():
    eng = _bare_engine(LABELS, tf_probs=[0.25, 0.20, 0.15, 0.40])
    result = eng.classify("completely unrelated gibberish text")
    assert result["intent"] == "unknown"
    assert result["model"] == "low_confidence_fallback"
    assert result["margin"] is not None
