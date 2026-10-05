"""
Unit test suite for Maya VE11 Batch 2 linguistic streaming segmentation,
senpai vocative protection, and ellipsis control.
"""

import pytest
from services.llm.llm_service import (
    _next_boundary,
    _normalize_pathological_dots,
    _count_spoken_words,
    _get_spoken_words,
    SegmentBoundary,
)


def _split_all_chunks(text: str) -> list[str]:
    """Helper to simulate full streaming consumption of text."""
    chunks = []
    buf = text
    while True:
        b = _next_boundary(buf, is_first_phrase=(len(chunks) == 0))
        if b is None:
            break
        idx, is_final = b
        chunks.append(buf[:idx].strip())
        buf = buf[idx:].lstrip()
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


class TestLinguisticSegmenter:
    def test_a_comma_coalescing_continuation(self):
        """Test A: Comma before continuation words ('rather', 'than') coalesces into one unit."""
        text = "Cloud computing lets you use servers and storage online, rather than keeping everything on your own device."
        chunks = _split_all_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_b_senpai_immediate(self):
        """Test B: Short greeting with senpai attaches as one single chunk."""
        text = "Okay, senpai!"
        chunks = _split_all_chunks(text)
        assert chunks == ["Okay, senpai!"]

    def test_c_senpai_after_clause(self):
        """Test C: Clause ending with vocative senpai is not split at the comma."""
        text = "I can explain that, senpai."
        chunks = _split_all_chunks(text)
        assert chunks == ["I can explain that, senpai."]

    def test_d_streaming_vocative_prefix(self):
        """Test D: Mid-stream sub-word token prefix of vocative ('sen') halts splitting until completed."""
        incomplete = "Okay, sen"
        assert _next_boundary(incomplete) is None

        complete = "Okay, senpai!"
        boundary = _next_boundary(complete)
        assert boundary is not None
        assert boundary[0] == len(complete)
        assert boundary[1] is True  # is_sentence_final

    def test_e_repeated_dots_ellipsis(self):
        """Test E: Repeated dots (..., ...., .....) produce single chunk and normalize cleanly."""
        for text in ["Okay...", "Okay....", "Okay....."]:
            chunks = _split_all_chunks(text)
            assert len(chunks) == 1
            assert chunks[0] == text

        # Verify normalization
        assert _normalize_pathological_dots("Okay.....") == "Okay."
        assert _normalize_pathological_dots("Okay...") == "Okay."
        assert _normalize_pathological_dots("Hmm... okay, senpai.") == "Hmm... okay, senpai."

    def test_f_normal_punctuation_separation(self):
        """Test F: Real sentence boundaries cleanly separate sentences."""
        text = "Hello, senpai. How are you?"
        chunks = _split_all_chunks(text)
        assert chunks == ["Hello, senpai.", "How are you?"]

    def test_g_short_standalone_responses(self):
        """Test G: Complete short utterances emit immediately for fast TTFA."""
        for text in ["Sure!", "Okay senpai.", "Got it."]:
            chunks = _split_all_chunks(text)
            assert chunks == [text]

    def test_h_continuation_conjunction_list(self):
        """Test H: Mid-sentence list continuation does not split before 'and'."""
        text = "servers, storage, and applications over the internet"
        chunks = _split_all_chunks(text)
        # Should coalesce without leaving stranded fragment before 'and'
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_expressive_pause_hmm(self):
        """Test Hmm: Intra-sentence pause is preserved as one cohesive chunk."""
        text = "Hmm... okay, senpai."
        chunks = _split_all_chunks(text)
        assert chunks == ["Hmm... okay, senpai."]

    def test_decimal_and_url_protection(self):
        """Test decimals and domain names are never split at periods."""
        text = "The ratio is 3.14, and the site is google.com today."
        # While '3.' is streaming at end of buffer
        assert _next_boundary("The ratio is 3.") is None
        # Complete text coalesces through 'and'
        chunks = _split_all_chunks(text)
        assert len(chunks) == 1
        assert "3.14" in chunks[0]
        assert "google.com" in chunks[0]

    def test_tagged_utterances(self):
        """Test emotion tags are ignored by punctuation search and spoken word counts."""
        text = "[excited] Good morning senpai! [happy] Ready to start?"
        chunks = _split_all_chunks(text)
        assert chunks == ["[excited] Good morning senpai!", "[happy] Ready to start?"]

    def test_segment_boundary_backward_compatibility(self):
        """Test SegmentBoundary tuple behaves as a 2-tuple and has reason attribute."""
        b = SegmentBoundary(10, True, "strong_punctuation")
        idx, is_final = b  # tuple unpacking
        assert idx == 10
        assert is_final is True
        assert b.reason == "strong_punctuation"

