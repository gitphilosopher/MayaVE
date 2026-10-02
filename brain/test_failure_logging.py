from __future__ import annotations

import json

from brain import intent_engine


def test_log_classification_failure_updates_duplicate_utterances(monkeypatch, tmp_path):
    failures_file = tmp_path / "intent_failures.jsonl"
    existing_rows = [
        {
            "utterance": "Example phrase",
            "predicted_intent": "old",
            "confidence": 0.2,
            "correct_intent": "reviewed_intent",
            "timestamp": 1,
        },
        {
            "utterance": " example   PHRASE ",
            "predicted_intent": "older",
            "confidence": 0.3,
            "correct_intent": None,
            "timestamp": 2,
        },
    ]
    failures_file.write_text(
        "\n".join(json.dumps(row) for row in existing_rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(intent_engine, "_FAILURES_FILE", failures_file)

    intent_engine.log_classification_failure("EXAMPLE phrase", "latest", 0.42)
    intent_engine.log_classification_failure("example phrase", "latest_again", 0.51)

    rows = [
        json.loads(line)
        for line in failures_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["predicted_intent"] == "latest_again"
    assert rows[0]["confidence"] == 0.51
    assert rows[0]["occurrences"] == 4
    assert rows[0]["correct_intent"] == "reviewed_intent"