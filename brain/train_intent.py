"""
brain/train_intent.py
=====================
Standalone training script — run this to retrain the intent models.

Usage:
    python -m brain.train_intent          # from project root
    python brain/train_intent.py          # direct

This deletes existing saved models and retrains from the TRAINING_DATA
in intent_engine.py. Useful when you add new intents/examples.

Note: intent_engine.py now also retrains automatically on the next
startup whenever TRAINING_DATA's content changes (it compares a
sha256 fingerprint against models/training_hash.txt) — you no longer
have to run this manually just to pick up new examples. This script is
still the right tool when you want a full clean wipe plus the
classification test report printed below.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

# Allow running from project root or brain/
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.intent_engine import (
    _PT_MODEL, _TF_MODEL, _VOCAB_FILE, _LABELS_FILE, _HASH_FILE, IntentEngine,
)

def retrain():
    print("🗑️  Removing saved models for full retrain…")
    # _HASH_FILE included so IntentEngine() below can't mistake this forced
    # retrain for a no-op — without deleting it, a matching leftover hash
    # from a previous run would make _load_or_train() think nothing changed.
    for f in [_PT_MODEL, _TF_MODEL, _VOCAB_FILE, _LABELS_FILE, _HASH_FILE]:
        if f.exists():
            f.unlink()
            print(f"   deleted {f.name}")

    print("\n🚀 Training…")
    engine = IntentEngine()

    print("\n✅ Done! Testing a few predictions:")
    tests = [
        # Existing
        "open youtube",
        "what time is it",
        "set a timer for 5 minutes",
        "search for python tutorials",
        "what is photosynthesis",
        "volume up",
        # Greet — should still work
        "hello",
        "hey maya",
        "hi",
        # Comparison — the previously broken cases
        "which one is better SSD or HDD",
        "which is better iphone or android",
        "compare ssd and hdd",
        "python vs javascript which is better",
        "is ssd better than hdd",
        "what is the difference between ram and rom",
        "which laptop should i buy",
        "amd or intel which is better",
        "are you excited today"
    ]
    for t in tests:
        r = engine.classify(t)
        print(f"  '{t}'  →  {r['intent']} ({r['confidence']:.2f} via {r['model']})")

if __name__ == "__main__":
    retrain()