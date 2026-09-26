"""
brain/train_intent.py
=====================
Training + evaluation CLI for Maya's intent classifier.

Usage:
    python -m brain.train_intent                # full retrain + report (from project root)
    python -m brain.train_intent --eval-only     # skip retrain, just evaluate saved models
    python brain/train_intent.py

Reads config/intents.json + config/{train,validation,test}_data.jsonl
(see brain/intent_engine.py's module docstring for the full data-source
contract). Deletes existing saved models and retrains from scratch by
default — use --eval-only to just score the currently-saved models.

Reports, in order:
  - split sizes (train/validation/test) and per-intent distribution
  - overall accuracy on validation and on test
  - per-intent precision/recall/F1 on test
  - confusion matrix (top confusions only, for readability)
  - 'general_query' catch-all class performance specifically (precision/
    recall) — the class most likely to silently rot if new intents crowd
    it out. general_query absorbed the former 'help' and 'unknown'
    intents (see docs/CHANGELOG.md); those ids no longer exist.

Note: 'note_create'/'note_append' -> 'note_write', 'note_read'/
'note_list'/'note_open' -> 'note_view', 'open_app'/'open_website' ->
'open_target', and 'set_reminder' -> 'set_timer' have likewise been
merged (see brain/intent_engine.py's config/intents.json and
brain/dataset_tools.py's _LEGACY_INTENT_MAP) — those ids no longer
exist either. Run `python -m brain.dataset_tools migrate-legacy-intents`
once against existing train/validation/test/candidates data before
retraining so old rows land under the new ids instead of failing
load_dataset()'s "intent not declared" check.
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.intent_engine import (
    _PT_MODEL, _TF_MODEL, _VOCAB_FILE, _LABELS_FILE, _HASH_FILE,
    _TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE,
    IntentEngine, load_intent_config, load_dataset,
)

logger = logging.getLogger("train_intent")


def _print_distribution(name: str, rows: list[dict]) -> None:
    counts = Counter(r["intent"] for r in rows)
    print(f"\n{name}: {len(rows)} examples across {len(counts)} intents")
    for intent, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {intent:<20} {n}")


def _evaluate(engine: IntentEngine, rows: list[dict], label: str) -> dict:
    """Runs classify() over every row, returns per-intent stats + confusion pairs."""
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    confusions: Counter = Counter()   # (true, predicted) -> count
    correct = 0

    for row in rows:
        true_intent = row["intent"]
        result = engine.classify(row["text"])
        pred = result["intent"]
        if pred == true_intent:
            correct += 1
            tp[true_intent] += 1
        else:
            fn[true_intent] += 1
            fp[pred] += 1
            confusions[(true_intent, pred)] += 1

    n = len(rows)
    accuracy = correct / n if n else 0.0

    print(f"\n── {label} — accuracy: {accuracy:.3f} ({correct}/{n}) ──")
    print(f"{'intent':<20}{'precision':>10}{'recall':>10}{'f1':>8}{'support':>9}")
    all_intents = sorted(set(r["intent"] for r in rows) | set(fp.keys()))
    for intent in all_intents:
        support = tp[intent] + fn[intent]
        precision = tp[intent] / (tp[intent] + fp[intent]) if (tp[intent] + fp[intent]) else 0.0
        recall = tp[intent] / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        marker = "  ⚠" if support > 0 and recall < 0.5 else ""
        print(f"{intent:<20}{precision:>10.2f}{recall:>10.2f}{f1:>8.2f}{support:>9}{marker}")

    if "general_query" in all_intents:
        support = tp["general_query"] + fn["general_query"]
        precision = (tp["general_query"] / (tp["general_query"] + fp["general_query"])
                     if (tp["general_query"] + fp["general_query"]) else 0.0)
        recall = tp["general_query"] / support if support else 0.0
        print(f"\n'general_query' catch-all — precision={precision:.2f} recall={recall:.2f} support={support}")
        if fp["general_query"] > 0:
            print(f"  {fp['general_query']} other-intent utterances were misclassified AS general_query "
                  f"(force-fitting risk in reverse — check these).")
        if fn["general_query"] > 0:
            print(f"  {fn['general_query']} genuinely off-topic/catch-all utterances were force-fit "
                  f"into a specific intent instead of falling back to general_query.")

    if confusions:
        print("\nTop confusions (true → predicted):")
        for (true_i, pred_i), count in confusions.most_common(10):
            print(f"    {true_i:<18} → {pred_i:<18} × {count}")

    return {"accuracy": accuracy, "confusions": confusions, "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Maya's intent classifier.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip retraining; just load saved models and evaluate.")
    args = parser.parse_args()

    cfg = load_intent_config()
    valid_ids = {e["id"] for e in cfg["intents"]}

    train_rows = load_dataset(_TRAIN_FILE, valid_ids)
    val_rows   = load_dataset(_VALIDATION_FILE, valid_ids)
    test_rows  = load_dataset(_TEST_FILE, valid_ids)

    _print_distribution("TRAIN", train_rows)
    _print_distribution("VALIDATION", val_rows)
    _print_distribution("TEST", test_rows)

    declared_with_zero = valid_ids - {r["intent"] for r in train_rows}
    if declared_with_zero:
        print(f"\n⚠ {len(declared_with_zero)} declared intent(s) have ZERO training examples: "
              f"{sorted(declared_with_zero)}")

    under_target = []
    by_id = {e["id"]: e for e in cfg["intents"]}
    train_counts = Counter(r["intent"] for r in train_rows)
    for iid, entry in by_id.items():
        got = train_counts.get(iid, 0)
        want = entry["min_examples"]
        if got < want:
            under_target.append((iid, got, want))
    if under_target:
        print(f"\n⚠ {len(under_target)} intent(s) below their config/intents.json min_examples target "
              f"(run brain/dataset_tools.py generate to close the gap):")
        for iid, got, want in sorted(under_target, key=lambda x: x[1] - x[2]):
            print(f"    {iid:<20} {got:>4} / {want:<4}")

    if not args.eval_only:
        print("\n🗑️  Removing saved models for full retrain…")
        for f in [_PT_MODEL, _TF_MODEL, _VOCAB_FILE, _LABELS_FILE, _HASH_FILE]:
            if f.exists():
                f.unlink()
                print(f"   deleted {f.name}")
        print("\n🚀 Training…")

    engine = IntentEngine()

    if val_rows:
        _evaluate(engine, val_rows, "VALIDATION")
    else:
        print("\n(no validation examples — skipping validation report)")

    if test_rows:
        _evaluate(engine, test_rows, "TEST (final, unbiased)")
    else:
        print("\n(no test examples — skipping test report)")

    print("\n✅ Done. Spot-check a few predictions:")
    smoke_tests = [
        "open youtube", "what time is it", "set a timer for 5 minutes",
        "search for python tutorials", "what is photosynthesis", "volume up",
        "hello", "hey maya", "hi",
        "which one is better SSD or HDD", "are you excited today",
        "my dog just knocked over a plant", "the sky looks orange today",
        # Merged-intent spot checks — all four below should land on the
        # new consolidated ids (open_target / set_timer / note_write /
        # note_view), never on a retired id.
        "open chrome", "go to reddit", "launch notepad",
        "remind me to call mom in ten minutes", "set a timer for 20 minutes",
        "take a note buy milk", "add to my note",
        "read my notes", "list my notes", "open my note in notepad",
    ]
    for t in smoke_tests:
        r = engine.classify(t)
        print(f"  '{t}'  →  {r['intent']} [{r['response_mode']}] ({r['confidence']:.2f} via {r['model']})")


if __name__ == "__main__":
    main()