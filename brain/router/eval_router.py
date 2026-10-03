"""
brain/router/eval_router.py
Offline evaluation + corpus seeding for the hybrid router. Run BEFORE
choosing thresholds (never tune by intuition).

    python -m brain.router.eval_router --seed      # (re)build command vectors via Ollama embeddings
    python -m brain.router.eval_router             # evaluate train/validation/test JSONL

Reports per split: total in-scope examples, top-1 accuracy (mapped through
legacy_intent), top-1/top-2/margin percentiles, and a sweep of
(min_similarity, min_margin) showing CONFIDENT coverage/precision plus the
LLM-fallback rate. Utterances whose labeled intent is not a command
(smalltalk, general_query, ...) are reported separately as "out-of-scope":
a confident match on one of those is a false positive.

top2/margin are COMMAND-level: the runner-up is the best DISTINCT command,
never another prototype of the winning command (see schemas.py). Each MISS
line prints the winning and runner-up command with their scores.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.embeddings import OllamaEmbedder, get_embedding_provider
from brain.router.command_vector_store import CommandVectorStore
from brain.router.registry import CommandRegistry, load_specs
from brain.router.semantic_router import SemanticRouter
from brain.intent_engine import _TRAIN_FILE, _VALIDATION_FILE, _TEST_FILE, load_dataset, load_intent_config
from config.settings import config


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--provider", default=getattr(config.router, "embedding_provider", "local"), choices=["local", "ollama"])
    args = ap.parse_args()

    registry = CommandRegistry(load_specs())
    store = CommandVectorStore(config.router.command_vector_db_path, provider=args.provider)
    embedder = get_embedding_provider(args.provider)
    model_id = getattr(embedder, "model_id", "")

    if args.seed or store.is_stale(registry.specs, model_id):
        print(f"Seeding vector index ({args.provider} / {model_id})...")
        if hasattr(embedder, "embed_sync"):
            seeded = store.reseed(registry.specs, embedder.embed_sync, model_id)
        else:
            seeded = await store.areseed(registry.specs, embedder.embed, model_id)
        print("seeded", seeded, "vectors")
        if args.seed:
            return

    router = SemanticRouter(embedder, store, registry)
    intent_to_key = {s.legacy_intent: s.key for s in registry.specs}
    cfg = load_intent_config()
    valid = {e["id"] for e in cfg["intents"]}

    for name, path in (("train", _TRAIN_FILE), ("validation", _VALIDATION_FILE), ("test", _TEST_FILE)):
        rows = load_dataset(path, valid)
        if not rows:
            print(f"{name}: (empty)"); continue
        recs = []
        for r in rows:
            res = await router.retrieve(r["text"])
            recs.append((r, res))
        inscope = [(r, res) for r, res in recs if r["intent"] in intent_to_key]
        oos = [(r, res) for r, res in recs if r["intent"] not in intent_to_key]
        correct = sum(1 for r, res in inscope if res.top1 and res.top1.spec.key == intent_to_key[r["intent"]])
        print(f"\n== {name}: {len(rows)} total, {len(inscope)} in-scope, {len(oos)} out-of-scope ==")
        if inscope:
            print(f"top-1 accuracy (in-scope): {correct/len(inscope):.3f}")
            t1 = [res.top1_similarity for _, res in inscope]; t2 = [res.top2_similarity for _, res in inscope]
            mg = [res.margin for _, res in inscope]
            for lbl, xs in (("top1", t1), ("top2", t2), ("margin", mg)):
                print(f"  {lbl}: p10={_pct(xs,.1):.3f} p50={_pct(xs,.5):.3f} p90={_pct(xs,.9):.3f}")
        print("sim  margin | confident% precision  oos_false_pos  llm_fallback%")
        for s in (0.70, 0.75, 0.80, 0.85, 0.90):
            for m in (0.03, 0.05, 0.08, 0.12):
                conf = [(r, res) for r, res in inscope if res.top1_similarity >= s and res.margin >= m]
                ok = sum(1 for r, res in conf if res.top1.spec.key == intent_to_key[r["intent"]])
                fp = sum(1 for r, res in oos if res.top1_similarity >= s and res.margin >= m)
                n = len(inscope) or 1
                print(f"{s:.2f} {m:.2f}   | {100*len(conf)/n:6.1f}%  {ok/len(conf) if conf else 0:8.3f}  {fp:6d}  {100*(1-len(conf)/n):8.1f}%")
        bad = [(r, res) for r, res in inscope if res.top1 and res.top1.spec.key != intent_to_key[r['intent']]][:15]
        for r, res in bad:
            d = res.diagnostics()
            print(f"  MISS '{r['text']}' want={intent_to_key[r['intent']]} "
                  f"got={d['winning_command']} ({d['winning_score']:.2f}) "
                  f"second={d['second_command']} ({d['second_score']:.2f}) margin={d['margin']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())