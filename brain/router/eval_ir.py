"""
brain/router/eval_ir.py
IR-level evaluation (extends eval_router.py, which only scores retrieval).
Cases: datasets/router_eval/cases.jsonl, one JSON object per line:
  {"text": "...", "category": "paraphrase|ambiguous|oos|adversarial|entity|clarification|context|llm",
   "history": ["earlier utterance", ...],            # optional, replayed in the same context
   "expect": {"status": "ready", "key": "timer.create", "entities": {"duration": 600}}}
Primary target: false-execution rate, NOT accuracy.

    python -m brain.router.eval_ir [path] [--semantic] [--llm]

Without --semantic/--llm only guards + classifier run; the LLM-fallback rate is
then trivially 0 and the output says so under "wired".

Metric definitions (every metric reports its support `n`; n < _MIN_MEANINGFUL_N
is flagged "low_sample"):
- wrong_execution_rate: READY with a key != the expected key (all cases).
- oos_false_execution_rate: READY on a case whose expected status is not "ready".
- wrong_entity_execution_rate: READY, right key, but a checked entity differs.
- oos_precision / oos_recall: "not executed" (UNKNOWN or REJECTED) vs. expected unknown.
- clarification_recall / false_clarification_rate: asked when expected / asked when not.
- llm_fallback_rate: LLM calls made while understanding each case's FINAL
  utterance, from the understander's real call counters (delta taken after
  history replay — BATCH 2 fix: history turns' calls were previously counted
  against the case count). Falls back to the ir.source label only if the
  understander exposes no counters.
- entity_accuracy: only over cases where the key matched; strings compare
  case-insensitively.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(__file__).parent.parent.parent / "datasets" / "router_eval" / "cases.jsonl"

_MIN_MEANINGFUL_N = 30
_COUNTERS = ("guard", "classifier", "semantic", "llm", "semantic_errors", "llm_errors", "internal_errors")

confirm_errors = 0

def load_cases(path: Path = DEFAULT) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _entities_match(got: dict, want: dict) -> bool:
    for k, v in want.items():
        gv = got.get(k)
        if isinstance(gv, str) and isinstance(v, str):
            if gv.strip().lower() != v.strip().lower():
                return False
        elif gv != v:
            return False
    return True


def _flag(n: int) -> dict:
    return {"n": n, "low_sample": n < _MIN_MEANINGFUL_N}


async def run_eval(make_understander, cases: list[dict], wired: dict | None = None, verbose=False) -> dict:
    n = correct = 0
    wrong_exec = oos_false_exec = wrong_entity_exec = 0
    oos_tp = oos_fp = oos_fn = 0
    clar_hit = clar_expected = clar_unexpected = 0
    status_breakdown = defaultdict(int)
    lat, by_cat = [], defaultdict(lambda: [0, 0])

    call_totals = {k: 0 for k in _COUNTERS}
    have_real_stats = False
    llm_source_hits = 0
    ent_total = ent_ok = 0

    for c in cases:
        u = make_understander()                      # fresh context per case
        for h in c.get("history", []):
            await u.understand(h)
        before = dict(getattr(u, "stats", None) or {})
        t0 = time.perf_counter()
        ir = await u.understand(c["text"])
        lat.append(time.perf_counter() - t0)
        exp = c["expect"]
        want_status, want_key = exp["status"], exp.get("key")

        ok = ir.status.value == want_status and (want_key is None or ir.key == want_key)
        n += 1; correct += ok
        by_cat[c.get("category", "?")][0] += ok; by_cat[c.get("category", "?")][1] += 1
        status_breakdown[ir.status.value] += 1

        got_ready = ir.status.value == "ready"
        want_ready = want_status == "ready"

        if "requires_confirmation" in exp and ir.requires_confirmation != exp["requires_confirmation"]:
            confirm_errors += 1

        if got_ready and want_key is not None and ir.key != want_key:
            wrong_exec += 1
        if got_ready and not want_ready:
            oos_false_exec += 1
        key_matched = want_key is not None and ir.key == want_key
        if got_ready and key_matched and exp.get("entities"):
            if not _entities_match(ir.entities, exp["entities"]):
                wrong_entity_exec += 1
        if key_matched:
            for ek, ev in exp.get("entities", {}).items():
                ent_total += 1
                gv = ir.entities.get(ek)
                if isinstance(gv, str) and isinstance(ev, str):
                    ent_ok += gv.strip().lower() == ev.strip().lower()
                else:
                    ent_ok += gv == ev

        pred_oos = ir.status.value in ("unknown", "rejected")
        true_oos = want_status == "unknown"
        oos_tp += pred_oos and true_oos
        oos_fp += pred_oos and not true_oos
        oos_fn += true_oos and not pred_oos

        want_clar = want_status == "needs_clarification"
        got_clar = ir.requires_clarification
        clar_expected += want_clar
        clar_hit += want_clar and got_clar
        clar_unexpected += got_clar and not want_clar

        if ir.source == "llm":
            llm_source_hits += 1
        after = getattr(u, "stats", None)
        if verbose:
            d = {k: after.get(k, 0) - before.get(k, 0) for k in _COUNTERS} if isinstance(after, dict) else {}
            print(f"[{'OK ' if ok else 'MISS'}] {c['text']!r} want={want_status}/{want_key} "
                  f"got={ir.status.value}/{ir.key} src={ir.source} reason={ir.reason!r} "
                  f"conf={ir.confidence:.2f} margin={ir.margin} {lat[-1]:.2f}s "
                  f"calls={ {k: v for k, v in d.items() if v} }", file=sys.stderr)
        if isinstance(after, dict):
            have_real_stats = True
            for k in call_totals:
                call_totals[k] += after.get(k, 0) - before.get(k, 0)

    lat.sort()
    n_ = max(n, 1)
    return {
        "cases": n,
        "wired": wired or {"semantic": None, "llm": None},
        "routing_accuracy": {"value": correct / n_, **_flag(n)},
        "wrong_execution_rate": {"value": wrong_exec / n_, **_flag(n)},
        "oos_false_execution_rate": {"value": oos_false_exec / n_, **_flag(n)},
        "wrong_entity_execution_rate": {"value": wrong_entity_exec / n_, **_flag(n)},
        "oos_precision": {"value": (oos_tp / (oos_tp + oos_fp)) if (oos_tp + oos_fp) else None, **_flag(oos_tp + oos_fp)},
        "oos_recall": {"value": (oos_tp / (oos_tp + oos_fn)) if (oos_tp + oos_fn) else None, **_flag(oos_tp + oos_fn)},
        "clarification_recall": {"value": (clar_hit / clar_expected) if clar_expected else None, **_flag(clar_expected)},
        "false_clarification_rate": {"value": clar_unexpected / n_, **_flag(n)},
        "llm_fallback_rate": {
            "value": (call_totals["llm"] / n_) if have_real_stats else (llm_source_hits / n_),
            "source": "call_counters" if have_real_stats else "source_label_proxy(undercounts)",
            **_flag(n),
        },
        "entity_accuracy": {"value": (ent_ok / ent_total) if ent_total else None, **_flag(ent_total)},
        "status_breakdown": dict(status_breakdown),
        "call_totals": call_totals if have_real_stats else None,
        "latency_p50_s": statistics.median(lat) if lat else None,
        "latency_p95_s": lat[min(n - 1, int(.95 * n))] if lat else None,
        "by_category": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in by_cat.items()},
    }


async def _main(path: Path, use_semantic: bool, use_llm: bool, verbose: bool = False) -> None:
    from brain.intent_engine import IntentEngine
    from brain.router import guards
    from brain.router.registry import CommandRegistry, load_specs
    from brain.router.understand import CommandUnderstander
    import logging

    legacy, reg = IntentEngine(), CommandRegistry(load_specs())
    semantic = llm = None
    if use_semantic:
        from brain.embeddings import OllamaEmbedder
        from brain.router.command_vector_store import CommandVectorStore
        from brain.router.semantic_router import SemanticRouter
        from config.settings import config
        store = CommandVectorStore(getattr(config.router, "command_vector_db_path", None))
        if store.is_stale(reg.specs):
            print("WARNING: command corpus stale/unseeded — run `python -m brain.router.eval_router --seed`.",
                  file=sys.stderr)
        semantic = SemanticRouter(OllamaEmbedder(), store, reg).retrieve
    if use_llm:
        from brain.router.llm_fallback import route as llm
    make = lambda: CommandUnderstander(legacy, reg, guard_fn=guards.check, semantic=semantic, llm_route=llm)
    result = await run_eval(
        make,
        load_cases(path),
        wired={"semantic": use_semantic, "llm": use_llm},
        verbose=verbose,
    )
    print(json.dumps(result, indent=2))
    if result["cases"] < _MIN_MEANINGFUL_N:
        print(
            f"\nWARNING: only {result['cases']} case(s) evaluated — below the "
            f"{_MIN_MEANINGFUL_N}-case floor. Metrics are marked \"low_sample\": true; "
            f"do not tune thresholds from this run.",
            file=sys.stderr,
        )
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DEFAULT))
    ap.add_argument("--semantic", action="store_true", help="wire semantic retrieval (needs Ollama + seeded corpus)")
    ap.add_argument("--llm", action="store_true", help="wire the LLM fallback (needs Ollama)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    asyncio.run(_main(Path(a.path), a.semantic, a.llm, verbose=a.verbose))