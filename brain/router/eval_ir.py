"""
brain/router/eval_ir.py
IR-level evaluation (extends eval_router.py, which only scores retrieval).
Cases: datasets/router_eval/cases.jsonl, one JSON object per line:
  {"text": "...", "category": "paraphrase|ambiguous|oos|adversarial|entity|clarification|context|llm",
   "history": ["earlier utterance", ...],            # optional, replayed in the same context
   "expect": {"status": "ready", "key": "timer.create", "entities": {"duration": 600}}}
Primary target: false-positive execution rate, NOT accuracy.

    python -m brain.router.eval_ir [path]
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

DEFAULT = Path(__file__).parent.parent.parent / "datasets" / "router_eval" / "cases.jsonl"


def load_cases(path: Path = DEFAULT) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


async def run_eval(make_understander, cases: list[dict]) -> dict:
    n = correct = fp_exec = oos_tp = oos_fp = oos_fn = clar = llm = 0
    ent_total = ent_ok = 0
    lat, by_cat = [], defaultdict(lambda: [0, 0])
    for c in cases:
        u = make_understander()                      # fresh context per case
        for h in c.get("history", []):
            await u.understand(h)
        t0 = time.perf_counter()
        ir = await u.understand(c["text"])
        lat.append(time.perf_counter() - t0)
        exp = c["expect"]
        ok = ir.status.value == exp["status"] and (exp.get("key") is None or ir.key == exp["key"])
        n += 1; correct += ok
        by_cat[c.get("category", "?")][0] += ok; by_cat[c.get("category", "?")][1] += 1
        got_ready, want_ready = ir.status.value == "ready", exp["status"] == "ready"
        if got_ready and not (want_ready and ir.key == exp.get("key")):
            fp_exec += 1                             # would have executed the wrong thing / an OOS request
        pred_oos, true_oos = ir.status.value == "unknown", exp["status"] == "unknown"
        oos_tp += pred_oos and true_oos; oos_fp += pred_oos and not true_oos; oos_fn += true_oos and not pred_oos
        clar += ir.requires_clarification
        llm += ir.source == "llm"
        for k, v in exp.get("entities", {}).items():
            ent_total += 1; ent_ok += ir.entities.get(k) == v
    lat.sort()
    return {
        "cases": n, "routing_accuracy": correct / n,
        "false_positive_execution_rate": fp_exec / n,
        "oos_precision": oos_tp / (oos_tp + oos_fp) if oos_tp + oos_fp else None,
        "oos_recall": oos_tp / (oos_tp + oos_fn) if oos_tp + oos_fn else None,
        "clarification_rate": clar / n, "llm_fallback_rate": llm / n,
        "entity_accuracy": ent_ok / ent_total if ent_total else None,
        "latency_p50_s": statistics.median(lat), "latency_p95_s": lat[min(n - 1, int(.95 * n))],
        "by_category": {k: v[0] / v[1] for k, v in by_cat.items()},
    }


async def _main(path: Path) -> None:
    from brain.intent_engine import IntentEngine
    from brain.router import guards
    from brain.router.registry import CommandRegistry, load_specs
    from brain.router.understand import CommandUnderstander
    legacy, reg = IntentEngine(), CommandRegistry(load_specs())
    make = lambda: CommandUnderstander(legacy, reg, guard_fn=guards.check)   # add semantic/llm to eval those levels
    print(json.dumps(await run_eval(make, load_cases(path)), indent=2))


if __name__ == "__main__":
    asyncio.run(_main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
