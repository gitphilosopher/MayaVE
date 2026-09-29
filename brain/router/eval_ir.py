"""
brain/router/eval_ir.py
IR-level evaluation (extends eval_router.py, which only scores retrieval).
Cases: datasets/router_eval/cases.jsonl, one JSON object per line:
  {"text": "...", "category": "paraphrase|ambiguous|oos|adversarial|entity|clarification|context|llm",
   "history": ["earlier utterance", ...],            # optional, replayed in the same context
   "expect": {"status": "ready", "key": "timer.create", "entities": {"duration": 600}}}
Primary target: false-positive execution rate, NOT accuracy.

    python -m brain.router.eval_ir [path]

PATCH (stabilization pass) — the original metric set had definitions that
could mislead:

- "false_positive_execution_rate" counted READY-but-wrong-key over ALL
  cases, including cases whose expectation isn't "ready" at all — an OOS
  case correctly landing on UNKNOWN was invisible to it, and a wrong-key
  READY on a genuinely in-scope case was mixed in with the OOS false-
  positive signal. Split into three:
    - wrong_execution_rate: READY but key != expected key, over ALL cases
      (the raw "did the router ever produce READY with the wrong command"
      number, independent of whether ready was even expected).
    - oos_false_execution_rate: READY on a case whose expectation is
      status != "ready" — the number that actually matters for safety
      (a command run when nothing should have run).
    - wrong_entity_execution_rate: READY, right key, but a checked entity
      value didn't match — a "ran the right command with wrong data" case,
      previously invisible to any execution metric.
- oos_precision/oos_recall previously used ir.status == "unknown" as the
  sole proxy for "predicted OOS", conflating REJECTED (a candidate that
  was actively considered and failed validation) with UNKNOWN (nothing
  usable was ever proposed). Both are now counted as "not executed",
  which is what OOS precision/recall are meant to measure, but each is
  also reported separately in `status_breakdown` so a REJECTED spike
  (malformed LLM output) doesn't hide inside an "OOS recall improved"
  headline.
- clarification_rate was a single number mixing "we asked when we should
  have" with "we asked when we shouldn't have". Split into
  clarification_recall (asked, when expected) and
  false_clarification_rate (asked, when NOT expected — a nuisance
  question on a case that should have just run or been rejected).
- llm_fallback_rate previously inferred LLM involvement from
  `ir.source == "llm"`, which misses "LLM was consulted, came back
  invalid/down, and the turn ended UNKNOWN/REJECTED with a different
  source label". It now reads real call counters off the understander
  (`self.stats`, added to CommandUnderstander in this same patch) when
  the understander instance exposes them, falling back to the source-sniff
  only if it doesn't (e.g. a bare-bones fake in a unit test).
- entity_accuracy now only scores entities on cases where the router's
  key actually matched the expected key (scoring entities against the
  wrong command was never meaningful) and compares case-insensitively for
  string values (a "London"/"london" mismatch is a normalization
  difference, not a real extraction failure).
- Every metric's support count (n) is now reported alongside its value,
  and metrics computed from fewer than _MIN_MEANINGFUL_N cases are
  flagged `"low_sample": true` in the JSON output and printed with a
  warning marker — the 8 seed cases in cases.jsonl are far below this for
  every metric and are NOT a basis for tuning thresholds (see module
  docstring's "Primary target" line and docs/CONTRIBUTING.md's
  "Measure before tuning" policy referenced by confidence.py).
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

# Below this many supporting cases, a metric is flagged rather than trusted.
_MIN_MEANINGFUL_N = 30


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
    return {"low_sample": n < _MIN_MEANINGFUL_N}


async def run_eval(make_understander, cases: list[dict]) -> dict:
    n = correct = 0
    wrong_exec = oos_false_exec = wrong_entity_exec = 0
    oos_tp = oos_fp = oos_fn = 0
    clar_hit = clar_expected = clar_unexpected = 0
    status_breakdown = defaultdict(int)
    lat, by_cat = [], defaultdict(lambda: [0, 0])

    # Real call-source counters, aggregated across every case's understander
    # instance (a fresh one is made per case, per the module docstring).
    call_totals = {"guard": 0, "classifier": 0, "semantic": 0, "llm": 0,
                   "semantic_errors": 0, "llm_errors": 0}
    have_real_stats = False
    llm_source_hits = 0   # fallback proxy if an understander has no .stats

    # Entity accuracy is scoped to cases where the router's key matched the
    # expected key — scoring entities against the wrong command was never
    # meaningful (see module docstring).
    ent_total = ent_ok = 0

    for c in cases:
        u = make_understander()                      # fresh context per case
        for h in c.get("history", []):
            await u.understand(h)
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

        # "predicted OOS" = nothing was proposed as executable — covers
        # both UNKNOWN (nothing usable proposed) and REJECTED (something
        # was proposed and failed validation); see module docstring.
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
        stats = getattr(u, "stats", None)
        if isinstance(stats, dict):
            have_real_stats = True
            for k in call_totals:
                call_totals[k] += stats.get(k, 0)

    lat.sort()
    n_ = max(n, 1)
    result = {
        "cases": n,
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
        "by_category": {k: v[0] / v[1] for k, v in by_cat.items()},
    }
    return result


async def _main(path: Path) -> None:
    from brain.intent_engine import IntentEngine
    from brain.router import guards
    from brain.router.registry import CommandRegistry, load_specs
    from brain.router.understand import CommandUnderstander
    legacy, reg = IntentEngine(), CommandRegistry(load_specs())
    make = lambda: CommandUnderstander(legacy, reg, guard_fn=guards.check)   # add semantic/llm to eval those levels
    result = await run_eval(make, load_cases(path))
    print(json.dumps(result, indent=2))
    if result["cases"] < _MIN_MEANINGFUL_N:
        print(
            f"\nWARNING: only {result['cases']} case(s) evaluated — below the "
            f"{_MIN_MEANINGFUL_N}-case floor this module treats as meaningful. "
            f"Every metric above is marked \"low_sample\": true. Do not tune "
            f"thresholds from this run; see confidence.py's docstring.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    asyncio.run(_main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
