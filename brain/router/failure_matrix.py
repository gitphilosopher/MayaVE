"""
brain/router/failure_matrix.py
Cross-seed failure analysis for eval_ir runs. Reads the `per_case` list from
each run JSON and writes logs/runs60/failure_matrix.txt.

    python -m brain.router.failure_matrix
    python -m brain.router.failure_matrix --dir logs/runs60

Symbols: OK = pass, X = fail, ~ = passed only because a REJECTED result on an
expected-unknown case is scored as a pass (LLM overreach).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SEEDS = ["seed1", "seed2", "seed3"]
REPEAT = "seed1_repeat"


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))   # files carry a BOM
    per_case = data.get("per_case")
    if not per_case:
        raise SystemExit(f"{path} has no 'per_case' list - re-run eval_ir with the patch applied.")
    return {
        "digest": data.get("model_digest", {}),
        "cases": {c["text"]: c for c in per_case},
    }


def sym(c) -> str:
    if not c["ok"]:
        return "X"
    return "~" if c["overreach"] else "OK"


def describe(c) -> str:
    return (f'want={c["want_status"]}/{c["want_key"]}  got={c["got_status"]}/{c["got_key"]}  '
            f'src={c["source"]}  reason={c["reason"]!r}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/runs60")
    a = ap.parse_args()
    d = Path(a.dir)

    runs = {name: load(d / f"{name}.json") for name in SEEDS + [REPEAT]}
    texts = list(runs[SEEDS[0]]["cases"])            # eval order
    for name, r in runs.items():
        if set(r["cases"]) != set(texts):
            raise SystemExit(f"{name} was run on a different case set - re-run all four on the same cases.jsonl.")

    out = []
    w = out.append

    # 0. determinism check
    w("=== Run identity ===")
    for name, r in runs.items():
        w(f"{name:<14} combined={r['digest'].get('combined')} seed={r['digest'].get('seed')} "
          f"training_hash={r['digest'].get('training_hash')}")
    same_model = runs["seed1"]["digest"].get("combined") == runs[REPEAT]["digest"].get("combined")
    diff = [t for t in texts if runs["seed1"]["cases"][t]["ok"] != runs[REPEAT]["cases"][t]["ok"]
            or runs["seed1"]["cases"][t]["got_key"] != runs[REPEAT]["cases"][t]["got_key"]
            or runs["seed1"]["cases"][t]["got_status"] != runs[REPEAT]["cases"][t]["got_status"]]
    w(f"\nseed1 vs seed1_repeat: same model={same_model}, differing case outcomes={len(diff)}")
    for t in diff:
        w(f"   NONDETERMINISTIC: {t!r}  {sym(runs['seed1']['cases'][t])} -> {sym(runs[REPEAT]['cases'][t])}"
          f"  (LLM fallback is temperature 0 but not guaranteed identical)")

    # 1. per-seed failure lists
    for name in SEEDS:
        cs = runs[name]["cases"]
        fails = [cs[t] for t in texts if not cs[t]["ok"]]
        w(f"\n=== {name}: {len(fails)} failure(s) of {len(texts)} ===")
        for c in fails:
            w(f'X [{c["category"]}] "{c["text"]}"')
            w(f"     {describe(c)}")
        over = [cs[t] for t in texts if cs[t]["overreach"]]
        if over:
            w(f"  -- {len(over)} overreach (scored pass, but REJECTED instead of UNKNOWN):")
            for c in over:
                w(f'~ [{c["category"]}] "{c["text"]}"  reason={c["reason"]!r}')

    # 2. matrix (only rows that are not clean passes everywhere)
    cols = SEEDS + [REPEAT]
    w("\n=== Cross-seed matrix (rows with at least one X or ~) ===")
    w(f'{"case":<62} ' + " ".join(f"{c:<12}" for c in cols) + " fails")
    freq = {}
    for t in texts:
        marks = [sym(runs[c]["cases"][t]) for c in cols]
        n_fail = sum(runs[s]["cases"][t]["ok"] is False for s in SEEDS)
        freq[t] = n_fail
        if any(m != "OK" for m in marks):
            w(f'{t[:60]:<62} ' + " ".join(f"{m:<12}" for m in marks) + f" {n_fail}/3")

    # 3. frequency buckets across the three DISTINCT seeds
    w("\n=== Failure frequency across seed1/seed2/seed3 ===")
    for k, label in ((3, "3/3  consistent: likely a real routing/data problem"),
                     (2, "2/3  seed-sensitive boundary"),
                     (1, "1/3  likely training variance")):
        rows = [t for t in texts if freq[t] == k]
        w(f"\n{label}  ({len(rows)})")
        for t in rows:
            c = runs["seed1"]["cases"][t]
            w(f'   [{c["category"]}] "{t}"  expected {c["want_status"]}/{c["want_key"]}')
            for s in SEEDS:
                cs = runs[s]["cases"][t]
                if not cs["ok"]:
                    w(f"        {s}: got {cs['got_status']}/{cs['got_key']} via {cs['source']}")

    # 4. category breakdown
    w("\n=== Failures by category (summed over the 3 seeds) ===")
    cat_fail, cat_total = defaultdict(int), defaultdict(int)
    for t in texts:
        cat = runs["seed1"]["cases"][t]["category"]
        cat_total[cat] += 3
        cat_fail[cat] += freq[t]
    for cat in sorted(cat_total, key=lambda c: -cat_fail[c]):
        w(f"{cat:<18} {cat_fail[cat]:>3}/{cat_total[cat]:<3} failing case-runs")

    # 5. failure mode summary: what kind of wrong answer is it?
    w("\n=== Failure modes (all seeds) ===")
    modes = defaultdict(int)
    for s in SEEDS:
        for c in runs[s]["cases"].values():
            if not c["ok"]:
                if c["want_status"] == "unknown" and c["got_status"] == "ready":
                    modes["false execution (OOS -> ready)"] += 1
                elif c["want_status"] == "ready" and c["got_status"] == "ready":
                    modes["wrong command"] += 1
                elif c["want_status"] == "ready":
                    modes[f"missed command (-> {c['got_status']})"] += 1
                else:
                    modes[f"expected {c['want_status']}, got {c['got_status']}"] += 1
    for m, n in sorted(modes.items(), key=lambda kv: -kv[1]):
        w(f"{n:>3}  {m}")

    text = "\n".join(out)
    (d / "failure_matrix.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {d / 'failure_matrix.txt'}")


if __name__ == "__main__":
    main()