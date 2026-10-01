#!/usr/bin/env python
"""
stage_data.py - safe, deduplicating staging of hand-written training rows and
eval cases. Run from the repo root. Dry run unless --apply is passed.

  python stage_data.py                 # report only
  python stage_data.py --apply         # append new rows / cases

Rows go to datasets/training/candidates.jsonl (append only, existing rows are
never rewritten); cases go to datasets/router_eval/cases.jsonl. The source
files are never modified. Text is normalized (case, apostrophes, punctuation,
whitespace) before comparison.

Rows are skipped when they: repeat an earlier source row, equal an eval-case
text, equal or are near-duplicates (>= --near) of trusted train/validation/test
rows, of candidates.jsonl rows, or of eval cases. Cases are skipped when their
text already exists, or when they equal / near-duplicate any training text
(that would leak the eval into training).
"""
import argparse, difflib, json, pathlib, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--rows",  default="datasets/training/hard_negatives.jsonl")
ap.add_argument("--cases", default="datasets/router_eval/cases_extension.jsonl")
ap.add_argument("--near",  type=float, default=0.90)
ap.add_argument("--apply", action="store_true")
ap.add_argument("--allow-other-verified", action="store_true")
a = ap.parse_args()

TRAIN = pathlib.Path("datasets/training")
CAND  = TRAIN / "candidates.jsonl"
CASES = pathlib.Path("datasets/router_eval/cases.jsonl")
SRC_ROWS, SRC_CASES = pathlib.Path(a.rows), pathlib.Path(a.cases)

def norm(s):
    s = s.lower().replace("\u2019", "'").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())

def rd(p):
    p = pathlib.Path(p)
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def near(n, pool):
    for m in pool:
        if difflib.SequenceMatcher(None, n, m).ratio() >= a.near:
            return m
    return None

def append(path, items):
    path = pathlib.Path(path)
    nl = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as f:
        if nl: f.write("\n")
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")

rows, new_cases = rd(SRC_ROWS), rd(SRC_CASES)
cand = rd(CAND)
trusted = {}
for f in ("train_data", "validation_data", "test_data"):
    for r in rd(TRAIN / f"{f}.jsonl"):
        trusted[norm(r["text"])] = (f, r["intent"])
cand_idx   = {norm(r["text"]): r for r in cand}
have_cases = {norm(c["text"]) for c in rd(CASES)}
eval_texts = have_cases | {norm(c["text"]) for c in new_cases}

others = [r for r in cand if r.get("verified")]
if others and not a.allow_other_verified:
    sys.exit(f"ABORT: candidates.jsonl already has {len(others)} verified row(s); "
             f"`candidates promote` would consume them too. Review, or pass --allow-other-verified.")

skip = {k: [] for k in ("dup_in_source", "eval_leak", "trusted_same_intent", "trusted_other_intent",
                         "cand_verified", "cand_unverified", "cand_other_intent", "near_dup")}
add_rows, seen = [], set()
train_pool = list(trusted) + list(cand_idx)
for r in rows:
    n = norm(r["text"])
    if n in seen: skip["dup_in_source"].append(r["text"]); continue
    seen.add(n)
    if n in eval_texts: skip["eval_leak"].append(r["text"]); continue
    if n in trusted:
        k = "trusted_same_intent" if trusted[n][1] == r["intent"] else "trusted_other_intent"
        skip[k].append(f'{r["text"]} [{trusted[n][0]}/{trusted[n][1]}]'); continue
    if n in cand_idx:
        c = cand_idx[n]
        k = "cand_other_intent" if c["intent"] != r["intent"] else ("cand_verified" if c.get("verified") else "cand_unverified")
        skip[k].append(r["text"]); continue
    m = near(n, train_pool + list(eval_texts))
    if m: skip["near_dup"].append(f'{r["text"]}  ~  {m}'); continue
    add_rows.append(r)

case_skip = {"already_present": [], "leaks_into_training": []}
add_cases, train_all = [], set(train_pool) | {norm(r["text"]) for r in add_rows}
for c in new_cases:
    n = norm(c["text"])
    if n in have_cases: case_skip["already_present"].append(c["text"]); continue
    if n in train_all or near(n, train_all):
        case_skip["leaks_into_training"].append(c["text"]); continue
    add_cases.append(c)

print(f"ROWS : source={len(rows)} new={len(add_rows)} skipped={sum(map(len, skip.values()))}")
for k, v in skip.items():
    if v:
        print(f"  skipped {k}: {len(v)}")
        for t in v: print("      -", t)
print(f"CASES: source={len(new_cases)} new={len(add_cases)} skipped={sum(map(len, case_skip.values()))}")
for k, v in case_skip.items():
    if v:
        print(f"  skipped {k}: {len(v)}")
        for t in v: print("      -", t)
if skip["cand_unverified"]:
    print("NOTE: rows under cand_unverified are NOT staged for promotion; verify by hand if wanted.")

if not a.apply:
    print("DRY RUN - nothing written. Re-run with --apply."); sys.exit(0)
if add_rows:  append(CAND, add_rows)
if add_cases: append(CASES, add_cases)
print(f"APPLIED: {len(add_rows)} row(s) -> {CAND} (existing {len(cand)} untouched); "
      f"{len(add_cases)} case(s) -> {CASES}. Source files unmodified.")
