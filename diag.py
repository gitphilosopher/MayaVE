"""
diag.py - one-shot diagnostics for the router eval. Run from the repo root:

    $env:PYTHONUTF8="1"; $env:MAYA_EMBEDDING_DEVICE="cpu"
    python diag.py 2>$null | Out-File -Encoding utf8 diag.txt

Part 1  IR misses on the CURRENT saved model (same pass rule as eval_ir:
        a REJECTED result on an expected-unknown case counts as a pass but is
        listed as OVERREACH). Prints classifier + semantic scores per miss.
Part 2  Classifier misses on validation/test for the rows this session added
        (variant hard_negative / positive_control) and for the intents whose
        test metrics moved (get_time, set_timer, timer_status, search_web,
        smalltalk, unknown). Read-only; note classify() appends low-confidence
        rows to logs/intent_failures.jsonl as it always does.
"""
import asyncio

from brain.intent_engine import (IntentEngine, load_intent_config, load_dataset,
                                 _VALIDATION_FILE, _TEST_FILE)
from brain.router import guards
from brain.router.registry import CommandRegistry, load_specs
from brain.router.understand import CommandUnderstander
from brain.router.normalize import normalize
from brain.router.eval_ir import load_cases
from brain.embeddings import OllamaEmbedder
from brain.router.command_vector_store import CommandVectorStore
from brain.router.semantic_router import SemanticRouter
from brain.router.llm_fallback import route as llm
from config.settings import config

legacy, reg = IntentEngine(), CommandRegistry(load_specs())
store = CommandVectorStore(getattr(config.router, "command_vector_db_path", None))
sem = SemanticRouter(OllamaEmbedder(), store, reg)
make = lambda: CommandUnderstander(legacy, reg, guard_fn=guards.check,
                                   semantic=sem.retrieve, llm_route=llm)


async def part1():
    print("=== PART 1: IR misses (current model) ===")
    n = bad = 0
    for c in load_cases():
        n += 1
        u = make()
        for h in c.get("history", []):
            await u.understand(h)
        ir = await u.understand(c["text"])
        exp = c["expect"]
        overreach = exp["status"] == "unknown" and ir.status.value == "rejected"
        got = "unknown" if overreach else ir.status.value
        ok = got == exp["status"] and (exp.get("key") is None or ir.key == exp["key"])
        if ok and not overreach:
            continue
        bad += (not ok)
        r = legacy.classify(c["text"])
        d = (await sem.retrieve(normalize(c["text"]))).diagnostics()
        print(f'{"OVERREACH" if ok else "MISS"} [{c["category"]}] {c["text"]!r}')
        print(f'   want={exp["status"]}/{exp.get("key")} got={ir.status.value}/{ir.key} '
              f'src={ir.source} reason={ir.reason!r}')
        print(f'   classifier: {r["intent"]} {r["confidence"]} margin={r["margin"]} via {r["model"]}')
        print(f'   semantic: {d["winning_command"]} {d["winning_score"]} / '
              f'{d["second_command"]} {d["second_score"]} margin={d["margin"]}')
    print(f"--- {n - bad}/{n} pass ---\n")


def part2():
    print("=== PART 2: classifier misses on val/test (new rows + moved intents) ===")
    valid = {e["id"] for e in load_intent_config()["intents"]}
    watch = {"get_time", "set_timer", "timer_status", "search_web", "smalltalk", "unknown"}
    for name, path in (("val", _VALIDATION_FILE), ("test", _TEST_FILE)):
        for r in load_dataset(path, valid):
            res = legacy.classify(r["text"])
            if res["intent"] == r["intent"]:
                continue
            new = r.get("variant") in ("hard_negative", "positive_control")
            if new or r["intent"] in watch or res["intent"] in watch:
                print(f'{name:<4} {"NEW" if new else "   "} {r["intent"]} -> {res["intent"]} '
                      f'({res["confidence"]} via {res["model"]}) | {r.get("variant")} | {r["text"]}')


asyncio.run(part1())
part2()
