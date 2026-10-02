# TEMPORARY DIAGNOSTIC - not production. Delete after use.
import asyncio, hashlib, sys
from brain.intent_engine import IntentEngine
from brain.router import guards, llm_fallback
from brain.router.registry import CommandRegistry, load_specs
from brain.router.understand import CommandUnderstander
from brain.router.normalize import normalize
from brain.router.confidence import ConfidenceThresholds, evaluate
from brain.router.semantic_router import SemanticRouter
from brain.router.command_vector_store import CommandVectorStore
from brain.embeddings import OllamaEmbedder
from config.settings import config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CASES = [  # (text, history)
    ("hey could you jot a quick memo that the dentist is at three", []),
    ("could you pull up spotify for me", []),
    ("what's it like outside right now", []),
    ("hit me up in twelve minutes", []),
    ("I have 5 minutes to kill before my meeting", ["set a timer"]),
]
legacy, reg = IntentEngine(), CommandRegistry(load_specs())
sem = SemanticRouter(OllamaEmbedder(), CommandVectorStore(config.router.command_vector_db_path), reg)
seen = []

async def spy(utt, domains, ops, top=None):
    p = llm_fallback._build_prompt(utt, domains, ops, top or [])
    raw = await llm_fallback.route(utt, domains, ops, top)
    seen.append({"prompt_sha": hashlib.sha256(p.encode()).hexdigest()[:10], "raw": raw})
    return raw

async def main():
    for text, hist in CASES:
        c = legacy.classify(text)
        r = await sem.retrieve(normalize(text))
        print(f"\n== {text!r}")
        print(f" classifier: {c['intent']} {c['confidence']} margin={c['margin']} via {c['model']} mode={c['response_mode']}")
        print(f" semantic: {[(x.spec.key, round(x.similarity,3)) for x in r.candidates[:3]]} "
              f"margin={r.margin:.3f} decision={evaluate(r, ConfidenceThresholds()).value}")
        for i in range(N):
            u = CommandUnderstander(legacy, reg, guard_fn=guards.check, semantic=sem.retrieve, llm_route=spy)
            for h in hist:
                await u.understand(h)
            seen.clear()
            ir = await u.understand(text)
            print(f" run{i}: status={ir.status.value} key={ir.key} src={ir.source} reason={ir.reason!r}")
            for s in seen:
                print(f"        llm prompt={s['prompt_sha']} raw={s['raw']}")
asyncio.run(main())