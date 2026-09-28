"""
brain/router/router_regression_test.py
Regression tests for the hybrid router's command-level retrieval, corpus
and confidence gate.

Named *_test.py (collected by pytest and `python -m unittest`) because
.gitignore excludes `tests/` and `test_*.py`.

Real repository components are used throughout: CommandVectorStore /
SQLiteVectorStore, CommandRegistry + config/command_domains.json,
SemanticRouter, confidence.evaluate, validate.validate and the adapter.
The ONLY substitute is the embedder: `LexicalEmbedder` implements the real
`Embedder` interface with a deterministic hashed word/bigram vector so the
suite needs no Ollama server. It therefore checks corpus coverage and the
retrieval/aggregation/confidence logic, not the quality of nomic-embed-text;
that is what brain/router/eval_router.py measures.
"""
import asyncio
import math
import re
import tempfile
import unittest
import zlib
from pathlib import Path

from brain.embeddings import Embedder
from brain.router import validate
from brain.router.adapter import to_legacy_intent
from brain.router.command_vector_store import CommandVectorStore
from brain.router.confidence import ConfidenceThresholds, Decision, evaluate
from brain.router.registry import CommandRegistry, CommandRegistryError, load_specs
from brain.router.schemas import CommandSpec, Command, aggregate_by_command
from brain.router.semantic_router import SemanticRouter

_THRESHOLDS = ConfidenceThresholds(min_similarity=0.80, min_margin=0.08, low_similarity_floor=0.55)

# ── Deterministic embedder (test double for the Embedder interface only) ──

_NUM_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "fifteen", "twenty", "thirty", "forty", "sixty", "ninety", "hundred"}
_STOP = {"a", "an", "the", "my", "me", "please", "is", "it", "to", "of", "for", "can", "you", "i", "do"}
_DIM = 1024


def _tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9']+", text.lower()):
        if t in _STOP:
            continue
        if t.isdigit() or t in _NUM_WORDS:
            t = "<num>"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def _lexical_vector(text: str) -> list[float]:
    toks = _tokens(text)
    vec = [0.0] * _DIM
    for t in toks:
        vec[zlib.crc32(t.encode()) % _DIM] += 1.0
    for a, b in zip(toks, toks[1:]):
        vec[zlib.crc32(f"{a}_{b}".encode()) % _DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class LexicalEmbedder(Embedder):
    async def embed(self, text: str):
        return _lexical_vector(text)


# ── Shared real-corpus fixture ───────────────────────────────────────────

_tmp = None
_registry = None
_router = None


def setUpModule():
    global _tmp, _registry, _router
    _tmp = tempfile.TemporaryDirectory()
    _registry = CommandRegistry(load_specs())
    store = CommandVectorStore(Path(_tmp.name) / "cmd.sqlite3")
    store.reseed(_registry.specs, _lexical_vector)
    _router = SemanticRouter(LexicalEmbedder(), store, _registry)


def tearDownModule():
    try:
        _tmp.cleanup()
    except OSError:
        pass   # Windows may hold the SQLite file briefly; the temp dir is disposable


def _retrieve(text: str):
    return asyncio.run(_router.retrieve(text))


# ── Helpers for controlled-vector tests ──────────────────────────────────

def _unit(cos_to_query: float) -> list[float]:
    """2-D unit vector whose cosine with the query (1, 0) is `cos_to_query`."""
    return [cos_to_query, math.sqrt(max(0.0, 1.0 - cos_to_query ** 2))]


def _spec(domain, op, seeds, legacy="x"):
    return CommandSpec(domain=domain, operation=op, legacy_intent=legacy, entities={},
                       target_mode="raw", seeds=tuple(seeds))


def _controlled_router(specs, scores: dict[str, float]):
    """Real store + real SemanticRouter; seed text -> chosen cosine to query."""
    tmp = tempfile.TemporaryDirectory()
    store = CommandVectorStore(Path(tmp.name) / "c.sqlite3")
    store.reseed(specs, lambda seed: _unit(scores[seed]))

    class _Q(Embedder):
        async def embed(self, text):
            return [1.0, 0.0]

    reg = CommandRegistry(specs)
    return SemanticRouter(_Q(), store, reg), store, tmp


# ── 1. Retrieval aggregation ─────────────────────────────────────────────

class RetrievalAggregationTests(unittest.TestCase):
    def test_spec_example_margin_is_command_level(self):
        specs = [_spec("timer", "create", ["c1", "c2"]), _spec("timer", "status", ["s1"]),
                 _spec("timer", "cancel", ["x1"])]
        scores = {"c1": 0.86, "c2": 0.84, "s1": 0.70, "x1": 0.61}
        router, _, tmp = _controlled_router(specs, scores)
        res = asyncio.run(router.retrieve("anything"))
        d = res.diagnostics()
        self.assertEqual(d["winning_command"], "timer.create")
        self.assertAlmostEqual(d["winning_score"], 0.86, places=3)
        self.assertEqual(d["second_command"], "timer.status")
        self.assertAlmostEqual(d["second_score"], 0.70, places=3)
        self.assertAlmostEqual(d["margin"], 0.16, places=3)
        self.assertAlmostEqual(res.margin, 0.16, places=3)
        keys = [c.spec.key for c in res.candidates]
        self.assertEqual(len(keys), len(set(keys)), "a command appeared twice as a candidate")

    def test_many_prototypes_do_not_crowd_out_other_commands(self):
        create_seeds = [f"c{i}" for i in range(30)]
        specs = [_spec("timer", "create", create_seeds), _spec("timer", "status", ["s1"])]
        scores = {s: 0.90 - i * 0.001 for i, s in enumerate(create_seeds)}
        scores["s1"] = 0.84
        router, _, tmp = _controlled_router(specs, scores)
        res = asyncio.run(router.retrieve("anything"))
        self.assertEqual(res.top2.spec.key, "timer.status")
        self.assertAlmostEqual(res.margin, 0.06, places=3)
        self.assertEqual(evaluate(res, _THRESHOLDS), Decision.AMBIGUOUS)

    def test_same_command_prototypes_never_create_ambiguity(self):
        specs = [_spec("timer", "create", ["c1", "c2"]), _spec("timer", "status", ["s1"])]
        router, _, tmp = _controlled_router(specs, {"c1": 0.90, "c2": 0.89, "s1": 0.50})
        res = asyncio.run(router.retrieve("anything"))
        self.assertAlmostEqual(res.margin, 0.40, places=3)
        self.assertEqual(evaluate(res, _THRESHOLDS), Decision.CONFIDENT)

    def test_stored_vectors_carry_grouping_metadata(self):
        specs = [_spec("timer", "create", ["c1"])]
        _, store, tmp = _controlled_router(specs, {"c1": 0.9})
        rows = store._store.search([1.0, 0.0], top_k=10, mem_type="command")
        self.assertEqual(len(rows), 1)
        meta = rows[0][0].metadata
        self.assertEqual((meta["domain"], meta["operation"], meta["prototype"]), ("timer", "create", "c1"))

    def test_aggregate_is_max_and_deterministic_on_ties(self):
        a, b = _spec("d", "a", ["p"]), _spec("d", "b", ["q"])
        res = aggregate_by_command([(b, 0.7, "q"), (a, 0.7, "p"), (a, 0.6, "p2")])
        self.assertEqual([c.spec.key for c in res.candidates], ["d.a", "d.b"])
        self.assertAlmostEqual(res.candidates[0].command_score, 0.7)

    def test_single_command_has_zero_second_score(self):
        res = aggregate_by_command([(_spec("d", "a", ["p"]), 0.8, "p")])
        self.assertIsNone(res.diagnostics()["second_command"])


# ── 2. Registry / corpus architecture ────────────────────────────────────

class RegistryTests(unittest.TestCase):
    def _write(self, ops):
        p = Path(_tmp.name) / "d.json"
        import json
        p.write_text(json.dumps({"domains": {"x": {"operations": ops}}}))
        return p

    def test_duplicate_seed_across_operations_is_rejected(self):
        p = self._write({"a": {"legacy_intent": "a", "seeds": ["same"]},
                         "b": {"legacy_intent": "b", "seeds": ["Same "]}})
        with self.assertRaises(CommandRegistryError):
            load_specs(p)

    def test_repeat_within_operation_is_dropped_in_order(self):
        p = self._write({"a": {"legacy_intent": "a", "seeds": ["one", "two", "one"]}})
        self.assertEqual(load_specs(p)[0].seeds, ("one", "two"))

    def test_real_corpus_has_multiple_prototypes_per_command(self):
        for s in _registry.specs:
            self.assertGreaterEqual(len(s.seeds), 2, s.key)

    def test_corpus_build_is_reproducible(self):
        a = CommandVectorStore(Path(_tmp.name) / "r1.sqlite3")
        b = CommandVectorStore(Path(_tmp.name) / "r2.sqlite3")
        a.reseed(_registry.specs, _lexical_vector)
        b.reseed(_registry.specs, _lexical_vector)
        self.assertEqual(a.stored_fingerprint(), b.stored_fingerprint())
        self.assertFalse(a.is_stale(_registry.specs))


# ── 3. Per-domain routing on held-out phrasings ──────────────────────────

class CommandRoutingTests(unittest.TestCase):
    def assertRoutes(self, cases):
        for text, expected in cases:
            with self.subTest(text=text):
                res = _retrieve(text)
                self.assertEqual(res.top1.spec.key, expected, res.diagnostics())

    def test_timer(self):
        self.assertRoutes([
            ("2 minute timer", "timer.create"),
            ("set timer for ten minutes", "timer.create"),
            ("start a five minute countdown", "timer.create"),
            ("how long is left on my countdown", "timer.status"),
            ("is the timer still going", "timer.status"),
            ("cancel that countdown", "timer.cancel"),
            ("please cancel my timer", "timer.cancel"),
        ])

    def test_datetime(self):
        self.assertRoutes([
            ("what is the current year", "datetime.date"),
            ("tell me today's date", "datetime.date"),
            ("what is the current time", "datetime.time"),
            ("can you tell me what time it is", "datetime.time"),
        ])

    def test_media(self):
        self.assertRoutes([
            ("turn the sound off", "media.mute"),
            ("silence everything", "media.mute"),
            ("pause my music", "media.pause"),
            ("turn down the volume", "media.volume_down"),
            ("lower the sound level", "media.volume_down"),
        ])

    def test_web_open_weather(self):
        self.assertRoutes([
            ("search for cheap laptops", "web.search"),
            ("look up the price of bitcoin", "web.search"),
            ("google best restaurants nearby", "web.search"),
            ("open netflix", "app_or_web.open"),
            ("what's the weather in berlin", "weather.current"),
            ("weather today", "weather.current"),
        ])

    def test_notes(self):
        self.assertRoutes([
            ("write down buy eggs", "notes.create"),
            ("take a quick note", "notes.create"),
            ("show me my notes", "notes.view"),
            ("list all saved notes", "notes.view"),
            ("remove my note", "notes.delete"),
            ("erase my saved note", "notes.delete"),
        ])

    def test_system(self):
        self.assertRoutes([
            ("shut the computer down", "system.shutdown"),
            ("turn off the computer", "system.shutdown"),
            ("restart my computer", "system.restart"),
            ("reboot the machine", "system.restart"),
            ("lock the pc", "system.lock"),
            ("what's my cpu usage", "system.info"),
            ("how much battery do i have", "system.info"),
        ])


# ── 4. Confidence / out-of-scope ─────────────────────────────────────────

class ConfidenceTests(unittest.TestCase):
    def test_close_commands_are_not_forced_into_direct_execution(self):
        specs = [_spec("a", "x", ["p1"]), _spec("a", "y", ["p2"])]
        router, _, tmp = _controlled_router(specs, {"p1": 0.88, "p2": 0.85})
        res = asyncio.run(router.retrieve("q"))
        self.assertEqual(evaluate(res, _THRESHOLDS), Decision.AMBIGUOUS)

    def test_weak_best_match_is_low(self):
        specs = [_spec("a", "x", ["p1"]), _spec("a", "y", ["p2"])]
        router, _, tmp = _controlled_router(specs, {"p1": 0.40, "p2": 0.10})
        self.assertEqual(evaluate(asyncio.run(router.retrieve("q")), _THRESHOLDS), Decision.LOW)

    def test_out_of_scope_utterances_are_never_confident(self):
        for text in ["what is photosynthesis", "tell me a joke about cats", "who won the game last night",
                     "i had a rough day at work", "explain how black holes form"]:
            with self.subTest(text=text):
                self.assertNotEqual(evaluate(_retrieve(text), _THRESHOLDS), Decision.CONFIDENT)

    def test_confident_result_validates_and_adapts_to_legacy_contract(self):
        res = _retrieve("set a timer for ten minutes")
        self.assertEqual(evaluate(res, _THRESHOLDS), Decision.CONFIDENT)
        top = res.top1
        cmd = Command(domain=top.spec.domain, operation=top.spec.operation, entities={},
                      confidence=top.similarity)
        v = validate.validate(cmd, _registry)
        self.assertTrue(v.ok)
        intent = to_legacy_intent(cmd, v.spec, "set a timer for ten minutes",
                                  confidence=top.similarity, model_source="semantic_retrieval")
        self.assertEqual(intent["intent"], "set_timer")
        self.assertEqual(intent["target"], "set a timer for ten minutes")
        self.assertEqual(intent["response_mode"], "skill")
        for k in ("intent", "target", "confidence", "raw", "model", "response_mode"):
            self.assertIn(k, intent)


if __name__ == "__main__":
    unittest.main()
