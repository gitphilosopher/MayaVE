import asyncio, json
import numpy as np
import pytest

from brain.router import validate, llm_fallback
from brain.router.adapter import to_legacy_intent
from brain.router.confidence import ConfidenceThresholds, Decision, evaluate
from brain.router.command_vector_store import CommandVectorStore, corpus_fingerprint
from brain.router.registry import CommandRegistry, load_specs
from brain.router.schemas import Command, CommandCandidate, RetrievalResult

REG = CommandRegistry(load_specs())
TH = ConfidenceThresholds()
LEGACY_KEYS = {"intent", "target", "confidence", "raw", "model", "response_mode"}


def _rr(*sims):
    spec = REG.specs[0]
    return RetrievalResult([CommandCandidate(spec, s, "x") for s in sims])


# ── registry / taxonomy drift ──
def test_registry_intents_exist_in_intents_json():
    ids = {e["id"] for e in json.load(open("datasets/intents.json"))["intents"]} if __import__("os").path.exists("datasets/intents.json") else None
    if ids is None:
        pytest.skip("datasets/intents.json not present in this checkout")
    for s in REG.specs:
        assert s.legacy_intent in ids


def test_legacy_intents_unique():
    li = [s.legacy_intent for s in REG.specs]
    assert len(li) == len(set(li))


# ── confidence ──
def test_confident():
    assert evaluate(_rr(0.9, 0.5), TH) == Decision.CONFIDENT

def test_ambiguous_close_top2():
    assert evaluate(_rr(0.9, 0.88), TH) == Decision.AMBIGUOUS

def test_low():
    assert evaluate(_rr(0.4, 0.1), TH) == Decision.LOW
    assert evaluate(_rr(), TH) == Decision.LOW


# ── validation ──
@pytest.mark.parametrize("raw", [None, [], {}, {"domain": "timer"}, {"domain": "", "operation": "x"}])
def test_raw_structural_rejects(raw):
    assert validate.validate_raw_llm_output(raw) is None

def test_bad_confidence_clamped():
    c = validate.validate_raw_llm_output({"domain": "timer", "operation": "create", "confidence": "abc"})
    assert c.confidence == 0.0

@pytest.mark.parametrize("cmd,err", [
    (Command("nope", "x", {}, 0.9), "unknown"),
    (Command("timer", "explode", {}, 0.9), "unknown"),
    (Command("unknown", "unknown", {}, 0.0), "not_a_command"),
    (Command("timer", "create", {"bogus": "1"}, 0.9), "unexpected entity"),
    (Command("timer", "create", {"duration": 5}, 0.9), "must be a string"),
    (Command("web", "search", {}, 0.9), "missing required"),
    (Command("timer", "create", {}, 0.2, True), "needs_clarification"),
])
def test_validation_failures(cmd, err):
    r = validate.validate(cmd, REG)
    assert not r.ok and err in r.error

def test_validation_ok():
    assert validate.validate(Command("web", "search", {"query": "cats"}, 0.9), REG).ok


# ── adapter ──
def test_adapter_contract_and_entity_target():
    cmd = Command("app_or_web", "open", {"target": "youtube"}, 0.9)
    spec = validate.validate(cmd, REG).spec
    d = to_legacy_intent(cmd, spec, "pull up youtube", confidence=0.9, model_source="semantic_retrieval")
    assert LEGACY_KEYS <= d.keys()
    assert d["intent"] == "open_target" and d["target"] == "youtube" and d["response_mode"] == "skill"
    assert d["_command"]["source"] == "semantic_retrieval"

def test_adapter_raw_for_timer():
    cmd = Command("timer", "create", {}, 0.9)
    spec = validate.validate(cmd, REG).spec
    d = to_legacy_intent(cmd, spec, "set a timer for ten minutes", confidence=0.9, model_source="x")
    assert d["intent"] == "set_timer" and d["target"] == "set a timer for ten minutes"

def test_adapter_every_spec_maps():
    for s in REG.specs:
        d = to_legacy_intent(Command(s.domain, s.operation, {}, 1.0), s, "raw text", confidence=1.0, model_source="t")
        assert d["intent"] == s.legacy_intent

def test_semantic_missing_required_entity_not_dispatched(monkeypatch, tmp_path):
    from brain.router import hybrid_engine as he
    called = []
    eng = _engine(monkeypatch, tmp_path,
                  _cands(("web.search", .95), ("app_or_web.open", .5)), None)
    async def boom(*a, **k): called.append(1); return None
    monkeypatch.setattr(he, "llm_route", boom)
    d = asyncio.run(eng.aclassify("search"))
    assert d["model"] == "legacy" and d["intent"] == "general_query"
    assert "_command" not in d and not called

# ── vector store isolation / fingerprint / search ──
def test_store_seed_search_and_stale(tmp_path):
    db = tmp_path / "cmd.sqlite3"
    st = CommandVectorStore(db)
    specs = REG.specs
    assert st.is_stale(specs)
    rng = np.random.default_rng(0)
    vecs = {}
    def emb(t):
        vecs.setdefault(t, rng.normal(size=16).tolist()); return vecs[t]
    n = st.reseed(specs, emb)
    assert n == sum(len(s.seeds) for s in specs)
    assert not st.is_stale(specs)
    res = st.search(vecs["set a timer for ten minutes"], REG.by_key, top_k=3)
    assert res.top1.spec.key == "timer.create" and res.top1_similarity > 0.99
    assert len(res.candidates) == len({c.spec.key for c in res.candidates})  # dedup per op
    assert corpus_fingerprint(specs) != corpus_fingerprint(specs[:-1])


# ── LLM fallback failure modes (never raise) ──
class _FakeResp:
    def __init__(self, body, status=200): self._b, self.status_code = body, status
    def raise_for_status(self):
        import httpx
        if self.status_code >= 400: raise httpx.HTTPStatusError("x", request=None, response=None)
    def json(self):
        if isinstance(self._b, Exception): raise self._b
        return self._b

def _patch_client(monkeypatch, behavior):
    class C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return behavior()
    monkeypatch.setattr(llm_fallback.httpx, "AsyncClient", C)

def _route():
    return asyncio.run(llm_fallback.route("x", REG.domains(), {d: REG.operations_for(d) for d in REG.domains()}))

def test_llm_timeout(monkeypatch):
    import httpx
    def b(): raise httpx.ReadTimeout("t")
    _patch_client(monkeypatch, b); assert _route() is None

def test_llm_connect_error(monkeypatch):
    import httpx
    def b(): raise httpx.ConnectError("c")
    _patch_client(monkeypatch, b); assert _route() is None

def test_llm_malformed_json(monkeypatch):
    _patch_client(monkeypatch, lambda: _FakeResp({"message": {"content": "not json"}})); assert _route() is None

def test_llm_non_object(monkeypatch):
    _patch_client(monkeypatch, lambda: _FakeResp({"message": {"content": "[1]"}})); assert _route() is None

def test_llm_model_missing_404(monkeypatch):
    _patch_client(monkeypatch, lambda: _FakeResp({}, 404)); assert _route() is None

def test_llm_good(monkeypatch):
    body = {"message": {"content": json.dumps({"domain": "weather", "operation": "current", "entities": {}, "confidence": .9})}}
    _patch_client(monkeypatch, lambda: _FakeResp(body)); assert _route()["domain"] == "weather"


# ── hybrid engine flow with fakes ──
class _FakeLegacy:
    _response_modes = {"general_query": "llm"}
    def __init__(self): self.calls = []
    def classify(self, t):
        self.calls.append(t)
        return {"intent": "general_query", "target": t, "confidence": .5, "raw": t, "model": "legacy", "response_mode": "llm"}
    def _is_dismissal(self, t): return False
    _PRESENCE_RE = __import__("re").compile(r"(?!x)x")
    _ACTION_WORD_RE = __import__("re").compile(r"(?!x)x")
    _ACTION_QUESTION_RE = __import__("re").compile(r"(?!x)x")
    def _keyword_fallback(self, t): return "unknown", 0.0, "keyword"

def _engine(monkeypatch, tmp_path, retrieval, llm_result, backend="hybrid", domains=()):
    from brain.router import hybrid_engine as he
    from config.settings import config
    monkeypatch.setattr(config.router, "backend", backend)
    monkeypatch.setattr(config.router, "hybrid_domains", list(domains))
    monkeypatch.setattr(config.router, "command_vector_db_path", tmp_path / "c.sqlite3")
    eng = he.HybridIntentEngine(legacy_engine=_FakeLegacy())
    async def retrieve(text, top_k=5): return retrieval
    monkeypatch.setattr(eng._semantic, "retrieve", retrieve)
    async def route(*a, **k): return llm_result
    monkeypatch.setattr(he, "llm_route", route)
    return eng

def _cands(*pairs):
    by = REG.by_key
    return RetrievalResult([CommandCandidate(by[k], s, "seed") for k, s in pairs])

def test_confident_skips_llm(monkeypatch, tmp_path):
    called = []
    eng = _engine(monkeypatch, tmp_path, _cands(("datetime.time", .95), ("datetime.date", .6)), None)
    from brain.router import hybrid_engine as he
    async def boom(*a, **k): called.append(1); return None
    monkeypatch.setattr(he, "llm_route", boom)
    d = asyncio.run(eng.aclassify("got the time?"))
    assert d["intent"] == "get_time" and not called and LEGACY_KEYS <= d.keys()

def test_ambiguous_goes_to_llm_and_validates(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, _cands(("media.volume_up", .85), ("media.volume_down", .84)),
                  {"domain": "media", "operation": "volume_down", "entities": {}, "confidence": .9})
    d = asyncio.run(eng.aclassify("make it quieter"))
    assert d["intent"] == "volume_down" and d["_command"]["source"] == "llm_fallback"

def test_hallucinated_llm_falls_to_legacy(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, _cands(("media.volume_up", .6), ("media.volume_down", .59)),
                  {"domain": "system", "operation": "format_disk", "entities": {}, "confidence": 1})
    d = asyncio.run(eng.aclassify("do something"))
    assert d["intent"] == "general_query"

def test_llm_down_falls_to_legacy(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, _cands(("media.volume_up", .6), ("media.volume_down", .59)), None)
    assert asyncio.run(eng.aclassify("hmm"))["intent"] == "general_query"

def test_legacy_backend_never_uses_hybrid(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, _cands(("datetime.time", .99), ("datetime.date", .1)), None, backend="legacy")
    assert asyncio.run(eng.aclassify("what time"))["model"] == "legacy"

def test_domain_allowlist(monkeypatch, tmp_path):
    eng = _engine(monkeypatch, tmp_path, _cands(("media.volume_up", .99), ("media.next", .1)), None, domains=["datetime"])
    assert asyncio.run(eng.aclassify("louder"))["model"] == "legacy"

def test_shadow_never_dispatches_hybrid(monkeypatch, tmp_path):
    from config.settings import config
    eng = _engine(monkeypatch, tmp_path, _cands(("datetime.time", .99), ("datetime.date", .1)), None, backend="legacy")
    monkeypatch.setattr(config.router, "shadow_mode", True)
    loop = asyncio.new_event_loop()
    import threading
    threading.Thread(target=loop.run_forever, daemon=True).start()
    eng.set_shadow_loop(loop)
    r = eng.classify("what time is it")
    assert r["model"] == "legacy" and "_command" not in r
    loop.call_soon_threadsafe(loop.stop)
