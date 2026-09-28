import sys, types, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# brain.intent_engine pulls torch/tensorflow-adjacent code; the router tests only
# need its name, so provide a light stub when the real one can't import.
try:
    import brain.intent_engine  # noqa
except Exception:
    m = types.ModuleType("brain.intent_engine")
    class IntentEngine: ...
    m.IntentEngine = IntentEngine
    sys.modules["brain.intent_engine"] = m
