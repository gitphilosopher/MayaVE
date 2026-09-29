"""
brain/router/
==============
Package containing BOTH routers:

  dispatch.py         — the legacy intent -> skill dispatcher (`Router`),
                        formerly the module brain/router.py. Moved here to
                        resolve the module/package name collision: with a
                        package `brain/router/` present, Python resolves
                        `brain.router` to the package and the old module was
                        unreachable.
  hybrid_engine.py    — HybridIntentEngine (semantic + LLM fallback wrapper).
  understand.py       — CommandUnderstander -> CommandIR (guards -> classifier
                        gate -> semantic -> LLM fallback -> validation).

Public names are exported LAZILY (PEP 562) so that importing a light
submodule (validate, entities, context, understand, ir) — e.g. in unit
tests — never drags in kokoro/sounddevice/torch via dispatch.py. Existing
imports keep working:

    from brain.router import Router                 # legacy dispatcher
    from brain.router import HybridIntentEngine
"""

_LAZY = {
    "Router": ("brain.router.dispatch", "Router"),
    "HybridIntentEngine": ("brain.router.hybrid_engine", "HybridIntentEngine"),
    "CommandUnderstander": ("brain.router.understand", "CommandUnderstander"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        # Must raise AttributeError so `from brain.router import <submodule>`
        # falls back to importing the submodule.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(target[0]), target[1])
