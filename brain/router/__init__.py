"""
brain/router/
==============
Hybrid semantic command router — an optional, drop-in-compatible
replacement for brain/intent_engine.py::IntentEngine.classify().

See docs/CONTRIBUTING.md and this package's module docstrings for the
migration rationale. The single public entry point most callers need is
HybridIntentEngine (brain/router/hybrid_engine.py), which:

  - wraps a real IntentEngine instance (never replaces it — guards and
    the conversational/LLM path are delegated to it unchanged),
  - is a no-op unless config.router.backend == "hybrid" (or "shadow"),
  - returns the exact same dict shape IntentEngine.classify() returns,
    plus an additive "_command" key.

Nothing in this package is imported by core/processor.py or
brain/router.py today — wiring HybridIntentEngine in as a replacement
for IntentEngine() in core/processor.py is a deliberate later step, not
part of this migration's Stage 1-4 scaffolding.
"""

from brain.router.hybrid_engine import HybridIntentEngine

__all__ = ["HybridIntentEngine"]