"""
skills/base.py
Skill boundary. A skill receives a validated CommandIR and returns a
SkillResult of FACTS (data) — not a final sentence. Legacy skills that still
return a tagged string are wrapped by LegacySkillAdapter (result.text is set,
generation is bypassed), so nothing existing has to change.

Source of truth for routing (BATCH 2)
-------------------------------------
brain/router/dispatch.py's Router._routes is the ONLY intent -> legacy-handler
map. SkillRegistry keeps just native (IR-native) registrations; for legacy
intents it holds a LIVE REFERENCE to Router._routes (bind_legacy_routes) and
wraps the handler at lookup time. Previously bind_legacy_routes() copied and
wrapped each handler once, so a later change to Router._routes was invisible
to (and could contradict) the registry.

Execution gate (BATCH 2)
------------------------
resolve(ir) returns None unless ir.executable (READY, no missing entities, has
a legacy_intent), and LegacySkillAdapter.run() refuses non-executable IRs too.
UNKNOWN / REJECTED / NEEDS_CLARIFICATION can never reach a skill through this
module. A native registration shadows a legacy one inside this registry only;
Router.dispatch does not consult the registry.

SkillResult.meta carries the legacy out-of-band keys a skill sets on the
intent dict: "action" (perform_action's animation) and "_no_history".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

_LEGACY_META_KEYS = ("action", "_no_history")


@dataclass
class SkillResult:
    ok: bool = True
    data: dict = field(default_factory=dict)   # structured facts for response generation
    text: str | None = None                    # pre-rendered tagged reply (legacy / fixed replies)
    error: str | None = None
    meta: dict = field(default_factory=dict)   # out-of-band signals, e.g. {"action": "nod"}

    def as_system_note(self) -> str:
        """Appended to the LLM system prompt so Maya answers from facts in her own voice."""
        if not self.ok:
            return f"\n\nSKILL FAILED: {self.error or 'unknown error'}. Tell senpai briefly and honestly."
        return ("\n\nSKILL RESULT (real data — use it to answer senpai's actual need in your own "
                "words; do not read the fields out): " + json.dumps(self.data, ensure_ascii=False))


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, object] = {}          # native (IR-native) skills only
        self._legacy_routes: dict | None = None       # live reference to Router._routes

    def register(self, key: str):
        def deco(obj):
            self._skills[key] = obj() if isinstance(obj, type) else obj
            return obj
        return deco

    def get(self, key: str | None):
        if not key:
            return None
        native = self._skills.get(key)
        if native is not None:
            return native
        if self._legacy_routes is not None:
            handler = self._legacy_routes.get(key)
            if handler is not None:
                return LegacySkillAdapter(handler)
        return None

    def bind_legacy_routes(self, routes: dict) -> None:
        """Bind (by reference, not copy) the legacy route map, e.g.
        Router._routes. A native registration for the same key still wins."""
        self._legacy_routes = routes

    def resolve(self, ir):
        """Skill for `ir`, or None. Only an executable IR ever resolves."""
        if not getattr(ir, "executable", False):
            return None
        return self.get(getattr(ir, "legacy_intent", None))


class LegacySkillAdapter:
    """Wrap `async execute(intent, text) -> str` in the SkillResult interface."""
    def __init__(self, execute):
        self._execute = execute

    async def run(self, ir, ctx=None) -> SkillResult:
        from brain.router.ir import to_legacy_intent
        if not getattr(ir, "executable", False):
            return SkillResult(ok=False, error=f"ir_not_executable:{getattr(getattr(ir, 'status', None), 'value', 'unknown')}")
        intent = to_legacy_intent(ir)
        try:
            text = await self._execute(intent, ir.effective_text or ir.raw_text)
            meta = {k: intent[k] for k in _LEGACY_META_KEYS if k in intent}
            return SkillResult(text=text, meta=meta)
        except Exception as e:  # skills must not crash the turn
            return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")


skill_registry = SkillRegistry()