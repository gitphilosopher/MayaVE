"""
skills/base.py
Skill boundary. A skill receives a validated CommandIR and returns a
SkillResult of FACTS (data) — not a final sentence. Legacy skills that still
return a tagged string are wrapped by LegacySkillAdapter (result.text is set,
generation is bypassed), so nothing existing has to change.

Registering a NEW skill (one with no legacy `execute(intent, text)` handler)
needs no router change:

    @skill_registry.register("weather.current")
    class WeatherSkill: async def run(self, ir, ctx): ...

PATCH (stabilization pass):
- SkillResult gained `meta: dict`. Verified against the real legacy skill
  contract: skills/system/perform_action.py sets `intent["action"]` for
  Processor.handle()'s on_audio_start animation hook, and several skills
  (llm_service.py's error paths, brain/router/dispatch.py's own exception
  handler) set `intent["_no_history"]` so Processor.handle() doesn't store
  the reply in conversation history. The original LegacySkillAdapter threw
  both away after calling execute() — perform_action would resolve an
  action but the caller could never see it, silently breaking the wave/nod/
  giggle/sigh/shrug/wink animation sync described in docs/architecture.md
  §9.2 and §4's routing table. LegacySkillAdapter.run() now copies both
  known out-of-band intent keys into result.meta after calling execute(),
  so a caller (e.g. a future ir-based Processor) can still apply them.
- SkillRegistry gained `bind_legacy_routes(routes: dict)` and
  `resolve(ir)`. Verified against the real registration mechanism:
  brain/router/dispatch.py's `Router.__init__` already builds the
  authoritative intent -> handler map (`self._routes`), including three
  different shapes (a skill's `execute`, a bound method like `self._greet`,
  and `llm_query`). SkillRegistry must not maintain a second, competing
  map of the same intents — that would drift the moment dispatch.py's
  table changes. `bind_legacy_routes()` wraps each existing handler in
  LegacySkillAdapter (if it isn't already IR-native) and keys the result by
  the SAME legacy intent name dispatch.py uses, so registering a new
  IR-native skill (`skill_registry.register(key)`) can shadow one legacy
  entry without touching dispatch.py, while every other intent keeps using
  Router._routes verbatim through `resolve()`'s fallback.
- Nothing existing is migrated: dispatch.py is unchanged apart from the
  clarify route and effective_text handling (see that module's docstring),
  and Processor.handle() still calls Router.dispatch(), not this registry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# Out-of-band intent keys a legacy skill may set as a side effect of
# execute() that a caller needs after the call returns. Kept as an
# explicit, closed list rather than copying the whole intent dict, so a
# skill can't leak router-internal state (_ir, _command, ...) into
# SkillResult.meta by accident.
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
        self._skills: dict[str, object] = {}

    def register(self, key: str):
        def deco(obj):
            self._skills[key] = obj() if isinstance(obj, type) else obj
            return obj
        return deco

    def get(self, key: str | None):
        return self._skills.get(key) if key else None

    def bind_legacy_routes(self, routes: dict[str, object]) -> None:
        """
        Register every entry of an existing legacy route map (e.g.
        brain/router/dispatch.py's Router._routes) under its own intent
        name, wrapped in LegacySkillAdapter, UNLESS that key was already
        registered natively via @register(). Never overwrites a native
        registration — legacy wins only where nothing IR-native exists yet.
        Idempotent: calling this again after Router._routes changes just
        re-wraps whatever is new; already-bound entries are left alone so a
        native registration made in between is never clobbered.
        """
        for legacy_intent, handler in routes.items():
            if legacy_intent in self._skills:
                continue   # a native IR skill already owns this key
            self._skills[legacy_intent] = LegacySkillAdapter(handler)

    def resolve(self, ir):
        """Look up a skill for `ir` by its legacy_intent (the only stable
        cross-reference an IR carries back to the existing route table —
        see brain/router/ir.py). Returns None if nothing is bound."""
        return self.get(getattr(ir, "legacy_intent", None))


class LegacySkillAdapter:
    """Wrap `async execute(intent, text) -> str` in the SkillResult interface."""
    def __init__(self, execute):
        self._execute = execute

    async def run(self, ir, ctx=None) -> SkillResult:
        from brain.router.ir import to_legacy_intent
        intent = to_legacy_intent(ir)
        try:
            text = await self._execute(intent, ir.effective_text or ir.raw_text)
            meta = {k: intent[k] for k in _LEGACY_META_KEYS if k in intent}
            return SkillResult(text=text, meta=meta)
        except Exception as e:  # skills must not crash the turn
            return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")


skill_registry = SkillRegistry()
