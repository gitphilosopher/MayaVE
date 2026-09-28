"""
skills/base.py
Skill boundary. A skill receives a validated CommandIR and returns a
SkillResult of FACTS (data) — not a final sentence. Legacy skills that still
return a tagged string are wrapped by LegacySkillAdapter (result.text is set,
generation is bypassed), so nothing existing has to change.

Registering a skill needs no router change:

    @skill_registry.register("weather.current")
    class WeatherSkill: async def run(self, ir, ctx): ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SkillResult:
    ok: bool = True
    data: dict = field(default_factory=dict)   # structured facts for response generation
    text: str | None = None                    # pre-rendered tagged reply (legacy / fixed replies)
    error: str | None = None

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


class LegacySkillAdapter:
    """Wrap `async execute(intent, text) -> str` in the SkillResult interface."""
    def __init__(self, execute):
        self._execute = execute

    async def run(self, ir, ctx=None) -> SkillResult:
        from brain.router.ir import to_legacy_intent
        try:
            return SkillResult(text=await self._execute(to_legacy_intent(ir), ir.effective_text or ir.raw_text))
        except Exception as e:  # skills must not crash the turn
            return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")


skill_registry = SkillRegistry()
