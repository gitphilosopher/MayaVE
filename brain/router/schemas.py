"""
brain/router/schemas.py
Shared data shapes passed between the hybrid router's stages.

PATCH (stabilization pass): CommandSpec gained `requires_confirmation`
(default False). understand.py's `_from_spec` and test_ir.py's
`test_confirmation_flag_carried` both reference it; registry.py's
load_specs() does not set it yet (config/command_domains.json has no
"requires_confirmation" key for any operation today), so every spec
loaded from JSON defaults to False until a domain actually needs it
(e.g. system.shutdown/system.restart, which already require confirmation
on the legacy skill side — see skills/system/power.py). Adding the JSON
key is a follow-up, not part of this stabilization pass.

Kept dependency-free (stdlib dataclasses only) so every other module in
this package — including the offline evaluator and the test suite — can
import from here without pulling in embeddings/Ollama/sqlite.

Command-level retrieval
-----------------------
The command corpus holds MANY prototype vectors per logical command
(`domain.operation`). Retrieval therefore works at the COMMAND level, not
the vector level:

    command_score = max similarity over every vector of that command
    ranking       = distinct commands ordered by command_score
    margin        = best command_score - second-best DISTINCT command_score

Two prototypes of the same command never compete with each other.
`aggregate_by_command()` is the single place that rule is implemented;
the vector store calls it and the confidence layer only reads its result.
"""

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CommandSpec:
    """One domain.operation entry from config/command_domains.json."""
    domain: str
    operation: str
    legacy_intent: str
    entities: dict          # name -> {"type": "string", "required": bool}
    target_mode: str        # "raw" | "entity:<name>"
    seeds: tuple[str, ...] = field(default_factory=tuple)
    requires_confirmation: bool = False   # PATCH — see module docstring

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.operation}"


@dataclass(frozen=True)
class CommandCandidate:
    """One DISTINCT command's best match from semantic retrieval.

    `similarity` is the command score (max over the command's prototype
    vectors); `seed_text` is the prototype that produced it."""
    spec: CommandSpec
    similarity: float
    seed_text: str

    @property
    def command_score(self) -> float:
        return self.similarity


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked, command-level retrieval outcome for one utterance.
    `candidates` holds at most one entry per domain.operation, best first."""
    candidates: list[CommandCandidate]

    @property
    def top1(self) -> CommandCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def top2(self) -> CommandCandidate | None:
        return self.candidates[1] if len(self.candidates) > 1 else None

    @property
    def top1_similarity(self) -> float:
        return self.top1.similarity if self.top1 else 0.0

    @property
    def top2_similarity(self) -> float:
        return self.top2.similarity if self.top2 else 0.0

    @property
    def margin(self) -> float:
        """Command-level margin: best command score minus the second-best
        DISTINCT command's score."""
        return self.top1_similarity - self.top2_similarity

    def diagnostics(self) -> dict:
        """Internal metadata for logging/evaluation; not part of the
        legacy intent contract."""
        return {
            "winning_command": self.top1.spec.key if self.top1 else None,
            "winning_score": round(self.top1_similarity, 4),
            "winning_prototype": self.top1.seed_text if self.top1 else None,
            "second_command": self.top2.spec.key if self.top2 else None,
            "second_score": round(self.top2_similarity, 4),
            "margin": round(self.margin, 4),
        }


def aggregate_by_command(
    scored: Iterable[tuple[CommandSpec, float, str]],
    top_k: int | None = None,
) -> RetrievalResult:
    """Collapse per-vector scores into a command-level ranking.

    `scored` yields (spec, similarity, prototype_text), one item per
    stored vector. Every vector of the same domain.operation is reduced
    to its maximum. Ties are broken by command key so results are
    deterministic."""
    best: dict[str, CommandCandidate] = {}
    for spec, sim, prototype in scored:
        cur = best.get(spec.key)
        if cur is None or sim > cur.similarity:
            best[spec.key] = CommandCandidate(spec=spec, similarity=sim, seed_text=prototype)
    ranked = sorted(best.values(), key=lambda c: (-c.similarity, c.spec.key))
    if top_k is not None:
        ranked = ranked[:top_k]
    return RetrievalResult(candidates=ranked)


@dataclass(frozen=True)
class Command:
    """The {domain, operation, entities} shape produced by either
    semantic retrieval (entities empty/best-effort) or the LLM fallback
    (entities populated per the schema), BEFORE validation."""
    domain: str
    operation: str
    entities: dict
    confidence: float
    needs_clarification: bool = False


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    spec: CommandSpec | None = None
    error: str | None = None
