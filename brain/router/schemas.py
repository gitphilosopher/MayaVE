"""
brain/router/schemas.py
Shared data shapes passed between the hybrid router's stages.

Kept dependency-free (stdlib dataclasses only) so every other module in
this package — including the offline evaluator and the test suite — can
import from here without pulling in embeddings/Ollama/sqlite.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandSpec:
    """One domain.operation entry from config/command_domains.json."""
    domain: str
    operation: str
    legacy_intent: str
    entities: dict          # name -> {"type": "string", "required": bool}
    target_mode: str        # "raw" | "entity:<name>"
    seeds: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.operation}"


@dataclass(frozen=True)
class CommandCandidate:
    """One scored match from semantic retrieval."""
    spec: CommandSpec
    similarity: float
    seed_text: str


@dataclass(frozen=True)
class RetrievalResult:
    """Top-k semantic retrieval outcome for one utterance."""
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
        return self.top1_similarity - self.top2_similarity


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


@dataclass(frozen=True)
class RouterDecision:
    """Final structured outcome handed to the adapter, carrying enough
    provenance for the '_command' block and for observability logging."""
    command: Command
    spec: CommandSpec
    source: str   # "guard" | "semantic" | "llm_fallback"
    top1_similarity: float = 0.0
    top2_similarity: float = 0.0
    margin: float = 0.0