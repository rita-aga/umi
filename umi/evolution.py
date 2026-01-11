"""EvolutionTracker - Memory relationship detection (ADR-011).

TigerStyle: Sim-first, deterministic, graceful degradation.

Detects how memories evolve over time:
- Update: New info replaces old
- Extend: New info adds to old
- Derive: New info concluded from old
- Contradict: New info conflicts with old

Example:
    >>> tracker = EvolutionTracker(llm, storage, seed=42)
    >>> evolution = await tracker.detect(new_entity, existing_entities)
    >>> # evolution.evolution_type: "update", "extend", "derive", or "contradict"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umi.providers.base import LLMProvider
    from umi.storage import Entity, SimStorage


# =============================================================================
# Constants (TigerStyle: explicit limits)
# =============================================================================

EXISTING_ENTITIES_MAX = 10  # Maximum entities to compare against
REASON_LENGTH_MAX = 500  # Maximum reason text length
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0
CONFIDENCE_DEFAULT = 0.5
CONFIDENCE_THRESHOLD_DEFAULT = 0.3  # Minimum confidence to return result

# Valid evolution types
EVOLUTION_TYPES = frozenset([
    "update",      # New info replaces old (e.g., job change)
    "extend",      # New info adds to old (e.g., additional detail)
    "derive",      # New info concluded from old (e.g., inference)
    "contradict",  # New info conflicts with old (e.g., disagreement)
])

# Detection prompt template
DETECTION_PROMPT = """Compare new information with existing memories and determine the relationship.

New information:
{new_content}

Existing memories:
{existing_list}

What is the relationship between the new information and existing memories?
- "update": New info replaces/corrects old (e.g., changed job, moved address)
- "extend": New info adds to old (e.g., more details, clarification)
- "derive": New info is conclusion from old (e.g., inference, deduction)
- "contradict": New info conflicts with old (e.g., disagreement, correction)
- "none": No significant relationship

Return JSON with this exact structure:
{{"type": "update|extend|derive|contradict|none", "reason": "brief explanation", "related_id": "id of most related memory or null", "confidence": 0.0-1.0}}

Only return the JSON, nothing else."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class EvolutionRelation:
    """Detected evolution relationship between memories.

    Attributes:
        source_id: ID of older/related memory.
        target_id: ID of newer memory.
        evolution_type: Type of evolution (update, extend, derive, contradict).
        reason: Brief explanation of the relationship.
        confidence: Detection confidence (0.0-1.0).
    """

    source_id: str
    target_id: str
    evolution_type: str
    reason: str
    confidence: float = CONFIDENCE_DEFAULT

    def __post_init__(self) -> None:
        """Validate evolution relation fields."""
        # TigerStyle: Preconditions
        assert self.source_id, "source_id must not be empty"
        assert self.target_id, "target_id must not be empty"
        assert self.evolution_type in EVOLUTION_TYPES, (
            f"invalid evolution_type: {self.evolution_type}"
        )
        assert CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX, (
            f"confidence must be {CONFIDENCE_MIN}-{CONFIDENCE_MAX}: {self.confidence}"
        )


# =============================================================================
# EvolutionTracker
# =============================================================================


@dataclass
class EvolutionTracker:
    """Track how memories evolve over time.

    Uses LLM to detect relationships between new and existing memories.

    Attributes:
        llm: LLM provider for detection.
        storage: Storage for looking up entities.
        seed: Optional seed for deterministic behavior.

    Example:
        >>> # Basic usage
        >>> tracker = EvolutionTracker(llm, storage, seed=42)
        >>> evolution = await tracker.detect(new_entity, existing)
        >>> if evolution:
        ...     print(f"Evolution: {evolution.evolution_type}")

        >>> # With confidence filtering
        >>> evolution = await tracker.detect(
        ...     new_entity, existing, min_confidence=0.5
        ... )
    """

    llm: "LLMProvider"
    storage: "SimStorage"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialize RNG for deterministic behavior."""
        if self.seed is not None:
            self._rng = Random(self.seed)
        else:
            self._rng = Random()

    async def detect(
        self,
        new_entity: "Entity",
        existing_entities: list["Entity"],
        *,
        min_confidence: float = CONFIDENCE_THRESHOLD_DEFAULT,
    ) -> EvolutionRelation | None:
        """Detect evolution relationship between new and existing entities.

        Args:
            new_entity: Newly created entity.
            existing_entities: Related existing entities to compare against.
            min_confidence: Minimum confidence threshold to return result.

        Returns:
            EvolutionRelation if detected above threshold, None otherwise.

        Raises:
            AssertionError: If preconditions not met.

        Example:
            >>> evolution = await tracker.detect(new_entity, existing)
            >>> if evolution and evolution.evolution_type == "contradict":
            ...     print(f"Contradiction detected: {evolution.reason}")
        """
        # TigerStyle: Preconditions
        assert new_entity is not None, "new_entity must not be None"
        assert new_entity.id, "new_entity must have id"
        assert isinstance(existing_entities, list), "existing_entities must be list"
        assert CONFIDENCE_MIN <= min_confidence <= CONFIDENCE_MAX, (
            f"min_confidence must be {CONFIDENCE_MIN}-{CONFIDENCE_MAX}"
        )

        # Nothing to compare against
        if not existing_entities:
            return None

        # Limit number of entities to compare
        existing_limited = existing_entities[:EXISTING_ENTITIES_MAX]

        # Build prompt
        prompt = self._build_prompt(new_entity, existing_limited)

        # Call LLM
        try:
            response = await self.llm.complete(prompt)
            result = self._parse_response(response, new_entity.id)
        except (RuntimeError, TimeoutError):
            # Graceful degradation: return None on failure
            return None

        # Apply confidence threshold
        if result is None or result.confidence < min_confidence:
            return None

        # TigerStyle: Postcondition
        assert result.evolution_type in EVOLUTION_TYPES, "invalid evolution type"
        assert CONFIDENCE_MIN <= result.confidence <= CONFIDENCE_MAX, "invalid confidence"

        return result

    async def find_related_and_detect(
        self,
        new_entity: "Entity",
        *,
        search_limit: int = 5,
        min_confidence: float = CONFIDENCE_THRESHOLD_DEFAULT,
    ) -> EvolutionRelation | None:
        """Find related entities and detect evolution in one call.

        Convenience method that searches storage for related entities
        before detecting evolution.

        Args:
            new_entity: Newly created entity.
            search_limit: Maximum entities to search for.
            min_confidence: Minimum confidence threshold.

        Returns:
            EvolutionRelation if detected above threshold, None otherwise.
        """
        # TigerStyle: Preconditions
        assert new_entity is not None, "new_entity must not be None"
        assert new_entity.id, "new_entity must have id"
        assert search_limit > 0, f"search_limit must be positive: {search_limit}"

        # Search for related entities by name
        existing = await self.storage.search(new_entity.name, limit=search_limit)

        # Filter out the new entity itself
        existing = [e for e in existing if e.id != new_entity.id]

        if not existing:
            return None

        return await self.detect(new_entity, existing, min_confidence=min_confidence)

    def _build_prompt(
        self,
        new_entity: "Entity",
        existing_entities: list["Entity"],
    ) -> str:
        """Build detection prompt.

        Args:
            new_entity: New entity to compare.
            existing_entities: Existing entities for context.

        Returns:
            Formatted prompt string.
        """
        # Format new entity
        new_content = f"{new_entity.name}: {new_entity.content}"

        # Format existing entities
        existing_lines = []
        for entity in existing_entities:
            content_preview = entity.content[:200] if entity.content else ""
            existing_lines.append(f"[{entity.id}] {entity.name}: {content_preview}")

        existing_list = "\n".join(existing_lines)

        return DETECTION_PROMPT.format(
            new_content=new_content,
            existing_list=existing_list,
        )

    def _parse_response(
        self,
        response: str,
        new_entity_id: str,
    ) -> EvolutionRelation | None:
        """Parse LLM response into EvolutionRelation.

        Args:
            response: Raw LLM response.
            new_entity_id: ID of the new entity.

        Returns:
            EvolutionRelation if valid, None otherwise.
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None

        # Extract fields
        evolution_type = str(data.get("type", "none")).lower()

        # "none" means no relationship detected
        if evolution_type == "none":
            return None

        # Validate evolution type
        if evolution_type not in EVOLUTION_TYPES:
            return None

        # Get related entity ID
        related_id = data.get("related_id")
        if not related_id:
            return None

        # Get reason
        reason = str(data.get("reason", ""))[:REASON_LENGTH_MAX]

        # Get confidence
        confidence = data.get("confidence", CONFIDENCE_DEFAULT)
        if not isinstance(confidence, (int, float)):
            confidence = CONFIDENCE_DEFAULT
        confidence = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, float(confidence)))

        try:
            return EvolutionRelation(
                source_id=str(related_id),
                target_id=new_entity_id,
                evolution_type=evolution_type,
                reason=reason,
                confidence=confidence,
            )
        except AssertionError:
            return None

    def reset(self) -> None:
        """Reset RNG to initial seed state."""
        if self.seed is not None:
            self._rng = Random(self.seed)
