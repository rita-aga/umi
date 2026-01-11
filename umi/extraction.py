"""EntityExtractor - LLM-powered entity and relation extraction (ADR-010).

TigerStyle: Sim-first, deterministic, graceful degradation.

Extracts structured entities and relations from text using LLM:
- Named entities (people, orgs, projects, topics)
- Relations between entities
- Confidence scores for filtering

Example:
    >>> extractor = EntityExtractor(llm, seed=42)
    >>> result = await extractor.extract("I met Alice at Acme Corp")
    >>> # result.entities: [Entity(name="Alice", type="person"), ...]
    >>> # result.relations: [Relation(source="Alice", target="Acme", ...)]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umi.providers.base import LLMProvider


# =============================================================================
# Constants (TigerStyle: explicit limits)
# =============================================================================

TEXT_BYTES_MAX = 100_000  # 100KB max input text
ENTITIES_MAX = 50  # Maximum entities per extraction
RELATIONS_MAX = 100  # Maximum relations per extraction
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0
CONFIDENCE_DEFAULT = 0.5

# Valid entity types
ENTITY_TYPES = frozenset([
    "person",      # People mentioned
    "org",         # Organizations, companies
    "project",     # Projects, initiatives
    "topic",       # Topics, concepts
    "preference",  # User preferences
    "task",        # Tasks, action items
    "event",       # Events, meetings
    "note",        # Fallback for unstructured
])

# Valid relation types
RELATION_TYPES = frozenset([
    "works_at",    # Person works at Org
    "knows",       # Person knows Person
    "manages",     # Person manages Project
    "relates_to",  # Generic relation
    "prefers",     # User prefers something
    "part_of",     # Entity is part of another
])

# Extraction prompt template
EXTRACTION_PROMPT = """Extract entities and relationships from this text.

Text: {text}

{context_section}

Return JSON with this exact structure:
{{
  "entities": [
    {{"name": "entity name", "type": "person|org|project|topic|preference|task|event", "content": "brief description", "confidence": 0.0-1.0}}
  ],
  "relations": [
    {{"source": "entity1 name", "target": "entity2 name", "type": "works_at|knows|manages|relates_to|prefers|part_of", "confidence": 0.0-1.0}}
  ]
}}

Rules:
- Only extract clear, factual entities
- Use confidence 0.9+ for explicit mentions, 0.5-0.8 for inferred
- Skip uncertain entities
- Return empty arrays if no entities found

Only return the JSON, nothing else."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class ExtractedEntity:
    """Entity extracted from text.

    Attributes:
        name: Entity name/identifier.
        entity_type: Type of entity (person, org, project, etc.).
        content: Brief description or context.
        confidence: Extraction confidence (0.0-1.0).
    """

    name: str
    entity_type: str
    content: str
    confidence: float = CONFIDENCE_DEFAULT

    def __post_init__(self) -> None:
        """Validate entity fields."""
        # TigerStyle: Preconditions
        assert self.name, "name must not be empty"
        assert self.entity_type in ENTITY_TYPES, f"invalid entity_type: {self.entity_type}"
        assert CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX, (
            f"confidence must be {CONFIDENCE_MIN}-{CONFIDENCE_MAX}: {self.confidence}"
        )


@dataclass(frozen=True)
class ExtractedRelation:
    """Relation between two entities.

    Attributes:
        source: Source entity name.
        target: Target entity name.
        relation_type: Type of relation.
        confidence: Extraction confidence (0.0-1.0).
    """

    source: str
    target: str
    relation_type: str
    confidence: float = CONFIDENCE_DEFAULT

    def __post_init__(self) -> None:
        """Validate relation fields."""
        # TigerStyle: Preconditions
        assert self.source, "source must not be empty"
        assert self.target, "target must not be empty"
        assert self.relation_type in RELATION_TYPES, f"invalid relation_type: {self.relation_type}"
        assert CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX, (
            f"confidence must be {CONFIDENCE_MIN}-{CONFIDENCE_MAX}: {self.confidence}"
        )


@dataclass
class ExtractionResult:
    """Result of entity extraction.

    Attributes:
        entities: List of extracted entities.
        relations: List of extracted relations.
        raw_text: Original input text.
    """

    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    raw_text: str


# =============================================================================
# EntityExtractor
# =============================================================================


@dataclass
class EntityExtractor:
    """Extract entities and relations from text using LLM.

    Uses LLM to identify named entities and their relationships
    from unstructured text.

    Attributes:
        llm: LLM provider for extraction.
        seed: Optional seed for deterministic behavior.

    Example:
        >>> # Basic usage
        >>> extractor = EntityExtractor(llm, seed=42)
        >>> result = await extractor.extract("Alice works at Acme")
        >>> assert len(result.entities) >= 1

        >>> # With existing context
        >>> result = await extractor.extract(
        ...     "She joined last month",
        ...     existing_entities=["Alice", "Acme Corp"]
        ... )
    """

    llm: "LLMProvider"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialize RNG for deterministic behavior."""
        if self.seed is not None:
            self._rng = Random(self.seed)
        else:
            self._rng = Random()

    async def extract(
        self,
        text: str,
        *,
        existing_entities: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> ExtractionResult:
        """Extract entities and relations from text.

        Args:
            text: Text to extract from.
            existing_entities: Known entity names for context.
            min_confidence: Minimum confidence threshold (0.0-1.0).

        Returns:
            ExtractionResult with entities and relations.

        Raises:
            AssertionError: If preconditions not met.

        Example:
            >>> result = await extractor.extract("I met Alice at Acme Corp")
            >>> assert any(e.name == "Alice" for e in result.entities)
        """
        # TigerStyle: Preconditions
        assert text, "text must not be empty"
        assert len(text) <= TEXT_BYTES_MAX, f"text exceeds {TEXT_BYTES_MAX} bytes"
        assert CONFIDENCE_MIN <= min_confidence <= CONFIDENCE_MAX, (
            f"min_confidence must be {CONFIDENCE_MIN}-{CONFIDENCE_MAX}"
        )

        # Build prompt
        prompt = self._build_prompt(text, existing_entities)

        # Call LLM
        try:
            response = await self.llm.complete(prompt)
            entities, relations = self._parse_response(response, text)
        except (RuntimeError, TimeoutError):
            # Graceful degradation: return empty result
            entities = []
            relations = []

        # Filter by confidence
        if min_confidence > 0.0:
            entities = [e for e in entities if e.confidence >= min_confidence]
            relations = [r for r in relations if r.confidence >= min_confidence]

        # Apply limits
        entities = entities[:ENTITIES_MAX]
        relations = relations[:RELATIONS_MAX]

        result = ExtractionResult(
            entities=entities,
            relations=relations,
            raw_text=text,
        )

        # TigerStyle: Postconditions
        assert isinstance(result.entities, list), "must return entities list"
        assert isinstance(result.relations, list), "must return relations list"
        assert len(result.entities) <= ENTITIES_MAX, "too many entities"
        assert len(result.relations) <= RELATIONS_MAX, "too many relations"

        return result

    async def extract_entities_only(
        self,
        text: str,
        *,
        min_confidence: float = 0.0,
    ) -> list[ExtractedEntity]:
        """Extract only entities (no relations).

        Convenience method for when relations aren't needed.

        Args:
            text: Text to extract from.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of extracted entities.
        """
        result = await self.extract(text, min_confidence=min_confidence)
        return result.entities

    def _build_prompt(
        self,
        text: str,
        existing_entities: list[str] | None,
    ) -> str:
        """Build extraction prompt.

        Args:
            text: Text to extract from.
            existing_entities: Known entity names for context.

        Returns:
            Formatted prompt string.
        """
        if existing_entities:
            context_section = f"Known entities (for context): {', '.join(existing_entities)}"
        else:
            context_section = ""

        return EXTRACTION_PROMPT.format(
            text=text,
            context_section=context_section,
        )

    def _parse_response(
        self,
        response: str,
        original_text: str,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Parse LLM response into entities and relations.

        Args:
            response: Raw LLM response.
            original_text: Original input text for fallback.

        Returns:
            Tuple of (entities, relations).
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: create note entity
            return self._create_fallback_entity(original_text), []

        entities = self._parse_entities(data, original_text)
        relations = self._parse_relations(data)

        return entities, relations

    def _parse_entities(
        self,
        data: dict,
        original_text: str,
    ) -> list[ExtractedEntity]:
        """Parse entities from response data.

        Args:
            data: Parsed JSON data.
            original_text: Original text for fallback content.

        Returns:
            List of ExtractedEntity objects.
        """
        raw_entities = data.get("entities", [])

        if not isinstance(raw_entities, list):
            return self._create_fallback_entity(original_text)

        entities = []
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue

            try:
                name = str(raw.get("name", "")).strip()
                if not name:
                    continue

                entity_type = str(raw.get("type", "note")).lower()
                if entity_type not in ENTITY_TYPES:
                    entity_type = "note"

                content = str(raw.get("content", original_text[:200]))

                confidence = raw.get("confidence", CONFIDENCE_DEFAULT)
                if not isinstance(confidence, (int, float)):
                    confidence = CONFIDENCE_DEFAULT
                confidence = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, float(confidence)))

                entity = ExtractedEntity(
                    name=name,
                    entity_type=entity_type,
                    content=content,
                    confidence=confidence,
                )
                entities.append(entity)

            except (AssertionError, ValueError, TypeError):
                # Skip invalid entities
                continue

        # If no valid entities, create fallback
        if not entities:
            return self._create_fallback_entity(original_text)

        return entities

    def _parse_relations(self, data: dict) -> list[ExtractedRelation]:
        """Parse relations from response data.

        Args:
            data: Parsed JSON data.

        Returns:
            List of ExtractedRelation objects.
        """
        raw_relations = data.get("relations", [])

        if not isinstance(raw_relations, list):
            return []

        relations = []
        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue

            try:
                source = str(raw.get("source", "")).strip()
                target = str(raw.get("target", "")).strip()

                if not source or not target:
                    continue

                relation_type = str(raw.get("type", "relates_to")).lower()
                if relation_type not in RELATION_TYPES:
                    relation_type = "relates_to"

                confidence = raw.get("confidence", CONFIDENCE_DEFAULT)
                if not isinstance(confidence, (int, float)):
                    confidence = CONFIDENCE_DEFAULT
                confidence = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, float(confidence)))

                relation = ExtractedRelation(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    confidence=confidence,
                )
                relations.append(relation)

            except (AssertionError, ValueError, TypeError):
                # Skip invalid relations
                continue

        return relations

    def _create_fallback_entity(self, text: str) -> list[ExtractedEntity]:
        """Create fallback note entity when extraction fails.

        Args:
            text: Original text.

        Returns:
            List with single note entity.
        """
        return [
            ExtractedEntity(
                name=f"Note: {text[:50]}",
                entity_type="note",
                content=text[:500],
                confidence=CONFIDENCE_DEFAULT,
            )
        ]

    def reset(self) -> None:
        """Reset RNG to initial seed state."""
        if self.seed is not None:
            self._rng = Random(self.seed)
