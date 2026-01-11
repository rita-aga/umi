"""Memory - Main interface for Umi memory system (ADR-008).

TigerStyle: Sim-first, preconditions/postconditions, explicit limits.

The Memory class orchestrates all components:
- LLM provider (SimLLMProvider or production)
- Storage (SimStorage or production)
- Entity extraction
- Evolution tracking (future)
- Dual retrieval (future)

Example:
    >>> # Simulation mode (deterministic)
    >>> memory = Memory(seed=42)
    >>> entities = await memory.remember("I met Alice at Acme Corp")
    >>> results = await memory.recall("Who do I know?")

    >>> # Production mode
    >>> memory = Memory(provider="anthropic")
    >>> entities = await memory.remember("I met Alice at Acme Corp")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from umi.evolution import EvolutionRelation, EvolutionTracker
from umi.extraction import EntityExtractor
from umi.faults import FaultConfig
from umi.providers.base import LLMProvider
from umi.providers.sim import SimLLMProvider
from umi.retrieval import DualRetriever
from umi.storage import Entity, SimStorage


# =============================================================================
# Constants (TigerStyle: explicit limits)
# =============================================================================

TEXT_BYTES_MAX = 100_000  # 100KB max input text
SEARCH_LIMIT_MAX = 100
IMPORTANCE_MIN = 0.0
IMPORTANCE_MAX = 1.0


# =============================================================================
# Memory Class
# =============================================================================


@dataclass
class Memory:
    """Main interface for Umi memory system.

    Provides simple remember/recall API with full simulation support.

    Attributes:
        seed: If provided, enables simulation mode with deterministic behavior.
        provider: LLM provider instance or name ("sim", "anthropic", "openai").
        faults: Fault injection configuration for simulation.

    Example:
        >>> # Simulation mode
        >>> memory = Memory(seed=42)
        >>> entities = await memory.remember("Alice works at Acme")
        >>> assert len(entities) >= 1

        >>> # With fault injection
        >>> memory = Memory(seed=42, faults=FaultConfig(llm_timeout=0.1))
    """

    seed: int | None = None
    provider: str | LLMProvider = "sim"
    faults: FaultConfig | None = None

    def __post_init__(self) -> None:
        """Initialize components based on mode."""
        # Default faults
        if self.faults is None:
            self.faults = FaultConfig()

        # Initialize LLM provider
        if self.seed is not None:
            # Simulation mode: use SimLLMProvider
            self._llm = SimLLMProvider(seed=self.seed, faults=self.faults)
            self._storage = SimStorage(seed=self.seed, faults=self.faults)
        elif isinstance(self.provider, str):
            # Production mode: create provider by name
            self._llm = self._create_provider(self.provider)
            self._storage = SimStorage(seed=0)  # TODO: real storage in future
        else:
            # Custom provider instance
            self._llm = self.provider
            self._storage = SimStorage(seed=0)

        # Initialize DualRetriever for smart recall
        self._retriever = DualRetriever(
            storage=self._storage,
            llm=self._llm,
            seed=self.seed,
        )

        # Initialize EntityExtractor for smart remember
        self._extractor = EntityExtractor(
            llm=self._llm,
            seed=self.seed,
        )

        # Initialize EvolutionTracker for memory relationships
        self._evolution = EvolutionTracker(
            llm=self._llm,
            storage=self._storage,
            seed=self.seed,
        )

    def _create_provider(self, name: str) -> LLMProvider:
        """Create LLM provider by name.

        Args:
            name: Provider name ("sim", "anthropic", "openai").

        Returns:
            LLMProvider instance.
        """
        if name == "sim":
            return SimLLMProvider(seed=42, faults=self.faults)
        elif name == "anthropic":
            from umi.providers.anthropic import AnthropicProvider
            return AnthropicProvider()
        elif name == "openai":
            from umi.providers.openai import OpenAIProvider
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {name}. Use 'sim', 'anthropic', or 'openai'")

    async def remember(
        self,
        text: str,
        *,
        importance: float = 0.5,
        document_time: datetime | None = None,
        event_time: datetime | None = None,
        extract_entities: bool = True,
        track_evolution: bool = True,
    ) -> list[Entity]:
        """Store information in memory.

        Extracts entities from text using LLM and stores them.
        Optionally detects evolution relationships with existing memories.

        Args:
            text: Text to remember.
            importance: Importance score (0.0-1.0).
            document_time: When source document was created.
            event_time: When the event actually occurred.
            extract_entities: Whether to use LLM for entity extraction.
            track_evolution: Whether to detect evolution with existing memories.

        Returns:
            List of stored entities.

        Raises:
            AssertionError: If preconditions not met.
            TimeoutError: If LLM times out.
            RuntimeError: If LLM or storage fails.

        Example:
            >>> memory = Memory(seed=42)
            >>> entities = await memory.remember("I met Alice at Acme Corp")
            >>> assert any(e.name == "Alice" for e in entities)

            >>> # With evolution tracking
            >>> await memory.remember("Alice works at Acme")
            >>> entities = await memory.remember("Alice left Acme for StartupX")
            >>> # May detect evolution: update relationship
        """
        # TigerStyle: Preconditions
        assert text, "text must not be empty"
        assert len(text) <= TEXT_BYTES_MAX, f"text exceeds {TEXT_BYTES_MAX} bytes"
        assert IMPORTANCE_MIN <= importance <= IMPORTANCE_MAX, (
            f"importance must be {IMPORTANCE_MIN}-{IMPORTANCE_MAX}: {importance}"
        )

        entities: list[Entity] = []

        if extract_entities:
            # Use EntityExtractor for smart extraction
            result = await self._extractor.extract(text)

            for extracted in result.entities:
                entity = Entity(
                    name=extracted.name,
                    content=extracted.content,
                    entity_type=extracted.entity_type,
                    importance=importance,
                    document_time=document_time,
                    event_time=event_time,
                )
                stored = await self._storage.store(entity)

                # Track evolution if enabled
                if track_evolution:
                    evolution = await self._evolution.find_related_and_detect(stored)
                    if evolution:
                        # Store evolution info in metadata
                        stored.metadata["evolution"] = {
                            "type": evolution.evolution_type,
                            "related_id": evolution.source_id,
                            "reason": evolution.reason,
                            "confidence": evolution.confidence,
                        }

                entities.append(stored)

            # If extraction returned nothing, create fallback
            if not entities:
                entity = Entity(
                    name=f"Note: {text[:50]}",
                    content=text,
                    entity_type="note",
                    importance=importance,
                    document_time=document_time,
                    event_time=event_time,
                )
                stored = await self._storage.store(entity)
                entities.append(stored)
        else:
            # No extraction: store as single note entity
            entity = Entity(
                name=f"Note: {text[:50]}",
                content=text,
                entity_type="note",
                importance=importance,
                document_time=document_time,
                event_time=event_time,
            )
            stored = await self._storage.store(entity)
            entities.append(stored)

        # TigerStyle: Postcondition
        assert isinstance(entities, list), "must return list"
        assert len(entities) >= 1, "must store at least one entity"

        return entities

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        deep_search: bool = False,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Entity]:
        """Retrieve memories matching query.

        Uses DualRetriever for smart search:
        - Fast path: Direct substring search
        - Deep path: LLM rewrites query into variations, merges results

        Args:
            query: Search query.
            limit: Maximum results.
            deep_search: Use LLM for enhanced search (default False).
            time_range: Filter by event_time (start, end).

        Returns:
            List of matching entities, sorted by relevance.

        Raises:
            AssertionError: If preconditions not met.
            RuntimeError: If storage fails.

        Example:
            >>> memory = Memory(seed=42)
            >>> await memory.remember("Alice works at Acme Corp")
            >>> results = await memory.recall("Acme")
            >>> assert len(results) >= 1

            >>> # Deep search for complex queries
            >>> results = await memory.recall("Who works at Acme?", deep_search=True)
        """
        # TigerStyle: Preconditions
        assert query, "query must not be empty"
        assert 0 < limit <= SEARCH_LIMIT_MAX, f"limit must be 1-{SEARCH_LIMIT_MAX}: {limit}"

        # Use DualRetriever for search
        results = await self._retriever.search(
            query,
            limit=limit,
            deep_search=deep_search,
            time_range=time_range,
        )

        # TigerStyle: Postcondition
        assert isinstance(results, list), "must return list"
        assert len(results) <= limit, f"results exceed limit: {len(results)} > {limit}"

        return results

    async def forget(self, entity_id: str) -> bool:
        """Delete an entity from memory.

        Args:
            entity_id: ID of entity to delete.

        Returns:
            True if deleted, False if not found.
        """
        # TigerStyle: Precondition
        assert entity_id, "entity_id must not be empty"

        return await self._storage.delete(entity_id)

    async def get(self, entity_id: str) -> Entity | None:
        """Get entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity if found, None otherwise.
        """
        # TigerStyle: Precondition
        assert entity_id, "entity_id must not be empty"

        return await self._storage.get(entity_id)

    async def count(self) -> int:
        """Count total stored entities.

        Returns:
            Number of entities in storage.
        """
        return await self._storage.count()

    async def clear(self) -> None:
        """Clear all stored entities."""
        await self._storage.clear()

    def reset(self) -> None:
        """Reset memory to initial state.

        Only applicable in simulation mode.
        """
        if hasattr(self._llm, "reset"):
            self._llm.reset()
        if hasattr(self._storage, "reset"):
            self._storage.reset()
        if hasattr(self._retriever, "reset"):
            self._retriever.reset()
        if hasattr(self._extractor, "reset"):
            self._extractor.reset()
        if hasattr(self._evolution, "reset"):
            self._evolution.reset()
