"""SimStorage - In-memory storage for simulation testing (ADR-008).

TigerStyle: Deterministic, fault-injectable, mirrors Rust SimStorageBackend.

Provides a pure-Python storage implementation for testing without
requiring Rust bindings or external databases.

Example:
    >>> storage = SimStorage(seed=42)
    >>> entity = Entity(name="Alice", content="My friend")
    >>> stored = await storage.store(entity)
    >>> retrieved = await storage.get(stored.id)
    >>> assert retrieved.name == "Alice"
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from random import Random
from typing import Optional

from umi.faults import FaultConfig


# =============================================================================
# Constants (TigerStyle: explicit limits)
# =============================================================================

ENTITY_NAME_BYTES_MAX = 256
ENTITY_CONTENT_BYTES_MAX = 1_000_000  # 1MB
SEARCH_RESULTS_COUNT_MAX = 100
SEARCH_QUERY_BYTES_MAX = 10_000


# =============================================================================
# Entity
# =============================================================================


@dataclass
class Entity:
    """An entity in memory storage.

    Mirrors the Rust Entity struct for Python-side testing.

    Attributes:
        id: Unique identifier (generated if not provided)
        entity_type: Type of entity (person, project, topic, note, task)
        name: Display name
        content: Main content
        metadata: Key-value metadata
        created_at: When entity was created
        updated_at: When entity was last modified
        document_time: When the source document was created
        event_time: When the event described actually occurred
        importance: Importance score (0.0-1.0)
    """

    name: str
    content: str
    entity_type: str = "note"
    id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document_time: Optional[datetime] = None
    event_time: Optional[datetime] = None
    importance: float = 0.5

    def __post_init__(self) -> None:
        """Generate ID if not provided and validate."""
        if not self.id:
            self.id = str(uuid.uuid4())

        # TigerStyle: Preconditions
        assert self.name, "name must not be empty"
        assert len(self.name) <= ENTITY_NAME_BYTES_MAX, (
            f"name exceeds {ENTITY_NAME_BYTES_MAX} bytes"
        )
        assert len(self.content) <= ENTITY_CONTENT_BYTES_MAX, (
            f"content exceeds {ENTITY_CONTENT_BYTES_MAX} bytes"
        )
        assert 0.0 <= self.importance <= 1.0, (
            f"importance must be 0-1: {self.importance}"
        )

    def has_temporal_metadata(self) -> bool:
        """Check if entity has temporal metadata."""
        return self.document_time is not None or self.event_time is not None


# =============================================================================
# SimStorage
# =============================================================================


@dataclass
class SimStorage:
    """In-memory storage backend for simulation testing.

    Deterministic behavior based on seed. Supports fault injection.

    Attributes:
        seed: Seed for deterministic RNG.
        faults: Fault injection configuration.

    Example:
        >>> storage = SimStorage(seed=42)
        >>> entity = Entity(name="Test", content="Content")
        >>> stored = await storage.store(entity)
        >>> assert stored.id
    """

    seed: int
    faults: FaultConfig | None = None

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._entities: dict[str, Entity] = {}
        self._rng = Random(self.seed)
        if self.faults is None:
            self.faults = FaultConfig()

    async def store(self, entity: Entity) -> Entity:
        """Store an entity.

        Args:
            entity: Entity to store.

        Returns:
            Stored entity (with ID).

        Raises:
            RuntimeError: If write fault triggered.
        """
        # TigerStyle: Precondition
        assert entity.name, "entity must have name"

        # Fault injection
        if self.faults and self.faults.should_fail("storage_write_error", self._rng):
            raise RuntimeError("Simulated storage write error")

        # Update timestamp
        entity.updated_at = datetime.now(timezone.utc)

        # Store
        self._entities[entity.id] = entity

        # TigerStyle: Postcondition
        assert entity.id in self._entities, "entity must be stored"

        return entity

    async def get(self, entity_id: str) -> Entity | None:
        """Get entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity if found, None otherwise.

        Raises:
            RuntimeError: If read fault triggered.
        """
        # TigerStyle: Precondition
        assert entity_id, "entity_id must not be empty"

        # Fault injection
        if self.faults and self.faults.should_fail("storage_read_error", self._rng):
            raise RuntimeError("Simulated storage read error")

        return self._entities.get(entity_id)

    async def search(self, query: str, limit: int = 10) -> list[Entity]:
        """Search entities by text.

        Simple substring matching on name and content.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching entities.

        Raises:
            RuntimeError: If read fault triggered.
        """
        # TigerStyle: Preconditions
        assert query, "query must not be empty"
        assert len(query) <= SEARCH_QUERY_BYTES_MAX, (
            f"query exceeds {SEARCH_QUERY_BYTES_MAX} bytes"
        )
        assert 0 < limit <= SEARCH_RESULTS_COUNT_MAX, (
            f"limit must be 1-{SEARCH_RESULTS_COUNT_MAX}: {limit}"
        )

        # Fault injection
        if self.faults and self.faults.should_fail("storage_read_error", self._rng):
            raise RuntimeError("Simulated storage read error")

        query_lower = query.lower()
        results = []

        for entity in self._entities.values():
            if query_lower in entity.name.lower() or query_lower in entity.content.lower():
                results.append(entity)
                if len(results) >= limit:
                    break

        # Sort by importance (descending), then by updated_at (descending)
        results.sort(key=lambda e: (-e.importance, -e.updated_at.timestamp()))

        # TigerStyle: Postcondition
        assert len(results) <= limit, f"results exceed limit: {len(results)} > {limit}"

        return results[:limit]

    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            True if entity was deleted, False if not found.

        Raises:
            RuntimeError: If write fault triggered.
        """
        # TigerStyle: Precondition
        assert entity_id, "entity_id must not be empty"

        # Fault injection
        if self.faults and self.faults.should_fail("storage_write_error", self._rng):
            raise RuntimeError("Simulated storage write error")

        if entity_id in self._entities:
            del self._entities[entity_id]
            # TigerStyle: Postcondition
            assert entity_id not in self._entities, "entity must be deleted"
            return True

        return False

    async def list_all(self, limit: int = 100) -> list[Entity]:
        """List all entities.

        Args:
            limit: Maximum results.

        Returns:
            List of entities.
        """
        # TigerStyle: Precondition
        assert 0 < limit <= SEARCH_RESULTS_COUNT_MAX, (
            f"limit must be 1-{SEARCH_RESULTS_COUNT_MAX}: {limit}"
        )

        # Fault injection
        if self.faults and self.faults.should_fail("storage_read_error", self._rng):
            raise RuntimeError("Simulated storage read error")

        entities = list(self._entities.values())
        entities.sort(key=lambda e: -e.updated_at.timestamp())

        return entities[:limit]

    async def count(self) -> int:
        """Count total entities.

        Returns:
            Number of stored entities.
        """
        return len(self._entities)

    async def clear(self) -> None:
        """Clear all entities."""
        self._entities.clear()

        # TigerStyle: Postcondition
        assert len(self._entities) == 0, "storage must be empty after clear"

    def reset(self) -> None:
        """Reset storage to initial state."""
        self._entities.clear()
        self._rng = Random(self.seed)
