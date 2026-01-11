"""Tests for Memory class and SimStorage.

TigerStyle: Simulation-first testing, determinism verification.
"""

from datetime import datetime, timezone, timedelta
import pytest

from umi.faults import FaultConfig
from umi.memory import Memory
from umi.storage import Entity, SimStorage


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntity:
    """Tests for Entity dataclass."""

    def test_creates_with_defaults(self) -> None:
        """Should create entity with default values."""
        entity = Entity(name="Test", content="Content")

        assert entity.name == "Test"
        assert entity.content == "Content"
        assert entity.entity_type == "note"
        assert entity.id  # Should have generated ID
        assert entity.importance == 0.5

    def test_validates_name(self) -> None:
        """Should reject empty name."""
        with pytest.raises(AssertionError):
            Entity(name="", content="Content")

    def test_validates_importance_range(self) -> None:
        """Should reject invalid importance."""
        with pytest.raises(AssertionError):
            Entity(name="Test", content="Content", importance=1.5)

        with pytest.raises(AssertionError):
            Entity(name="Test", content="Content", importance=-0.1)

    def test_temporal_metadata(self) -> None:
        """Should track temporal metadata."""
        now = datetime.now(timezone.utc)

        entity = Entity(
            name="Test",
            content="Content",
            document_time=now,
            event_time=now - timedelta(days=1),
        )

        assert entity.has_temporal_metadata()
        assert entity.document_time == now
        assert entity.event_time == now - timedelta(days=1)

    def test_no_temporal_metadata(self) -> None:
        """Should report no temporal metadata when none set."""
        entity = Entity(name="Test", content="Content")
        assert not entity.has_temporal_metadata()


# =============================================================================
# SimStorage Tests
# =============================================================================


class TestSimStorage:
    """Tests for SimStorage."""

    @pytest.mark.asyncio
    async def test_store_and_get(self) -> None:
        """Should store and retrieve entity."""
        storage = SimStorage(seed=42)
        entity = Entity(name="Alice", content="My friend")

        stored = await storage.store(entity)
        assert stored.id

        retrieved = await storage.get(stored.id)
        assert retrieved is not None
        assert retrieved.name == "Alice"

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        """Should find entities by text."""
        storage = SimStorage(seed=42)

        await storage.store(Entity(name="Alice", content="Works at Acme"))
        await storage.store(Entity(name="Bob", content="Works at TechCorp"))
        await storage.store(Entity(name="Charlie", content="Works at Acme too"))

        results = await storage.search("Acme", limit=10)
        assert len(results) == 2

        names = [e.name for e in results]
        assert "Alice" in names
        assert "Charlie" in names

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Should delete entity."""
        storage = SimStorage(seed=42)
        entity = Entity(name="Test", content="Content")

        stored = await storage.store(entity)
        assert await storage.get(stored.id) is not None

        deleted = await storage.delete(stored.id)
        assert deleted is True

        assert await storage.get(stored.id) is None

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        """Should count entities."""
        storage = SimStorage(seed=42)

        assert await storage.count() == 0

        await storage.store(Entity(name="A", content="A"))
        await storage.store(Entity(name="B", content="B"))

        assert await storage.count() == 2

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Should clear all entities."""
        storage = SimStorage(seed=42)

        await storage.store(Entity(name="A", content="A"))
        await storage.store(Entity(name="B", content="B"))

        await storage.clear()
        assert await storage.count() == 0

    @pytest.mark.asyncio
    async def test_fault_injection_write(self) -> None:
        """Should inject write faults."""
        faults = FaultConfig(storage_write_error=1.0)
        storage = SimStorage(seed=42, faults=faults)

        with pytest.raises(RuntimeError, match="write error"):
            await storage.store(Entity(name="Test", content="Content"))

    @pytest.mark.asyncio
    async def test_fault_injection_read(self) -> None:
        """Should inject read faults."""
        faults = FaultConfig(storage_read_error=1.0)
        storage = SimStorage(seed=42, faults=faults)

        with pytest.raises(RuntimeError, match="read error"):
            await storage.get("some-id")

    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        """Same seed should produce same behavior."""
        storage1 = SimStorage(seed=42)
        storage2 = SimStorage(seed=42)

        entity1 = Entity(name="Test", content="Content")
        entity2 = Entity(name="Test", content="Content")

        # IDs are UUIDs so will differ, but storage behavior is deterministic
        await storage1.store(entity1)
        await storage2.store(entity2)

        # Both should have same count
        assert await storage1.count() == await storage2.count()


# =============================================================================
# Memory Determinism Tests
# =============================================================================


class TestMemoryDeterminism:
    """Tests for deterministic behavior of Memory class."""

    @pytest.mark.asyncio
    async def test_same_seed_same_entities(self) -> None:
        """Same seed should extract same entities."""
        memory1 = Memory(seed=42)
        memory2 = Memory(seed=42)

        text = "I met Alice at Acme Corp"

        entities1 = await memory1.remember(text)
        entities2 = await memory2.remember(text)

        # Same number of entities
        assert len(entities1) == len(entities2)

        # Same names extracted
        names1 = {e.name for e in entities1}
        names2 = {e.name for e in entities2}
        assert names1 == names2

    @pytest.mark.asyncio
    async def test_reproducibility_with_reset(self) -> None:
        """Should reproduce exact sequence after reset."""
        memory = Memory(seed=42)

        # First run
        entities1 = await memory.remember("Alice works at Acme")
        count1 = await memory.count()

        # Reset
        memory.reset()

        # Second run
        entities2 = await memory.remember("Alice works at Acme")
        count2 = await memory.count()

        # Should be identical
        assert count1 == count2
        names1 = {e.name for e in entities1}
        names2 = {e.name for e in entities2}
        assert names1 == names2


# =============================================================================
# Memory Remember Tests
# =============================================================================


class TestMemoryRemember:
    """Tests for Memory.remember()."""

    @pytest.mark.asyncio
    async def test_extracts_entities(self) -> None:
        """Should extract entities from text."""
        memory = Memory(seed=42)

        entities = await memory.remember("I met Alice at Acme Corp")

        assert len(entities) >= 1

        # Should extract known names
        names = {e.name for e in entities}
        assert "Alice" in names or "Acme" in names

    @pytest.mark.asyncio
    async def test_stores_with_importance(self) -> None:
        """Should store with specified importance."""
        memory = Memory(seed=42)

        entities = await memory.remember("Important note", importance=0.9)

        assert all(e.importance == 0.9 for e in entities)

    @pytest.mark.asyncio
    async def test_stores_with_temporal_metadata(self) -> None:
        """Should store with temporal metadata."""
        memory = Memory(seed=42)
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=7)

        entities = await memory.remember(
            "Meeting happened last week",
            document_time=now,
            event_time=past,
        )

        assert all(e.document_time == now for e in entities)
        assert all(e.event_time == past for e in entities)

    @pytest.mark.asyncio
    async def test_without_extraction(self) -> None:
        """Should store without entity extraction."""
        memory = Memory(seed=42)

        entities = await memory.remember(
            "Just a note without extraction",
            extract_entities=False,
        )

        assert len(entities) == 1
        assert entities[0].entity_type == "note"

    @pytest.mark.asyncio
    async def test_validates_text(self) -> None:
        """Should reject empty text."""
        memory = Memory(seed=42)

        with pytest.raises(AssertionError):
            await memory.remember("")

    @pytest.mark.asyncio
    async def test_validates_importance(self) -> None:
        """Should reject invalid importance."""
        memory = Memory(seed=42)

        with pytest.raises(AssertionError):
            await memory.remember("Text", importance=1.5)


# =============================================================================
# Memory Recall Tests
# =============================================================================


class TestMemoryRecall:
    """Tests for Memory.recall()."""

    @pytest.mark.asyncio
    async def test_finds_stored_entities(self) -> None:
        """Should find previously stored entities."""
        memory = Memory(seed=42)

        await memory.remember("Alice works at Acme Corp")
        await memory.remember("Bob works at TechCorp")

        results = await memory.recall("Acme")

        names = {e.name for e in results}
        # Should find entities related to Acme
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_respects_limit(self) -> None:
        """Should respect result limit."""
        memory = Memory(seed=42)

        # Store many entities
        for i in range(10):
            await memory.remember(f"Note {i} about topic", extract_entities=False)

        results = await memory.recall("topic", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_time_range_filter(self) -> None:
        """Should filter by time range."""
        memory = Memory(seed=42)
        now = datetime.now(timezone.utc)

        # Store entity from last week
        await memory.remember(
            "Old event",
            event_time=now - timedelta(days=7),
            extract_entities=False,
        )

        # Store entity from today
        await memory.remember(
            "New event",
            event_time=now,
            extract_entities=False,
        )

        # Query for recent events only
        results = await memory.recall(
            "event",
            time_range=(now - timedelta(days=1), now + timedelta(days=1)),
        )

        # Should only find the new event
        for entity in results:
            if entity.event_time:
                assert entity.event_time >= now - timedelta(days=1)

    @pytest.mark.asyncio
    async def test_validates_query(self) -> None:
        """Should reject empty query."""
        memory = Memory(seed=42)

        with pytest.raises(AssertionError):
            await memory.recall("")

    @pytest.mark.asyncio
    async def test_validates_limit(self) -> None:
        """Should reject invalid limit."""
        memory = Memory(seed=42)

        with pytest.raises(AssertionError):
            await memory.recall("query", limit=0)

        with pytest.raises(AssertionError):
            await memory.recall("query", limit=1000)


# =============================================================================
# Memory Other Operations Tests
# =============================================================================


class TestMemoryOperations:
    """Tests for other Memory operations."""

    @pytest.mark.asyncio
    async def test_forget(self) -> None:
        """Should delete entity by ID."""
        memory = Memory(seed=42)

        entities = await memory.remember("Alice", extract_entities=False)
        entity_id = entities[0].id

        assert await memory.get(entity_id) is not None

        deleted = await memory.forget(entity_id)
        assert deleted is True

        assert await memory.get(entity_id) is None

    @pytest.mark.asyncio
    async def test_get(self) -> None:
        """Should get entity by ID."""
        memory = Memory(seed=42)

        entities = await memory.remember("Test", extract_entities=False)
        entity_id = entities[0].id

        retrieved = await memory.get(entity_id)
        assert retrieved is not None
        assert retrieved.id == entity_id

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        """Should count stored entities."""
        memory = Memory(seed=42)

        assert await memory.count() == 0

        await memory.remember("A", extract_entities=False)
        await memory.remember("B", extract_entities=False)

        assert await memory.count() == 2

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Should clear all entities."""
        memory = Memory(seed=42)

        await memory.remember("A", extract_entities=False)
        await memory.remember("B", extract_entities=False)

        await memory.clear()
        assert await memory.count() == 0


# =============================================================================
# Memory Fault Injection Tests
# =============================================================================


class TestMemoryFaultInjection:
    """Tests for fault injection in Memory."""

    @pytest.mark.asyncio
    async def test_llm_timeout_during_remember(self) -> None:
        """Should return fallback entity on LLM timeout (graceful degradation)."""
        faults = FaultConfig(llm_timeout=1.0)
        memory = Memory(seed=42, faults=faults)

        # EntityExtractor has graceful degradation - returns empty on timeout
        # Memory.remember() creates fallback note entity
        entities = await memory.remember("Text that will timeout")

        # Should return fallback note entity, not raise
        assert len(entities) == 1
        assert entities[0].entity_type == "note"
        assert "Text that will timeout" in entities[0].content

    @pytest.mark.asyncio
    async def test_storage_error_during_remember(self) -> None:
        """Should propagate storage error."""
        faults = FaultConfig(storage_write_error=1.0)
        memory = Memory(seed=42, faults=faults)

        # First LLM call succeeds, but storage fails
        # Need to set LLM timeout to 0 so LLM succeeds
        memory._llm.faults = FaultConfig()  # No LLM faults

        with pytest.raises(RuntimeError, match="write error"):
            await memory.remember("Text", extract_entities=False)


# =============================================================================
# Memory Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Integration tests for Memory workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Complete remember/recall workflow."""
        memory = Memory(seed=42)

        # Remember multiple things
        await memory.remember("Alice works at Acme Corp as an engineer")
        await memory.remember("Bob is a data scientist at TechCorp")
        await memory.remember("I had lunch with Alice yesterday")

        # Recall
        results = await memory.recall("Alice")
        assert len(results) >= 1

        # Count
        count = await memory.count()
        assert count >= 3

    @pytest.mark.asyncio
    async def test_provider_sim_default(self) -> None:
        """Should use sim provider by default."""
        memory = Memory(seed=42)

        # Should not fail (using sim provider)
        entities = await memory.remember("Test")
        assert len(entities) >= 1
