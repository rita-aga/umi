"""Tests for EvolutionTracker.

TigerStyle: Simulation-first testing, determinism verification.
"""

import pytest

from umi.evolution import (
    EvolutionTracker,
    EvolutionRelation,
    EVOLUTION_TYPES,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
)
from umi.faults import FaultConfig
from umi.providers.sim import SimLLMProvider
from umi.storage import Entity, SimStorage


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def llm() -> SimLLMProvider:
    """Create a SimLLMProvider with seed."""
    return SimLLMProvider(seed=42)


@pytest.fixture
def storage() -> SimStorage:
    """Create a SimStorage with seed."""
    return SimStorage(seed=42)


@pytest.fixture
def tracker(llm: SimLLMProvider, storage: SimStorage) -> EvolutionTracker:
    """Create an EvolutionTracker with seeded components."""
    return EvolutionTracker(llm=llm, storage=storage, seed=42)


@pytest.fixture
def sample_entity() -> Entity:
    """Create a sample entity."""
    return Entity(
        name="Alice",
        content="Alice works at Acme Corp as an engineer",
        entity_type="person",
    )


@pytest.fixture
def sample_existing_entities(storage: SimStorage) -> list[Entity]:
    """Create sample existing entities."""
    entities = [
        Entity(
            name="Alice",
            content="Alice is a software developer",
            entity_type="person",
        ),
        Entity(
            name="Alice",
            content="Alice graduated from MIT",
            entity_type="person",
        ),
    ]
    return entities


# =============================================================================
# EvolutionRelation Tests
# =============================================================================


class TestEvolutionRelation:
    """Tests for EvolutionRelation dataclass."""

    def test_creates_valid_relation(self) -> None:
        """Should create relation with valid fields."""
        relation = EvolutionRelation(
            source_id="entity-123",
            target_id="entity-456",
            evolution_type="update",
            reason="Job change supersedes previous",
            confidence=0.85,
        )

        assert relation.source_id == "entity-123"
        assert relation.target_id == "entity-456"
        assert relation.evolution_type == "update"
        assert relation.reason == "Job change supersedes previous"
        assert relation.confidence == 0.85

    def test_default_confidence(self) -> None:
        """Should use default confidence when not specified."""
        relation = EvolutionRelation(
            source_id="entity-123",
            target_id="entity-456",
            evolution_type="extend",
            reason="Additional detail",
        )

        assert relation.confidence == 0.5

    def test_validates_empty_source_id(self) -> None:
        """Should reject empty source_id."""
        with pytest.raises(AssertionError):
            EvolutionRelation(
                source_id="",
                target_id="entity-456",
                evolution_type="update",
                reason="Test",
            )

    def test_validates_empty_target_id(self) -> None:
        """Should reject empty target_id."""
        with pytest.raises(AssertionError):
            EvolutionRelation(
                source_id="entity-123",
                target_id="",
                evolution_type="update",
                reason="Test",
            )

    def test_validates_evolution_type(self) -> None:
        """Should reject invalid evolution type."""
        with pytest.raises(AssertionError):
            EvolutionRelation(
                source_id="entity-123",
                target_id="entity-456",
                evolution_type="invalid",
                reason="Test",
            )

    def test_validates_confidence_range(self) -> None:
        """Should reject confidence outside 0-1."""
        with pytest.raises(AssertionError):
            EvolutionRelation(
                source_id="entity-123",
                target_id="entity-456",
                evolution_type="update",
                reason="Test",
                confidence=1.5,
            )

        with pytest.raises(AssertionError):
            EvolutionRelation(
                source_id="entity-123",
                target_id="entity-456",
                evolution_type="update",
                reason="Test",
                confidence=-0.1,
            )

    def test_all_evolution_types_valid(self) -> None:
        """Should accept all valid evolution types."""
        for evolution_type in EVOLUTION_TYPES:
            relation = EvolutionRelation(
                source_id="entity-123",
                target_id="entity-456",
                evolution_type=evolution_type,
                reason="Test reason",
            )
            assert relation.evolution_type == evolution_type


# =============================================================================
# EvolutionTracker Basic Tests
# =============================================================================


class TestEvolutionTrackerBasic:
    """Basic tests for EvolutionTracker."""

    @pytest.mark.asyncio
    async def test_detect_returns_relation_or_none(
        self,
        tracker: EvolutionTracker,
        sample_entity: Entity,
        sample_existing_entities: list[Entity],
    ) -> None:
        """Should return EvolutionRelation or None."""
        # Store the sample entity first to give it an ID
        stored = await tracker.storage.store(sample_entity)

        result = await tracker.detect(stored, sample_existing_entities)

        # Result is either EvolutionRelation or None
        assert result is None or isinstance(result, EvolutionRelation)

    @pytest.mark.asyncio
    async def test_detect_with_no_existing_returns_none(
        self,
        tracker: EvolutionTracker,
        sample_entity: Entity,
    ) -> None:
        """Should return None when no existing entities."""
        stored = await tracker.storage.store(sample_entity)

        result = await tracker.detect(stored, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_detect_validates_new_entity_not_none(
        self,
        tracker: EvolutionTracker,
        sample_existing_entities: list[Entity],
    ) -> None:
        """Should reject None as entity."""
        with pytest.raises(AssertionError):
            await tracker.detect(None, sample_existing_entities)  # type: ignore

    @pytest.mark.asyncio
    async def test_detect_validates_existing_is_list(
        self,
        tracker: EvolutionTracker,
        sample_entity: Entity,
    ) -> None:
        """Should reject non-list existing_entities."""
        stored = await tracker.storage.store(sample_entity)

        with pytest.raises(AssertionError):
            await tracker.detect(stored, "not a list")  # type: ignore


# =============================================================================
# EvolutionTracker Confidence Tests
# =============================================================================


class TestConfidenceFiltering:
    """Tests for confidence filtering."""

    @pytest.mark.asyncio
    async def test_filters_by_min_confidence(
        self,
        tracker: EvolutionTracker,
    ) -> None:
        """Should filter results below min_confidence."""
        # Create entities that might have low confidence
        new_entity = Entity(
            name="Generic Note",
            content="Some random note",
            entity_type="note",
        )
        stored = await tracker.storage.store(new_entity)

        existing = [
            Entity(
                name="Other Note",
                content="Unrelated content",
                entity_type="note",
            )
        ]

        # With very high threshold, should return None
        result = await tracker.detect(stored, existing, min_confidence=0.99)

        # Either None or has high confidence
        if result is not None:
            assert result.confidence >= 0.99

    @pytest.mark.asyncio
    async def test_validates_min_confidence_range(
        self,
        tracker: EvolutionTracker,
        sample_entity: Entity,
        sample_existing_entities: list[Entity],
    ) -> None:
        """Should reject invalid min_confidence."""
        stored = await tracker.storage.store(sample_entity)

        with pytest.raises(AssertionError):
            await tracker.detect(stored, sample_existing_entities, min_confidence=1.5)

        with pytest.raises(AssertionError):
            await tracker.detect(stored, sample_existing_entities, min_confidence=-0.1)


# =============================================================================
# EvolutionTracker Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_same_seed_same_result(self) -> None:
        """Same seed should produce same detection."""
        llm1 = SimLLMProvider(seed=42)
        llm2 = SimLLMProvider(seed=42)
        storage1 = SimStorage(seed=42)
        storage2 = SimStorage(seed=42)

        tracker1 = EvolutionTracker(llm=llm1, storage=storage1, seed=42)
        tracker2 = EvolutionTracker(llm=llm2, storage=storage2, seed=42)

        entity1 = Entity(name="Alice", content="Alice works at Acme", entity_type="person")
        entity2 = Entity(name="Alice", content="Alice works at Acme", entity_type="person")

        stored1 = await storage1.store(entity1)
        stored2 = await storage2.store(entity2)

        existing1 = [Entity(name="Alice", content="Alice is a developer", entity_type="person")]
        existing2 = [Entity(name="Alice", content="Alice is a developer", entity_type="person")]

        result1 = await tracker1.detect(stored1, existing1, min_confidence=0.0)
        result2 = await tracker2.detect(stored2, existing2, min_confidence=0.0)

        # Both should have same result type
        if result1 is None:
            assert result2 is None
        else:
            assert result2 is not None
            assert result1.evolution_type == result2.evolution_type

    @pytest.mark.asyncio
    async def test_reset_restores_state(self) -> None:
        """Reset should restore to initial state."""
        llm = SimLLMProvider(seed=42)
        storage = SimStorage(seed=42)
        tracker = EvolutionTracker(llm=llm, storage=storage, seed=42)

        entity = Entity(name="Bob", content="Bob knows Alice", entity_type="person")
        stored = await storage.store(entity)
        existing = [Entity(name="Bob", content="Bob is a friend", entity_type="person")]

        # First detection
        result1 = await tracker.detect(stored, existing, min_confidence=0.0)

        # Reset all
        tracker.reset()
        llm.reset()
        storage.reset()

        # Re-store and detect
        stored2 = await storage.store(entity)
        result2 = await tracker.detect(stored2, existing, min_confidence=0.0)

        # Should have same type
        if result1 is None:
            assert result2 is None
        else:
            assert result2 is not None
            assert result1.evolution_type == result2.evolution_type


# =============================================================================
# EvolutionTracker Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation on errors."""

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_timeout(self) -> None:
        """Should return None on LLM timeout."""
        faults = FaultConfig(llm_timeout=1.0)  # Always timeout
        llm = SimLLMProvider(seed=42, faults=faults)
        storage = SimStorage(seed=42)
        tracker = EvolutionTracker(llm=llm, storage=storage, seed=42)

        entity = Entity(name="Test", content="Test content", entity_type="note")
        stored = await storage.store(entity)
        existing = [Entity(name="Test", content="Old content", entity_type="note")]

        result = await tracker.detect(stored, existing)

        # Should return None, not raise
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self) -> None:
        """Should return None on LLM error."""
        faults = FaultConfig(llm_error=1.0)  # Always error
        llm = SimLLMProvider(seed=42, faults=faults)
        storage = SimStorage(seed=42)
        tracker = EvolutionTracker(llm=llm, storage=storage, seed=42)

        entity = Entity(name="Test", content="Test content", entity_type="note")
        stored = await storage.store(entity)
        existing = [Entity(name="Test", content="Old content", entity_type="note")]

        result = await tracker.detect(stored, existing)

        assert result is None


# =============================================================================
# EvolutionTracker Convenience Method Tests
# =============================================================================


class TestFindRelatedAndDetect:
    """Tests for find_related_and_detect convenience method."""

    @pytest.mark.asyncio
    async def test_finds_related_and_detects(
        self,
        tracker: EvolutionTracker,
    ) -> None:
        """Should search storage and detect evolution."""
        # Store some existing entities
        existing1 = Entity(name="Alice", content="Alice works at Acme", entity_type="person")
        existing2 = Entity(name="Alice", content="Alice is an engineer", entity_type="person")
        await tracker.storage.store(existing1)
        await tracker.storage.store(existing2)

        # Create and store new entity
        new_entity = Entity(
            name="Alice",
            content="Alice left Acme and joined StartupX",
            entity_type="person",
        )
        stored = await tracker.storage.store(new_entity)

        # Find related and detect
        result = await tracker.find_related_and_detect(stored, min_confidence=0.0)

        # Result should be None or EvolutionRelation
        assert result is None or isinstance(result, EvolutionRelation)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_related(
        self,
        tracker: EvolutionTracker,
    ) -> None:
        """Should return None when no related entities found."""
        # Create entity with unique name
        new_entity = Entity(
            name="UniqueNameXYZ123",
            content="Unique content",
            entity_type="note",
        )
        stored = await tracker.storage.store(new_entity)

        result = await tracker.find_related_and_detect(stored)

        assert result is None

    @pytest.mark.asyncio
    async def test_validates_search_limit(
        self,
        tracker: EvolutionTracker,
    ) -> None:
        """Should reject invalid search_limit."""
        entity = Entity(name="Test", content="Test", entity_type="note")
        stored = await tracker.storage.store(entity)

        with pytest.raises(AssertionError):
            await tracker.find_related_and_detect(stored, search_limit=0)

        with pytest.raises(AssertionError):
            await tracker.find_related_and_detect(stored, search_limit=-1)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for EvolutionTracker."""

    @pytest.mark.asyncio
    async def test_full_evolution_workflow(self) -> None:
        """Complete evolution detection workflow."""
        llm = SimLLMProvider(seed=42)
        storage = SimStorage(seed=42)
        tracker = EvolutionTracker(llm=llm, storage=storage, seed=42)

        # Store initial memory
        initial = Entity(
            name="Alice",
            content="Alice works at Acme Corp as a junior developer",
            entity_type="person",
        )
        await storage.store(initial)

        # Create updated memory
        updated = Entity(
            name="Alice",
            content="Alice was promoted to senior developer at Acme Corp",
            entity_type="person",
        )
        stored_updated = await storage.store(updated)

        # Detect evolution
        evolution = await tracker.find_related_and_detect(
            stored_updated,
            min_confidence=0.0,
        )

        # Should detect some relationship (update or extend)
        if evolution is not None:
            assert evolution.evolution_type in EVOLUTION_TYPES
            assert evolution.target_id == stored_updated.id
            assert CONFIDENCE_MIN <= evolution.confidence <= CONFIDENCE_MAX

    @pytest.mark.asyncio
    async def test_evolution_types_from_sim(self) -> None:
        """SimLLMProvider should generate valid evolution types."""
        llm = SimLLMProvider(seed=42)
        storage = SimStorage(seed=42)
        tracker = EvolutionTracker(llm=llm, storage=storage, seed=42)

        # Test multiple scenarios
        for i in range(5):
            entity = Entity(
                name=f"Person{i}",
                content=f"Person {i} works at Company {i}",
                entity_type="person",
            )
            stored = await storage.store(entity)

            existing = [
                Entity(
                    name=f"Person{i}",
                    content=f"Person {i} is an employee",
                    entity_type="person",
                )
            ]

            result = await tracker.detect(stored, existing, min_confidence=0.0)

            if result is not None:
                # Should have valid evolution type
                assert result.evolution_type in EVOLUTION_TYPES
                assert CONFIDENCE_MIN <= result.confidence <= CONFIDENCE_MAX
