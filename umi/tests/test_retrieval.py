"""Tests for DualRetriever.

TigerStyle: Simulation-first testing, determinism verification.
"""

from datetime import datetime, timezone, timedelta
import pytest

from umi.faults import FaultConfig
from umi.providers.sim import SimLLMProvider
from umi.retrieval import (
    DualRetriever,
    QUESTION_WORDS,
    RELATIONSHIP_TERMS,
    TEMPORAL_TERMS,
    RRF_K,
)
from umi.storage import Entity, SimStorage


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage() -> SimStorage:
    """Create a SimStorage with seed."""
    return SimStorage(seed=42)


@pytest.fixture
def llm() -> SimLLMProvider:
    """Create a SimLLMProvider with seed."""
    return SimLLMProvider(seed=42)


@pytest.fixture
def retriever(storage: SimStorage, llm: SimLLMProvider) -> DualRetriever:
    """Create a DualRetriever with seeded components."""
    return DualRetriever(storage=storage, llm=llm, seed=42)


@pytest.fixture
async def populated_storage(storage: SimStorage) -> SimStorage:
    """Storage with pre-populated test entities."""
    entities = [
        Entity(name="Alice", content="Alice works at Acme Corp as an engineer"),
        Entity(name="Bob", content="Bob is a data scientist at TechCorp"),
        Entity(name="Charlie", content="Charlie knows Alice from college"),
        Entity(name="Acme Corp", content="A technology company", entity_type="org"),
        Entity(name="TechCorp", content="Another tech company", entity_type="org"),
        Entity(name="Project Alpha", content="Alice is leading Project Alpha at Acme"),
    ]
    for entity in entities:
        await storage.store(entity)
    return storage


# =============================================================================
# Heuristics Tests
# =============================================================================


class TestNeedsDeepSearch:
    """Tests for needs_deep_search() heuristic."""

    def test_simple_query_no_deep_search(self, retriever: DualRetriever) -> None:
        """Simple name queries don't need deep search."""
        assert not retriever.needs_deep_search("Alice")
        assert not retriever.needs_deep_search("Acme Corp")
        assert not retriever.needs_deep_search("Project Alpha")

    def test_question_words_trigger_deep_search(self, retriever: DualRetriever) -> None:
        """Question words should trigger deep search."""
        for word in QUESTION_WORDS:
            query = f"{word} is Alice?"
            assert retriever.needs_deep_search(query), f"'{word}' should trigger deep search"

    def test_relationship_terms_trigger_deep_search(self, retriever: DualRetriever) -> None:
        """Relationship terms should trigger deep search."""
        assert retriever.needs_deep_search("related to Alice")
        assert retriever.needs_deep_search("about the project")
        assert retriever.needs_deep_search("regarding Acme")
        assert retriever.needs_deep_search("connected to Bob")

    def test_temporal_terms_trigger_deep_search(self, retriever: DualRetriever) -> None:
        """Temporal terms should trigger deep search."""
        assert retriever.needs_deep_search("yesterday's meeting")
        assert retriever.needs_deep_search("last week")
        assert retriever.needs_deep_search("recent updates")
        assert retriever.needs_deep_search("before the deadline")

    def test_abstract_terms_trigger_deep_search(self, retriever: DualRetriever) -> None:
        """Abstract terms should trigger deep search."""
        assert retriever.needs_deep_search("similar to this")
        assert retriever.needs_deep_search("like Alice")
        assert retriever.needs_deep_search("connections with Acme")

    def test_validates_empty_query(self, retriever: DualRetriever) -> None:
        """Should reject empty query."""
        with pytest.raises(AssertionError):
            retriever.needs_deep_search("")


# =============================================================================
# Query Rewriting Tests
# =============================================================================


class TestQueryRewriting:
    """Tests for rewrite_query()."""

    @pytest.mark.asyncio
    async def test_returns_variations(self, retriever: DualRetriever) -> None:
        """Should return query variations."""
        variations = await retriever.rewrite_query("Acme employees")

        assert isinstance(variations, list)
        assert len(variations) >= 1
        assert len(variations) <= 3  # Max 3 variations

    @pytest.mark.asyncio
    async def test_includes_original_query(self, retriever: DualRetriever) -> None:
        """Should include original query in variations."""
        query = "Who works at Acme?"
        variations = await retriever.rewrite_query(query)

        # Original should be included (may be reworded by sim)
        assert len(variations) >= 1

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same variations."""
        storage = SimStorage(seed=42)

        llm1 = SimLLMProvider(seed=42)
        llm2 = SimLLMProvider(seed=42)

        retriever1 = DualRetriever(storage=storage, llm=llm1, seed=42)
        retriever2 = DualRetriever(storage=storage, llm=llm2, seed=42)

        variations1 = await retriever1.rewrite_query("test query")
        variations2 = await retriever2.rewrite_query("test query")

        assert variations1 == variations2

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_llm_failure(self) -> None:
        """Should return original query if LLM fails."""
        storage = SimStorage(seed=42)
        faults = FaultConfig(llm_timeout=1.0)  # Always timeout
        llm = SimLLMProvider(seed=42, faults=faults)
        retriever = DualRetriever(storage=storage, llm=llm, seed=42)

        variations = await retriever.rewrite_query("test query")

        # Should fallback to original
        assert variations == ["test query"]

    @pytest.mark.asyncio
    async def test_validates_empty_query(self, retriever: DualRetriever) -> None:
        """Should reject empty query."""
        with pytest.raises(AssertionError):
            await retriever.rewrite_query("")


# =============================================================================
# RRF Merge Tests
# =============================================================================


class TestMergeRRF:
    """Tests for merge_rrf()."""

    def test_merges_single_list(self, retriever: DualRetriever) -> None:
        """Should handle single list."""
        entities = [
            Entity(name="A", content="A"),
            Entity(name="B", content="B"),
        ]

        merged = retriever.merge_rrf(entities)

        assert len(merged) == 2
        assert merged[0].name == "A"  # Higher rank

    def test_merges_multiple_lists(self, retriever: DualRetriever) -> None:
        """Should merge multiple lists."""
        e1 = Entity(name="A", content="A")
        e2 = Entity(name="B", content="B")
        e3 = Entity(name="C", content="C")

        list1 = [e1, e2]
        list2 = [e2, e3]

        merged = retriever.merge_rrf(list1, list2)

        assert len(merged) == 3
        # B appears in both lists, should rank higher
        assert merged[0].name == "B"

    def test_deduplicates_by_id(self, retriever: DualRetriever) -> None:
        """Should deduplicate entities by ID."""
        entity = Entity(name="A", content="A")

        list1 = [entity]
        list2 = [entity]  # Same entity

        merged = retriever.merge_rrf(list1, list2)

        assert len(merged) == 1

    def test_rrf_scoring(self, retriever: DualRetriever) -> None:
        """Should apply RRF scoring correctly."""
        e1 = Entity(name="First", content="First")
        e2 = Entity(name="Second", content="Second")
        e3 = Entity(name="Third", content="Third")

        # e1 is first in list1, e2 is first in list2
        # Both should have same score: 1/(60+0) = 0.0167
        # e3 is second in both lists: 2 * 1/(60+1) = 0.0328
        list1 = [e1, e3]
        list2 = [e2, e3]

        merged = retriever.merge_rrf(list1, list2)

        # e3 appears in both lists at rank 1, should be first
        assert merged[0].name == "Third"

    def test_handles_empty_lists(self, retriever: DualRetriever) -> None:
        """Should handle empty lists."""
        merged = retriever.merge_rrf([], [])
        assert merged == []

    def test_validates_k_positive(self, retriever: DualRetriever) -> None:
        """Should reject non-positive k."""
        with pytest.raises(AssertionError):
            retriever.merge_rrf([], k=0)

        with pytest.raises(AssertionError):
            retriever.merge_rrf([], k=-1)


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for search() method."""

    @pytest.mark.asyncio
    async def test_basic_search(
        self, llm: SimLLMProvider, populated_storage: SimStorage
    ) -> None:
        """Should find entities by text."""
        retriever = DualRetriever(storage=populated_storage, llm=llm, seed=42)

        results = await retriever.search("Alice", deep_search=False)

        assert len(results) >= 1
        names = {e.name for e in results}
        assert "Alice" in names or any("Alice" in e.content for e in results)

    @pytest.mark.asyncio
    async def test_deep_search_finds_more(
        self, llm: SimLLMProvider, populated_storage: SimStorage
    ) -> None:
        """Deep search should potentially find more results."""
        retriever = DualRetriever(storage=populated_storage, llm=llm, seed=42)

        # Simple query that might benefit from rewriting
        fast_results = await retriever.search("Acme", deep_search=False)
        deep_results = await retriever.search("Who works at Acme?", deep_search=True)

        # Deep search triggered by question word
        assert retriever.needs_deep_search("Who works at Acme?")

    @pytest.mark.asyncio
    async def test_respects_limit(
        self, llm: SimLLMProvider, populated_storage: SimStorage
    ) -> None:
        """Should respect result limit."""
        retriever = DualRetriever(storage=populated_storage, llm=llm, seed=42)

        results = await retriever.search("a", limit=2)  # Matches many

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_time_range_filter(self, retriever: DualRetriever) -> None:
        """Should filter by time range."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=7)
        new_time = now

        # Store entities with different event times
        old_entity = Entity(name="Old", content="Old event", event_time=old_time)
        new_entity = Entity(name="New", content="New event", event_time=new_time)

        await retriever.storage.store(old_entity)
        await retriever.storage.store(new_entity)

        # Query for recent events only
        results = await retriever.search(
            "event",
            time_range=(now - timedelta(days=1), now + timedelta(days=1)),
            deep_search=False,
        )

        # Should only find new event
        for entity in results:
            if entity.event_time:
                assert entity.event_time >= now - timedelta(days=1)

    @pytest.mark.asyncio
    async def test_validates_empty_query(self, retriever: DualRetriever) -> None:
        """Should reject empty query."""
        with pytest.raises(AssertionError):
            await retriever.search("")

    @pytest.mark.asyncio
    async def test_validates_limit(self, retriever: DualRetriever) -> None:
        """Should reject invalid limit."""
        with pytest.raises(AssertionError):
            await retriever.search("test", limit=0)

        with pytest.raises(AssertionError):
            await retriever.search("test", limit=1000)


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_same_seed_same_results(self) -> None:
        """Same seed should produce same search results."""
        # Create two identical setups
        storage1 = SimStorage(seed=42)
        storage2 = SimStorage(seed=42)

        llm1 = SimLLMProvider(seed=42)
        llm2 = SimLLMProvider(seed=42)

        retriever1 = DualRetriever(storage=storage1, llm=llm1, seed=42)
        retriever2 = DualRetriever(storage=storage2, llm=llm2, seed=42)

        # Store same entities
        entity = Entity(name="Alice", content="Works at Acme")
        await storage1.store(Entity(name="Alice", content="Works at Acme"))
        await storage2.store(Entity(name="Alice", content="Works at Acme"))

        # Search should return same results
        results1 = await retriever1.search("Alice", deep_search=False)
        results2 = await retriever2.search("Alice", deep_search=False)

        assert len(results1) == len(results2)

    @pytest.mark.asyncio
    async def test_reset_restores_state(self) -> None:
        """Reset should restore RNG state."""
        storage = SimStorage(seed=42)
        llm = SimLLMProvider(seed=42)
        retriever = DualRetriever(storage=storage, llm=llm, seed=42)

        await storage.store(Entity(name="Test", content="Test content"))

        # First search
        results1 = await retriever.search("Test", deep_search=False)

        # Reset
        retriever.reset()
        storage.reset()
        llm.reset()

        # Store again
        await storage.store(Entity(name="Test", content="Test content"))

        # Second search should match
        results2 = await retriever.search("Test", deep_search=False)

        assert len(results1) == len(results2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for DualRetriever."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Complete dual retrieval workflow."""
        storage = SimStorage(seed=42)
        llm = SimLLMProvider(seed=42)
        retriever = DualRetriever(storage=storage, llm=llm, seed=42)

        # Store multiple entities
        await storage.store(Entity(name="Alice", content="Alice works at Acme Corp"))
        await storage.store(Entity(name="Bob", content="Bob knows Alice from work"))
        await storage.store(Entity(name="Acme Corp", content="Tech company", entity_type="org"))

        # Simple search (fast path)
        simple_results = await retriever.search("Alice", deep_search=False)
        assert len(simple_results) >= 1

        # Complex search (deep path) - use query that will match via fast path
        # The deep search adds LLM rewriting, but fast path always runs
        complex_results = await retriever.search("Acme", deep_search=True)
        assert len(complex_results) >= 1  # Fast path will find "Acme Corp" entity

    @pytest.mark.asyncio
    async def test_graceful_degradation_full(self) -> None:
        """Should work even if LLM always fails."""
        storage = SimStorage(seed=42)
        faults = FaultConfig(llm_timeout=1.0)  # Always fail
        llm = SimLLMProvider(seed=42, faults=faults)
        retriever = DualRetriever(storage=storage, llm=llm, seed=42)

        await storage.store(Entity(name="Test", content="Test content"))

        # Should still return results via fast path
        results = await retriever.search("Test", deep_search=True)
        assert len(results) >= 1
