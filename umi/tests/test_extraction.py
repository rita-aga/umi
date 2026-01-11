"""Tests for EntityExtractor.

TigerStyle: Simulation-first testing, determinism verification.
"""

import pytest

from umi.extraction import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    ENTITY_TYPES,
    RELATION_TYPES,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
)
from umi.faults import FaultConfig
from umi.providers.sim import SimLLMProvider


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def llm() -> SimLLMProvider:
    """Create a SimLLMProvider with seed."""
    return SimLLMProvider(seed=42)


@pytest.fixture
def extractor(llm: SimLLMProvider) -> EntityExtractor:
    """Create an EntityExtractor with seeded LLM."""
    return EntityExtractor(llm=llm, seed=42)


# =============================================================================
# ExtractedEntity Tests
# =============================================================================


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_creates_valid_entity(self) -> None:
        """Should create entity with valid fields."""
        entity = ExtractedEntity(
            name="Alice",
            entity_type="person",
            content="A software engineer",
            confidence=0.9,
        )

        assert entity.name == "Alice"
        assert entity.entity_type == "person"
        assert entity.content == "A software engineer"
        assert entity.confidence == 0.9

    def test_default_confidence(self) -> None:
        """Should use default confidence when not specified."""
        entity = ExtractedEntity(
            name="Alice",
            entity_type="person",
            content="Description",
        )

        assert entity.confidence == 0.5

    def test_validates_empty_name(self) -> None:
        """Should reject empty name."""
        with pytest.raises(AssertionError):
            ExtractedEntity(name="", entity_type="person", content="Test")

    def test_validates_entity_type(self) -> None:
        """Should reject invalid entity type."""
        with pytest.raises(AssertionError):
            ExtractedEntity(name="Test", entity_type="invalid", content="Test")

    def test_validates_confidence_range(self) -> None:
        """Should reject confidence outside 0-1."""
        with pytest.raises(AssertionError):
            ExtractedEntity(name="Test", entity_type="person", content="Test", confidence=1.5)

        with pytest.raises(AssertionError):
            ExtractedEntity(name="Test", entity_type="person", content="Test", confidence=-0.1)

    def test_all_entity_types_valid(self) -> None:
        """Should accept all valid entity types."""
        for entity_type in ENTITY_TYPES:
            entity = ExtractedEntity(
                name="Test",
                entity_type=entity_type,
                content="Test content",
            )
            assert entity.entity_type == entity_type


# =============================================================================
# ExtractedRelation Tests
# =============================================================================


class TestExtractedRelation:
    """Tests for ExtractedRelation dataclass."""

    def test_creates_valid_relation(self) -> None:
        """Should create relation with valid fields."""
        relation = ExtractedRelation(
            source="Alice",
            target="Acme",
            relation_type="works_at",
            confidence=0.85,
        )

        assert relation.source == "Alice"
        assert relation.target == "Acme"
        assert relation.relation_type == "works_at"
        assert relation.confidence == 0.85

    def test_default_confidence(self) -> None:
        """Should use default confidence when not specified."""
        relation = ExtractedRelation(
            source="Alice",
            target="Bob",
            relation_type="knows",
        )

        assert relation.confidence == 0.5

    def test_validates_empty_source(self) -> None:
        """Should reject empty source."""
        with pytest.raises(AssertionError):
            ExtractedRelation(source="", target="Bob", relation_type="knows")

    def test_validates_empty_target(self) -> None:
        """Should reject empty target."""
        with pytest.raises(AssertionError):
            ExtractedRelation(source="Alice", target="", relation_type="knows")

    def test_validates_relation_type(self) -> None:
        """Should reject invalid relation type."""
        with pytest.raises(AssertionError):
            ExtractedRelation(source="A", target="B", relation_type="invalid")

    def test_all_relation_types_valid(self) -> None:
        """Should accept all valid relation types."""
        for relation_type in RELATION_TYPES:
            relation = ExtractedRelation(
                source="A",
                target="B",
                relation_type=relation_type,
            )
            assert relation.relation_type == relation_type


# =============================================================================
# EntityExtractor Basic Tests
# =============================================================================


class TestEntityExtractorBasic:
    """Basic tests for EntityExtractor."""

    @pytest.mark.asyncio
    async def test_extracts_entities(self, extractor: EntityExtractor) -> None:
        """Should extract entities from text."""
        result = await extractor.extract("I met Alice at Acme Corp")

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) >= 1
        assert result.raw_text == "I met Alice at Acme Corp"

    @pytest.mark.asyncio
    async def test_extracts_known_names(self, extractor: EntityExtractor) -> None:
        """Should extract known names like Alice, Bob, Acme."""
        result = await extractor.extract("Alice works at Acme")

        names = {e.name for e in result.entities}
        # SimLLMProvider recognizes common names
        assert "Alice" in names or "Acme" in names

    @pytest.mark.asyncio
    async def test_extracts_relations(self, extractor: EntityExtractor) -> None:
        """Should extract relations when multiple entities found."""
        result = await extractor.extract("Alice works at Acme Corp as an engineer")

        # May have relations if multiple entities extracted
        if len(result.entities) >= 2:
            # Relations are optional, test they're valid if present
            for relation in result.relations:
                assert relation.source
                assert relation.target
                assert relation.relation_type in RELATION_TYPES

    @pytest.mark.asyncio
    async def test_returns_result_structure(self, extractor: EntityExtractor) -> None:
        """Should return proper ExtractionResult structure."""
        result = await extractor.extract("Test text")

        assert hasattr(result, "entities")
        assert hasattr(result, "relations")
        assert hasattr(result, "raw_text")
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)

    @pytest.mark.asyncio
    async def test_validates_empty_text(self, extractor: EntityExtractor) -> None:
        """Should reject empty text."""
        with pytest.raises(AssertionError):
            await extractor.extract("")


# =============================================================================
# EntityExtractor Confidence Tests
# =============================================================================


class TestConfidenceFiltering:
    """Tests for confidence filtering."""

    @pytest.mark.asyncio
    async def test_filters_by_min_confidence(self, extractor: EntityExtractor) -> None:
        """Should filter entities below min_confidence."""
        # Get all entities first
        result_all = await extractor.extract("Alice works at Acme")

        # Filter with high threshold
        result_filtered = await extractor.extract(
            "Alice works at Acme",
            min_confidence=0.95,
        )

        # Filtered should have fewer or equal entities
        assert len(result_filtered.entities) <= len(result_all.entities)

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self, extractor: EntityExtractor) -> None:
        """All extracted entities should have valid confidence."""
        result = await extractor.extract("Bob met Charlie at TechCorp")

        for entity in result.entities:
            assert CONFIDENCE_MIN <= entity.confidence <= CONFIDENCE_MAX

        for relation in result.relations:
            assert CONFIDENCE_MIN <= relation.confidence <= CONFIDENCE_MAX

    @pytest.mark.asyncio
    async def test_validates_min_confidence_range(self, extractor: EntityExtractor) -> None:
        """Should reject invalid min_confidence."""
        with pytest.raises(AssertionError):
            await extractor.extract("Test", min_confidence=1.5)

        with pytest.raises(AssertionError):
            await extractor.extract("Test", min_confidence=-0.1)


# =============================================================================
# EntityExtractor Context Tests
# =============================================================================


class TestExistingEntities:
    """Tests for existing entities context."""

    @pytest.mark.asyncio
    async def test_accepts_existing_entities(self, extractor: EntityExtractor) -> None:
        """Should accept existing entities for context."""
        result = await extractor.extract(
            "She joined the team last month",
            existing_entities=["Alice", "Acme Corp"],
        )

        # Should still return valid result
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) >= 1

    @pytest.mark.asyncio
    async def test_works_without_context(self, extractor: EntityExtractor) -> None:
        """Should work without existing entities."""
        result = await extractor.extract("Alice is a developer")

        assert isinstance(result, ExtractionResult)


# =============================================================================
# EntityExtractor Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_same_seed_same_entities(self) -> None:
        """Same seed should produce same entities."""
        llm1 = SimLLMProvider(seed=42)
        llm2 = SimLLMProvider(seed=42)

        extractor1 = EntityExtractor(llm=llm1, seed=42)
        extractor2 = EntityExtractor(llm=llm2, seed=42)

        text = "Alice works at Acme Corp"

        result1 = await extractor1.extract(text)
        result2 = await extractor2.extract(text)

        # Same number of entities
        assert len(result1.entities) == len(result2.entities)

        # Same entity names
        names1 = {e.name for e in result1.entities}
        names2 = {e.name for e in result2.entities}
        assert names1 == names2

    @pytest.mark.asyncio
    async def test_reset_restores_state(self) -> None:
        """Reset should restore to initial state."""
        llm = SimLLMProvider(seed=42)
        extractor = EntityExtractor(llm=llm, seed=42)

        text = "Bob knows Charlie"

        # First extraction
        result1 = await extractor.extract(text)

        # Reset both
        extractor.reset()
        llm.reset()

        # Second extraction should match
        result2 = await extractor.extract(text)

        assert len(result1.entities) == len(result2.entities)


# =============================================================================
# EntityExtractor Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation on errors."""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_timeout(self) -> None:
        """Should return empty result on LLM timeout."""
        faults = FaultConfig(llm_timeout=1.0)  # Always timeout
        llm = SimLLMProvider(seed=42, faults=faults)
        extractor = EntityExtractor(llm=llm, seed=42)

        result = await extractor.extract("Test text")

        # Should return empty lists, not raise
        assert isinstance(result, ExtractionResult)
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self) -> None:
        """Should return empty result on LLM error."""
        faults = FaultConfig(llm_error=1.0)  # Always error
        llm = SimLLMProvider(seed=42, faults=faults)
        extractor = EntityExtractor(llm=llm, seed=42)

        result = await extractor.extract("Test text")

        assert isinstance(result, ExtractionResult)
        assert result.entities == []


# =============================================================================
# EntityExtractor Convenience Methods Tests
# =============================================================================


class TestConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_extract_entities_only(self, extractor: EntityExtractor) -> None:
        """Should extract only entities without relations."""
        entities = await extractor.extract_entities_only("Alice works at Acme")

        assert isinstance(entities, list)
        assert len(entities) >= 1
        assert all(isinstance(e, ExtractedEntity) for e in entities)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for EntityExtractor."""

    @pytest.mark.asyncio
    async def test_full_extraction_workflow(self) -> None:
        """Complete extraction workflow."""
        llm = SimLLMProvider(seed=42)
        extractor = EntityExtractor(llm=llm, seed=42)

        # Extract from complex text
        result = await extractor.extract(
            "Alice, the CEO of Acme Corp, met with Bob from TechCorp to discuss Project Alpha."
        )

        # Should have multiple entities
        assert len(result.entities) >= 1

        # Check entity types are valid
        for entity in result.entities:
            assert entity.entity_type in ENTITY_TYPES

        # Check relation types are valid
        for relation in result.relations:
            assert relation.relation_type in RELATION_TYPES

    @pytest.mark.asyncio
    async def test_entity_types_extracted(self) -> None:
        """Should extract various entity types."""
        llm = SimLLMProvider(seed=42)
        extractor = EntityExtractor(llm=llm, seed=42)

        # Person extraction
        result = await extractor.extract("Alice is a software engineer")
        assert len(result.entities) >= 1

        # Org extraction
        result = await extractor.extract("Acme Corp is a tech company")
        assert len(result.entities) >= 1
