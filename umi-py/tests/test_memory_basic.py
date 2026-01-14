"""
Basic Memory Tests

Tests for core Memory functionality with Sim providers.
"""

import pytest
import umi


def test_memory_sim_constructor():
    """Test Memory.sim() constructor."""
    memory = umi.Memory.sim(seed=42)
    assert memory is not None


def test_remember_sync():
    """Test remember_sync stores information."""
    memory = umi.Memory.sim(seed=42)
    options = umi.RememberOptions()

    result = memory.remember_sync("Alice works at Acme", options)

    assert result is not None
    assert result.entity_count() > 0
    assert isinstance(result.entities, list)


def test_recall_sync():
    """Test recall_sync retrieves information."""
    memory = umi.Memory.sim(seed=42)

    # Store some data
    memory.remember_sync("Alice works at Acme Corp", umi.RememberOptions())

    # Recall it
    entities = memory.recall_sync("Alice", umi.RecallOptions())

    assert isinstance(entities, list)
    assert len(entities) > 0


def test_count_sync():
    """Test count_sync returns entity count."""
    memory = umi.Memory.sim(seed=42)

    initial_count = memory.count_sync()
    assert initial_count == 0

    # Add some entities
    memory.remember_sync("Test data", umi.RememberOptions())

    new_count = memory.count_sync()
    assert new_count > initial_count


def test_determinism():
    """Test same seed produces same results."""
    memory1 = umi.Memory.sim(seed=12345)
    memory2 = umi.Memory.sim(seed=12345)

    text = "Bob is a developer"
    options = umi.RememberOptions()

    result1 = memory1.remember_sync(text, options)
    result2 = memory2.remember_sync(text, options)

    assert result1.entity_count() == result2.entity_count()


def test_options():
    """Test RememberOptions and RecallOptions."""
    options = umi.RememberOptions()
    assert options is not None

    # Test builder pattern
    options = options.without_extraction().with_importance(0.8)
    assert options is not None

    recall_opts = umi.RecallOptions()
    recall_opts = recall_opts.with_limit(5).fast_only()
    assert recall_opts is not None


def test_result_type():
    """Test RememberResult properties."""
    memory = umi.Memory.sim(seed=42)
    result = memory.remember_sync("Test", umi.RememberOptions())

    assert hasattr(result, 'entities')
    assert hasattr(result, 'evolutions')
    assert hasattr(result, 'entity_count')
    assert hasattr(result, 'has_evolutions')


def test_entity_properties():
    """Test Entity has expected properties."""
    entity = umi.Entity("person", "Alice", "Friend")

    assert entity.entity_type == "person"
    assert entity.name == "Alice"
    assert entity.content == "Friend"
    assert entity.id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
