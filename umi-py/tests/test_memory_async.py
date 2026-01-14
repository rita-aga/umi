"""
Async Memory Tests

Tests for Memory async API.
"""

import pytest
import umi


@pytest.mark.asyncio
async def test_remember_async():
    """Test remember async stores information."""
    memory = umi.Memory.sim(seed=42)
    options = umi.RememberOptions()

    result = await memory.remember("Alice works at Acme", options)

    assert result is not None
    assert result.entity_count() > 0


@pytest.mark.asyncio
async def test_recall_async():
    """Test recall async retrieves information."""
    memory = umi.Memory.sim(seed=42)

    # Store some data
    await memory.remember("Alice works at Acme Corp", umi.RememberOptions())

    # Recall it
    entities = await memory.recall("Alice", umi.RecallOptions())

    assert isinstance(entities, list)
    assert len(entities) > 0


@pytest.mark.asyncio
async def test_count_async():
    """Test count async returns entity count."""
    memory = umi.Memory.sim(seed=42)

    initial_count = await memory.count()
    assert initial_count == 0

    # Add some entities
    await memory.remember("Test data", umi.RememberOptions())

    new_count = await memory.count()
    assert new_count > initial_count


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test multiple concurrent async operations."""
    import asyncio

    memory = umi.Memory.sim(seed=42)

    # Store multiple items concurrently
    results = await asyncio.gather(
        memory.remember("Alice works at Acme", umi.RememberOptions()),
        memory.remember("Bob is CEO", umi.RememberOptions()),
        memory.remember("Charlie is CTO", umi.RememberOptions()),
    )

    assert len(results) == 3
    assert all(r.entity_count() > 0 for r in results)

    # Verify all stored
    total = await memory.count()
    assert total > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
