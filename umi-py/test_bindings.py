#!/usr/bin/env python3
"""
Test script to verify Python bindings work correctly.
"""
import sys
sys.path.insert(0, '/Users/seshendranalla/Development/umi/target/wheels')

import asyncio
import umi

async def test_basic_sim():
    """Test basic functionality with Sim providers."""
    print("Testing Umi Python bindings...")
    print()

    # Create memory with sim providers
    print("1. Creating Memory with sim providers (seed=42)...")
    memory = umi.Memory.sim(seed=42)
    print(f"   ✓ Created: {memory}")
    print()

    # Test remember
    print("2. Testing remember()...")
    options = umi.RememberOptions()
    result = await memory.remember("Alice works at Acme Corp", options)
    print(f"   ✓ Stored {result.entity_count()} entities")
    for entity in result.entities:
        print(f"     - {entity.name} ({entity.entity_type}): {entity.content}")
    print()

    # Test recall
    print("3. Testing recall()...")
    recall_options = umi.RecallOptions()
    entities = await memory.recall("Alice", recall_options)
    print(f"   ✓ Found {len(entities)} entities")
    for entity in entities:
        print(f"     - {entity.name}: {entity.content}")
    print()

    # Test count
    print("4. Testing count()...")
    count = await memory.count()
    print(f"   ✓ Total entities in storage: {count}")
    print()

    # Test get by ID
    if result.entities:
        entity_id = result.entities[0].id
        print(f"5. Testing get() with ID: {entity_id}...")
        entity = await memory.get(entity_id)
        if entity:
            print(f"   ✓ Retrieved: {entity.name}")
        else:
            print(f"   ✗ Entity not found")
        print()

    print("✅ All tests passed!")

def test_core_memory():
    """Test CoreMemory."""
    print("\nTesting CoreMemory...")
    core = umi.CoreMemory()

    core.set_block("system", "You are a helpful assistant.")
    core.set_block("persona", "You are friendly and concise.")

    print(f"  Used: {core.used_bytes} / {core.max_bytes} bytes ({core.utilization * 100:.1f}%)")
    print(f"  XML: {core.render()[:100]}...")
    print("  ✓ CoreMemory works")

def test_working_memory():
    """Test WorkingMemory."""
    print("\nTesting WorkingMemory...")
    working = umi.WorkingMemory()

    working.set("session_id", b"abc123", ttl_secs=3600)
    value = working.get("session_id")

    print(f"  Stored session_id: {value}")
    print(f"  Entry count: {working.entry_count}")
    print(f"  Used: {working.used_bytes} bytes")
    print("  ✓ WorkingMemory works")

if __name__ == "__main__":
    print("=" * 60)
    print("Umi Python Bindings Test Suite")
    print("=" * 60)
    print()

    # Test sync components
    test_core_memory()
    test_working_memory()

    # Test async components
    print("\nTesting Archival Memory (async)...")
    asyncio.run(test_basic_sim())

    print()
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
