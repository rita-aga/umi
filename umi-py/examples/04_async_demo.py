#!/usr/bin/env python3
"""
Async API Demo

Demonstrates native Python async/await support with Umi Memory.
"""

import asyncio
import umi


async def main():
    # Create memory with deterministic seed
    print("Creating memory with Sim providers (seed=42)...")
    memory = umi.Memory.sim(seed=42)

    # Remember some information (async)
    print("\nRemembering: 'Alice works at Acme Corp'")
    options = umi.RememberOptions()
    result = await memory.remember("Alice works at Acme Corp", options)
    print(f"✅ Stored {result.entity_count()} entities")
    for entity in result.entities:
        print(f"   - {entity.name} ({entity.entity_type}): {entity.content}")

    # Recall information (async)
    print("\nRecalling: 'Who works at Acme?'")
    recall_options = umi.RecallOptions().with_limit(10)
    entities = await memory.recall("Who works at Acme?", recall_options)
    print(f"✅ Found {len(entities)} results:")
    for entity in entities:
        print(f"   - {entity.name}: {entity.content}")

    # Count total entities (async)
    total = await memory.count()
    print(f"\n📊 Total entities in memory: {total}")

    # Demonstrate concurrent operations
    print("\n🔄 Testing concurrent async operations...")
    results = await asyncio.gather(
        memory.remember("Bob is CEO", umi.RememberOptions()),
        memory.remember("Charlie is CTO", umi.RememberOptions()),
        memory.remember("Diana is CFO", umi.RememberOptions()),
    )
    print(f"✅ Stored {sum(r.entity_count() for r in results)} entities concurrently")

    # Final count
    total = await memory.count()
    print(f"📊 Final total: {total} entities")


if __name__ == "__main__":
    asyncio.run(main())
