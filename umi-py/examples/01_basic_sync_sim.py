#!/usr/bin/env python3
"""
Basic Sync Usage with Sim Providers

Demonstrates basic remember/recall operations using simulation providers
for deterministic testing.
"""

import umi

def main():
    # Create memory with deterministic seed
    print("Creating memory with Sim providers (seed=42)...")
    memory = umi.Memory.sim(seed=42)

    # Remember some information
    print("\nRemembering: 'Alice works at Acme Corp as a software engineer'")
    options = umi.RememberOptions()
    result = memory.remember_sync(
        "Alice works at Acme Corp as a software engineer",
        options
    )
    print(f"✅ Stored {result.entity_count()} entities")
    for entity in result.entities:
        print(f"   - {entity.name} ({entity.entity_type}): {entity.content}")

    # Recall information
    print("\nRecalling: 'Who works at Acme?'")
    recall_options = umi.RecallOptions().with_limit(10)
    entities = memory.recall_sync("Who works at Acme?", recall_options)
    print(f"✅ Found {len(entities)} results:")
    for entity in entities:
        print(f"   - {entity.name}: {entity.content}")

    # Count total entities
    total = memory.count_sync()
    print(f"\n📊 Total entities in memory: {total}")

if __name__ == "__main__":
    main()
