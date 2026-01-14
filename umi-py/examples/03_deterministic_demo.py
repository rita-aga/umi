#!/usr/bin/env python3
"""
Deterministic Demo

Demonstrates that same seed produces same results (DST principle).
"""

import umi

def main():
    # Create two memories with same seed
    print("Creating two memories with same seed (42)...")
    memory1 = umi.Memory.sim(seed=42)
    memory2 = umi.Memory.sim(seed=42)

    # Same input
    text = "Alice works at Acme Corp"
    options = umi.RememberOptions()

    # Remember in both
    print(f"\nRemembering in both: '{text}'")
    result1 = memory1.remember_sync(text, options)
    result2 = memory2.remember_sync(text, options)

    # Compare results
    print(f"\nMemory 1: {result1.entity_count()} entities")
    print(f"Memory 2: {result2.entity_count()} entities")

    if result1.entity_count() == result2.entity_count():
        print("✅ Same seed = same results! (Deterministic)")
    else:
        print("❌ Different results (not deterministic)")

    # Recall should also be deterministic
    print("\nRecalling in both: 'Acme'")
    entities1 = memory1.recall_sync("Acme", umi.RecallOptions())
    entities2 = memory2.recall_sync("Acme", umi.RecallOptions())

    print(f"Memory 1 found: {len(entities1)} entities")
    print(f"Memory 2 found: {len(entities2)} entities")

    if len(entities1) == len(entities2):
        print("✅ Recall is also deterministic!")

if __name__ == "__main__":
    main()
