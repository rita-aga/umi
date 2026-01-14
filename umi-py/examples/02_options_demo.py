#!/usr/bin/env python3
"""
Options Demo

Demonstrates using RememberOptions and RecallOptions to control behavior.
"""

import umi

def main():
    memory = umi.Memory.sim(seed=42)

    # Example 1: Remember without entity extraction
    print("Example 1: Store as raw text (no extraction)")
    options = umi.RememberOptions().without_extraction()
    result = memory.remember_sync("This is raw text", options)
    print(f"Stored {result.entity_count()} entities")

    # Example 2: Remember with importance
    print("\nExample 2: Store with high importance")
    options = umi.RememberOptions().with_importance(0.9)
    result = memory.remember_sync("Critical information", options)
    print(f"Stored {result.entity_count()} entities")

    # Example 3: Recall with limit
    print("\nExample 3: Recall with limit=5")
    recall_opts = umi.RecallOptions().with_limit(5)
    entities = memory.recall_sync("information", recall_opts)
    print(f"Found {len(entities)} entities (limited to 5)")

    # Example 4: Fast search (no deep LLM rewrite)
    print("\nExample 4: Fast text-only search")
    recall_opts = umi.RecallOptions().fast_only()
    entities = memory.recall_sync("text", recall_opts)
    print(f"Found {len(entities)} entities (fast search)")

if __name__ == "__main__":
    main()
