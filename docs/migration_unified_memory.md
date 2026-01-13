# Migration Guide: Memory to UnifiedMemory

This guide helps you migrate from the basic `Memory` API to the new `UnifiedMemory` orchestrator for automatic tier management, promotion, eviction, and category evolution.

## Overview

| Feature | Memory | UnifiedMemory |
|---------|--------|---------------|
| **Tier Management** | Manual | Automatic |
| **Promotion** | Manual | Configurable auto |
| **Eviction** | Manual | Configurable auto |
| **Access Tracking** | None | Full frequency, recency, importance |
| **Category Evolution** | None | Automatic suggestions |
| **Policies** | Fixed | Pluggable (LRU, Importance, Hybrid) |

## When to Use UnifiedMemory

Use `UnifiedMemory` when you need:

- **Automatic tier management**: Entities promoted and evicted based on access patterns
- **Access pattern analysis**: Track which entities are accessed frequently vs rarely
- **Memory block optimization**: Get suggestions for splitting/merging blocks
- **Co-occurrence detection**: Identify frequently accessed entity type pairs

Use basic `Memory` when you need:

- **Simple remember/recall**: No automatic tier management needed
- **Full control**: Manual promotion/eviction decisions
- **Minimal overhead**: Lower memory footprint

## Quick Migration

### Before (Memory)

```rust
use umi_memory::umi::{Memory, RememberOptions, RecallOptions};

let mut memory = Memory::sim(42);

// Remember
memory.remember("Alice works at Acme", RememberOptions::default()).await?;

// Recall
let results = memory.recall("Acme", RecallOptions::default()).await?;
```

### After (UnifiedMemory)

```rust
use umi_memory::dst::{SimClock, SimConfig};
use umi_memory::orchestration::{UnifiedMemory, UnifiedMemoryConfig};
// ... provider imports

let clock = SimClock::new();
let config = UnifiedMemoryConfig::default();
let mut memory = UnifiedMemory::new(llm, embedder, vector, storage, clock, config);

// Remember (returns additional promotion/eviction stats)
let result = memory.remember("Alice works at Acme").await?;
println!("Stored {} entities, promoted {}", result.entity_count(), result.promoted_count);

// Recall (same interface)
let results = memory.recall("Acme", 10).await?;
```

## Enabling the Feature

Add the `unified-memory` feature to your `Cargo.toml`:

```toml
[dependencies]
umi-memory = { version = "0.2", features = ["unified-memory"] }
```

## Configuration Options

### UnifiedMemoryConfig Builder

```rust
let config = UnifiedMemoryConfig::new()
    // Core memory limits
    .with_core_size_limit_bytes(32 * 1024)  // 32KB core limit
    .with_core_entity_limit(100)             // Max 100 entities in core

    // Promotion/eviction intervals (ms)
    .with_promotion_interval_ms(5_000)       // Check every 5s
    .with_eviction_interval_ms(10_000)       // Evict every 10s

    // Disable automatic operations (for manual control)
    .without_auto_promote()
    .without_auto_evict();
```

### Custom Policies

```rust
use umi_memory::orchestration::{HybridPolicy, HybridEvictionPolicy, PromotionPolicy, EvictionPolicy};

// Create custom promotion policy (weighted by frequency, recency, importance)
let promotion_policy = HybridPolicy::new(0.3, 0.3, 0.4);

// Create custom eviction policy (LRU + Importance hybrid)
let eviction_policy = HybridEvictionPolicy::new(0.5);

// Use with UnifiedMemory
let memory = UnifiedMemory::with_policies(
    llm, embedder, vector, storage, clock, config,
    promotion_policy,
    eviction_policy,
);
```

## Access Pattern Monitoring

```rust
// Get entity access pattern
let pattern = memory.access_tracker().get_access_pattern(&entity_id);
if let Some(p) = pattern {
    println!("Access count: {}", p.access_count);
    println!("Frequency: {:.2}", p.frequency_score);
    println!("Recency: {:.2}", p.recency_score);
    println!("Importance: {:.2}", p.importance_score);
}

// Get total access stats
let total = memory.category_evolver().total_accesses();
println!("Total accesses tracked: {}", total);
```

## Category Evolution

```rust
// Get evolution suggestions after 100+ accesses
let suggestions = memory.get_evolution_suggestions();

for suggestion in suggestions {
    match suggestion {
        EvolutionSuggestion::CreateBlock { name, types, reason, confidence } => {
            println!("Create block '{}' for {:?}: {} (conf: {:.2})",
                     name, types, reason, confidence);
        }
        EvolutionSuggestion::MergeBlocks { block1, block2, reason, confidence } => {
            println!("Merge {:?} + {:?}: {} (conf: {:.2})",
                     block1, block2, reason, confidence);
        }
        EvolutionSuggestion::SplitBlock { block, types, reason, confidence } => {
            println!("Split {:?} into {:?}: {} (conf: {:.2})",
                     block, types, reason, confidence);
        }
    }
}

// Check block usage distribution
use umi_memory::memory::MemoryBlockType;

println!("Scratch: {:.1}%", memory.block_usage(MemoryBlockType::Scratch) * 100.0);
println!("Facts: {:.1}%", memory.block_usage(MemoryBlockType::Facts) * 100.0);
println!("Goals: {:.1}%", memory.block_usage(MemoryBlockType::Goals) * 100.0);

// Check co-occurrence patterns
use umi_memory::storage::EntityType;

let score = memory.entity_co_occurrence(&EntityType::Person, &EntityType::Project);
println!("Person-Project co-occurrence: {:.2}", score);
```

## Manual Tier Operations

Even with `without_auto_promote()`/`without_auto_evict()`, you can trigger operations manually:

```rust
// Manual promotion
let promoted_count = memory.promote_to_core().await?;
println!("Promoted {} entities", promoted_count);

// Manual eviction
let evicted_count = memory.evict_from_core().await?;
println!("Evicted {} entities", evicted_count);

// Check if entity is in core
if memory.is_in_core(&entity_id) {
    println!("Entity is in core memory");
}

// Get core entity count
let core_count = memory.core_entity_count();
```

## DST Testing Pattern

UnifiedMemory works seamlessly with the DST framework:

```rust
use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};

#[tokio::test]
async fn test_unified_memory_with_faults() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        // Create memory with fault-injected storage
        let fault_config = FaultConfig::new(FaultType::StorageWriteFail, 0.1);
        let storage = SimStorageBackend::new(SimConfig::with_seed(42))
            .with_faults(fault_config);

        let mut memory = UnifiedMemory::new(
            llm, embedder, vector, storage, env.clock.clone(), config
        );

        // Test behavior under faults
        let result = memory.remember("Test data").await;
        // Should handle gracefully

        Ok::<(), anyhow::Error>(())
    })
    .await
    .unwrap();
}
```

## Complete Example

See the full working example at:

```bash
cargo run --example unified_memory_basic --features unified-memory
```

## Summary

1. Enable `unified-memory` feature
2. Replace `Memory::new()` with `UnifiedMemory::new()`
3. Configure limits and intervals via `UnifiedMemoryConfig`
4. Use access tracking for monitoring
5. Review evolution suggestions periodically
6. Manual operations available even with auto mode disabled
7. Same DST testing patterns work
