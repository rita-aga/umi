//! Basic UnifiedMemory Example with DST Verification
//!
//! This example demonstrates the UnifiedMemory orchestrator which provides:
//! - Automatic promotion from archival to core based on access patterns
//! - Automatic eviction from core when limits are reached
//! - Access tracking with frequency, recency, and importance scoring
//! - Category evolution for optimizing memory organization
//!
//! Run with: cargo run --example unified_memory_basic --features unified-memory
//!
//! `TigerStyle`: DST-first, deterministic, reproducible with seed.

use umi_memory::dst::{SimClock, SimConfig, Simulation};
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::memory::MemoryBlockType;
use umi_memory::orchestration::{UnifiedMemory, UnifiedMemoryConfig};
use umi_memory::storage::{EntityType, SimStorageBackend, SimVectorBackend};

/// Create a UnifiedMemory instance for deterministic testing.
fn create_unified(seed: u64, clock: SimClock, config: UnifiedMemoryConfig) -> UnifiedMemory {
    let llm = SimLLMProvider::with_seed(seed);
    let embedder = SimEmbeddingProvider::with_seed(seed);
    let vector = SimVectorBackend::new(seed);
    let storage = SimStorageBackend::new(SimConfig::with_seed(seed));

    UnifiedMemory::new(llm, embedder, vector, storage, clock, config)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== UnifiedMemory Basic Example ===\n");

    // Example 1: Basic remember and recall
    println!("--- Example 1: Basic Remember and Recall ---");
    example_basic_remember_recall().await?;

    // Example 2: Access tracking and promotion
    println!("\n--- Example 2: Access Tracking and Promotion ---");
    example_access_tracking().await?;

    // Example 3: Category evolution suggestions
    println!("\n--- Example 3: Category Evolution ---");
    example_category_evolution().await?;

    // Example 4: Deterministic simulation verification
    println!("\n--- Example 4: Determinism Verification ---");
    example_determinism_verification().await?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Example 1: Basic remember and recall operations.
async fn example_basic_remember_recall() -> anyhow::Result<()> {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let config = UnifiedMemoryConfig::default();
        let mut memory = create_unified(42, env.clock.clone(), config);

        // Remember some information
        let result = memory
            .remember("Alice is a software engineer at Acme")
            .await
            .map_err(|e| anyhow::anyhow!("remember failed: {}", e))?;
        println!("Remembered {} entities", result.entities.len());

        for entity in &result.entities {
            println!("  - {} ({:?})", entity.name, entity.entity_type);
        }

        // Recall related information
        let results = memory
            .recall("Who works at Acme?", 10)
            .await
            .map_err(|e| anyhow::anyhow!("recall failed: {}", e))?;
        println!("Recalled {} entities", results.len());

        Ok::<(), anyhow::Error>(())
    })
    .await?;

    Ok(())
}

/// Example 2: Access tracking with promotion.
async fn example_access_tracking() -> anyhow::Result<()> {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let config = UnifiedMemoryConfig::new().with_core_entity_limit(5); // Small limit to demonstrate promotion

        let mut memory = create_unified(42, env.clock.clone(), config);

        // Remember multiple things
        for i in 0..10 {
            let text = format!("Entity {} is important information", i);
            let _ = memory.remember(&text).await;
        }

        // Check access tracking stats
        let total_accesses = memory.category_evolver().total_accesses();
        println!("Total tracked accesses: {}", total_accesses);

        // Trigger manual promotion
        let promoted = memory
            .promote_to_core()
            .await
            .map_err(|e| anyhow::anyhow!("promotion failed: {}", e))?;
        println!("Promoted {} entities to core memory", promoted);

        // Check tier stats
        let core_count = memory.core_entity_count();
        println!("Core entity count: {}", core_count);

        Ok::<(), anyhow::Error>(())
    })
    .await?;

    Ok(())
}

/// Example 3: Category evolution suggestions.
async fn example_category_evolution() -> anyhow::Result<()> {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let config = UnifiedMemoryConfig::default();
        let mut memory = create_unified(42, env.clock.clone(), config);

        // Generate enough accesses for evolution analysis (need 100+)
        for i in 0..120 {
            let text = if i % 3 == 0 {
                format!("Person {} works on project", i)
            } else if i % 3 == 1 {
                format!("Project {} has deadline", i)
            } else {
                format!("Task {} needs review", i)
            };
            let _ = memory.remember(&text).await;
        }

        // Check block usage patterns
        println!(
            "Block usage - Scratch: {:.1}%, Facts: {:.1}%, Goals: {:.1}%",
            memory.block_usage(MemoryBlockType::Scratch) * 100.0,
            memory.block_usage(MemoryBlockType::Facts) * 100.0,
            memory.block_usage(MemoryBlockType::Goals) * 100.0
        );

        // Check co-occurrence patterns
        let person_project = memory.entity_co_occurrence(&EntityType::Person, &EntityType::Project);
        println!("Person-Project co-occurrence: {:.2}", person_project);

        // Get evolution suggestions
        let suggestions = memory.get_evolution_suggestions();
        println!("Evolution suggestions: {}", suggestions.len());

        for suggestion in &suggestions {
            match suggestion {
                umi_memory::orchestration::EvolutionSuggestion::CreateBlock {
                    name,
                    reason,
                    ..
                } => {
                    println!("  - Create block '{}': {}", name, reason);
                }
                umi_memory::orchestration::EvolutionSuggestion::MergeBlocks { reason, .. } => {
                    println!("  - Merge blocks: {}", reason);
                }
                umi_memory::orchestration::EvolutionSuggestion::SplitBlock { reason, .. } => {
                    println!("  - Split block: {}", reason);
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })
    .await?;

    Ok(())
}

/// Example 4: Verify deterministic behavior.
async fn example_determinism_verification() -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    // Run 1 with seed 42
    let accesses1 = Arc::new(AtomicU64::new(0));
    let accesses1_clone = accesses1.clone();

    {
        let sim = Simulation::new(SimConfig::with_seed(42));
        sim.run(|env| {
            let accesses = accesses1_clone.clone();
            async move {
                let config = UnifiedMemoryConfig::default();
                let mut memory = create_unified(42, env.clock.clone(), config);

                let _ = memory.remember("Alice works on Alpha project").await;
                let _ = memory.remember("Bob collaborates with Alice").await;
                let _ = memory.recall("project", 10).await;

                accesses.store(memory.category_evolver().total_accesses(), Ordering::SeqCst);
                Ok::<(), anyhow::Error>(())
            }
        })
        .await?;
    }

    // Run 2 with same seed 42
    let accesses2 = Arc::new(AtomicU64::new(0));
    let accesses2_clone = accesses2.clone();

    {
        let sim = Simulation::new(SimConfig::with_seed(42));
        sim.run(|env| {
            let accesses = accesses2_clone.clone();
            async move {
                let config = UnifiedMemoryConfig::default();
                let mut memory = create_unified(42, env.clock.clone(), config);

                let _ = memory.remember("Alice works on Alpha project").await;
                let _ = memory.remember("Bob collaborates with Alice").await;
                let _ = memory.recall("project", 10).await;

                accesses.store(memory.category_evolver().total_accesses(), Ordering::SeqCst);
                Ok::<(), anyhow::Error>(())
            }
        })
        .await?;
    }

    let result1 = accesses1.load(Ordering::SeqCst);
    let result2 = accesses2.load(Ordering::SeqCst);

    println!("Run 1 (seed 42): {} accesses", result1);
    println!("Run 2 (seed 42): {} accesses", result2);

    if result1 == result2 {
        println!("VERIFIED: Same seed produces identical results!");
    } else {
        println!("ERROR: Results differ! {} != {}", result1, result2);
    }

    Ok(())
}
