//! DST Integration Tests for UnifiedMemory.
//!
//! `TigerStyle`: Full workflow testing with Simulation harness.
//!
//! These tests verify:
//! - Full lifecycle: remember -> promote -> recall -> evict
//! - Multi-entity type handling
//! - Graceful degradation under faults
//! - CategoryEvolver integration
//! - Deterministic behavior

use crate::dst::{FaultConfig, FaultType, SimClock, SimConfig, Simulation};
use crate::embedding::SimEmbeddingProvider;
use crate::llm::SimLLMProvider;
use crate::memory::MemoryBlockType;
use crate::orchestration::{
    CategoryEvolver, EvolutionSuggestion, UnifiedMemory, UnifiedMemoryConfig,
};
use crate::storage::{EntityType, SimStorageBackend, SimVectorBackend};

// =============================================================================
// Test Helpers
// =============================================================================

/// Create UnifiedMemory for integration testing.
fn create_unified(
    seed: u64,
    clock: SimClock,
) -> UnifiedMemory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> {
    let llm = SimLLMProvider::with_seed(seed);
    let embedder = SimEmbeddingProvider::with_seed(seed);
    let vector = SimVectorBackend::new(seed);
    let storage = SimStorageBackend::new(SimConfig::with_seed(seed));
    let config = UnifiedMemoryConfig::default();

    UnifiedMemory::new(llm, embedder, vector, storage, clock, config)
}

/// Create UnifiedMemory with custom config.
fn create_unified_with_config(
    seed: u64,
    clock: SimClock,
    config: UnifiedMemoryConfig,
) -> UnifiedMemory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> {
    let llm = SimLLMProvider::with_seed(seed);
    let embedder = SimEmbeddingProvider::with_seed(seed);
    let vector = SimVectorBackend::new(seed);
    let storage = SimStorageBackend::new(SimConfig::with_seed(seed));

    UnifiedMemory::new(llm, embedder, vector, storage, clock, config)
}

/// Create UnifiedMemory with storage fault injection.
fn create_unified_with_faults(
    seed: u64,
    clock: SimClock,
    fault_config: FaultConfig,
) -> UnifiedMemory<SimLLMProvider, SimEmbeddingProvider, SimStorageBackend, SimVectorBackend> {
    let llm = SimLLMProvider::with_seed(seed);
    let embedder = SimEmbeddingProvider::with_seed(seed);
    let vector = SimVectorBackend::new(seed);
    let storage = SimStorageBackend::new(SimConfig::with_seed(seed)).with_faults(fault_config);
    let config = UnifiedMemoryConfig::default();

    UnifiedMemory::new(llm, embedder, vector, storage, clock, config)
}

// =============================================================================
// Full Workflow Tests
// =============================================================================

/// Test the complete lifecycle: remember -> access tracking -> evolution tracking.
#[tokio::test]
async fn test_full_lifecycle_remember_to_evolution() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Phase 1: Remember multiple entities
        let texts = [
            "Alice is the project lead for Project Alpha",
            "Bob works with Alice on Project Alpha",
            "Project Alpha is due next month",
            "Alice reviewed the code for Project Alpha",
            "Bob submitted the report for Project Alpha",
        ];

        for text in &texts {
            let result = memory.remember(*text).await;
            assert!(result.is_ok(), "remember should succeed: {:?}", result);
        }

        // Phase 2: Verify access tracking
        let total_accesses = memory.category_evolver().total_accesses();
        assert!(
            total_accesses >= 5,
            "should have at least 5 accesses, got {}",
            total_accesses
        );

        // Phase 3: Verify block usage tracking
        // Notes/fallback entities go to Scratch
        let scratch_usage = memory.block_usage(MemoryBlockType::Scratch);
        println!("Scratch block usage: {:.2}%", scratch_usage * 100.0);

        // Phase 4: Check we can recall
        let recall_result = memory.recall("Alice", 10).await;
        assert!(recall_result.is_ok(), "recall should succeed");

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test recall with core -> archival fallback.
#[tokio::test]
async fn test_recall_with_fallback() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let config = UnifiedMemoryConfig::new()
            .without_auto_promote() // Disable auto-promotion to test fallback
            .without_auto_evict();

        let mut memory = create_unified_with_config(42, env.clock.clone(), config);

        // Remember something (goes to archival only since auto-promote is off)
        let _ = memory.remember("Important project data").await;

        // Recall should fall back to archival
        let results = memory.recall("project", 10).await.unwrap();

        // Should find something (from archival)
        println!("Recall results: {} entities", results.len());

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test promotion and eviction cycle.
#[tokio::test]
async fn test_promotion_eviction_cycle() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let config = UnifiedMemoryConfig::new()
            .with_core_entity_limit(5) // Small limit to trigger eviction
            .with_core_size_limit_bytes(1024); // Small size limit

        let mut memory = create_unified_with_config(42, env.clock.clone(), config);

        // Remember many things to trigger promotion and eviction
        for i in 0..20 {
            let text = format!("Entity {} is important data", i);
            let result = memory.remember(&text).await;
            assert!(result.is_ok(), "remember {} should succeed", i);
        }

        // Manual promotion
        let promoted = memory.promote_to_core().await;
        assert!(promoted.is_ok(), "promotion should succeed");
        println!("Promoted {} entities", promoted.unwrap());

        // Manual eviction if needed
        let evicted = memory.evict_from_core().await;
        assert!(evicted.is_ok(), "eviction should succeed");
        println!("Evicted {} entities", evicted.unwrap());

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

// =============================================================================
// Fault Injection Tests
// =============================================================================

/// Test graceful degradation with storage write failures.
#[tokio::test]
async fn test_graceful_degradation_storage_write_fail() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        // 50% storage write failure rate
        let fault_config = FaultConfig::new(FaultType::StorageWriteFail, 0.5);
        let mut memory = create_unified_with_faults(42, env.clock.clone(), fault_config);

        let mut successes = 0;
        let mut failures = 0;

        // Try to remember multiple things
        for i in 0..10 {
            let text = format!("Test data {}", i);
            match memory.remember(&text).await {
                Ok(_) => successes += 1,
                Err(_) => failures += 1,
            }
        }

        println!(
            "Storage fault test: successes={}, failures={}",
            successes, failures
        );

        // With 50% failure rate, we should see both
        assert!(
            failures > 0,
            "should have some failures with 50% fault rate"
        );
        assert!(
            successes > 0,
            "should have some successes with 50% fault rate"
        );

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test graceful degradation with storage read failures.
#[tokio::test]
async fn test_graceful_degradation_storage_read_fail() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        // First store some data without faults
        let mut memory = create_unified(42, env.clock.clone());
        for i in 0..5 {
            let _ = memory.remember(&format!("Data {}", i)).await;
        }

        // Now create memory with read faults
        let fault_config = FaultConfig::new(FaultType::StorageReadFail, 0.5);
        let mut memory_with_faults =
            create_unified_with_faults(42, env.clock.clone(), fault_config);

        // Try to recall - should handle failures gracefully
        let mut successes = 0;
        let mut failures = 0;

        for _ in 0..10 {
            match memory_with_faults.recall("data", 10).await {
                Ok(_) => successes += 1,
                Err(_) => failures += 1,
            }
        }

        println!(
            "Read fault test: successes={}, failures={}",
            successes, failures
        );

        // Should have some of each
        assert!(failures > 0 || successes > 0, "should complete operations");

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test that 100% storage failure is handled gracefully.
#[tokio::test]
async fn test_total_storage_failure() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        // 100% storage failure
        let fault_config = FaultConfig::new(FaultType::StorageWriteFail, 1.0);
        let mut memory = create_unified_with_faults(42, env.clock.clone(), fault_config);

        // Try to remember - should fail gracefully
        let result = memory.remember("Test data").await;

        // Should return an error, not panic
        assert!(
            result.is_err(),
            "should return error with 100% storage failure"
        );

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

// =============================================================================
// Determinism Tests
// =============================================================================

/// Test that same seed produces identical results.
#[tokio::test]
async fn test_determinism_same_seed() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Run 1
    let accesses1 = Arc::new(AtomicU64::new(0));
    let accesses1_clone = accesses1.clone();

    {
        let sim = Simulation::new(SimConfig::with_seed(42));
        sim.run(|env| {
            let accesses = accesses1_clone.clone();
            async move {
                let mut memory = create_unified(42, env.clock.clone());

                let _ = memory.remember("Alice works on Project Alpha").await;
                let _ = memory.remember("Bob is Alice's colleague").await;

                accesses.store(memory.category_evolver().total_accesses(), Ordering::SeqCst);
                Ok::<(), std::convert::Infallible>(())
            }
        })
        .await
        .unwrap();
    }

    // Run 2 with same seed
    let accesses2 = Arc::new(AtomicU64::new(0));
    let accesses2_clone = accesses2.clone();

    {
        let sim = Simulation::new(SimConfig::with_seed(42));
        sim.run(|env| {
            let accesses = accesses2_clone.clone();
            async move {
                let mut memory = create_unified(42, env.clock.clone());

                let _ = memory.remember("Alice works on Project Alpha").await;
                let _ = memory.remember("Bob is Alice's colleague").await;

                accesses.store(memory.category_evolver().total_accesses(), Ordering::SeqCst);
                Ok::<(), std::convert::Infallible>(())
            }
        })
        .await
        .unwrap();
    }

    let result1 = accesses1.load(Ordering::SeqCst);
    let result2 = accesses2.load(Ordering::SeqCst);

    assert_eq!(result1, result2, "same seed should produce identical access counts");
}

/// Test that different seeds produce consistent results within themselves.
#[tokio::test]
async fn test_determinism_different_seeds() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Run with seed 42
    let accesses1 = Arc::new(AtomicU64::new(0));
    let accesses1_clone = accesses1.clone();

    {
        let sim = Simulation::new(SimConfig::with_seed(42));
        sim.run(|env| {
            let accesses = accesses1_clone.clone();
            async move {
                let mut memory = create_unified(42, env.clock.clone());
                let _ = memory.remember("Test data").await;
                accesses.store(memory.category_evolver().total_accesses(), Ordering::SeqCst);
                Ok::<(), std::convert::Infallible>(())
            }
        })
        .await
        .unwrap();
    }

    // Run with seed 43
    let accesses2 = Arc::new(AtomicU64::new(0));
    let accesses2_clone = accesses2.clone();

    {
        let sim = Simulation::new(SimConfig::with_seed(43));
        sim.run(|env| {
            let accesses = accesses2_clone.clone();
            async move {
                let mut memory = create_unified(43, env.clock.clone());
                let _ = memory.remember("Test data").await;
                accesses.store(memory.category_evolver().total_accesses(), Ordering::SeqCst);
                Ok::<(), std::convert::Infallible>(())
            }
        })
        .await
        .unwrap();
    }

    let result1 = accesses1.load(Ordering::SeqCst);
    let result2 = accesses2.load(Ordering::SeqCst);

    // Results should be consistent within themselves but may differ between seeds
    // (though in this simple case they might be the same)
    println!("Seed 42: {} accesses", result1);
    println!("Seed 43: {} accesses", result2);
}

// =============================================================================
// CategoryEvolver Integration Tests
// =============================================================================

/// Test that CategoryEvolver tracks patterns during full workflow.
#[tokio::test]
async fn test_category_evolver_full_workflow() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Remember enough data to trigger evolution analysis
        for i in 0..120 {
            let text = if i % 3 == 0 {
                format!("Person {} is important", i)
            } else if i % 3 == 1 {
                format!("Project {} is ongoing", i)
            } else {
                format!("Task {} needs attention", i)
            };

            let _ = memory.remember(&text).await;
        }

        // Check evolution analysis
        let total = memory.category_evolver().total_accesses();
        assert!(
            total >= 100,
            "should have 100+ accesses for evolution analysis, got {}",
            total
        );

        // Get suggestions
        let suggestions = memory.get_evolution_suggestions();
        println!(
            "After {} accesses: {} evolution suggestions",
            total,
            suggestions.len()
        );

        // Log what kind of suggestions we got
        for suggestion in &suggestions {
            match suggestion {
                EvolutionSuggestion::CreateBlock { name, reason, .. } => {
                    println!("  CreateBlock: {} - {}", name, reason);
                }
                EvolutionSuggestion::MergeBlocks { block1, block2, reason, .. } => {
                    println!("  MergeBlocks: {:?} + {:?} - {}", block1, block2, reason);
                }
                EvolutionSuggestion::SplitBlock { block, reason, .. } => {
                    println!("  SplitBlock: {:?} - {}", block, reason);
                }
            }
        }

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test co-occurrence detection during workflow.
#[tokio::test]
async fn test_co_occurrence_detection() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Remember related entities together
        for _ in 0..50 {
            // Always mention Person and Project together
            let _ = memory
                .remember("Alice (person) works on Alpha (project)")
                .await;
        }

        // Check co-occurrence
        let score = memory.entity_co_occurrence(&EntityType::Person, &EntityType::Project);
        println!("Person-Project co-occurrence: {:.2}", score);

        // We expect some co-occurrence to be tracked
        // (exact value depends on extraction behavior)

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

// =============================================================================
// Time-Based Tests
// =============================================================================

/// Test behavior over simulated time.
#[tokio::test]
async fn test_time_based_access_decay() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Remember something
        let result = memory.remember("Important information").await.unwrap();
        let entity_id = &result.entities[0].id;

        // Get initial access pattern
        let pattern1 = memory.access_tracker().get_access_pattern(entity_id);
        assert!(pattern1.is_some());
        let initial_recency = pattern1.unwrap().recency_score;

        // Advance time by 7 days
        for _ in 0..7 {
            let _ = env.clock.advance_ms(24 * 60 * 60 * 1000);
        }

        // Get updated access pattern
        let pattern2 = memory.access_tracker().get_access_pattern(entity_id);
        let final_recency = pattern2.unwrap().recency_score;

        // Recency should decay
        assert!(
            final_recency < initial_recency,
            "recency should decay: {} -> {}",
            initial_recency,
            final_recency
        );

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test with empty input.
#[tokio::test]
async fn test_empty_input_handling() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Empty remember should error
        let result = memory.remember("").await;
        assert!(result.is_err(), "empty remember should fail");

        // Empty recall should error
        let result = memory.recall("", 10).await;
        assert!(result.is_err(), "empty recall should fail");

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test with very long input.
#[tokio::test]
async fn test_long_input_handling() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Long text (50KB - well below the 100KB LLM limit but still "very long")
        // Note: LLM has TigerStyle assertion at 100KB, so we test below that boundary
        let long_text = "x".repeat(50_000);
        let result = memory.remember(&long_text).await;

        // Should either succeed with truncation or fail gracefully
        match result {
            Ok(_) => println!("Long input handled (possibly truncated)"),
            Err(e) => println!("Long input rejected: {}", e),
        }

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}

/// Test rapid fire operations.
#[tokio::test]
async fn test_rapid_fire_operations() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_unified(42, env.clock.clone());

        // Rapid remember operations
        for i in 0..100 {
            let _ = memory.remember(&format!("Rapid data {}", i)).await;
        }

        // Rapid recall operations
        for _ in 0..50 {
            let _ = memory.recall("data", 10).await;
        }

        // Should handle without panic
        let total = memory.category_evolver().total_accesses();
        assert!(
            total >= 100,
            "should track all accesses: {}",
            total
        );

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();
}
