// Aggressive DST Performance Bug Hunt
// Goal: Actually TRY to break the system, not just verify invariants

use std::sync::Arc;
use umi_memory::dst::{
    DeterministicRng, FaultConfig, FaultInjector, FaultType, SimConfig, Simulation,
};
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::orchestration::{UnifiedMemory, UnifiedMemoryConfig};
use umi_memory::storage::{SimStorageBackend, SimVectorBackend};

#[tokio::main]
async fn main() {
    println!("=== Aggressive DST Performance Bug Hunt ===\n");

    // Test 1: Extreme fault rate (90%) - does the system degrade gracefully or crash?
    println!("Test 1: 90% storage fault rate - hunting for crashes or hangs");
    test_extreme_fault_rate().await;

    // Test 2: Cascading faults (storage + LLM simultaneously)
    println!("\nTest 2: Cascading faults (storage 50% + LLM 50%) - hunting for state corruption");
    test_cascading_faults().await;

    // Test 3: Rapid operations with faults - race conditions?
    println!("\nTest 3: 500 rapid operations with 40% faults - hunting for race conditions");
    test_rapid_operations_under_faults().await;

    // Test 4: Alternating fault states - does recovery work correctly?
    println!("\nTest 4: Alternating fault on/off - hunting for lingering corruption");
    test_alternating_faults().await;

    // Test 5: Memory pressure with faults - eviction under stress
    println!("\nTest 5: Memory pressure + 60% faults - hunting for memory leaks");
    test_memory_pressure_with_faults().await;

    println!("\n=== Done ===");
}

async fn test_extreme_fault_rate() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    let result = sim
        .run(|env| async move {
            // 90% fault rate - almost always fails
            let storage = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.9));

            let llm = SimLLMProvider::with_seed(42);
            let embedder = SimEmbeddingProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let config = UnifiedMemoryConfig::default();

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut successes = 0;
            let mut failures = 0;

            for i in 0..100 {
                match memory.remember(&format!("Test {}", i)).await {
                    Ok(_) => successes += 1,
                    Err(_) => failures += 1,
                }
                let _ = env.clock.advance_ms(5);
            }

            // With 90% fault and 2 stores per op: P(success) = 0.1^2 = 1%
            let success_rate = successes as f64 / 100.0;
            println!(
                "  90% fault rate: {} successes, {} failures (rate={:.1}%)",
                successes,
                failures,
                success_rate * 100.0
            );

            // BUG CHECK: Did any operation hang? (we completed all 100)
            println!("  All 100 operations completed (no hangs)");

            // BUG CHECK: Is state consistent after extreme faults?
            let accesses = memory.category_evolver().total_accesses();
            let core_count = memory.core_entity_count();
            println!(
                "  State: {} accesses, {} core entities",
                accesses, core_count
            );

            // Verify we can still recall after extreme faults
            let recall_result = memory.recall("Test", 5).await;
            match recall_result {
                Ok(entities) => println!("  Recall works: {} entities found", entities.len()),
                Err(e) => println!("  BUG? Recall failed: {}", e),
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG FOUND: Test crashed: {}", e);
    }
}

async fn test_cascading_faults() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    let result = sim
        .run(|env| async move {
            // Storage faults
            let storage = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.5));

            // LLM faults
            let mut injector = FaultInjector::new(DeterministicRng::new(42));
            injector.register(FaultConfig::new(FaultType::LlmTimeout, 0.5));
            let llm = SimLLMProvider::with_faults(42, Arc::new(injector));

            let embedder = SimEmbeddingProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let config = UnifiedMemoryConfig::default();

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut storage_errors = 0;
            let mut successes = 0;

            for i in 0..50 {
                match memory.remember(&format!("Alice cascading {}", i)).await {
                    Ok(result) => {
                        successes += 1;
                        // Check if we got fallback Notes (LLM failed)
                        let has_note = result
                            .entities
                            .iter()
                            .any(|e| format!("{:?}", e.entity_type).contains("Note"));
                        if has_note && i < 5 {
                            println!("  Op {}: LLM failed -> Note fallback", i);
                        }
                    }
                    Err(_) => storage_errors += 1,
                }
                let _ = env.clock.advance_ms(10);
            }

            println!(
                "  Cascading faults: {} successes, {} storage errors",
                successes, storage_errors
            );

            // BUG CHECK: State should be consistent
            let accesses = memory.category_evolver().total_accesses();

            // With 50% storage fault and 2 stores: ~25% success
            // With additional LLM faults, successful ones may have degraded quality
            println!(
                "  Final state: {} accesses after {} successes",
                accesses, successes
            );

            // BUG CHECK: Access count should match successful operations
            // Each success creates ~2 accesses
            let max_expected = (successes as u64) * 4;
            if accesses > max_expected {
                println!(
                    "  POTENTIAL BUG: {} accesses > {} expected for {} successes",
                    accesses, max_expected, successes
                );
            } else {
                println!("  Access count is bounded correctly");
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG FOUND: Test crashed: {}", e);
    }
}

async fn test_rapid_operations_under_faults() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    let result = sim
        .run(|env| async move {
            let storage = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.4));

            let llm = SimLLMProvider::with_seed(42);
            let embedder = SimEmbeddingProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(20);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut successes = 0;
            let mut failures = 0;

            // 500 rapid operations with minimal time between
            for i in 0..500 {
                match memory.remember(&format!("Rapid {}", i)).await {
                    Ok(_) => successes += 1,
                    Err(_) => failures += 1,
                }

                // Minimal time advance - stress test
                let _ = env.clock.advance_ms(1);

                // Interleave recalls
                if i % 10 == 0 {
                    let _ = memory.recall("Rapid", 5).await;
                }

                // Periodic promotion/eviction
                if i % 50 == 0 {
                    let _ = env.clock.advance_ms(100);
                    let _ = memory.promote_to_core().await;
                    let _ = memory.evict_from_core().await;
                }
            }

            println!(
                "  500 rapid ops: {} successes, {} failures",
                successes, failures
            );

            // BUG CHECK: Core count should be bounded
            let core_count = memory.core_entity_count();
            if core_count > 20 {
                println!("  BUG: Core count {} exceeds limit 20", core_count);
            } else {
                println!("  Core count bounded: {}/20", core_count);
            }

            // BUG CHECK: Total accesses should be reasonable
            let accesses = memory.category_evolver().total_accesses();
            println!("  Total accesses: {}", accesses);

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG FOUND: Test crashed: {}", e);
    }
}

async fn test_alternating_faults() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    let result = sim
        .run(|env| async move {
            // Phase 1: Heavy faults (80%)
            let storage_faulty = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.8));

            let llm = SimLLMProvider::with_seed(42);
            let embedder = SimEmbeddingProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let config = UnifiedMemoryConfig::default();

            let mut memory_faulty = UnifiedMemory::new(
                llm.clone(),
                embedder.clone(),
                vector.clone(),
                storage_faulty,
                env.clock.clone(),
                config.clone(),
            );

            let mut phase1_failures = 0;
            for i in 0..30 {
                if memory_faulty
                    .remember(&format!("Faulty {}", i))
                    .await
                    .is_err()
                {
                    phase1_failures += 1;
                }
                let _ = env.clock.advance_ms(10);
            }
            println!("  Phase 1 (80% faults): {} failures / 30", phase1_failures);

            // Phase 2: No faults - should recover completely
            let storage_healthy = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory_healthy = UnifiedMemory::new(
                llm.clone(),
                embedder.clone(),
                vector.clone(),
                storage_healthy,
                env.clock.clone(),
                config.clone(),
            );

            let mut phase2_failures = 0;
            for i in 0..30 {
                if memory_healthy
                    .remember(&format!("Healthy {}", i))
                    .await
                    .is_err()
                {
                    phase2_failures += 1;
                }
                let _ = env.clock.advance_ms(10);
            }
            println!("  Phase 2 (0% faults): {} failures / 30", phase2_failures);

            // BUG CHECK: Phase 2 should have 0 failures
            if phase2_failures > 0 {
                println!(
                    "  BUG: {} failures in healthy phase - lingering corruption?",
                    phase2_failures
                );
            } else {
                println!("  Recovery verified: 0 failures after fault period");
            }

            // Phase 3: Faults again (50%)
            let storage_partial = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.5));
            let mut memory_partial = UnifiedMemory::new(
                llm,
                embedder,
                vector,
                storage_partial,
                env.clock.clone(),
                config,
            );

            let mut phase3_successes = 0;
            for i in 0..30 {
                if memory_partial
                    .remember(&format!("Partial {}", i))
                    .await
                    .is_ok()
                {
                    phase3_successes += 1;
                }
                let _ = env.clock.advance_ms(10);
            }
            println!(
                "  Phase 3 (50% faults): {} successes / 30",
                phase3_successes
            );

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG FOUND: Test crashed: {}", e);
    }
}

async fn test_memory_pressure_with_faults() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    let result = sim
        .run(|env| async move {
            let storage = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.6));

            let llm = SimLLMProvider::with_seed(42);
            let embedder = SimEmbeddingProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);

            // Very small limits to force pressure
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(5);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut max_core = 0usize;
            let mut evictions = 0;

            for i in 0..200 {
                let _ = memory.remember(&format!("Pressure {}", i)).await;

                // Track max core size
                let core = memory.core_entity_count();
                max_core = max_core.max(core);

                // Force promotion/eviction frequently
                if i % 5 == 0 {
                    let _ = env.clock.advance_ms(500);
                    let _ = memory.recall("Pressure", 3).await;
                    let _ = env.clock.advance_ms(500);
                    let _ = memory.promote_to_core().await;
                    let evicted = memory.evict_from_core().await.unwrap_or(0);
                    evictions += evicted;
                }
            }

            let final_core = memory.core_entity_count();
            let accesses = memory.category_evolver().total_accesses();

            println!("  200 ops with 60% faults and limit=5:");
            println!("    Final core: {}, max core: {}", final_core, max_core);
            println!("    Total evictions: {}", evictions);
            println!("    Total accesses: {}", accesses);

            // BUG CHECK: Core should never exceed limit
            if max_core > 5 {
                println!("  BUG: Max core {} exceeded limit 5", max_core);
            } else {
                println!("  Core limit respected throughout");
            }

            // BUG CHECK: Should have some evictions given the pressure
            if evictions == 0 && max_core >= 5 {
                println!("  WARNING: No evictions despite reaching limit");
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG FOUND: Test crashed: {}", e);
    }
}
