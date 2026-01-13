//! Full Workflow DST Simulation
//!
//! This is a COMPREHENSIVE deterministic simulation that exercises
//! the entire UnifiedMemory system with fault injection.
//!
//! Run with: cargo run --example full_workflow_dst --all-features
//!
//! DST-First Approach:
//! 1. Run full lifecycle: remember → promote → recall → evict
//! 2. Inject faults at multiple layers
//! 3. Verify invariants throughout
//! 4. Hunt for bugs in real workflows

use std::sync::Arc;
use umi_memory::dst::{
    DeterministicRng, FaultConfig, FaultInjector, FaultType, SimConfig, Simulation,
};
use umi_memory::embedding::SimEmbeddingProvider;
use umi_memory::llm::SimLLMProvider;
use umi_memory::orchestration::{UnifiedMemory, UnifiedMemoryConfig};
use umi_memory::storage::{EntityType, SimStorageBackend, SimVectorBackend};

#[tokio::main]
async fn main() {
    println!("=== Full Workflow DST Simulation ===\n");
    println!("Running comprehensive simulation with fault injection...\n");

    // Run multiple simulation scenarios
    run_scenario_1_extended_lifecycle().await;
    run_scenario_2_cascading_faults().await;
    run_scenario_3_rapid_operations().await;
    run_scenario_4_memory_pressure_extended().await;
    run_scenario_5_multi_seed_verification().await;
    run_scenario_6_edge_cases_under_faults().await;

    println!("\n=== Simulation Complete ===");
}

/// Scenario 1: Extended Lifecycle (1000+ operations)
/// Hunt for: State corruption, memory leaks, inconsistent behavior
async fn run_scenario_1_extended_lifecycle() {
    println!("Scenario 1: Extended Lifecycle (1000+ operations)");
    println!("  Hunting for: State corruption, memory leaks");

    let sim = Simulation::new(SimConfig::with_seed(42));

    let result = sim
        .run(|env| async move {
            // 30% storage faults to stress the system
            let storage = SimStorageBackend::new(SimConfig::with_seed(42))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.3));

            let llm = SimLLMProvider::with_seed(42);
            let embedder = SimEmbeddingProvider::with_seed(42);
            let vector = SimVectorBackend::new(42);
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(20);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut total_remembers = 0u64;
            let mut total_recalls = 0u64;
            let mut total_promotions = 0u64;
            let mut total_evictions = 0u64;
            let mut max_core_count = 0usize;
            let mut max_access_count = 0u64;

            // Run 100 iterations of full lifecycle
            for iteration in 0..100 {
                // Phase 1: Remember (10 operations per iteration)
                for i in 0..10 {
                    let text = format!(
                        "Alice works with Bob on project {} iteration {}",
                        i, iteration
                    );
                    if memory.remember(&text).await.is_ok() {
                        total_remembers += 1;
                    }
                }

                // Advance time
                let _ = env.clock.advance_ms(500);

                // Phase 2: Recall
                let queries = ["Alice", "Bob", "project", "works"];
                for query in queries {
                    if memory.recall(query, 5).await.is_ok() {
                        total_recalls += 1;
                    }
                }

                // More time advancement
                let _ = env.clock.advance_ms(500);

                // Phase 3: Promote
                if let Ok(promoted) = memory.promote_to_core().await {
                    total_promotions += promoted as u64;
                }

                // Phase 4: Evict
                if let Ok(evicted) = memory.evict_from_core().await {
                    total_evictions += evicted as u64;
                }

                // Track maximums
                max_core_count = max_core_count.max(memory.core_entity_count());
                max_access_count = max_access_count.max(memory.category_evolver().total_accesses());

                // INVARIANT CHECK: Core should never exceed limit
                let core_count = memory.core_entity_count();
                if core_count > 20 {
                    println!(
                        "  BUG: Core count {} exceeds limit 20 at iteration {}",
                        core_count, iteration
                    );
                }
            }

            let final_core = memory.core_entity_count();
            let final_accesses = memory.category_evolver().total_accesses();

            println!("  1000 operations completed:");
            println!("    Remembers: {} successful", total_remembers);
            println!("    Recalls: {} successful", total_recalls);
            println!("    Promotions: {}, Evictions: {}", total_promotions, total_evictions);
            println!(
                "    Max core: {}/20, Final core: {}",
                max_core_count, final_core
            );
            println!(
                "    Max accesses: {}, Final accesses: {}",
                max_access_count, final_accesses
            );

            // Final invariant checks
            if final_core > 20 {
                println!("  BUG: Final core count {} exceeds limit", final_core);
            } else {
                println!("  PASS: Core limit respected throughout");
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG: Simulation crashed: {}", e);
    }
    println!();
}

/// Scenario 2: Cascading Faults (Storage + LLM)
/// Hunt for: Error handling gaps, state corruption under multiple fault types
async fn run_scenario_2_cascading_faults() {
    println!("Scenario 2: Cascading Faults (Storage 40% + LLM 40%)");
    println!("  Hunting for: Error handling gaps, state corruption");

    let sim = Simulation::new(SimConfig::with_seed(99));

    let result = sim
        .run(|env| async move {
            // Storage faults
            let storage = SimStorageBackend::new(SimConfig::with_seed(99))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.4));

            // LLM faults
            let mut injector = FaultInjector::new(DeterministicRng::new(99));
            injector.register(FaultConfig::new(FaultType::LlmTimeout, 0.4));
            let llm = SimLLMProvider::with_faults(99, Arc::new(injector));

            let embedder = SimEmbeddingProvider::with_seed(99);
            let vector = SimVectorBackend::new(99);
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(15);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut storage_errors = 0;
            let mut successes = 0;
            let mut note_fallbacks = 0;

            for i in 0..200 {
                match memory.remember(&format!("Carol manages team {} at Acme", i)).await {
                    Ok(result) => {
                        successes += 1;
                        // Check for LLM fallback (Note entities)
                        if result
                            .entities
                            .iter()
                            .any(|e| e.entity_type == EntityType::Note)
                        {
                            note_fallbacks += 1;
                        }
                    }
                    Err(_) => storage_errors += 1,
                }

                // Interleave recalls
                if i % 5 == 0 {
                    let _ = memory.recall("Carol", 3).await;
                    let _ = env.clock.advance_ms(100);
                }
            }

            // Try promotion and eviction under faults
            let _ = memory.promote_to_core().await;
            let _ = memory.evict_from_core().await;

            let final_accesses = memory.category_evolver().total_accesses();
            let final_core = memory.core_entity_count();

            println!("  200 operations with cascading faults:");
            println!(
                "    {} successes, {} storage errors",
                successes, storage_errors
            );
            println!("    {} used LLM fallback (Note entities)", note_fallbacks);
            println!(
                "    Final state: {} accesses, {} core entities",
                final_accesses, final_core
            );

            // INVARIANT: Access count should be bounded
            let max_expected = (successes * 4) as u64;
            if final_accesses > max_expected {
                println!(
                    "  BUG: Access count {} exceeds expected {} for {} successes",
                    final_accesses, max_expected, successes
                );
            } else {
                println!("  PASS: Access count bounded correctly");
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG: Simulation crashed: {}", e);
    }
    println!();
}

/// Scenario 3: Rapid Operations (Minimal Time Between)
/// Hunt for: Race conditions, timing bugs
async fn run_scenario_3_rapid_operations() {
    println!("Scenario 3: Rapid Operations (500 ops, 1ms intervals)");
    println!("  Hunting for: Race conditions, timing bugs");

    let sim = Simulation::new(SimConfig::with_seed(123));

    let result = sim
        .run(|env| async move {
            let storage = SimStorageBackend::new(SimConfig::with_seed(123))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.2));

            let llm = SimLLMProvider::with_seed(123);
            let embedder = SimEmbeddingProvider::with_seed(123);
            let vector = SimVectorBackend::new(123);
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(30);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut success_count = 0;
            let mut prev_accesses = 0u64;
            let mut access_decreased = false;

            for i in 0..500 {
                // Rapid remember
                if memory
                    .remember(&format!("Rapid entity {}", i))
                    .await
                    .is_ok()
                {
                    success_count += 1;
                }

                // Minimal time advance
                let _ = env.clock.advance_ms(1);

                // Track access count - should never decrease
                let current_accesses = memory.category_evolver().total_accesses();
                if current_accesses < prev_accesses {
                    println!(
                        "  BUG: Access count decreased from {} to {} at op {}",
                        prev_accesses, current_accesses, i
                    );
                    access_decreased = true;
                }
                prev_accesses = current_accesses;

                // Rapid recall every 10 ops
                if i % 10 == 0 {
                    let _ = memory.recall("Rapid", 5).await;
                }

                // Promotion/eviction every 50 ops
                if i % 50 == 0 {
                    let _ = memory.promote_to_core().await;
                    let _ = memory.evict_from_core().await;
                }
            }

            let final_core = memory.core_entity_count();
            let final_accesses = memory.category_evolver().total_accesses();

            println!("  500 rapid operations completed:");
            println!("    {} successful remembers", success_count);
            println!(
                "    Final state: {} core, {} accesses",
                final_core, final_accesses
            );

            if access_decreased {
                println!("  BUG: Access count decreased during operation");
            } else {
                println!("  PASS: Access count monotonically increased");
            }

            if final_core > 30 {
                println!("  BUG: Core count {} exceeds limit 30", final_core);
            } else {
                println!("  PASS: Core limit respected");
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG: Simulation crashed: {}", e);
    }
    println!();
}

/// Scenario 4: Extended Memory Pressure
/// Hunt for: Memory leaks, eviction failures under sustained load
async fn run_scenario_4_memory_pressure_extended() {
    println!("Scenario 4: Extended Memory Pressure (500 ops, limit=5)");
    println!("  Hunting for: Memory leaks, eviction failures");

    let sim = Simulation::new(SimConfig::with_seed(456));

    let result = sim
        .run(|env| async move {
            let storage = SimStorageBackend::new(SimConfig::with_seed(456))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.3));

            let llm = SimLLMProvider::with_seed(456);
            let embedder = SimEmbeddingProvider::with_seed(456);
            let vector = SimVectorBackend::new(456);

            // Very small limit to force continuous eviction
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(5);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let mut max_core_ever = 0usize;
            let mut core_exceeded_limit = false;
            let mut total_evictions = 0u64;
            let mut total_promotions = 0u64;

            // Names that SimLLM actually extracts
            let common_names = ["Alice", "Bob", "Charlie", "David", "Eve",
                               "Acme", "Google", "Microsoft", "Apple", "Amazon"];

            for i in 0..500 {
                // Use text that will extract real entity names
                let name1 = common_names[i % 10];
                let name2 = common_names[(i + 5) % 10];
                let _ = memory
                    .remember(&format!("{} works at {}", name1, name2))
                    .await;

                // Track core count
                let core = memory.core_entity_count();
                max_core_ever = max_core_ever.max(core);
                if core > 5 {
                    println!(
                        "  BUG: Core {} exceeded limit 5 at iteration {}",
                        core, i
                    );
                    core_exceeded_limit = true;
                }

                // Aggressive promotion/eviction with MATCHING queries
                if i % 3 == 0 {
                    let _ = env.clock.advance_ms(200);
                    // Recall with actual entity name to build access history
                    let query_name = common_names[i % 10];
                    let _ = memory.recall(query_name, 3).await;
                    let _ = env.clock.advance_ms(200);
                    if let Ok(promoted) = memory.promote_to_core().await {
                        total_promotions += promoted as u64;
                    }
                    if let Ok(evicted) = memory.evict_from_core().await {
                        total_evictions += evicted as u64;
                    }
                }
            }

            let final_core = memory.core_entity_count();
            let final_accesses = memory.category_evolver().total_accesses();

            println!("  500 operations with limit=5:");
            println!("    Max core ever: {}, Final core: {}", max_core_ever, final_core);
            println!("    Total promotions: {}, Total evictions: {}", total_promotions, total_evictions);
            println!("    Final accesses: {}", final_accesses);

            if core_exceeded_limit {
                println!("  BUG: Core limit was exceeded during operation");
            } else {
                println!("  PASS: Core limit always respected");
            }

            if total_promotions == 0 {
                println!("  WARNING: No promotions occurred - check access patterns");
            }

            Ok::<(), anyhow::Error>(())
        })
        .await;

    if let Err(e) = result {
        println!("  BUG: Simulation crashed: {}", e);
    }
    println!();
}

/// Scenario 5: Multi-Seed Verification
/// Hunt for: Non-determinism, seed-dependent bugs
async fn run_scenario_5_multi_seed_verification() {
    println!("Scenario 5: Multi-Seed Verification (10 seeds)");
    println!("  Hunting for: Non-determinism, seed-dependent bugs");

    let seeds = [1, 42, 100, 256, 1000, 5555, 12345, 99999, 1000000, 7777777];
    let mut all_deterministic = true;

    for seed in seeds {
        let result1 = run_deterministic_check(seed).await;
        let result2 = run_deterministic_check(seed).await;

        if result1 != result2 {
            println!(
                "  BUG: Non-deterministic with seed {}: {:?} vs {:?}",
                seed, result1, result2
            );
            all_deterministic = false;
        }
    }

    if all_deterministic {
        println!("  PASS: All 10 seeds produce deterministic results");
    } else {
        println!("  BUG: Non-determinism detected");
    }
    println!();
}

#[derive(Debug, PartialEq)]
struct DeterministicResult {
    successes: u64,
    final_accesses: u64,
    final_core: usize,
}

async fn run_deterministic_check(seed: u64) -> DeterministicResult {
    let sim = Simulation::new(SimConfig::with_seed(seed));

    let successes = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let accesses = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let core = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let s = successes.clone();
    let a = accesses.clone();
    let c = core.clone();

    sim.run(|env| async move {
        let storage = SimStorageBackend::new(SimConfig::with_seed(seed))
            .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.3));

        let llm = SimLLMProvider::with_seed(seed);
        let embedder = SimEmbeddingProvider::with_seed(seed);
        let vector = SimVectorBackend::new(seed);
        let config = UnifiedMemoryConfig::new().with_core_entity_limit(10);

        let mut memory =
            UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

        for i in 0..50 {
            if memory.remember(&format!("Test {}", i)).await.is_ok() {
                s.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            let _ = env.clock.advance_ms(10);

            if i % 10 == 0 {
                let _ = memory.promote_to_core().await;
            }
        }

        a.store(
            memory.category_evolver().total_accesses(),
            std::sync::atomic::Ordering::SeqCst,
        );
        c.store(
            memory.core_entity_count(),
            std::sync::atomic::Ordering::SeqCst,
        );

        Ok::<(), std::convert::Infallible>(())
    })
    .await
    .unwrap();

    DeterministicResult {
        successes: successes.load(std::sync::atomic::Ordering::SeqCst),
        final_accesses: accesses.load(std::sync::atomic::Ordering::SeqCst),
        final_core: core.load(std::sync::atomic::Ordering::SeqCst),
    }
}

/// Scenario 6: Edge Cases Under Faults
/// Hunt for: Crashes, panics, unexpected behavior with edge inputs
async fn run_scenario_6_edge_cases_under_faults() {
    println!("Scenario 6: Edge Cases Under Faults");
    println!("  Hunting for: Crashes, panics with edge inputs");

    let sim = Simulation::new(SimConfig::with_seed(789));

    let result = sim
        .run(|env| async move {
            let storage = SimStorageBackend::new(SimConfig::with_seed(789))
                .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.5));

            // LLM with rate limits
            let mut injector = FaultInjector::new(DeterministicRng::new(789));
            injector.register(FaultConfig::new(FaultType::LlmRateLimit, 0.3));
            let llm = SimLLMProvider::with_faults(789, Arc::new(injector));

            let embedder = SimEmbeddingProvider::with_seed(789);
            let vector = SimVectorBackend::new(789);
            let config = UnifiedMemoryConfig::new().with_core_entity_limit(10);

            let mut memory =
                UnifiedMemory::new(llm, embedder, vector, storage, env.clock.clone(), config);

            let very_long_input = "x".repeat(10000);
            let edge_cases = vec![
                ("empty", ""),
                ("whitespace", "   \t\n  "),
                ("unicode", "日本語テスト 🎉 émojis"),
                ("special chars", "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"),
                ("very long", very_long_input.as_str()),
                ("newlines", "line1\nline2\nline3"),
                ("tabs", "col1\tcol2\tcol3"),
                ("mixed", "Alice 日本語 @#$ works at Acme 🏢"),
                ("numbers", "12345 67890 3.14159"),
                ("null-like", "null undefined NaN"),
            ];

            let mut successes = 0;
            let mut failures = 0;

            for (name, input) in edge_cases {
                match memory.remember(input).await {
                    Ok(_) => {
                        successes += 1;
                        println!("    {} -> OK", name);
                    }
                    Err(e) => {
                        failures += 1;
                        println!("    {} -> Error: {}", name, e);
                    }
                }
                let _ = env.clock.advance_ms(50);
            }

            // Recall with edge cases
            let long_query = "x".repeat(100);
            let recall_tests = ["", "日本語", "🎉", "@#$", long_query.as_str()];
            let mut recall_successes = 0;

            for query in recall_tests {
                if memory.recall(query, 5).await.is_ok() {
                    recall_successes += 1;
                }
            }

            println!(
                "  Remember: {} ok, {} errors",
                successes, failures
            );
            println!("  Recall edge cases: {} ok", recall_successes);

            // Final state check
            let accesses = memory.category_evolver().total_accesses();
            let core = memory.core_entity_count();
            println!(
                "  Final state: {} accesses, {} core (no crashes!)",
                accesses, core
            );

            Ok::<(), anyhow::Error>(())
        })
        .await;

    match result {
        Ok(_) => println!("  PASS: All edge cases handled without crashing"),
        Err(e) => println!("  BUG: Simulation crashed: {}", e),
    }
    println!();
}
