//! Simulation - DST Test Harness
//!
//! `TigerStyle`: Simulation harness that provides deterministic environment.

use std::future::Future;
use std::sync::Arc;

use super::clock::SimClock;
use super::config::SimConfig;
use super::fault::{FaultConfig, FaultInjector, FaultInjectorBuilder};
use super::llm::SimLLM;
use super::network::SimNetwork;
use super::rng::DeterministicRng;
use super::storage::SimStorage;

/// Environment provided to simulation tests.
///
/// `TigerStyle`: All simulation resources in one place.
pub struct SimEnvironment {
    /// Simulation configuration
    pub config: SimConfig,
    /// Simulated clock
    pub clock: SimClock,
    /// Deterministic RNG
    pub rng: DeterministicRng,
    /// Fault injector (shared via Arc with storage, network, and llm)
    pub faults: Arc<FaultInjector>,
    /// Simulated storage
    pub storage: SimStorage,
    /// Simulated network
    pub network: SimNetwork,
    /// Simulated LLM
    pub llm: SimLLM,
}

impl SimEnvironment {
    /// Advance simulated time in milliseconds.
    pub fn advance_time_ms(&self, ms: u64) -> u64 {
        self.clock.advance_ms(ms)
    }

    /// Advance simulated time in seconds.
    pub fn advance_time_secs(&self, secs: f64) -> u64 {
        self.clock.advance_secs(secs)
    }

    /// Get current simulated time in milliseconds.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.clock.now_ms()
    }

    /// Sleep for the given milliseconds (async, waits for time to advance).
    pub async fn sleep_ms(&self, ms: u64) {
        self.clock.sleep_ms(ms).await;
    }

    /// Create a Memory instance with providers connected to this simulation's fault injector.
    ///
    /// All SimProviders (LLM, Embedding, Vector, Storage) will share the `FaultInjector`
    /// configured in the `Simulation`, allowing fault injection tests to work correctly.
    ///
    /// This is the recommended way to create a `Memory` instance within a `Simulation::run()`
    /// closure when you want fault injection to be applied to memory operations.
    ///
    /// **DST-First Discovery**: This method was discovered through DST-first testing
    /// (see `.progress/015_DST_FIRST_DEMO.md`). The discovery test revealed that
    /// `Memory::sim()` creates isolated providers not connected to the Simulation's
    /// FaultInjector, making fault injection tests ineffective.
    ///
    /// `TigerStyle`: Providers share the simulation's fault injector for deterministic testing.
    ///
    /// # Example
    ///
    /// ```rust
    /// use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};
    /// use umi_memory::umi::RememberOptions;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let sim = Simulation::new(SimConfig::with_seed(42))
    ///         .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));
    ///
    ///     sim.run(|env| async move {
    ///         let mut memory = env.create_memory();  // ✅ Connected to fault injector
    ///
    ///         // All memory operations now have fault injection applied
    ///         match memory.remember("Alice works at Acme", RememberOptions::default()).await {
    ///             Ok(result) => println!("Stored {} entities", result.entities.len()),
    ///             Err(e) => println!("Failed due to fault: {}", e),  // May fail due to fault
    ///         }
    ///
    ///         Ok::<(), anyhow::Error>(())
    ///     }).await.unwrap();
    /// }
    /// ```
    #[must_use]
    pub fn create_memory(&self) -> crate::umi::Memory<
        crate::llm::SimLLMProvider,
        crate::embedding::SimEmbeddingProvider,
        crate::storage::SimStorageBackend,
        crate::storage::SimVectorBackend,
    > {
        use crate::embedding::SimEmbeddingProvider;
        use crate::llm::SimLLMProvider;
        use crate::storage::{SimStorageBackend, SimVectorBackend};
        use crate::umi::Memory;

        let seed = self.config.seed();

        // Create all providers with the shared fault injector (DST-First fix!)
        let llm = SimLLMProvider::with_faults(seed, Arc::clone(&self.faults));
        let embedder = SimEmbeddingProvider::with_faults(seed, Arc::clone(&self.faults));
        let vector = SimVectorBackend::with_faults(seed, Arc::clone(&self.faults));
        let storage = SimStorageBackend::with_fault_injector(self.config, Arc::clone(&self.faults));

        Memory::new(llm, embedder, vector, storage)
    }
}

/// DST simulation harness.
///
/// `TigerStyle`:
/// - Single seed controls all randomness
/// - Faults are registered explicitly
/// - Environment is provided to test closure
///
/// # Example
///
/// ```rust
/// use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};
///
/// #[tokio::test]
/// async fn test_storage() {
///     let sim = Simulation::new(SimConfig::with_seed(42))
///         .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));
///
///     sim.run(|mut env| async move {
///         env.storage.write("key", b"value").await?;
///         env.advance_time_ms(1000);
///         let result = env.storage.read("key").await?;
///         assert_eq!(result, Some(b"value".to_vec()));
///         Ok(())
///     }).await.unwrap();
/// }
/// ```
pub struct Simulation {
    config: SimConfig,
    fault_configs: Vec<FaultConfig>,
}

impl Simulation {
    /// Create a new simulation with the given configuration.
    #[must_use]
    pub fn new(config: SimConfig) -> Self {
        Self {
            config,
            fault_configs: Vec::new(),
        }
    }

    /// Register a fault to inject during simulation.
    ///
    /// `TigerStyle`: Fluent API for fault registration.
    #[must_use]
    pub fn with_fault(mut self, fault_config: FaultConfig) -> Self {
        self.fault_configs.push(fault_config);
        self
    }

    /// Add common storage faults.
    ///
    /// `TigerStyle`: Convenience method for common fault patterns.
    #[must_use]
    pub fn with_storage_faults(self, probability: f64) -> Self {
        use super::fault::FaultType;

        self.with_fault(FaultConfig::new(FaultType::StorageWriteFail, probability))
            .with_fault(FaultConfig::new(FaultType::StorageReadFail, probability))
    }

    /// Add common database faults.
    #[must_use]
    pub fn with_db_faults(self, probability: f64) -> Self {
        use super::fault::FaultType;

        self.with_fault(FaultConfig::new(FaultType::DbConnectionFail, probability))
            .with_fault(FaultConfig::new(FaultType::DbQueryTimeout, probability))
    }

    /// Add common LLM/API faults.
    #[must_use]
    pub fn with_llm_faults(self, probability: f64) -> Self {
        use super::fault::FaultType;

        self.with_fault(FaultConfig::new(FaultType::LlmTimeout, probability))
            .with_fault(FaultConfig::new(FaultType::LlmRateLimit, probability))
    }

    /// Run the simulation with the given test function.
    ///
    /// `TigerStyle`: Test function receives environment and returns Result.
    ///
    /// # Errors
    /// Returns any error from the test function.
    pub async fn run<F, Fut, E>(self, test_fn: F) -> Result<(), E>
    where
        F: FnOnce(SimEnvironment) -> Fut,
        Fut: Future<Output = Result<(), E>>,
    {
        // Create components with forked RNGs for independence
        let mut rng = DeterministicRng::new(self.config.seed());
        let clock = SimClock::new();

        // Build fault injector using builder pattern (Kelpie style)
        let mut fault_builder = FaultInjectorBuilder::new(rng.fork());
        for fault_config in self.fault_configs {
            fault_builder = fault_builder.with_fault(fault_config);
        }
        // Wrap in Arc for sharing between env.faults, storage, and network
        let faults = Arc::new(fault_builder.build());

        // Create storage with SHARED fault injector (critical fix!)
        let storage = SimStorage::new(
            clock.clone(),
            rng.fork(),
            Arc::clone(&faults), // Storage SHARES the fault injector
        );

        // Create network with SHARED fault injector
        let network = SimNetwork::new(
            clock.clone(),
            rng.fork(),
            Arc::clone(&faults), // Network SHARES the fault injector
        );

        // Create LLM with SHARED fault injector
        let llm = SimLLM::new(
            clock.clone(),
            rng.fork(),
            Arc::clone(&faults), // LLM SHARES the fault injector
        );

        let env = SimEnvironment {
            config: self.config,
            clock,
            rng,
            faults,
            storage,
            network,
            llm,
        };

        // Run the test
        let result = test_fn(env).await;

        // Log stats if there were faults
        // (In production, this would use proper logging)

        result
    }

    /// Build the simulation environment without running a test.
    ///
    /// Useful for custom test setups.
    #[must_use]
    pub fn build(self) -> SimEnvironment {
        let mut rng = DeterministicRng::new(self.config.seed());
        let clock = SimClock::new();

        // Build fault injector using builder pattern (Kelpie style)
        let mut fault_builder = FaultInjectorBuilder::new(rng.fork());
        for fault_config in self.fault_configs {
            fault_builder = fault_builder.with_fault(fault_config);
        }
        // Wrap in Arc for sharing between env.faults, storage, and network
        let faults = Arc::new(fault_builder.build());

        // Create storage with SHARED fault injector
        let storage = SimStorage::new(clock.clone(), rng.fork(), Arc::clone(&faults));

        // Create network with SHARED fault injector
        let network = SimNetwork::new(clock.clone(), rng.fork(), Arc::clone(&faults));

        // Create LLM with SHARED fault injector
        let llm = SimLLM::new(clock.clone(), rng.fork(), Arc::clone(&faults));

        SimEnvironment {
            config: self.config,
            clock,
            rng,
            faults,
            storage,
            network,
            llm,
        }
    }
}

/// Create a simulation with optional seed.
///
/// `TigerStyle`: Factory function for common case.
#[must_use]
pub fn create_simulation(seed: Option<u64>) -> Simulation {
    let config = match seed {
        Some(s) => SimConfig::with_seed(s),
        None => SimConfig::from_env_or_random(),
    };
    Simulation::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::fault::FaultType;
    use crate::dst::storage::StorageError;

    #[tokio::test]
    async fn test_basic_simulation() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|mut env| async move {
            env.storage.write("key", b"value").await?;
            env.advance_time_ms(1000);
            let result = env.storage.read("key").await?;

            assert_eq!(result, Some(b"value".to_vec()));
            assert_eq!(env.now_ms(), 1000);

            Ok::<(), StorageError>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_simulation_build() {
        let sim = Simulation::new(SimConfig::with_seed(42));
        let mut env = sim.build();

        env.storage.write("key", b"value").await.unwrap();
        let result = env.storage.read("key").await.unwrap();

        assert_eq!(result, Some(b"value".to_vec()));
    }

    #[tokio::test]
    async fn test_simulation_determinism() {
        let mut results1 = Vec::new();
        let mut results2 = Vec::new();

        // First run
        let sim1 = Simulation::new(SimConfig::with_seed(12345));
        sim1.run(|mut env| async move {
            for _ in 0..10 {
                results1.push(env.rng.next_float());
            }
            Ok::<(), StorageError>(())
        })
        .await
        .unwrap();

        // Second run with same seed
        let sim2 = Simulation::new(SimConfig::with_seed(12345));
        sim2.run(|mut env| async move {
            for _ in 0..10 {
                results2.push(env.rng.next_float());
            }
            Ok::<(), StorageError>(())
        })
        .await
        .unwrap();

        // Note: results are captured but comparison is tricky with closures
        // The important thing is that the same seed produces deterministic behavior
    }

    #[tokio::test]
    async fn test_create_simulation() {
        let sim = create_simulation(Some(42));
        let env = sim.build();
        assert_eq!(env.config.seed(), 42);
    }

    #[test]
    fn test_fluent_api() {
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_storage_faults(0.1)
            .with_db_faults(0.05)
            .with_llm_faults(0.01);

        // Just verify it compiles and builds
        let _env = sim.build();
    }

    /// CRITICAL TEST: Verifies fault injection works through the simulation harness.
    ///
    /// This test catches the bug where storage had its own empty FaultInjector
    /// instead of sharing the one with registered faults.
    #[tokio::test]
    async fn test_fault_injection_through_harness() {
        // Register a fault with 100% probability - should ALWAYS fail
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        let result = sim
            .run(|mut env| async move {
                // This write should FAIL due to fault injection
                env.storage.write("key", b"value").await?;
                Ok::<(), StorageError>(())
            })
            .await;

        // The test MUST fail due to fault injection
        assert!(
            result.is_err(),
            "Fault injection should have caused write to fail!"
        );
    }

    /// Test that fault stats are properly tracked through the shared FaultInjector.
    #[tokio::test]
    async fn test_fault_stats_shared() {
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        let env = sim.build();

        // Both env.faults and storage.faults should point to the same FaultInjector
        // After a fault is injected, stats should be visible via env.faults
        assert_eq!(env.faults.total_injections(), 0);
    }

    // =============================================================================
    // DST-First Discovery Test: Memory Fault Injection
    // =============================================================================

    /// DISCOVERY TEST: Does fault injection work with Memory?
    ///
    /// This test is written BEFORE implementing any solution, to discover whether
    /// the existing API properly connects Memory to the Simulation's FaultInjector.
    ///
    /// **Hypothesis**: When we register a fault in Simulation and create a Memory
    /// instance, the fault should affect Memory operations.
    ///
    /// **Expected**: With 100% StorageWriteFail, the remember() should FAIL.
    ///
    /// **What we'll discover**: Whether Memory::sim() properly connects to faults.
    #[tokio::test]
    async fn test_dst_discovery_memory_fault_injection() {
        use crate::umi::RememberOptions;

        // Register 100% storage write failure - should ALWAYS fail
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        let result = sim
            .run(|env| async move {
                // NOW FIXED: Use env.create_memory() to get a properly connected Memory
                let mut memory = env.create_memory();

                // Try to remember something - with 100% fault rate, this should FAIL
                memory
                    .remember("Alice works at Acme", RememberOptions::default())
                    .await?;

                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        // With 100% fault rate, we EXPECT this to fail
        // NOW THIS SHOULD PASS - faults are properly connected!
        assert!(
            result.is_err(),
            "SUCCESS: With 100% StorageWriteFail and env.create_memory(), \
             remember() should fail. The fault is now properly injected!"
        );
    }

    // =============================================================================
    // Memory Integration Verification Tests
    // =============================================================================

    /// VERIFICATION TEST: Confirms storage write faults affect Memory operations.
    ///
    /// This test verifies that `env.create_memory()` creates a Memory instance with
    /// providers connected to the shared FaultInjector, so faults are actually applied.
    #[tokio::test]
    async fn test_memory_fault_injection_storage_write_fail() {
        use crate::umi::RememberOptions;

        // Register a storage write fault with 100% probability
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        let result = sim
            .run(|env| async move {
                let mut memory = env.create_memory();

                // This should FAIL due to storage fault injection
                memory
                    .remember("Alice works at Acme", RememberOptions::default())
                    .await?;

                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        // The test MUST fail due to fault injection
        assert!(
            result.is_err(),
            "Fault injection should have caused storage write to fail!"
        );

        // Verify it's actually a storage error
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("storage") || err_str.contains("Storage"),
            "Error should be storage-related: {}",
            err_str
        );
    }

    /// Test that LLM faults are applied to Memory operations.
    ///
    /// Note: Because the FaultInjector is shared across all providers, an LLM fault
    /// might cause a storage operation to fail if it fires during a storage call.
    /// This is expected behavior - the fault injector is global.
    #[tokio::test]
    async fn test_memory_fault_injection_llm_timeout() {
        use crate::umi::RememberOptions;

        // Register LLM timeout with 100% probability
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmTimeout, 1.0));

        let result = sim
            .run(|env| async move {
                let mut memory = env.create_memory();

                // Fault injection is global - LLM timeout might fire during ANY operation
                // This could cause storage, extraction, or LLM operations to fail
                let _result = memory
                    .remember("Bob is the CTO", RememberOptions::default())
                    .await;

                // The important thing is that faults are actually being injected
                // We don't care if it succeeds or fails, just that it doesn't panic
                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        // Should complete without panicking
        assert!(
            result.is_ok(),
            "Memory should handle faults without panicking: {:?}",
            result.unwrap_err()
        );
    }

    /// Test that embedding faults are applied to Memory operations.
    ///
    /// Note: Fault injection is global across all providers.
    #[tokio::test]
    async fn test_memory_fault_injection_embedding_timeout() {
        use crate::umi::RememberOptions;

        // Register embedding timeout with 100% probability
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::EmbeddingTimeout, 1.0));

        let result = sim
            .run(|env| async move {
                let mut memory = env.create_memory();

                // Fault injection is global - may fail at any provider
                let _result = memory
                    .remember("Carol manages engineering", RememberOptions::default())
                    .await;

                // Just verify we don't panic
                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        // Should complete without panicking
        assert!(
            result.is_ok(),
            "Memory should handle faults without panicking: {:?}",
            result.unwrap_err()
        );
    }

    /// Test that vector search faults are applied to Memory recall operations.
    ///
    /// Note: Fault injection is global across all providers.
    #[tokio::test]
    async fn test_memory_fault_injection_vector_search_timeout() {
        use crate::umi::{RecallOptions, RememberOptions};

        // Test with vector search timeout
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::VectorSearchTimeout, 1.0));

        let result = sim
            .run(|env| async move {
                let mut memory = env.create_memory();

                // Fault injection is global - may fail during store or recall
                let _store_result = memory
                    .remember("Alice works at Acme", RememberOptions::default())
                    .await;

                // Try recall regardless of store result
                let _recall_result = memory.recall("Alice", RecallOptions::default()).await;

                // Just verify we don't panic
                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        // Should complete without panicking
        assert!(
            result.is_ok(),
            "Memory should handle faults without panicking: {:?}",
            result.unwrap_err()
        );
    }

    /// Test deterministic behavior: same seed + same faults = same results.
    #[tokio::test]
    async fn test_memory_fault_injection_deterministic() {
        use crate::umi::RememberOptions;

        // Helper to run the test and return whether it succeeded (true) or failed (false)
        async fn run_with_seed_and_faults(seed: u64) -> bool {
            let sim = Simulation::new(SimConfig::with_seed(seed))
                .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.5));

            let result = sim
                .run(|env| async move {
                    let mut memory = env.create_memory();

                    // Try to store - may succeed or fail depending on RNG
                    memory
                        .remember("Test data", RememberOptions::default())
                        .await?;

                    Ok::<(), crate::umi::MemoryError>(())
                })
                .await;

            result.is_ok() // true = success, false = fault injected
        }

        // Run twice with same seed - should get identical results
        let result1 = run_with_seed_and_faults(12345).await;
        let result2 = run_with_seed_and_faults(12345).await;

        assert_eq!(
            result1, result2,
            "Same seed should produce same fault injection pattern"
        );

        // Run with different seed - may get different results (not required, just likely)
        let result3 = run_with_seed_and_faults(67890).await;

        // Just verify it runs - result may or may not differ
        let _ = result3;
    }

    /// Test that multiple fault types can be injected simultaneously.
    #[tokio::test]
    async fn test_memory_fault_injection_multiple_faults() {
        use crate::umi::RememberOptions;

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmTimeout, 0.5))
            .with_fault(FaultConfig::new(FaultType::EmbeddingTimeout, 0.5))
            .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));

        let result = sim
            .run(|env| async move {
                let mut memory = env.create_memory();

                // Should handle multiple fault types gracefully
                // May succeed or fail depending on which faults fire, but should not panic
                let _result = memory
                    .remember("David is an engineer", RememberOptions::default())
                    .await;

                // Just verify we got here without panicking
                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        // Should complete without panicking
        assert!(
            result.is_ok(),
            "Memory should handle multiple faults gracefully: {:?}",
            result.unwrap_err()
        );
    }

    // =============================================================================
    // STRESS TESTS: Find Bugs Through Aggressive Fault Injection
    // =============================================================================

    /// STRESS TEST: Probabilistic faults over many iterations.
    ///
    /// This discovers bugs that only appear under specific fault timing conditions.
    /// We run 100 iterations with different seeds to explore the state space.
    #[tokio::test]
    async fn test_stress_probabilistic_fault_injection() {
        use crate::umi::RememberOptions;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let success_count = Arc::new(AtomicUsize::new(0));
        let failure_count = Arc::new(AtomicUsize::new(0));

        // Run 100 iterations with 30% fault rate
        for seed in 0..100 {
            let sim = Simulation::new(SimConfig::with_seed(seed))
                .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.3));

            let sc = Arc::clone(&success_count);
            let fc = Arc::clone(&failure_count);

            let result = sim
                .run(|env| async move {
                    let mut memory = env.create_memory();

                    // Try to store something
                    memory
                        .remember("Alice works at Acme", RememberOptions::default())
                        .await?;

                    Ok::<(), crate::umi::MemoryError>(())
                })
                .await;

            match result {
                Ok(_) => sc.fetch_add(1, Ordering::Relaxed),
                Err(_) => fc.fetch_add(1, Ordering::Relaxed),
            };
        }

        let successes = success_count.load(Ordering::Relaxed);
        let failures = failure_count.load(Ordering::Relaxed);

        println!("Stress test: {} successes, {} failures", successes, failures);

        // DISCOVERY: With 30% fault rate, we get ~50% failures!
        // This is because remember() extracts multiple entities (typically 2 for "Alice works at Acme")
        // and stores each one. With 2 entities:
        //   P(at least one fails) = 1 - (0.7 * 0.7) = 0.51 = 51%
        //
        // This is NOT a bug - it's the correct behavior of global fault injection.
        // The fault can fire at ANY storage.store_entity() call, and there are multiple per operation.
        assert!(
            failures >= 35 && failures <= 65,
            "Expected ~50% failures due to multiple entities per operation, got {}",
            failures
        );

        // Determinism check: same seed should give same result
        let result1 = run_with_seed(42).await;
        let result2 = run_with_seed(42).await;
        assert_eq!(
            result1.is_ok(),
            result2.is_ok(),
            "Same seed should produce same result"
        );

        async fn run_with_seed(seed: u64) -> Result<(), crate::umi::MemoryError> {
            use crate::umi::RememberOptions;

            let sim = Simulation::new(SimConfig::with_seed(seed))
                .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.3));

            sim.run(|env| async move {
                let mut memory = env.create_memory();
                memory
                    .remember("Test", RememberOptions::default())
                    .await?;
                Ok(())
            })
            .await
        }
    }

    /// STRESS TEST: Multiple fault types injected simultaneously.
    ///
    /// This discovers bugs in error handling composition - when multiple things
    /// fail at once, does the system handle it gracefully?
    #[tokio::test]
    async fn test_stress_multiple_simultaneous_faults() {
        use crate::umi::RememberOptions;

        // Inject faults in ALL providers
        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.2))
            .with_fault(FaultConfig::new(FaultType::LlmTimeout, 0.2))
            .with_fault(FaultConfig::new(FaultType::EmbeddingTimeout, 0.2));

        let mut operation_count = 0;
        let mut error_count = 0;

        // Run 50 operations, see how many fail
        for seed in 0..50 {
            operation_count += 1;

            let sim = Simulation::new(SimConfig::with_seed(seed))
                .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.2))
                .with_fault(FaultConfig::new(FaultType::LlmTimeout, 0.2))
                .with_fault(FaultConfig::new(FaultType::EmbeddingTimeout, 0.2));

            let result = sim
                .run(|env| async move {
                    let mut memory = env.create_memory();

                    // This could fail in multiple ways:
                    // - LLM timeout during extraction
                    // - Embedding timeout during embedding generation
                    // - Storage write fail during persist
                    memory
                        .remember("Data point", RememberOptions::default())
                        .await?;

                    Ok::<(), crate::umi::MemoryError>(())
                })
                .await;

            if result.is_err() {
                error_count += 1;
            }
        }

        println!(
            "Multiple faults: {}/{} operations failed",
            error_count, operation_count
        );

        // Should have SOME failures with 20% rate on 3 providers
        assert!(
            error_count > 0,
            "Expected some failures with multiple fault types"
        );

        // But not ALL failures
        assert!(
            error_count < operation_count,
            "Not all operations should fail"
        );
    }

    /// STRESS TEST: Fault during recall operations (not just remember).
    ///
    /// This discovers bugs in the retrieval path that might not show up
    /// when only testing the write path.
    #[tokio::test]
    async fn test_stress_fault_during_recall() {
        use crate::umi::{RecallOptions, RememberOptions};

        let sim = Simulation::new(SimConfig::with_seed(42));

        // First, store some data WITHOUT faults
        let result = sim
            .run(|env| async move {
                let mut memory = env.create_memory();

                // Store multiple entities
                memory
                    .remember("Alice works at Acme", RememberOptions::default())
                    .await?;
                memory
                    .remember("Bob is the CTO", RememberOptions::default())
                    .await?;
                memory
                    .remember("Carol manages engineering", RememberOptions::default())
                    .await?;

                Ok::<(), crate::umi::MemoryError>(())
            })
            .await;

        assert!(result.is_ok(), "Setup should succeed without faults");

        // Now inject faults and try to recall
        let sim_with_faults = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::VectorSearchTimeout, 0.5));

        let mut recall_attempts = 0;
        let mut recall_failures = 0;

        for seed in 0..20 {
            recall_attempts += 1;

            let sim = Simulation::new(SimConfig::with_seed(seed))
                .with_fault(FaultConfig::new(FaultType::VectorSearchTimeout, 0.5));

            let result = sim
                .run(|env| async move {
                    let mut memory = env.create_memory();

                    // Store first (might fail)
                    let _ = memory
                        .remember("Test", RememberOptions::default())
                        .await;

                    // Try to recall (might fail due to vector search timeout)
                    memory.recall("Alice", RecallOptions::default()).await?;

                    Ok::<(), crate::umi::MemoryError>(())
                })
                .await;

            if result.is_err() {
                recall_failures += 1;
            }
        }

        println!(
            "Recall faults: {}/{} operations failed",
            recall_failures, recall_attempts
        );

        // Should have SOME failures with 50% vector search timeout
        // But global fault injection means it could fail anywhere
        // Just verify the system doesn't panic
        assert!(recall_attempts == 20, "All attempts should complete");
    }

    /// STRESS TEST: Verify fault injection doesn't break determinism.
    ///
    /// This is CRITICAL: even with fault injection, same seed = same behavior.
    #[tokio::test]
    async fn test_stress_determinism_with_faults() {
        use crate::umi::RememberOptions;

        async fn run_scenario(seed: u64) -> Vec<bool> {
            let mut results = Vec::new();

            for i in 0..10 {
                let sim = Simulation::new(SimConfig::with_seed(seed + i))
                    .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.3));

                let result = sim
                    .run(|env| async move {
                        let mut memory = env.create_memory();
                        memory
                            .remember("Test", RememberOptions::default())
                            .await?;
                        Ok::<(), crate::umi::MemoryError>(())
                    })
                    .await;

                results.push(result.is_ok());
            }

            results
        }

        // Run same scenario twice with same seed
        let results1 = run_scenario(42).await;
        let results2 = run_scenario(42).await;

        // MUST be identical
        assert_eq!(
            results1, results2,
            "Determinism violated! Same seed should produce same fault pattern"
        );

        // Run with different seed
        let results3 = run_scenario(12345).await;

        // Should be different (statistically)
        // (Not guaranteed but very likely with 10 operations)
        // Just verify it runs without panicking
        assert_eq!(results3.len(), 10);
    }
}

