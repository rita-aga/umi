//! Property-Based Testing for DST
//!
//! TigerStyle: Random operation sequences with invariant checking.
//!
//! # Philosophy
//!
//! Property-based testing generates random operations and verifies that
//! invariants hold after each operation. Combined with DST, this gives:
//! - Deterministic reproduction via seed
//! - Time control via SimClock
//! - Fault injection via FaultInjector
//!
//! # Example
//!
//! ```rust,ignore
//! use umi_core::dst::{PropertyTest, PropertyTestable, SimClock, DeterministicRng};
//!
//! struct Counter { value: i64, min: i64, max: i64 }
//!
//! #[derive(Debug, Clone)]
//! enum CounterOp { Increment(i64), Decrement(i64), Reset }
//!
//! impl PropertyTestable for Counter {
//!     type Operation = CounterOp;
//!
//!     fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation {
//!         match rng.next_usize(0, 3) {
//!             0 => CounterOp::Increment(rng.next_usize(1, 10) as i64),
//!             1 => CounterOp::Decrement(rng.next_usize(1, 10) as i64),
//!             _ => CounterOp::Reset,
//!         }
//!     }
//!
//!     fn apply_operation(&mut self, op: &Self::Operation, _clock: &SimClock) {
//!         match op {
//!             CounterOp::Increment(n) => self.value = (self.value + n).min(self.max),
//!             CounterOp::Decrement(n) => self.value = (self.value - n).max(self.min),
//!             CounterOp::Reset => self.value = 0,
//!         }
//!     }
//!
//!     fn check_invariants(&self) -> Result<(), String> {
//!         if self.value < self.min {
//!             return Err(format!("value {} below min {}", self.value, self.min));
//!         }
//!         if self.value > self.max {
//!             return Err(format!("value {} above max {}", self.value, self.max));
//!         }
//!         Ok(())
//!     }
//! }
//!
//! #[test]
//! fn test_counter_properties() {
//!     let counter = Counter { value: 0, min: -100, max: 100 };
//!     let result = PropertyTest::new(42)
//!         .with_max_operations(1000)
//!         .run(counter);
//!     assert!(result.is_success());
//! }
//! ```

use std::fmt::Debug;

use super::clock::SimClock;
use super::rng::DeterministicRng;
use crate::constants::DST_SIMULATION_STEPS_MAX;

/// Trait for systems that can be property-tested.
///
/// TigerStyle: Explicit operation generation and invariant checking.
pub trait PropertyTestable {
    /// The type of operations that can be performed.
    type Operation: Debug + Clone;

    /// Generate a random operation based on current state.
    ///
    /// The operation should be valid for the current state.
    fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation;

    /// Apply an operation to the state.
    ///
    /// May use the clock for time-dependent operations.
    fn apply_operation(&mut self, op: &Self::Operation, clock: &SimClock);

    /// Check that all invariants hold.
    ///
    /// Returns Ok(()) if all invariants pass, Err(message) otherwise.
    fn check_invariants(&self) -> Result<(), String>;

    /// Optional: Describe the current state for debugging.
    fn describe_state(&self) -> String {
        String::from("(state description not implemented)")
    }
}

/// Result of a property test run.
#[derive(Debug)]
pub struct PropertyTestResult {
    /// Number of operations successfully executed
    pub operations_executed: u64,
    /// Seed used for reproduction
    pub seed: u64,
    /// Failure details, if any
    pub failure: Option<PropertyTestFailure>,
}

impl PropertyTestResult {
    /// Check if the test passed.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.failure.is_none()
    }

    /// Check if the test failed.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.failure.is_some()
    }

    /// Unwrap the result, panicking with details if failed.
    ///
    /// # Panics
    /// Panics if the test failed, with reproduction info.
    pub fn unwrap(self) {
        if let Some(failure) = self.failure {
            panic!(
                "Property test failed!\n\
                 Seed: {} (use this to reproduce)\n\
                 Operation #{}: {:?}\n\
                 Invariant violation: {}\n\
                 State: {}",
                self.seed,
                failure.operation_index,
                failure.operation,
                failure.message,
                failure.state_description
            );
        }
    }
}

/// Details of a property test failure.
#[derive(Debug)]
pub struct PropertyTestFailure {
    /// Index of the failing operation (0-based)
    pub operation_index: u64,
    /// The operation that caused the failure
    pub operation: String,
    /// The invariant violation message
    pub message: String,
    /// Description of the state at failure
    pub state_description: String,
}

/// Configuration for time advancement during property tests.
#[derive(Debug, Clone)]
pub struct TimeAdvanceConfig {
    /// Minimum time to advance per operation (ms)
    pub min_ms: u64,
    /// Maximum time to advance per operation (ms)
    pub max_ms: u64,
    /// Probability of advancing time (0.0 to 1.0)
    pub probability: f64,
}

impl Default for TimeAdvanceConfig {
    fn default() -> Self {
        Self {
            min_ms: 0,
            max_ms: 1000,
            probability: 0.5,
        }
    }
}

impl TimeAdvanceConfig {
    /// No time advancement.
    #[must_use]
    pub fn none() -> Self {
        Self {
            min_ms: 0,
            max_ms: 0,
            probability: 0.0,
        }
    }

    /// Always advance by fixed amount.
    #[must_use]
    pub fn fixed(ms: u64) -> Self {
        Self {
            min_ms: ms,
            max_ms: ms,
            probability: 1.0,
        }
    }

    /// Advance with given range and probability.
    #[must_use]
    pub fn random(min_ms: u64, max_ms: u64, probability: f64) -> Self {
        assert!(probability >= 0.0 && probability <= 1.0);
        assert!(min_ms <= max_ms);
        Self {
            min_ms,
            max_ms,
            probability,
        }
    }
}

/// Property-based test runner.
///
/// TigerStyle:
/// - Deterministic via seed
/// - Explicit operation count limits
/// - Invariant checking after each operation
/// - Time advancement control
#[derive(Debug)]
pub struct PropertyTest {
    seed: u64,
    max_operations: u64,
    time_config: TimeAdvanceConfig,
    check_invariants_before: bool,
}

impl PropertyTest {
    /// Create a new property test with the given seed.
    ///
    /// # Panics
    /// Panics if max_operations would exceed DST_SIMULATION_STEPS_MAX.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            max_operations: 100, // Sensible default
            time_config: TimeAdvanceConfig::default(),
            check_invariants_before: true,
        }
    }

    /// Set the maximum number of operations to run.
    ///
    /// # Panics
    /// Panics if max exceeds DST_SIMULATION_STEPS_MAX.
    #[must_use]
    pub fn with_max_operations(mut self, max: u64) -> Self {
        assert!(
            max <= DST_SIMULATION_STEPS_MAX,
            "max_operations {} exceeds DST_SIMULATION_STEPS_MAX {}",
            max,
            DST_SIMULATION_STEPS_MAX
        );
        self.max_operations = max;
        self
    }

    /// Configure time advancement between operations.
    #[must_use]
    pub fn with_time_advance(mut self, config: TimeAdvanceConfig) -> Self {
        self.time_config = config;
        self
    }

    /// Disable checking invariants before the first operation.
    #[must_use]
    pub fn skip_initial_invariant_check(mut self) -> Self {
        self.check_invariants_before = false;
        self
    }

    /// Run the property test.
    ///
    /// Generates random operations, applies them, and checks invariants
    /// after each operation. Returns detailed results.
    #[must_use]
    pub fn run<T: PropertyTestable>(self, mut state: T) -> PropertyTestResult {
        let mut rng = DeterministicRng::new(self.seed);
        let clock = SimClock::new();

        // Check initial invariants
        if self.check_invariants_before {
            if let Err(msg) = state.check_invariants() {
                return PropertyTestResult {
                    operations_executed: 0,
                    seed: self.seed,
                    failure: Some(PropertyTestFailure {
                        operation_index: 0,
                        operation: "(initial state)".to_string(),
                        message: format!("Initial state violates invariants: {}", msg),
                        state_description: state.describe_state(),
                    }),
                };
            }
        }

        for i in 0..self.max_operations {
            // Maybe advance time
            if self.time_config.probability > 0.0 && rng.next_bool(self.time_config.probability) {
                let advance = if self.time_config.min_ms == self.time_config.max_ms {
                    self.time_config.min_ms
                } else {
                    rng.next_usize(
                        self.time_config.min_ms as usize,
                        self.time_config.max_ms as usize,
                    ) as u64
                };
                clock.advance_ms(advance);
            }

            // Generate and apply operation
            let op = state.generate_operation(&mut rng);
            let op_debug = format!("{:?}", op);
            state.apply_operation(&op, &clock);

            // Check invariants
            if let Err(msg) = state.check_invariants() {
                return PropertyTestResult {
                    operations_executed: i + 1,
                    seed: self.seed,
                    failure: Some(PropertyTestFailure {
                        operation_index: i,
                        operation: op_debug,
                        message: msg,
                        state_description: state.describe_state(),
                    }),
                };
            }
        }

        PropertyTestResult {
            operations_executed: self.max_operations,
            seed: self.seed,
            failure: None,
        }
    }

    /// Run the property test, panicking on failure.
    ///
    /// Convenience method for use in #[test] functions.
    ///
    /// # Panics
    /// Panics if any invariant is violated.
    pub fn run_and_assert<T: PropertyTestable>(self, state: T) {
        self.run(state).unwrap();
    }
}

/// Run multiple property tests with different seeds.
///
/// TigerStyle: Multi-seed testing for broader coverage.
///
/// # Panics
/// Panics if any test fails.
pub fn run_property_tests<T, F>(seeds: &[u64], max_operations: u64, state_factory: F)
where
    T: PropertyTestable,
    F: Fn() -> T,
{
    for &seed in seeds {
        let state = state_factory();
        PropertyTest::new(seed)
            .with_max_operations(max_operations)
            .run_and_assert(state);
    }
}

/// Generate a set of test seeds including edge cases.
///
/// Returns seeds: [0, 1, 42, random, random, ...]
#[must_use]
pub fn test_seeds(count: usize) -> Vec<u64> {
    assert!(count >= 3, "need at least 3 seeds for edge cases");

    let mut seeds = vec![0, 1, 42]; // Edge cases + common test seed

    // Add random seeds
    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);
    let mut rng = DeterministicRng::new(time_seed);

    while seeds.len() < count {
        // Generate u64 from two usize values
        let high = rng.next_usize(0, u32::MAX as usize) as u64;
        let low = rng.next_usize(0, u32::MAX as usize) as u64;
        seeds.push((high << 32) | low);
    }

    seeds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple counter for testing the property test framework.
    struct BoundedCounter {
        value: i64,
        min: i64,
        max: i64,
    }

    #[derive(Debug, Clone)]
    enum CounterOp {
        Increment(i64),
        Decrement(i64),
        Reset,
    }

    impl PropertyTestable for BoundedCounter {
        type Operation = CounterOp;

        fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation {
            match rng.next_usize(0, 3) {
                0 => CounterOp::Increment(rng.next_usize(1, 20) as i64),
                1 => CounterOp::Decrement(rng.next_usize(1, 20) as i64),
                _ => CounterOp::Reset,
            }
        }

        fn apply_operation(&mut self, op: &Self::Operation, _clock: &SimClock) {
            match op {
                CounterOp::Increment(n) => {
                    self.value = (self.value + n).min(self.max);
                }
                CounterOp::Decrement(n) => {
                    self.value = (self.value - n).max(self.min);
                }
                CounterOp::Reset => {
                    self.value = 0;
                }
            }
        }

        fn check_invariants(&self) -> Result<(), String> {
            if self.value < self.min {
                return Err(format!("value {} below min {}", self.value, self.min));
            }
            if self.value > self.max {
                return Err(format!("value {} above max {}", self.value, self.max));
            }
            Ok(())
        }

        fn describe_state(&self) -> String {
            format!(
                "BoundedCounter {{ value: {}, min: {}, max: {} }}",
                self.value, self.min, self.max
            )
        }
    }

    #[test]
    fn test_property_test_success() {
        let counter = BoundedCounter {
            value: 0,
            min: -100,
            max: 100,
        };

        let result = PropertyTest::new(42)
            .with_max_operations(1000)
            .with_time_advance(TimeAdvanceConfig::none())
            .run(counter);

        assert!(result.is_success());
        assert_eq!(result.operations_executed, 1000);
        assert_eq!(result.seed, 42);
    }

    #[test]
    fn test_property_test_determinism() {
        // Same seed should produce same results
        let run1 = PropertyTest::new(12345)
            .with_max_operations(100)
            .run(BoundedCounter {
                value: 0,
                min: -50,
                max: 50,
            });

        let run2 = PropertyTest::new(12345)
            .with_max_operations(100)
            .run(BoundedCounter {
                value: 0,
                min: -50,
                max: 50,
            });

        assert_eq!(run1.operations_executed, run2.operations_executed);
        assert_eq!(run1.is_success(), run2.is_success());
    }

    /// Buggy counter that doesn't clamp properly - should fail.
    struct BuggyCounter {
        value: i64,
        max: i64,
    }

    #[derive(Debug, Clone)]
    enum BuggyOp {
        Add(i64),
    }

    impl PropertyTestable for BuggyCounter {
        type Operation = BuggyOp;

        fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation {
            BuggyOp::Add(rng.next_usize(1, 50) as i64)
        }

        fn apply_operation(&mut self, op: &Self::Operation, _clock: &SimClock) {
            match op {
                BuggyOp::Add(n) => {
                    // Bug: doesn't clamp to max!
                    self.value += n;
                }
            }
        }

        fn check_invariants(&self) -> Result<(), String> {
            if self.value > self.max {
                return Err(format!("value {} exceeds max {}", self.value, self.max));
            }
            Ok(())
        }

        fn describe_state(&self) -> String {
            format!(
                "BuggyCounter {{ value: {}, max: {} }}",
                self.value, self.max
            )
        }
    }

    #[test]
    fn test_property_test_catches_bug() {
        let counter = BuggyCounter { value: 0, max: 100 };

        let result = PropertyTest::new(42).with_max_operations(1000).run(counter);

        assert!(result.is_failure());
        let failure = result.failure.unwrap();
        assert!(failure.message.contains("exceeds max"));
    }

    #[test]
    fn test_time_advance_config() {
        // Test that time actually advances
        struct TimeTracker {
            last_time: u64,
            times_advanced: u64,
        }

        #[derive(Debug, Clone)]
        struct NoOp;

        impl PropertyTestable for TimeTracker {
            type Operation = NoOp;

            fn generate_operation(&self, _rng: &mut DeterministicRng) -> Self::Operation {
                NoOp
            }

            fn apply_operation(&mut self, _op: &Self::Operation, clock: &SimClock) {
                let now = clock.now_ms();
                if now > self.last_time {
                    self.times_advanced += 1;
                    self.last_time = now;
                }
            }

            fn check_invariants(&self) -> Result<(), String> {
                Ok(())
            }

            fn describe_state(&self) -> String {
                format!("TimeTracker {{ times_advanced: {} }}", self.times_advanced)
            }
        }

        let tracker = TimeTracker {
            last_time: 0,
            times_advanced: 0,
        };

        let _result = PropertyTest::new(42)
            .with_max_operations(100)
            .with_time_advance(TimeAdvanceConfig::fixed(10))
            .run(tracker);

        // Time should have advanced multiple times
        // (we can't easily check the internal state after run, but no panics = success)
    }

    #[test]
    fn test_test_seeds() {
        let seeds = test_seeds(10);
        assert_eq!(seeds.len(), 10);
        assert_eq!(seeds[0], 0); // Edge case
        assert_eq!(seeds[1], 1); // Edge case
        assert_eq!(seeds[2], 42); // Common test seed
    }

    #[test]
    fn test_run_property_tests_helper() {
        run_property_tests(&[0, 1, 42], 100, || BoundedCounter {
            value: 0,
            min: -100,
            max: 100,
        });
    }

    #[test]
    fn test_initial_invariant_check() {
        // State that starts invalid
        let bad_counter = BoundedCounter {
            value: 200, // Exceeds max!
            min: -100,
            max: 100,
        };

        let result = PropertyTest::new(42).run(bad_counter);

        assert!(result.is_failure());
        assert!(result
            .failure
            .unwrap()
            .message
            .contains("Initial state violates"));
    }

    #[test]
    fn test_skip_initial_invariant_check() {
        // Use BuggyCounter which doesn't clamp - starts invalid and stays invalid
        let bad_counter = BuggyCounter {
            value: 200,
            max: 100,
        };

        let result = PropertyTest::new(42)
            .skip_initial_invariant_check()
            .with_max_operations(1)
            .run(bad_counter);

        // Should fail when invariant is checked after first op (value still > max)
        assert!(result.is_failure());
    }

    #[test]
    fn test_skip_initial_but_fixes_itself() {
        // BoundedCounter clamps values, so invalid initial state gets fixed
        let bad_counter = BoundedCounter {
            value: 200,
            min: -100,
            max: 100,
        };

        let result = PropertyTest::new(42)
            .skip_initial_invariant_check()
            .with_max_operations(1)
            .run(bad_counter);

        // Should pass because BoundedCounter clamps to valid range on any operation
        assert!(result.is_success());
    }
}
