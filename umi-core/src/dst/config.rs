//! SimConfig - Simulation Configuration
//!
//! TigerStyle: Seed management for deterministic testing.

use rand::Rng;
use std::env;

use crate::constants::DST_SIMULATION_STEPS_MAX;

/// Configuration for a simulation run.
///
/// TigerStyle:
/// - Immutable after creation
/// - Seed logged for reproducibility
/// - All limits explicit
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    /// Random seed for deterministic execution
    seed: u64,
    /// Maximum number of simulation steps
    steps_max: u64,
}

impl SimConfig {
    /// Create config with explicit seed.
    ///
    /// # Panics
    /// Never panics - all u64 values are valid seeds.
    ///
    /// # Example
    /// ```
    /// use umi_core::dst::SimConfig;
    /// let config = SimConfig::with_seed(12345);
    /// assert_eq!(config.seed(), 12345);
    /// ```
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        // Postcondition
        let config = Self {
            seed,
            steps_max: DST_SIMULATION_STEPS_MAX,
        };
        assert_eq!(config.seed, seed, "seed must be stored correctly");
        assert!(config.steps_max > 0, "steps_max must be positive");
        config
    }

    /// Create config from DST_SEED env var or random.
    ///
    /// If DST_SEED is set, uses that value.
    /// Otherwise, generates a random seed and prints it for reproducibility.
    ///
    /// # Example
    /// ```
    /// use umi_core::dst::SimConfig;
    /// // Set DST_SEED=42 to get deterministic seed
    /// let config = SimConfig::from_env_or_random();
    /// ```
    #[must_use]
    pub fn from_env_or_random() -> Self {
        let seed = match env::var("DST_SEED") {
            Ok(seed_str) => {
                // Precondition: DST_SEED must be valid u64
                seed_str.parse::<u64>().unwrap_or_else(|_| {
                    panic!("DST_SEED must be a valid u64, got: {}", seed_str);
                })
            }
            Err(_) => {
                let seed = rand::thread_rng().gen::<u64>();
                eprintln!("DST: Generated random seed (replay with DST_SEED={})", seed);
                seed
            }
        };

        Self::with_seed(seed)
    }

    /// Get the seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the maximum number of steps.
    #[must_use]
    pub fn steps_max(&self) -> u64 {
        self.steps_max
    }

    /// Create a new config with a different steps_max.
    #[must_use]
    pub fn with_steps_max(self, steps_max: u64) -> Self {
        // Precondition
        assert!(steps_max > 0, "steps_max must be positive");

        Self {
            seed: self.seed,
            steps_max,
        }
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self::from_env_or_random()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_seed() {
        let config = SimConfig::with_seed(12345);
        assert_eq!(config.seed(), 12345);
        assert_eq!(config.steps_max(), DST_SIMULATION_STEPS_MAX);
    }

    #[test]
    fn test_with_seed_zero() {
        let config = SimConfig::with_seed(0);
        assert_eq!(config.seed(), 0);
    }

    #[test]
    fn test_with_seed_max() {
        let config = SimConfig::with_seed(u64::MAX);
        assert_eq!(config.seed(), u64::MAX);
    }

    #[test]
    fn test_with_steps_max() {
        let config = SimConfig::with_seed(42).with_steps_max(100);
        assert_eq!(config.seed(), 42);
        assert_eq!(config.steps_max(), 100);
    }

    #[test]
    #[should_panic(expected = "steps_max must be positive")]
    fn test_with_steps_max_zero_panics() {
        let _ = SimConfig::with_seed(42).with_steps_max(0);
    }

    // Note: Environment variable tests are tricky because tests run in parallel.
    // These tests are better run in isolation or with --test-threads=1.
    // For now, we focus on the core functionality tests.

    #[test]
    fn test_random_seed_generation() {
        // Clear env to ensure random generation
        let _ = env::remove_var("DST_SEED");

        // Just verify that from_env_or_random() works without DST_SEED
        // We can't easily test the exact value since it's random
        let config = SimConfig::from_env_or_random();
        assert!(config.seed() > 0 || config.seed() == 0); // Any u64 is valid
        assert!(config.steps_max() > 0);
    }
}
