//! DeterministicRng - Seeded Random Number Generator
//!
//! TigerStyle: ChaCha20-based RNG for deterministic simulation.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// A deterministic random number generator.
///
/// TigerStyle:
/// - Same seed always produces same sequence
/// - Fork creates independent streams
/// - All randomness flows through this
#[derive(Debug)]
pub struct DeterministicRng {
    rng: ChaCha20Rng,
    seed: u64,
    /// Counter for generating fork seeds
    fork_counter: u64,
}

impl DeterministicRng {
    /// Create a new RNG with the given seed.
    ///
    /// # Example
    /// ```
    /// use umi_core::dst::DeterministicRng;
    /// let mut rng = DeterministicRng::new(42);
    /// let value = rng.next_float();
    /// ```
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);

        // Postcondition
        let result = Self {
            rng,
            seed,
            fork_counter: 0,
        };
        assert_eq!(result.seed, seed, "seed must be stored");
        result
    }

    /// Get the original seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate a random float in [0, 1).
    pub fn next_float(&mut self) -> f64 {
        let value = self.rng.gen::<f64>();

        // Postcondition
        assert!((0.0..1.0).contains(&value), "float must be in [0, 1)");
        value
    }

    /// Generate a random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.rng.gen()
    }

    /// Generate a random integer in [min, max] (inclusive).
    ///
    /// # Panics
    /// Panics if min > max.
    pub fn next_int(&mut self, min: i64, max: i64) -> i64 {
        // Precondition
        assert!(min <= max, "min ({}) must be <= max ({})", min, max);

        let value = self.rng.gen_range(min..=max);

        // Postcondition
        assert!(value >= min && value <= max, "value must be in range");
        value
    }

    /// Generate a random usize in [min, max] (inclusive).
    ///
    /// # Panics
    /// Panics if min > max.
    pub fn next_usize(&mut self, min: usize, max: usize) -> usize {
        // Precondition
        assert!(min <= max, "min ({}) must be <= max ({})", min, max);

        let value = self.rng.gen_range(min..=max);

        // Postcondition
        assert!(value >= min && value <= max, "value must be in range");
        value
    }

    /// Generate a random boolean with the given probability of true.
    ///
    /// # Panics
    /// Panics if probability is not in [0, 1].
    pub fn next_bool(&mut self, probability: f64) -> bool {
        // Precondition
        assert!(
            (0.0..=1.0).contains(&probability),
            "probability must be in [0, 1], got {}",
            probability
        );

        self.next_float() < probability
    }

    /// Choose a random element from a slice.
    ///
    /// # Panics
    /// Panics if the slice is empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        // Precondition
        assert!(!items.is_empty(), "cannot choose from empty slice");

        let index = self.next_usize(0, items.len() - 1);
        &items[index]
    }

    /// Shuffle a mutable slice in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        // Fisher-Yates shuffle
        for i in (1..items.len()).rev() {
            let j = self.next_usize(0, i);
            items.swap(i, j);
        }
    }

    /// Create an independent fork of this RNG.
    ///
    /// TigerStyle: Forks have independent sequences derived from parent.
    ///
    /// # Example
    /// ```
    /// use umi_core::dst::DeterministicRng;
    /// let mut rng = DeterministicRng::new(42);
    /// let mut fork1 = rng.fork();
    /// let mut fork2 = rng.fork();
    /// // fork1 and fork2 have independent sequences
    /// ```
    pub fn fork(&mut self) -> Self {
        // Generate a new seed by combining original seed with fork counter
        // Using golden ratio constant for good distribution
        let fork_seed = self.seed.wrapping_add(
            self.fork_counter
                .wrapping_add(1)
                .wrapping_mul(0x9E3779B97F4A7C15),
        );
        self.fork_counter += 1;

        // Create fork with derived seed
        Self::new(fork_seed)
    }

    /// Generate random bytes.
    pub fn next_bytes(&mut self, len: usize) -> Vec<u8> {
        // Precondition
        assert!(len <= 1_000_000, "len must be <= 1MB");

        let mut bytes = vec![0u8; len];
        self.rng.fill(&mut bytes[..]);

        // Postcondition
        assert_eq!(bytes.len(), len, "must generate requested bytes");
        bytes
    }
}

impl Clone for DeterministicRng {
    fn clone(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            seed: self.seed,
            fork_counter: self.fork_counter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_seed_same_sequence() {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_float(), rng2.next_float());
        }
    }

    #[test]
    fn test_different_seeds_different_sequence() {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(54321);

        let differs = (0..10).any(|_| rng1.next_float() != rng2.next_float());
        assert!(
            differs,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_next_int_bounds() {
        let mut rng = DeterministicRng::new(42);

        for _ in 0..100 {
            let val = rng.next_int(5, 10);
            assert!((5..=10).contains(&val));
        }
    }

    #[test]
    fn test_next_bool_always_false() {
        let mut rng = DeterministicRng::new(42);

        for _ in 0..100 {
            assert!(!rng.next_bool(0.0));
        }
    }

    #[test]
    fn test_next_bool_always_true() {
        let mut rng = DeterministicRng::new(42);

        for _ in 0..100 {
            assert!(rng.next_bool(1.0));
        }
    }

    #[test]
    fn test_fork_independence() {
        let mut rng = DeterministicRng::new(42);

        let mut fork1 = rng.fork();
        let mut fork2 = rng.fork();

        // Forks should have different seeds (derived from parent)
        assert_ne!(
            fork1.seed(),
            fork2.seed(),
            "forks should have different seeds"
        );

        // Forks should produce different sequences
        let fork1_vals: Vec<f64> = (0..5).map(|_| fork1.next_float()).collect();
        let fork2_vals: Vec<f64> = (0..5).map(|_| fork2.next_float()).collect();

        assert_ne!(
            fork1_vals, fork2_vals,
            "forks should have different sequences"
        );

        // Original RNG should still work
        let _ = rng.next_float();
    }

    #[test]
    fn test_choose() {
        let mut rng = DeterministicRng::new(42);
        let items = vec![1, 2, 3, 4, 5];

        for _ in 0..100 {
            let chosen = rng.choose(&items);
            assert!(items.contains(chosen));
        }
    }

    #[test]
    fn test_shuffle() {
        let mut rng = DeterministicRng::new(42);
        let mut items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let original = items.clone();

        rng.shuffle(&mut items);

        // Should be different order (with very high probability)
        assert_ne!(items, original, "shuffle should change order");
        // But same elements
        items.sort();
        assert_eq!(items, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_next_bytes() {
        let mut rng = DeterministicRng::new(42);
        let bytes = rng.next_bytes(32);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    #[should_panic(expected = "min (10) must be <= max (5)")]
    fn test_next_int_invalid_range() {
        let mut rng = DeterministicRng::new(42);
        rng.next_int(10, 5);
    }

    #[test]
    #[should_panic(expected = "probability must be in [0, 1]")]
    fn test_next_bool_invalid_probability() {
        let mut rng = DeterministicRng::new(42);
        rng.next_bool(1.5);
    }

    #[test]
    #[should_panic(expected = "cannot choose from empty slice")]
    fn test_choose_empty() {
        let mut rng = DeterministicRng::new(42);
        let items: Vec<i32> = vec![];
        rng.choose(&items);
    }
}
