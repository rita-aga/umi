//! Working Memory - Session-Scoped KV Store with TTL
//!
//! TigerStyle: Explicit limits, TTL expiration, simulation-first testing.
//!
//! # Design
//!
//! Working memory is a bounded KV store (~1MB) for session state.
//! Entries expire after TTL (default 1 hour).
//!
//! # Simulation-First
//!
//! Tests are written BEFORE implementation. This file starts with tests
//! and minimal stubs. Implementation follows to make tests pass.

use std::collections::HashMap;

use crate::constants::{
    WORKING_MEMORY_ENTRIES_COUNT_MAX, WORKING_MEMORY_ENTRY_SIZE_BYTES_MAX,
    WORKING_MEMORY_SIZE_BYTES_MAX, WORKING_MEMORY_TTL_SECS_DEFAULT,
};

// =============================================================================
// Error Types
// =============================================================================

/// Errors from working memory operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WorkingMemoryError {
    /// Entry too large
    #[error("entry too large: {size_bytes} bytes exceeds max {max_bytes}")]
    EntryTooLarge {
        /// Size of the entry
        size_bytes: usize,
        /// Maximum allowed
        max_bytes: usize,
    },

    /// Memory full
    #[error("working memory full: {current_bytes}/{max_bytes} bytes")]
    MemoryFull {
        /// Current used bytes
        current_bytes: usize,
        /// Maximum allowed bytes
        max_bytes: usize,
    },

    /// Too many entries
    #[error("too many entries: {count} exceeds max {max_count}")]
    TooManyEntries {
        /// Current entry count
        count: usize,
        /// Maximum allowed
        max_count: usize,
    },

    /// Key too long
    #[error("key too long: {len} bytes exceeds max {max_len}")]
    KeyTooLong {
        /// Key length
        len: usize,
        /// Maximum allowed
        max_len: usize,
    },

    /// TTL too long
    #[error("TTL too long: {ttl_secs} seconds exceeds max {max_secs}")]
    TtlTooLong {
        /// Requested TTL
        ttl_secs: u64,
        /// Maximum allowed
        max_secs: u64,
    },
}

/// Result type for working memory operations.
pub type WorkingMemoryResult<T> = Result<T, WorkingMemoryError>;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for working memory.
#[derive(Debug, Clone)]
pub struct WorkingMemoryConfig {
    /// Maximum total size in bytes
    pub max_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Default TTL in milliseconds
    pub default_ttl_ms: u64,
    /// Maximum key length
    pub max_key_len: usize,
}

impl Default for WorkingMemoryConfig {
    fn default() -> Self {
        Self {
            max_bytes: WORKING_MEMORY_SIZE_BYTES_MAX,
            max_entries: WORKING_MEMORY_ENTRIES_COUNT_MAX,
            default_ttl_ms: WORKING_MEMORY_TTL_SECS_DEFAULT * 1000,
            max_key_len: 256,
        }
    }
}

// =============================================================================
// Entry Type
// =============================================================================

/// A single entry in working memory.
#[derive(Debug, Clone)]
struct Entry {
    /// The value bytes
    value: Vec<u8>,
    /// Size in bytes (cached for quick access)
    size_bytes: usize,
    /// Creation timestamp (ms since epoch) - kept for debugging/stats
    #[allow(dead_code)]
    created_at_ms: u64,
    /// Expiration timestamp (ms since epoch)
    expires_at_ms: u64,
}

// =============================================================================
// Working Memory
// =============================================================================

/// Working memory - session-scoped KV store with TTL.
///
/// TigerStyle:
/// - Bounded capacity (~1MB)
/// - TTL-based expiration
/// - Explicit size tracking
/// - DST-compatible via set_clock_ms()
#[derive(Debug)]
pub struct WorkingMemory {
    /// Configuration
    config: WorkingMemoryConfig,
    /// Entries indexed by key
    entries: HashMap<String, Entry>,
    /// Current total size in bytes (keys + values)
    current_bytes: usize,
    /// Clock source for timestamps (ms since epoch)
    clock_ms: u64,
}

impl WorkingMemory {
    /// Create a new working memory with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(WorkingMemoryConfig::default())
    }

    /// Create a new working memory with custom configuration.
    #[must_use]
    pub fn with_config(config: WorkingMemoryConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            current_bytes: 0,
            clock_ms: 0,
        }
    }

    /// Set the internal clock (for DST).
    ///
    /// TigerStyle: Explicit time control for simulation.
    pub fn set_clock_ms(&mut self, ms: u64) {
        self.clock_ms = ms;
    }

    /// Get the internal clock value.
    #[must_use]
    pub fn clock_ms(&self) -> u64 {
        self.clock_ms
    }

    /// Set an entry with optional TTL.
    ///
    /// If TTL is None, uses the default TTL from config.
    /// If key already exists, the entry is replaced and TTL is reset.
    ///
    /// # Errors
    /// Returns error if entry is too large, memory is full, or too many entries.
    pub fn set(&mut self, key: &str, value: &[u8], ttl_ms: Option<u64>) -> WorkingMemoryResult<()> {
        let value_len = value.len();
        let key_len = key.len();
        let entry_size = key_len + value_len;

        // Precondition: key length
        if key_len > self.config.max_key_len {
            return Err(WorkingMemoryError::KeyTooLong {
                len: key_len,
                max_len: self.config.max_key_len,
            });
        }

        // Precondition: entry size
        if value_len > WORKING_MEMORY_ENTRY_SIZE_BYTES_MAX {
            return Err(WorkingMemoryError::EntryTooLarge {
                size_bytes: value_len,
                max_bytes: WORKING_MEMORY_ENTRY_SIZE_BYTES_MAX,
            });
        }

        // Calculate size delta (account for existing entry if overwriting)
        let old_size = self
            .entries
            .get(key)
            .map(|e| key_len + e.size_bytes)
            .unwrap_or(0);
        let projected_size = self.current_bytes - old_size + entry_size;

        // Check capacity
        if projected_size > self.config.max_bytes {
            return Err(WorkingMemoryError::MemoryFull {
                current_bytes: self.current_bytes,
                max_bytes: self.config.max_bytes,
            });
        }

        // Check entry count (only if not overwriting)
        let is_new_key = !self.entries.contains_key(key);
        if is_new_key && self.entries.len() >= self.config.max_entries {
            return Err(WorkingMemoryError::TooManyEntries {
                count: self.entries.len(),
                max_count: self.config.max_entries,
            });
        }

        // Calculate TTL
        let ttl = ttl_ms.unwrap_or(self.config.default_ttl_ms);
        let expires_at_ms = self.clock_ms.saturating_add(ttl);

        // Create entry
        let entry = Entry {
            value: value.to_vec(),
            size_bytes: value_len,
            created_at_ms: self.clock_ms,
            expires_at_ms,
        };

        self.entries.insert(key.to_string(), entry);
        self.current_bytes = projected_size;

        // Postcondition
        assert!(
            self.current_bytes <= self.config.max_bytes,
            "size invariant violated"
        );

        Ok(())
    }

    /// Get an entry by key.
    ///
    /// Returns None if key doesn't exist or entry is expired.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.entries.get(key).and_then(|entry| {
            if entry.expires_at_ms > self.clock_ms {
                Some(entry.value.as_slice())
            } else {
                None // Expired
            }
        })
    }

    /// Delete an entry by key.
    ///
    /// Returns true if entry existed (even if expired), false otherwise.
    pub fn delete(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            let entry_size = key.len() + entry.size_bytes;
            self.current_bytes = self.current_bytes.saturating_sub(entry_size);
            true
        } else {
            false
        }
    }

    /// Check if a key exists (and is not expired).
    #[must_use]
    pub fn exists(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .map(|entry| entry.expires_at_ms > self.clock_ms)
            .unwrap_or(false)
    }

    /// Remove all expired entries.
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let clock = self.clock_ms;
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.expires_at_ms <= clock)
            .map(|(key, _)| key.clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.delete(&key);
        }

        count
    }

    /// Get used bytes (keys + values).
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Get available bytes.
    #[must_use]
    pub fn available_bytes(&self) -> usize {
        self.config.max_bytes.saturating_sub(self.current_bytes)
    }

    /// Get entry count (including expired).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &WorkingMemoryConfig {
        &self.config
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS - Written FIRST (Simulation-First)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic CRUD Tests
    // =========================================================================

    #[test]
    fn test_new_working_memory() {
        let wm = WorkingMemory::new();
        assert_eq!(wm.used_bytes(), 0);
        assert_eq!(wm.entry_count(), 0);
        assert!(wm.is_empty());
    }

    #[test]
    fn test_set_and_get() {
        let mut wm = WorkingMemory::new();
        wm.set("key1", b"value1", None).unwrap();

        assert_eq!(wm.get("key1"), Some(b"value1".as_slice()));
        assert!(wm.exists("key1"));
        assert_eq!(wm.entry_count(), 1);
    }

    #[test]
    fn test_set_overwrites() {
        let mut wm = WorkingMemory::new();
        wm.set("key1", b"value1", None).unwrap();
        wm.set("key1", b"new_value", None).unwrap();

        assert_eq!(wm.get("key1"), Some(b"new_value".as_slice()));
        assert_eq!(wm.entry_count(), 1);
    }

    #[test]
    fn test_get_nonexistent() {
        let wm = WorkingMemory::new();
        assert_eq!(wm.get("nonexistent"), None);
        assert!(!wm.exists("nonexistent"));
    }

    #[test]
    fn test_delete() {
        let mut wm = WorkingMemory::new();
        wm.set("key1", b"value1", None).unwrap();

        assert!(wm.delete("key1"));
        assert_eq!(wm.get("key1"), None);
        assert!(!wm.exists("key1"));
        assert_eq!(wm.entry_count(), 0);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut wm = WorkingMemory::new();
        assert!(!wm.delete("nonexistent"));
    }

    #[test]
    fn test_clear() {
        let mut wm = WorkingMemory::new();
        wm.set("key1", b"value1", None).unwrap();
        wm.set("key2", b"value2", None).unwrap();

        wm.clear();

        assert!(wm.is_empty());
        assert_eq!(wm.used_bytes(), 0);
    }

    // =========================================================================
    // Size Tracking Tests
    // =========================================================================

    #[test]
    fn test_size_tracking() {
        let mut wm = WorkingMemory::new();

        // key1 (4 bytes) + value1 (6 bytes) = 10 bytes
        wm.set("key1", b"value1", None).unwrap();
        assert_eq!(wm.used_bytes(), 10);

        // key2 (4 bytes) + value2 (6 bytes) = 10 bytes, total = 20
        wm.set("key2", b"value2", None).unwrap();
        assert_eq!(wm.used_bytes(), 20);

        // Delete key1, should free 10 bytes
        wm.delete("key1");
        assert_eq!(wm.used_bytes(), 10);
    }

    #[test]
    fn test_overwrite_size_tracking() {
        let mut wm = WorkingMemory::new();

        // key1 (4) + short (5) = 9 bytes
        wm.set("key1", b"short", None).unwrap();
        assert_eq!(wm.used_bytes(), 9);

        // key1 (4) + much_longer_value (17) = 21 bytes
        wm.set("key1", b"much_longer_value", None).unwrap();
        assert_eq!(wm.used_bytes(), 21);
    }

    // =========================================================================
    // Capacity Limit Tests
    // =========================================================================

    #[test]
    fn test_entry_too_large() {
        let mut wm = WorkingMemory::new();
        let large_value = vec![0u8; WORKING_MEMORY_ENTRY_SIZE_BYTES_MAX + 1];

        let result = wm.set("key", &large_value, None);
        assert!(matches!(
            result,
            Err(WorkingMemoryError::EntryTooLarge { .. })
        ));
    }

    #[test]
    fn test_memory_full() {
        let config = WorkingMemoryConfig {
            max_bytes: 100, // Very small for testing
            ..Default::default()
        };
        let mut wm = WorkingMemory::with_config(config);

        // Fill up most of the space
        wm.set("key1", &vec![0u8; 80], None).unwrap();

        // This should fail - not enough space
        let result = wm.set("key2", &vec![0u8; 50], None);
        assert!(matches!(result, Err(WorkingMemoryError::MemoryFull { .. })));
    }

    #[test]
    fn test_too_many_entries() {
        let config = WorkingMemoryConfig {
            max_entries: 3,
            ..Default::default()
        };
        let mut wm = WorkingMemory::with_config(config);

        wm.set("key1", b"v", None).unwrap();
        wm.set("key2", b"v", None).unwrap();
        wm.set("key3", b"v", None).unwrap();

        let result = wm.set("key4", b"v", None);
        assert!(matches!(
            result,
            Err(WorkingMemoryError::TooManyEntries { .. })
        ));
    }

    #[test]
    fn test_overwrite_does_not_increase_entry_count() {
        let config = WorkingMemoryConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut wm = WorkingMemory::with_config(config);

        wm.set("key1", b"v", None).unwrap();
        wm.set("key2", b"v", None).unwrap();

        // This should succeed - overwriting existing key
        wm.set("key1", b"new_value", None).unwrap();
        assert_eq!(wm.entry_count(), 2);
    }
}

// =============================================================================
// DST Tests - Use SimClock for TTL testing
// =============================================================================

#[cfg(test)]
mod dst_tests {
    use super::*;
    use crate::dst::{SimConfig, Simulation};

    #[tokio::test]
    async fn test_ttl_expiration() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|env| async move {
            let mut wm = WorkingMemory::new();
            wm.set_clock_ms(env.clock.now_ms());

            // Set entry with 1 second TTL
            wm.set("key1", b"value1", Some(1000)).unwrap();
            assert!(wm.exists("key1"));
            assert_eq!(wm.get("key1"), Some(b"value1".as_slice()));

            // Advance time by 500ms - not expired yet
            env.clock.advance_ms(500);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(wm.exists("key1"));

            // Advance time by 600ms (total 1100ms) - now expired
            env.clock.advance_ms(600);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(!wm.exists("key1"));
            assert_eq!(wm.get("key1"), None);

            Ok::<(), std::convert::Infallible>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_overwrite_resets_ttl() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|env| async move {
            let mut wm = WorkingMemory::new();
            wm.set_clock_ms(env.clock.now_ms());

            // Set entry with 1 second TTL
            wm.set("key1", b"value1", Some(1000)).unwrap();

            // Advance time by 800ms
            env.clock.advance_ms(800);
            wm.set_clock_ms(env.clock.now_ms());

            // Overwrite - should reset TTL
            wm.set("key1", b"new_value", Some(1000)).unwrap();

            // Advance time by 800ms (total 1600ms from start, but only 800ms from overwrite)
            env.clock.advance_ms(800);
            wm.set_clock_ms(env.clock.now_ms());

            // Should still exist because TTL was reset
            assert!(wm.exists("key1"));

            // Advance another 300ms (1100ms from overwrite) - now expired
            env.clock.advance_ms(300);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(!wm.exists("key1"));

            Ok::<(), std::convert::Infallible>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|env| async move {
            let mut wm = WorkingMemory::new();
            wm.set_clock_ms(env.clock.now_ms());

            // Set entries with different TTLs
            wm.set("short", b"v", Some(500)).unwrap(); // 500ms TTL
            wm.set("medium", b"v", Some(1000)).unwrap(); // 1s TTL
            wm.set("long", b"v", Some(2000)).unwrap(); // 2s TTL

            assert_eq!(wm.entry_count(), 3);

            // Advance 600ms - "short" expired
            env.clock.advance_ms(600);
            wm.set_clock_ms(env.clock.now_ms());

            let removed = wm.cleanup_expired();
            assert_eq!(removed, 1);
            assert_eq!(wm.entry_count(), 2);

            // Advance another 500ms (1100ms total) - "medium" expired
            env.clock.advance_ms(500);
            wm.set_clock_ms(env.clock.now_ms());

            let removed = wm.cleanup_expired();
            assert_eq!(removed, 1);
            assert_eq!(wm.entry_count(), 1);

            // "long" should still exist
            assert!(wm.exists("long"));

            Ok::<(), std::convert::Infallible>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_cleanup_frees_memory() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|env| async move {
            let mut wm = WorkingMemory::new();
            wm.set_clock_ms(env.clock.now_ms());

            // Set entry
            wm.set("key1", b"value1", Some(500)).unwrap();
            let used_before = wm.used_bytes();
            assert!(used_before > 0);

            // Expire it
            env.clock.advance_ms(600);
            wm.set_clock_ms(env.clock.now_ms());

            wm.cleanup_expired();
            assert_eq!(wm.used_bytes(), 0);

            Ok::<(), std::convert::Infallible>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_default_ttl() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|env| async move {
            let config = WorkingMemoryConfig {
                default_ttl_ms: 1000, // 1 second default
                ..Default::default()
            };
            let mut wm = WorkingMemory::with_config(config);
            wm.set_clock_ms(env.clock.now_ms());

            // Set without explicit TTL - should use default
            wm.set("key1", b"value1", None).unwrap();

            // After 900ms - still exists
            env.clock.advance_ms(900);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(wm.exists("key1"));

            // After 200ms more (1100ms total) - expired
            env.clock.advance_ms(200);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(!wm.exists("key1"));

            Ok::<(), std::convert::Infallible>(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_multiple_entries_different_ttls() {
        let sim = Simulation::new(SimConfig::with_seed(42));

        sim.run(|env| async move {
            let mut wm = WorkingMemory::new();
            wm.set_clock_ms(env.clock.now_ms());

            // Set entries at different times with same TTL
            wm.set("first", b"v", Some(1000)).unwrap();

            env.clock.advance_ms(300);
            wm.set_clock_ms(env.clock.now_ms());
            wm.set("second", b"v", Some(1000)).unwrap();

            env.clock.advance_ms(300);
            wm.set_clock_ms(env.clock.now_ms());
            wm.set("third", b"v", Some(1000)).unwrap();

            // Now at 600ms from start
            // first expires at 1000ms, second at 1300ms, third at 1600ms

            // At 800ms - all exist
            env.clock.advance_ms(200);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(wm.exists("first"));
            assert!(wm.exists("second"));
            assert!(wm.exists("third"));

            // At 1100ms - first expired
            env.clock.advance_ms(300);
            wm.set_clock_ms(env.clock.now_ms());
            assert!(!wm.exists("first"));
            assert!(wm.exists("second"));
            assert!(wm.exists("third"));

            Ok::<(), std::convert::Infallible>(())
        })
        .await
        .unwrap();
    }
}

// =============================================================================
// Property-Based Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::dst::{
        DeterministicRng, PropertyTest, PropertyTestable, SimClock, TimeAdvanceConfig,
    };

    /// Operations that can be performed on WorkingMemory
    #[derive(Debug, Clone)]
    enum WorkingMemoryOp {
        Set {
            key: String,
            value_len: usize,
            ttl_ms: u64,
        },
        Get {
            key: String,
        },
        Delete {
            key: String,
        },
        Cleanup,
    }

    /// Wrapper for property testing
    struct WorkingMemoryWrapper {
        inner: WorkingMemory,
        /// Track keys we've set (for generating realistic ops)
        known_keys: Vec<String>,
    }

    impl PropertyTestable for WorkingMemoryWrapper {
        type Operation = WorkingMemoryOp;

        fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation {
            let op_type = rng.next_usize(0, 3); // 0-3 inclusive = 4 options

            match op_type {
                0 => {
                    // Set - either new key or existing
                    let key = if !self.known_keys.is_empty() && rng.next_bool(0.3) {
                        // Use existing key (next_usize is inclusive, so use len-1)
                        let idx = rng.next_usize(0, self.known_keys.len() - 1);
                        self.known_keys[idx].clone()
                    } else {
                        // New key
                        format!("key_{}", rng.next_usize(0, 999))
                    };
                    let value_len = rng.next_usize(1, 1000);
                    let ttl_ms = rng.next_usize(100, 10000) as u64;
                    WorkingMemoryOp::Set {
                        key,
                        value_len,
                        ttl_ms,
                    }
                }
                1 => {
                    // Get - prefer existing keys
                    let key = if !self.known_keys.is_empty() && rng.next_bool(0.7) {
                        let idx = rng.next_usize(0, self.known_keys.len() - 1);
                        self.known_keys[idx].clone()
                    } else {
                        format!("key_{}", rng.next_usize(0, 999))
                    };
                    WorkingMemoryOp::Get { key }
                }
                2 => {
                    // Delete - prefer existing keys
                    let key = if !self.known_keys.is_empty() && rng.next_bool(0.5) {
                        let idx = rng.next_usize(0, self.known_keys.len() - 1);
                        self.known_keys[idx].clone()
                    } else {
                        format!("key_{}", rng.next_usize(0, 999))
                    };
                    WorkingMemoryOp::Delete { key }
                }
                _ => WorkingMemoryOp::Cleanup,
            }
        }

        fn apply_operation(&mut self, op: &Self::Operation, clock: &SimClock) {
            self.inner.set_clock_ms(clock.now_ms());

            match op {
                WorkingMemoryOp::Set {
                    key,
                    value_len,
                    ttl_ms,
                } => {
                    let value = vec![0u8; *value_len];
                    if self.inner.set(key, &value, Some(*ttl_ms)).is_ok() {
                        if !self.known_keys.contains(key) {
                            self.known_keys.push(key.clone());
                        }
                    }
                }
                WorkingMemoryOp::Get { key } => {
                    let _ = self.inner.get(key);
                }
                WorkingMemoryOp::Delete { key } => {
                    if self.inner.delete(key) {
                        self.known_keys.retain(|k| k != key);
                    }
                }
                WorkingMemoryOp::Cleanup => {
                    self.inner.cleanup_expired();
                    // Update known_keys based on what still exists
                    self.known_keys.retain(|k| self.inner.exists(k));
                }
            }
        }

        fn check_invariants(&self) -> Result<(), String> {
            // Invariant 1: used_bytes <= max_bytes
            if self.inner.used_bytes() > self.inner.config().max_bytes {
                return Err(format!(
                    "used_bytes {} exceeds max {}",
                    self.inner.used_bytes(),
                    self.inner.config().max_bytes
                ));
            }

            // Invariant 2: entry_count <= max_entries
            if self.inner.entry_count() > self.inner.config().max_entries {
                return Err(format!(
                    "entry_count {} exceeds max {}",
                    self.inner.entry_count(),
                    self.inner.config().max_entries
                ));
            }

            // Invariant 3: if empty, used_bytes should be 0
            if self.inner.is_empty() && self.inner.used_bytes() != 0 {
                return Err(format!(
                    "is_empty() but used_bytes is {}",
                    self.inner.used_bytes()
                ));
            }

            Ok(())
        }

        fn describe_state(&self) -> String {
            format!(
                "WorkingMemory {{ entries: {}, bytes: {}/{}, known_keys: {} }}",
                self.inner.entry_count(),
                self.inner.used_bytes(),
                self.inner.config().max_bytes,
                self.known_keys.len()
            )
        }
    }

    #[test]
    fn test_property_invariants() {
        let wm = WorkingMemoryWrapper {
            inner: WorkingMemory::new(),
            known_keys: Vec::new(),
        };

        PropertyTest::new(42)
            .with_max_operations(500)
            .with_time_advance(TimeAdvanceConfig::random(0, 5000, 0.3))
            .run_and_assert(wm);
    }

    #[test]
    fn test_property_invariants_small_capacity() {
        let config = WorkingMemoryConfig {
            max_bytes: 10_000, // 10KB
            max_entries: 50,
            ..Default::default()
        };
        let wm = WorkingMemoryWrapper {
            inner: WorkingMemory::with_config(config),
            known_keys: Vec::new(),
        };

        PropertyTest::new(12345)
            .with_max_operations(1000)
            .with_time_advance(TimeAdvanceConfig::random(0, 2000, 0.5))
            .run_and_assert(wm);
    }

    #[test]
    fn test_property_multi_seed() {
        for seed in [0, 1, 42, 12345, 99999] {
            let wm = WorkingMemoryWrapper {
                inner: WorkingMemory::new(),
                known_keys: Vec::new(),
            };

            PropertyTest::new(seed)
                .with_max_operations(200)
                .with_time_advance(TimeAdvanceConfig::random(0, 1000, 0.4))
                .run_and_assert(wm);
        }
    }
}
