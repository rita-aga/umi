//! SimStorage - Simulated Storage with Fault Injection
//!
//! TigerStyle: In-memory storage that can fail deterministically.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

use super::clock::SimClock;
use super::fault::{FaultInjector, FaultType};
use super::rng::DeterministicRng;

/// Storage errors.
#[derive(Error, Debug, Clone)]
pub enum StorageError {
    /// Write operation failed.
    #[error("storage write failed: {0}")]
    Write(String),

    /// Read operation failed.
    #[error("storage read failed: {0}")]
    Read(String),

    /// Delete operation failed.
    #[error("storage delete failed: {0}")]
    Delete(String),

    /// Storage corruption detected.
    #[error("storage corruption detected: {0}")]
    Corruption(String),

    /// Disk is full.
    #[error("disk full: {0}")]
    DiskFull(String),
}

/// Write-specific error alias.
pub type StorageWriteError = StorageError;

/// Read-specific error alias.
pub type StorageReadError = StorageError;

/// Storage statistics.
#[derive(Debug, Default)]
pub struct StorageStats {
    pub writes_count: u64,
    pub reads_count: u64,
    pub deletes_count: u64,
    pub entries_count: u64,
    pub bytes_total: u64,
    pub faults_injected_count: u64,
}

/// Entry metadata.
#[derive(Debug, Clone)]
struct StorageEntry {
    data: Vec<u8>,
    #[allow(dead_code)] // For future temporal queries
    created_at_ms: u64,
    #[allow(dead_code)] // For future temporal queries
    updated_at_ms: u64,
}

/// Simulated storage for DST testing.
///
/// TigerStyle:
/// - In-memory HashMap storage
/// - Fault injection at every operation
/// - Full statistics tracking
/// - Shared FaultInjector via Arc (Kelpie pattern)
#[derive(Debug)]
pub struct SimStorage {
    data: HashMap<String, StorageEntry>,
    clock: SimClock,
    #[allow(dead_code)] // For future random delays/corruption
    rng: DeterministicRng,
    /// Shared fault injector (via Arc for sharing with simulation harness)
    faults: Arc<FaultInjector>,
    // Statistics
    writes_count: AtomicU64,
    reads_count: AtomicU64,
    deletes_count: AtomicU64,
    faults_injected_count: AtomicU64,
}

impl SimStorage {
    /// Create a new simulated storage.
    ///
    /// TigerStyle: Takes Arc<FaultInjector> to share with simulation harness.
    #[must_use]
    pub fn new(clock: SimClock, rng: DeterministicRng, faults: Arc<FaultInjector>) -> Self {
        Self {
            data: HashMap::new(),
            clock,
            rng,
            faults,
            writes_count: AtomicU64::new(0),
            reads_count: AtomicU64::new(0),
            deletes_count: AtomicU64::new(0),
            faults_injected_count: AtomicU64::new(0),
        }
    }

    /// Write a value to storage.
    ///
    /// # Errors
    /// Returns error if fault injection triggers a failure.
    pub async fn write(&mut self, key: &str, value: &[u8]) -> Result<(), StorageError> {
        // Preconditions
        assert!(!key.is_empty(), "key must not be empty");
        assert!(value.len() <= 10_000_000, "value too large");

        // Check for fault injection
        if let Some(fault) = self.faults.should_inject("storage_write") {
            self.faults_injected_count.fetch_add(1, Ordering::Relaxed);
            return Err(self.fault_to_error(fault, "write"));
        }

        let now = self.clock.now_ms();
        let entry = StorageEntry {
            data: value.to_vec(),
            created_at_ms: now,
            updated_at_ms: now,
        };

        self.data.insert(key.to_string(), entry);
        self.writes_count.fetch_add(1, Ordering::Relaxed);

        // Postcondition
        assert!(self.data.contains_key(key), "key must exist after write");

        Ok(())
    }

    /// Read a value from storage.
    ///
    /// # Errors
    /// Returns error if fault injection triggers a failure.
    pub async fn read(&mut self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        // Precondition
        assert!(!key.is_empty(), "key must not be empty");

        // Check for fault injection
        if let Some(fault) = self.faults.should_inject("storage_read") {
            self.faults_injected_count.fetch_add(1, Ordering::Relaxed);
            return Err(self.fault_to_error(fault, "read"));
        }

        self.reads_count.fetch_add(1, Ordering::Relaxed);

        Ok(self.data.get(key).map(|entry| entry.data.clone()))
    }

    /// Delete a value from storage.
    ///
    /// # Errors
    /// Returns error if fault injection triggers a failure.
    pub async fn delete(&mut self, key: &str) -> Result<bool, StorageError> {
        // Precondition
        assert!(!key.is_empty(), "key must not be empty");

        // Check for fault injection
        if let Some(fault) = self.faults.should_inject("storage_delete") {
            self.faults_injected_count.fetch_add(1, Ordering::Relaxed);
            return Err(self.fault_to_error(fault, "delete"));
        }

        self.deletes_count.fetch_add(1, Ordering::Relaxed);

        Ok(self.data.remove(key).is_some())
    }

    /// Check if a key exists.
    ///
    /// # Errors
    /// Returns error if fault injection triggers a failure.
    pub async fn exists(&mut self, key: &str) -> Result<bool, StorageError> {
        // Precondition
        assert!(!key.is_empty(), "key must not be empty");

        // Check for fault injection
        if let Some(fault) = self.faults.should_inject("storage_read") {
            self.faults_injected_count.fetch_add(1, Ordering::Relaxed);
            return Err(self.fault_to_error(fault, "exists"));
        }

        Ok(self.data.contains_key(key))
    }

    /// List keys matching a prefix.
    ///
    /// # Errors
    /// Returns error if fault injection triggers a failure.
    pub async fn keys(&mut self, prefix: Option<&str>) -> Result<Vec<String>, StorageError> {
        // Check for fault injection
        if let Some(fault) = self.faults.should_inject("storage_read") {
            self.faults_injected_count.fetch_add(1, Ordering::Relaxed);
            return Err(self.fault_to_error(fault, "keys"));
        }

        let keys: Vec<String> = match prefix {
            Some(p) => self
                .data
                .keys()
                .filter(|k| k.starts_with(p))
                .cloned()
                .collect(),
            None => self.data.keys().cloned().collect(),
        };

        Ok(keys)
    }

    /// Get storage statistics.
    #[must_use]
    pub fn stats(&self) -> StorageStats {
        let bytes_total: u64 = self.data.values().map(|e| e.data.len() as u64).sum();

        StorageStats {
            writes_count: self.writes_count.load(Ordering::Relaxed),
            reads_count: self.reads_count.load(Ordering::Relaxed),
            deletes_count: self.deletes_count.load(Ordering::Relaxed),
            entries_count: self.data.len() as u64,
            bytes_total,
            faults_injected_count: self.faults_injected_count.load(Ordering::Relaxed),
        }
    }

    /// Get the clock.
    #[must_use]
    pub fn clock(&self) -> &SimClock {
        &self.clock
    }

    /// Get mutable clock.
    pub fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Convert a fault type to an error.
    fn fault_to_error(&self, fault: FaultType, operation: &str) -> StorageError {
        match fault {
            FaultType::StorageWriteFail => {
                StorageError::Write(format!("injected {} fault", operation))
            }
            FaultType::StorageReadFail => {
                StorageError::Read(format!("injected {} fault", operation))
            }
            FaultType::StorageDeleteFail => {
                StorageError::Delete(format!("injected {} fault", operation))
            }
            FaultType::StorageCorruption => {
                StorageError::Corruption(format!("injected {} fault", operation))
            }
            FaultType::StorageDiskFull => {
                StorageError::DiskFull(format!("injected {} fault", operation))
            }
            _ => StorageError::Write(format!("unexpected fault type for {}", operation)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::fault::{FaultConfig, FaultInjectorBuilder};

    fn create_storage() -> SimStorage {
        let clock = SimClock::new();
        let mut rng = DeterministicRng::new(42);
        let faults = Arc::new(FaultInjectorBuilder::new(rng.fork()).build());
        SimStorage::new(clock, rng, faults)
    }

    fn create_faulty_storage(fault_type: FaultType) -> SimStorage {
        let clock = SimClock::new();
        let mut rng = DeterministicRng::new(42);
        let faults = Arc::new(
            FaultInjectorBuilder::new(rng.fork())
                .with_fault(FaultConfig::new(fault_type, 1.0))
                .build(),
        );
        SimStorage::new(clock, rng, faults)
    }

    #[tokio::test]
    async fn test_write_and_read() {
        let mut storage = create_storage();

        storage.write("key1", b"value1").await.unwrap();
        let result = storage.read("key1").await.unwrap();

        assert_eq!(result, Some(b"value1".to_vec()));
    }

    #[tokio::test]
    async fn test_read_nonexistent() {
        let mut storage = create_storage();

        let result = storage.read("nonexistent").await.unwrap();

        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_delete() {
        let mut storage = create_storage();

        storage.write("key1", b"value1").await.unwrap();
        let deleted = storage.delete("key1").await.unwrap();

        assert!(deleted);
        assert_eq!(storage.read("key1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let mut storage = create_storage();

        let deleted = storage.delete("nonexistent").await.unwrap();

        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_exists() {
        let mut storage = create_storage();

        assert!(!storage.exists("key1").await.unwrap());

        storage.write("key1", b"value1").await.unwrap();

        assert!(storage.exists("key1").await.unwrap());
    }

    #[tokio::test]
    async fn test_keys() {
        let mut storage = create_storage();

        storage.write("user:1", b"alice").await.unwrap();
        storage.write("user:2", b"bob").await.unwrap();
        storage.write("session:1", b"data").await.unwrap();

        let mut all_keys = storage.keys(None).await.unwrap();
        all_keys.sort();
        assert_eq!(all_keys, vec!["session:1", "user:1", "user:2"]);

        let mut user_keys = storage.keys(Some("user:")).await.unwrap();
        user_keys.sort();
        assert_eq!(user_keys, vec!["user:1", "user:2"]);
    }

    #[tokio::test]
    async fn test_write_fault_injection() {
        let mut storage = create_faulty_storage(FaultType::StorageWriteFail);

        let result = storage.write("key", b"value").await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::Write(_)));
    }

    #[tokio::test]
    async fn test_read_fault_injection() {
        let mut storage = create_faulty_storage(FaultType::StorageReadFail);

        let result = storage.read("key").await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::Read(_)));
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let mut storage = create_storage();

        storage.write("k1", b"v1").await.unwrap();
        storage.write("k2", b"v2").await.unwrap();
        storage.read("k1").await.unwrap();
        storage.read("k3").await.unwrap(); // nonexistent
        storage.delete("k1").await.unwrap();

        let stats = storage.stats();

        assert_eq!(stats.writes_count, 2);
        assert_eq!(stats.reads_count, 2);
        assert_eq!(stats.deletes_count, 1);
        assert_eq!(stats.entries_count, 1); // k2 remains
    }

    #[tokio::test]
    async fn test_clear() {
        let mut storage = create_storage();

        storage.write("k1", b"v1").await.unwrap();
        storage.write("k2", b"v2").await.unwrap();

        storage.clear();

        assert_eq!(storage.stats().entries_count, 0);
    }

    #[test]
    #[should_panic(expected = "key must not be empty")]
    fn test_write_empty_key() {
        let mut storage = create_storage();
        let _ = tokio_test::block_on(storage.write("", b"value"));
    }
}
