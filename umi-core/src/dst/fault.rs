//! FaultInjector - Probabilistic Fault Injection
//!
//! TigerStyle: Explicit fault injection for chaos testing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::rng::DeterministicRng;
use crate::constants::DST_FAULT_PROBABILITY_MAX;

/// Types of faults that can be injected.
///
/// TigerStyle: Every fault type is explicit and documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultType {
    // =========================================================================
    // Storage Faults
    // =========================================================================
    /// Write operation fails
    StorageWriteFail,
    /// Read operation fails
    StorageReadFail,
    /// Delete operation fails
    StorageDeleteFail,
    /// Storage corruption (data garbled)
    StorageCorruption,
    /// Disk full error
    StorageDiskFull,
    /// Storage latency spike
    StorageLatency,

    // =========================================================================
    // Database Faults
    // =========================================================================
    /// Connection fails
    DbConnectionFail,
    /// Query timeout
    DbQueryTimeout,
    /// Deadlock detected
    DbDeadlock,
    /// Serialization failure (retry needed)
    DbSerializationFail,
    /// Connection pool exhausted
    DbPoolExhausted,

    // =========================================================================
    // Network Faults
    // =========================================================================
    /// Connection timeout
    NetworkTimeout,
    /// Connection refused
    NetworkConnectionRefused,
    /// DNS resolution fails
    NetworkDnsFail,
    /// Partial write (incomplete data)
    NetworkPartialWrite,
    /// Connection reset
    NetworkReset,

    // =========================================================================
    // LLM/API Faults
    // =========================================================================
    /// LLM request timeout
    LlmTimeout,
    /// Rate limit exceeded
    LlmRateLimit,
    /// Context length exceeded
    LlmContextOverflow,
    /// Invalid response format
    LlmInvalidResponse,
    /// Service unavailable
    LlmServiceUnavailable,

    // =========================================================================
    // Resource Faults
    // =========================================================================
    /// Out of memory
    ResourceOom,
    /// Too many open files
    ResourceFileLimit,
    /// CPU throttling
    ResourceCpuThrottle,

    // =========================================================================
    // Time Faults
    // =========================================================================
    /// Clock skew (time jumps)
    TimeClockSkew,
    /// Leap second handling
    TimeLeapSecond,
}

impl FaultType {
    /// Get the fault type name as a string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::StorageWriteFail => "storage_write_fail",
            Self::StorageReadFail => "storage_read_fail",
            Self::StorageDeleteFail => "storage_delete_fail",
            Self::StorageCorruption => "storage_corruption",
            Self::StorageDiskFull => "storage_disk_full",
            Self::StorageLatency => "storage_latency",
            Self::DbConnectionFail => "db_connection_fail",
            Self::DbQueryTimeout => "db_query_timeout",
            Self::DbDeadlock => "db_deadlock",
            Self::DbSerializationFail => "db_serialization_fail",
            Self::DbPoolExhausted => "db_pool_exhausted",
            Self::NetworkTimeout => "network_timeout",
            Self::NetworkConnectionRefused => "network_connection_refused",
            Self::NetworkDnsFail => "network_dns_fail",
            Self::NetworkPartialWrite => "network_partial_write",
            Self::NetworkReset => "network_reset",
            Self::LlmTimeout => "llm_timeout",
            Self::LlmRateLimit => "llm_rate_limit",
            Self::LlmContextOverflow => "llm_context_overflow",
            Self::LlmInvalidResponse => "llm_invalid_response",
            Self::LlmServiceUnavailable => "llm_service_unavailable",
            Self::ResourceOom => "resource_oom",
            Self::ResourceFileLimit => "resource_file_limit",
            Self::ResourceCpuThrottle => "resource_cpu_throttle",
            Self::TimeClockSkew => "time_clock_skew",
            Self::TimeLeapSecond => "time_leap_second",
        }
    }
}

/// Configuration for a specific fault.
#[derive(Debug, Clone)]
pub struct FaultConfig {
    /// The type of fault
    pub fault_type: FaultType,
    /// Probability of injection (0.0 to 1.0)
    pub probability: f64,
    /// Optional operation filter (substring match)
    pub operation_filter: Option<String>,
    /// Maximum number of injections (None = unlimited)
    pub max_injections: Option<u64>,
}

impl FaultConfig {
    /// Create a new fault configuration.
    ///
    /// # Panics
    /// Panics if probability is not in [0, 1].
    #[must_use]
    pub fn new(fault_type: FaultType, probability: f64) -> Self {
        // Precondition
        assert!(
            (0.0..=DST_FAULT_PROBABILITY_MAX).contains(&probability),
            "probability must be in [0, {}], got {}",
            DST_FAULT_PROBABILITY_MAX,
            probability
        );

        Self {
            fault_type,
            probability,
            operation_filter: None,
            max_injections: None,
        }
    }

    /// Set operation filter (fault only applies to matching operations).
    #[must_use]
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.operation_filter = Some(filter.into());
        self
    }

    /// Set maximum number of injections.
    #[must_use]
    pub fn with_max_injections(mut self, max: u64) -> Self {
        // Precondition
        assert!(max > 0, "max_injections must be positive");
        self.max_injections = Some(max);
        self
    }
}

/// Fault injection statistics.
#[derive(Debug, Default)]
struct FaultStats {
    injection_count: AtomicU64,
}

/// Fault injector for simulation testing.
///
/// TigerStyle:
/// - Explicit fault registration
/// - Deterministic through RNG
/// - Statistics tracked
/// - Interior mutability for sharing via Arc
#[derive(Debug)]
pub struct FaultInjector {
    /// RNG wrapped in Mutex for interior mutability (allows sharing via Arc)
    rng: Mutex<DeterministicRng>,
    configs: Vec<FaultConfig>,
    stats: HashMap<FaultType, FaultStats>,
    /// Current injection counts (wrapped in Mutex for interior mutability)
    injection_counts: Mutex<HashMap<FaultType, u64>>,
}

impl FaultInjector {
    /// Create a new fault injector with the given RNG.
    #[must_use]
    pub fn new(rng: DeterministicRng) -> Self {
        Self {
            rng: Mutex::new(rng),
            configs: Vec::new(),
            stats: HashMap::new(),
            injection_counts: Mutex::new(HashMap::new()),
        }
    }

    /// Register a fault configuration.
    ///
    /// Note: Registration must happen before sharing via Arc.
    pub fn register(&mut self, config: FaultConfig) {
        // Precondition
        assert!(
            config.probability >= 0.0,
            "probability must be non-negative"
        );
        assert!(config.probability <= 1.0, "probability must be <= 1.0");

        // Initialize stats for this fault type
        self.stats.entry(config.fault_type).or_default();
        self.injection_counts
            .lock()
            .unwrap()
            .entry(config.fault_type)
            .or_insert(0);

        self.configs.push(config);
    }

    /// Check if a fault should be injected for the given operation.
    ///
    /// Returns the fault type if one should be injected, None otherwise.
    ///
    /// TigerStyle: Uses interior mutability (Mutex) so can be called on &self,
    /// allowing FaultInjector to be shared via Arc.
    pub fn should_inject(&self, operation: &str) -> Option<FaultType> {
        for config in &self.configs {
            // Check operation filter
            if let Some(ref filter) = config.operation_filter {
                if !operation.contains(filter) {
                    continue;
                }
            }

            // Check max injections
            if let Some(max) = config.max_injections {
                let counts = self.injection_counts.lock().unwrap();
                let count = counts.get(&config.fault_type).copied().unwrap_or(0);
                if count >= max {
                    continue;
                }
            }

            // Roll for injection (uses interior mutability)
            let should_inject = {
                let mut rng = self.rng.lock().unwrap();
                rng.next_bool(config.probability)
            };

            if should_inject {
                // Update stats
                if let Some(stats) = self.stats.get(&config.fault_type) {
                    stats.injection_count.fetch_add(1, Ordering::Relaxed);
                }
                {
                    let mut counts = self.injection_counts.lock().unwrap();
                    if let Some(count) = counts.get_mut(&config.fault_type) {
                        *count += 1;
                    }
                }

                return Some(config.fault_type);
            }
        }

        None
    }

    /// Get injection statistics.
    #[must_use]
    pub fn injection_stats(&self) -> HashMap<String, u64> {
        self.stats
            .iter()
            .map(|(fault_type, stats)| {
                (
                    fault_type.as_str().to_string(),
                    stats.injection_count.load(Ordering::Relaxed),
                )
            })
            .collect()
    }

    /// Get total number of injections.
    #[must_use]
    pub fn total_injections(&self) -> u64 {
        self.stats
            .values()
            .map(|s| s.injection_count.load(Ordering::Relaxed))
            .sum()
    }

    /// Reset all statistics.
    pub fn reset_stats(&self) {
        for stats in self.stats.values() {
            stats.injection_count.store(0, Ordering::Relaxed);
        }
        let mut counts = self.injection_counts.lock().unwrap();
        for count in counts.values_mut() {
            *count = 0;
        }
    }
}

/// Builder for FaultInjector (Kelpie pattern).
///
/// TigerStyle: Builder pattern for clean configuration before sharing via Arc.
pub struct FaultInjectorBuilder {
    rng: DeterministicRng,
    configs: Vec<FaultConfig>,
}

impl FaultInjectorBuilder {
    /// Create a new builder with the given RNG.
    #[must_use]
    pub fn new(rng: DeterministicRng) -> Self {
        Self {
            rng,
            configs: Vec::new(),
        }
    }

    /// Add a fault configuration.
    #[must_use]
    pub fn with_fault(mut self, config: FaultConfig) -> Self {
        self.configs.push(config);
        self
    }

    /// Add common storage faults.
    #[must_use]
    pub fn with_storage_faults(self, probability: f64) -> Self {
        self.with_fault(FaultConfig::new(FaultType::StorageWriteFail, probability))
            .with_fault(FaultConfig::new(FaultType::StorageReadFail, probability))
    }

    /// Add common database faults.
    #[must_use]
    pub fn with_db_faults(self, probability: f64) -> Self {
        self.with_fault(FaultConfig::new(FaultType::DbConnectionFail, probability))
            .with_fault(FaultConfig::new(FaultType::DbQueryTimeout, probability))
    }

    /// Add common LLM/API faults.
    #[must_use]
    pub fn with_llm_faults(self, probability: f64) -> Self {
        self.with_fault(FaultConfig::new(FaultType::LlmTimeout, probability))
            .with_fault(FaultConfig::new(FaultType::LlmRateLimit, probability))
    }

    /// Build the FaultInjector.
    #[must_use]
    pub fn build(self) -> FaultInjector {
        let mut injector = FaultInjector::new(self.rng);
        for config in self.configs {
            injector.register(config);
        }
        injector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_no_faults_registered() {
        let rng = DeterministicRng::new(42);
        let injector = FaultInjector::new(rng);

        for _ in 0..100 {
            assert!(injector.should_inject("any_operation").is_none());
        }
    }

    #[test]
    fn test_always_inject() {
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(rng);
        injector.register(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        for _ in 0..10 {
            assert_eq!(
                injector.should_inject("storage_write"),
                Some(FaultType::StorageWriteFail)
            );
        }
    }

    #[test]
    fn test_never_inject() {
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(rng);
        injector.register(FaultConfig::new(FaultType::StorageWriteFail, 0.0));

        for _ in 0..100 {
            assert!(injector.should_inject("storage_write").is_none());
        }
    }

    #[test]
    fn test_operation_filter() {
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(rng);
        injector.register(FaultConfig::new(FaultType::StorageWriteFail, 1.0).with_filter("write"));

        // Should inject for write operations
        assert_eq!(
            injector.should_inject("storage_write"),
            Some(FaultType::StorageWriteFail)
        );

        // Should not inject for read operations
        assert!(injector.should_inject("storage_read").is_none());
    }

    #[test]
    fn test_max_injections() {
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(rng);
        injector
            .register(FaultConfig::new(FaultType::StorageWriteFail, 1.0).with_max_injections(2));

        // First two should inject
        assert_eq!(
            injector.should_inject("op"),
            Some(FaultType::StorageWriteFail)
        );
        assert_eq!(
            injector.should_inject("op"),
            Some(FaultType::StorageWriteFail)
        );

        // Third should not
        assert!(injector.should_inject("op").is_none());
    }

    #[test]
    fn test_injection_stats() {
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(rng);
        injector.register(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        injector.should_inject("op");
        injector.should_inject("op");
        injector.should_inject("op");

        let stats = injector.injection_stats();
        assert_eq!(stats.get("storage_write_fail"), Some(&3));
        assert_eq!(injector.total_injections(), 3);
    }

    #[test]
    fn test_reset_stats() {
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(rng);
        injector.register(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

        injector.should_inject("op");
        assert_eq!(injector.total_injections(), 1);

        injector.reset_stats();
        assert_eq!(injector.total_injections(), 0);
    }

    #[test]
    fn test_fault_type_as_str() {
        assert_eq!(FaultType::StorageWriteFail.as_str(), "storage_write_fail");
        assert_eq!(FaultType::DbDeadlock.as_str(), "db_deadlock");
        assert_eq!(FaultType::LlmRateLimit.as_str(), "llm_rate_limit");
    }

    #[test]
    #[should_panic(expected = "probability must be in")]
    fn test_invalid_probability() {
        let _ = FaultConfig::new(FaultType::StorageWriteFail, 1.5);
    }

    #[test]
    #[should_panic(expected = "max_injections must be positive")]
    fn test_invalid_max_injections() {
        let _ = FaultConfig::new(FaultType::StorageWriteFail, 0.5).with_max_injections(0);
    }

    #[test]
    fn test_builder_pattern() {
        let rng = DeterministicRng::new(42);
        let injector = FaultInjectorBuilder::new(rng)
            .with_storage_faults(0.1)
            .with_db_faults(0.05)
            .build();

        // Just verify it builds
        assert_eq!(injector.total_injections(), 0);
    }

    #[test]
    fn test_arc_sharing() {
        // Verify FaultInjector can be shared via Arc
        let rng = DeterministicRng::new(42);
        let injector = Arc::new(
            FaultInjectorBuilder::new(rng)
                .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 1.0))
                .build(),
        );

        // Can call should_inject on shared Arc
        assert_eq!(
            injector.should_inject("storage_write"),
            Some(FaultType::StorageWriteFail)
        );

        // Clone and use
        let injector2 = Arc::clone(&injector);
        assert_eq!(
            injector2.should_inject("storage_write"),
            Some(FaultType::StorageWriteFail)
        );

        // Stats are shared
        assert_eq!(injector.total_injections(), 2);
    }
}
