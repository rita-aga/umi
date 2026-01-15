//! `SimStorageBackend` - In-Memory Storage for Testing
//!
//! `TigerStyle`: Deterministic testing with fault injection.
//!
//! # Simulation-First
//!
//! This file follows simulation-first development:
//! 1. Tests are written FIRST (below)
//! 2. Implementation follows to make tests pass
//! 3. DST integration enables fault injection

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use crate::dst::{DeterministicRng, FaultConfig, FaultInjector, SimClock, SimConfig};

use super::backend::StorageBackend;
use super::entity::{Entity, EntityType};
use super::error::{StorageError, StorageResult};

// =============================================================================
// SimStorageBackend
// =============================================================================

/// In-memory storage backend for testing.
///
/// `TigerStyle`:
/// - Deterministic via `SimClock` and `DeterministicRng`
/// - Fault injection via `FaultInjector`
/// - Thread-safe with `RwLock`
#[derive(Debug, Clone)]
pub struct SimStorageBackend {
    /// Stored entities indexed by ID
    storage: Arc<RwLock<HashMap<String, Entity>>>,
    /// Fault injector for simulating failures
    fault_injector: Arc<FaultInjector>,
    /// Simulated clock
    clock: SimClock,
    /// Deterministic RNG for operations
    #[allow(dead_code)]
    rng: Arc<RwLock<DeterministicRng>>,
}

impl SimStorageBackend {
    /// Create a new `SimStorageBackend` with given config.
    #[must_use]
    pub fn new(config: SimConfig) -> Self {
        let mut rng = DeterministicRng::new(config.seed());
        let fault_rng = rng.fork();

        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            fault_injector: Arc::new(FaultInjector::new(fault_rng)),
            clock: SimClock::new(),
            rng: Arc::new(RwLock::new(rng)),
        }
    }

    /// Create a new `SimStorageBackend` with a shared fault injector.
    ///
    /// This constructor accepts an external `FaultInjector` (typically shared
    /// from a `Simulation`), allowing fault injection tests to work correctly.
    ///
    /// **DST-First Discovery**: This method was added after discovering that
    /// `Memory::sim()` created isolated providers not connected to the Simulation's
    /// FaultInjector. See `.progress/015_DST_FIRST_DEMO.md` for the discovery process.
    ///
    /// # Arguments
    /// * `config` - Simulation configuration
    /// * `fault_injector` - Shared fault injector from the simulation environment
    ///
    /// # Example
    /// ```rust,ignore
    /// use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};
    /// use umi_memory::storage::SimStorageBackend;
    ///
    /// let sim = Simulation::new(SimConfig::with_seed(42))
    ///     .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));
    ///
    /// sim.run(|env| async move {
    ///     let storage = SimStorageBackend::with_fault_injector(
    ///         env.config,
    ///         Arc::clone(&env.faults)
    ///     );
    ///     // Storage operations now have fault injection applied
    ///     Ok(())
    /// }).await;
    /// ```
    #[must_use]
    pub fn with_fault_injector(config: SimConfig, fault_injector: Arc<FaultInjector>) -> Self {
        let rng = DeterministicRng::new(config.seed());

        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            fault_injector, // Use the provided one instead of creating new
            clock: SimClock::new(),
            rng: Arc::new(RwLock::new(rng)),
        }
    }

    /// Add fault configuration.
    ///
    /// Note: Creates a new backend with the fault registered.
    /// `FaultInjector` uses interior mutability, but register needs &mut self
    /// which we can't do through Arc. So we create with faults upfront.
    #[must_use]
    pub fn with_faults(mut self, config: FaultConfig) -> Self {
        // Get mutable access before wrapping in Arc
        // This only works because we haven't shared the Arc yet
        Arc::get_mut(&mut self.fault_injector)
            .expect("cannot add faults after backend is shared")
            .register(config);
        self
    }

    /// Get the simulated clock.
    #[must_use]
    pub fn clock(&self) -> &SimClock {
        &self.clock
    }

    /// Get fault injector for inspection.
    #[must_use]
    pub fn fault_injector(&self) -> &Arc<FaultInjector> {
        &self.fault_injector
    }

    /// Check if a fault should be injected for an operation.
    fn maybe_inject_fault(&self, operation: &str) -> StorageResult<()> {
        if let Some(fault_type) = self.fault_injector.should_inject(operation) {
            Err(StorageError::simulated_fault(format!(
                "{fault_type:?} during {operation}"
            )))
        } else {
            Ok(())
        }
    }

    /// Get entity count (for testing).
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.storage.read().unwrap().len()
    }
}

#[async_trait]
impl StorageBackend for SimStorageBackend {
    #[tracing::instrument(skip(self, entity), fields(entity_id = %entity.id))]
    async fn store_entity(&self, entity: &Entity) -> StorageResult<String> {
        // Check for faults
        self.maybe_inject_fault("store")?;

        // Preconditions
        assert!(!entity.id.is_empty(), "entity must have id");

        let mut storage = self.storage.write().unwrap();
        storage.insert(entity.id.clone(), entity.clone());

        Ok(entity.id.clone())
    }

    #[tracing::instrument(skip(self))]
    async fn get_entity(&self, id: &str) -> StorageResult<Option<Entity>> {
        // Check for faults
        self.maybe_inject_fault("get")?;

        let storage = self.storage.read().unwrap();
        Ok(storage.get(id).cloned())
    }

    async fn delete_entity(&self, id: &str) -> StorageResult<bool> {
        // Check for faults
        self.maybe_inject_fault("delete")?;

        let mut storage = self.storage.write().unwrap();
        Ok(storage.remove(id).is_some())
    }

    #[tracing::instrument(skip(self), fields(query_len = query.len()))]
    async fn search(&self, query: &str, limit: usize) -> StorageResult<Vec<Entity>> {
        // Check for faults
        self.maybe_inject_fault("search")?;

        let storage = self.storage.read().unwrap();
        let query_lower = query.to_lowercase();

        let mut results: Vec<Entity> = storage
            .values()
            .filter(|e| {
                e.name.to_lowercase().contains(&query_lower)
                    || e.content.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect();

        // Sort by name for determinism
        results.sort_by(|a, b| a.name.cmp(&b.name));

        // Apply limit
        results.truncate(limit);

        Ok(results)
    }

    async fn list_entities(
        &self,
        entity_type: Option<EntityType>,
        limit: usize,
        offset: usize,
    ) -> StorageResult<Vec<Entity>> {
        // Check for faults
        self.maybe_inject_fault("list")?;

        let storage = self.storage.read().unwrap();

        let mut results: Vec<Entity> = storage
            .values()
            .filter(|e| entity_type.map_or(true, |t| e.entity_type == t))
            .cloned()
            .collect();

        // Sort by created_at for determinism
        results.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        // Apply offset and limit
        let results: Vec<Entity> = results.into_iter().skip(offset).take(limit).collect();

        Ok(results)
    }

    async fn count_entities(&self, entity_type: Option<EntityType>) -> StorageResult<usize> {
        // Check for faults
        self.maybe_inject_fault("count")?;

        let storage = self.storage.read().unwrap();

        let count = storage
            .values()
            .filter(|e| entity_type.map_or(true, |t| e.entity_type == t))
            .count();

        Ok(count)
    }

    async fn clear(&self) -> StorageResult<()> {
        // Check for faults
        self.maybe_inject_fault("clear")?;

        let mut storage = self.storage.write().unwrap();
        storage.clear();
        Ok(())
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

    #[tokio::test]
    async fn test_store_and_get() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        let entity = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "Friend".to_string(),
        );
        let id = entity.id.clone();

        backend.store_entity(&entity).await.unwrap();

        let retrieved = backend.get_entity(&id).await.unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.entity_type, EntityType::Person);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        let result = backend.get_entity("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_store_updates_existing() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        let mut entity = Entity::new(EntityType::Note, "Note".to_string(), "Original".to_string());
        let id = entity.id.clone();

        backend.store_entity(&entity).await.unwrap();

        // Update content
        entity.update_content("Updated".to_string());
        backend.store_entity(&entity).await.unwrap();

        let retrieved = backend.get_entity(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated");

        // Should still be just one entity
        assert_eq!(backend.entity_count(), 1);
    }

    #[tokio::test]
    async fn test_delete() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        let entity = Entity::new(EntityType::Task, "Task".to_string(), "Do it".to_string());
        let id = entity.id.clone();

        backend.store_entity(&entity).await.unwrap();
        assert_eq!(backend.entity_count(), 1);

        let deleted = backend.delete_entity(&id).await.unwrap();
        assert!(deleted);
        assert_eq!(backend.entity_count(), 0);

        // Deleting again returns false
        let deleted = backend.delete_entity(&id).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_clear() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Note,
                "A".to_string(),
                "a".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Note,
                "B".to_string(),
                "b".to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(backend.entity_count(), 2);

        backend.clear().await.unwrap();

        assert_eq!(backend.entity_count(), 0);
    }

    // =========================================================================
    // Search Tests
    // =========================================================================

    #[tokio::test]
    async fn test_search_by_name() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "Friend".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Bob".to_string(),
                "Colleague".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Charlie".to_string(),
                "Neighbor".to_string(),
            ))
            .await
            .unwrap();

        let results = backend.search("alice", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");
    }

    #[tokio::test]
    async fn test_search_by_content() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Note,
                "Work".to_string(),
                "Meeting with team".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Note,
                "Personal".to_string(),
                "Call mom".to_string(),
            ))
            .await
            .unwrap();

        let results = backend.search("meeting", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Work");
    }

    #[tokio::test]
    async fn test_search_case_insensitive() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Topic,
                "Rust".to_string(),
                "Programming language".to_string(),
            ))
            .await
            .unwrap();

        let results = backend.search("RUST", 10).await.unwrap();
        assert_eq!(results.len(), 1);

        let results = backend.search("rust", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_limit() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        for i in 0..10 {
            backend
                .store_entity(&Entity::new(
                    EntityType::Note,
                    format!("Note {i}"),
                    "common content".to_string(),
                ))
                .await
                .unwrap();
        }

        let results = backend.search("common", 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    // =========================================================================
    // List Tests
    // =========================================================================

    #[tokio::test]
    async fn test_list_all() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "a".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Project,
                "Umi".to_string(),
                "b".to_string(),
            ))
            .await
            .unwrap();

        let results = backend.list_entities(None, 100, 0).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_list_by_type() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "a".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Bob".to_string(),
                "b".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Project,
                "Umi".to_string(),
                "c".to_string(),
            ))
            .await
            .unwrap();

        let results = backend
            .list_entities(Some(EntityType::Person), 100, 0)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        let results = backend
            .list_entities(Some(EntityType::Project), 100, 0)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_list_with_offset() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        for i in 0..5 {
            // Small delay to ensure different timestamps
            backend
                .store_entity(&Entity::new(
                    EntityType::Note,
                    format!("Note {i}"),
                    "content".to_string(),
                ))
                .await
                .unwrap();
        }

        let all = backend.list_entities(None, 100, 0).await.unwrap();
        assert_eq!(all.len(), 5);

        let offset_2 = backend.list_entities(None, 100, 2).await.unwrap();
        assert_eq!(offset_2.len(), 3);
    }

    #[tokio::test]
    async fn test_count() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42));

        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Alice".to_string(),
                "a".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Person,
                "Bob".to_string(),
                "b".to_string(),
            ))
            .await
            .unwrap();
        backend
            .store_entity(&Entity::new(
                EntityType::Task,
                "Task".to_string(),
                "c".to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(backend.count_entities(None).await.unwrap(), 3);
        assert_eq!(
            backend
                .count_entities(Some(EntityType::Person))
                .await
                .unwrap(),
            2
        );
        assert_eq!(
            backend
                .count_entities(Some(EntityType::Task))
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            backend
                .count_entities(Some(EntityType::Project))
                .await
                .unwrap(),
            0
        );
    }
}

// =============================================================================
// DST Tests - Fault Injection
// =============================================================================

#[cfg(test)]
mod dst_tests {
    use super::*;
    use crate::dst::FaultType;

    #[tokio::test]
    async fn test_fault_injection_on_store() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42)).with_faults(
            FaultConfig::new(FaultType::StorageWriteFail, 1.0) // Always fail
                .with_filter("store"),
        );

        let entity = Entity::new(EntityType::Note, "Test".to_string(), "content".to_string());
        let result = backend.store_entity(&entity).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(StorageError::SimulatedFault { .. })));
    }

    #[tokio::test]
    async fn test_fault_injection_on_get() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42))
            .with_faults(FaultConfig::new(FaultType::StorageReadFail, 1.0).with_filter("get"));

        // Store should work (no fault on store)
        let entity = Entity::new(EntityType::Note, "Test".to_string(), "content".to_string());
        backend.store_entity(&entity).await.unwrap();

        // Get should fail
        let result = backend.get_entity(&entity.id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fault_injection_probability() {
        // 50% fault probability
        let backend = SimStorageBackend::new(SimConfig::with_seed(42))
            .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 0.5).with_filter("store"));

        let mut successes = 0;
        let mut failures = 0;

        for i in 0..100 {
            let entity = Entity::new(EntityType::Note, format!("Test {i}"), "content".to_string());
            match backend.store_entity(&entity).await {
                Ok(_) => successes += 1,
                Err(_) => failures += 1,
            }
        }

        // With 50% probability, we should see both successes and failures
        // (statistically very unlikely to see all one or the other with 100 trials)
        assert!(successes > 0, "expected some successes");
        assert!(failures > 0, "expected some failures");
    }

    #[tokio::test]
    async fn test_fault_injection_stats() {
        let backend = SimStorageBackend::new(SimConfig::with_seed(42))
            .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 1.0).with_filter("store"));

        let entity = Entity::new(EntityType::Note, "Test".to_string(), "content".to_string());

        // Try 5 times
        for _ in 0..5 {
            let _ = backend.store_entity(&entity).await;
        }

        let total = backend.fault_injector().total_injections();
        assert_eq!(total, 5);
    }
}

// =============================================================================
// Property-Based Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::dst::{PropertyTest, PropertyTestable, TimeAdvanceConfig};

    /// Operations that can be performed on storage
    #[derive(Debug, Clone)]
    enum StorageOp {
        Store {
            entity_type: EntityType,
            name: String,
        },
        Get {
            id: String,
        },
        Delete {
            id: String,
        },
        Search {
            query: String,
        },
        List {
            entity_type: Option<EntityType>,
        },
    }

    /// Wrapper for property testing
    struct StorageWrapper {
        backend: SimStorageBackend,
        known_ids: Vec<String>,
    }

    impl PropertyTestable for StorageWrapper {
        type Operation = StorageOp;

        fn generate_operation(&self, rng: &mut DeterministicRng) -> Self::Operation {
            let op_type = rng.next_usize(0, 4);

            match op_type {
                0 => {
                    // Store
                    let types = EntityType::all();
                    let type_idx = rng.next_usize(0, types.len() - 1);
                    StorageOp::Store {
                        entity_type: types[type_idx],
                        name: format!("entity_{}", rng.next_usize(0, 999)),
                    }
                }
                1 => {
                    // Get
                    let id = if !self.known_ids.is_empty() && rng.next_bool(0.7) {
                        let idx = rng.next_usize(0, self.known_ids.len() - 1);
                        self.known_ids[idx].clone()
                    } else {
                        format!("unknown_{}", rng.next_usize(0, 99))
                    };
                    StorageOp::Get { id }
                }
                2 => {
                    // Delete
                    let id = if !self.known_ids.is_empty() && rng.next_bool(0.5) {
                        let idx = rng.next_usize(0, self.known_ids.len() - 1);
                        self.known_ids[idx].clone()
                    } else {
                        format!("unknown_{}", rng.next_usize(0, 99))
                    };
                    StorageOp::Delete { id }
                }
                3 => {
                    // Search
                    StorageOp::Search {
                        query: format!("entity_{}", rng.next_usize(0, 9)),
                    }
                }
                _ => {
                    // List
                    let types = EntityType::all();
                    let entity_type = if rng.next_bool(0.3) {
                        Some(types[rng.next_usize(0, types.len() - 1)])
                    } else {
                        None
                    };
                    StorageOp::List { entity_type }
                }
            }
        }

        fn apply_operation(&mut self, op: &Self::Operation, _clock: &SimClock) {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                match op {
                    StorageOp::Store { entity_type, name } => {
                        let entity = Entity::new(*entity_type, name.clone(), "content".to_string());
                        if self.backend.store_entity(&entity).await.is_ok()
                            && !self.known_ids.contains(&entity.id)
                        {
                            self.known_ids.push(entity.id);
                        }
                    }
                    StorageOp::Get { id } => {
                        let _ = self.backend.get_entity(id).await;
                    }
                    StorageOp::Delete { id } => {
                        if self.backend.delete_entity(id).await.unwrap_or(false) {
                            self.known_ids.retain(|i| i != id);
                        }
                    }
                    StorageOp::Search { query } => {
                        let _ = self.backend.search(query, 10).await;
                    }
                    StorageOp::List { entity_type } => {
                        let _ = self.backend.list_entities(*entity_type, 10, 0).await;
                    }
                }
            });
        }

        fn check_invariants(&self) -> Result<(), String> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                // Invariant 1: count matches known_ids (after filtering deleted)
                let count = self
                    .backend
                    .count_entities(None)
                    .await
                    .map_err(|e| e.to_string())?;

                // known_ids might include some that were deleted by parallel ops,
                // so we just check count <= known_ids.len() doesn't hold (they should be close)
                // Actually, we maintain known_ids properly, so count should equal known_ids.len()
                if count != self.known_ids.len() {
                    // This could happen if store failed silently - but we track correctly
                    // So this is an invariant violation
                    return Err(format!(
                        "count {} != known_ids.len() {}",
                        count,
                        self.known_ids.len()
                    ));
                }

                Ok(())
            })
        }

        fn describe_state(&self) -> String {
            format!(
                "SimStorageBackend {{ entities: {}, known_ids: {} }}",
                self.backend.entity_count(),
                self.known_ids.len()
            )
        }
    }

    #[test]
    fn test_property_invariants() {
        let wrapper = StorageWrapper {
            backend: SimStorageBackend::new(SimConfig::with_seed(42)),
            known_ids: Vec::new(),
        };

        PropertyTest::new(42)
            .with_max_operations(200)
            .with_time_advance(TimeAdvanceConfig::none())
            .run_and_assert(wrapper);
    }

    #[test]
    fn test_property_multi_seed() {
        for seed in [0, 1, 42, 12345, 99999] {
            let wrapper = StorageWrapper {
                backend: SimStorageBackend::new(SimConfig::with_seed(seed)),
                known_ids: Vec::new(),
            };

            PropertyTest::new(seed)
                .with_max_operations(100)
                .with_time_advance(TimeAdvanceConfig::none())
                .run_and_assert(wrapper);
        }
    }
}
