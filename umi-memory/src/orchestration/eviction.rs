//! Eviction policies for automatic Core Memory management.
//!
//! `TigerStyle`: Protected entity types, deterministic selection, configurable thresholds.
//!
//! This module defines policies that determine when entities should be evicted
//! from Core Memory based on recency, importance, and entity type.

use crate::constants::*;
use crate::orchestration::AccessTracker;
use crate::storage::{Entity, EntityType};

/// Policy for determining entity eviction from Core Memory.
///
/// Implementations decide which entities should be evicted when Core Memory
/// reaches capacity limits.
pub trait EvictionPolicy: Send + Sync {
    /// Select entities to evict from Core Memory.
    ///
    /// # Arguments
    /// * `core_entities` - Current entities in Core Memory
    /// * `access_tracker` - Access pattern tracking
    /// * `count` - Number of entities to evict
    ///
    /// # Returns
    /// Vector of entity IDs to evict (length <= count)
    ///
    /// # Preconditions
    /// - `count` must be > 0
    /// - `core_entities` must not be empty
    ///
    /// # Postconditions
    /// - Returns at most `count` entity IDs
    /// - Never returns Self_ entity IDs (protected)
    /// - Returns empty vector if no entities can be evicted
    fn select_eviction_candidates(
        &self,
        core_entities: &[Entity],
        access_tracker: &AccessTracker,
        count: usize,
    ) -> Vec<String>;
}

/// LRU (Least Recently Used) eviction policy.
///
/// Evicts entities that haven't been accessed recently, based on last access time.
/// Self_ entities are never evicted (protected).
#[derive(Debug, Clone)]
pub struct LRUEvictionPolicy {
    /// Optional time threshold - only evict if not accessed within this time
    threshold_ms: Option<u64>,
}

impl LRUEvictionPolicy {
    /// Create a new LRU eviction policy with no time threshold.
    ///
    /// # Postconditions
    /// - Returns policy with no threshold (will evict oldest regardless of time)
    #[must_use]
    pub fn new() -> Self {
        Self { threshold_ms: None }
    }

    /// Create a new LRU eviction policy with time threshold.
    ///
    /// # Arguments
    /// * `threshold_ms` - Only evict entities not accessed within this time
    ///
    /// # Preconditions
    /// - `threshold_ms` must be > 0
    ///
    /// # Postconditions
    /// - Returns policy with validated threshold
    #[must_use]
    pub fn with_threshold(threshold_ms: u64) -> Self {
        // Preconditions
        assert!(threshold_ms > 0, "threshold_ms must be > 0");

        Self {
            threshold_ms: Some(threshold_ms),
        }
    }

    /// Check if entity type is protected from eviction.
    fn is_protected(&self, entity_type: EntityType) -> bool {
        matches!(entity_type, EntityType::Self_)
    }
}

impl Default for LRUEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for LRUEvictionPolicy {
    fn select_eviction_candidates(
        &self,
        core_entities: &[Entity],
        access_tracker: &AccessTracker,
        count: usize,
    ) -> Vec<String> {
        // Preconditions
        assert!(count > 0, "count must be > 0");

        if core_entities.is_empty() {
            return Vec::new();
        }

        // Filter out protected entities and get access patterns
        let mut candidates: Vec<(String, u64)> = core_entities
            .iter()
            .filter(|e| !self.is_protected(e.entity_type))
            .filter_map(|e| {
                access_tracker
                    .get_access_pattern(&e.id)
                    .map(|p| (e.id.clone(), p.last_access_ms))
            })
            .collect();

        // Apply time threshold if set
        if let Some(threshold) = self.threshold_ms {
            // Get current time from the clock
            let current_time = access_tracker.clock().now_ms();
            candidates.retain(|(_, last_access)| {
                current_time.saturating_sub(*last_access) >= threshold
            });
        }

        // Sort by last access time (oldest first)
        candidates.sort_by_key(|(_, last_access)| *last_access);

        // Take up to count entities
        let result: Vec<String> = candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect();

        // Postconditions
        assert!(
            result.len() <= count,
            "returned more than requested count"
        );
        assert!(
            result.iter().all(|id| {
                core_entities
                    .iter()
                    .find(|e| &e.id == id)
                    .map_or(false, |e| !self.is_protected(e.entity_type))
            }),
            "protected entities in eviction list"
        );

        result
    }
}

/// Importance-based eviction policy.
///
/// Evicts entities with lowest importance scores.
/// Self_ entities are never evicted (protected).
#[derive(Debug, Clone)]
pub struct ImportanceEvictionPolicy {
    /// Optional importance threshold - never evict above this
    min_importance: Option<f64>,
}

impl ImportanceEvictionPolicy {
    /// Create a new importance-based eviction policy with no threshold.
    ///
    /// # Postconditions
    /// - Returns policy with no threshold (will evict lowest regardless of importance)
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_importance: None,
        }
    }

    /// Create a new importance-based eviction policy with minimum importance threshold.
    ///
    /// # Arguments
    /// * `min_importance` - Never evict entities above this importance
    ///
    /// # Preconditions
    /// - `min_importance` must be in range [0.0, 1.0]
    ///
    /// # Postconditions
    /// - Returns policy with validated threshold
    #[must_use]
    pub fn with_threshold(min_importance: f64) -> Self {
        // Preconditions
        assert!(
            min_importance >= 0.0 && min_importance <= 1.0,
            "min_importance must be in range [0.0, 1.0], got {}",
            min_importance
        );

        Self {
            min_importance: Some(min_importance),
        }
    }

    /// Check if entity type is protected from eviction.
    fn is_protected(&self, entity_type: EntityType) -> bool {
        matches!(entity_type, EntityType::Self_)
    }
}

impl Default for ImportanceEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for ImportanceEvictionPolicy {
    fn select_eviction_candidates(
        &self,
        core_entities: &[Entity],
        access_tracker: &AccessTracker,
        count: usize,
    ) -> Vec<String> {
        // Preconditions
        assert!(count > 0, "count must be > 0");

        if core_entities.is_empty() {
            return Vec::new();
        }

        // Filter out protected entities and get importance scores
        let mut candidates: Vec<(String, f64)> = core_entities
            .iter()
            .filter(|e| !self.is_protected(e.entity_type))
            .filter_map(|e| {
                access_tracker
                    .get_access_pattern(&e.id)
                    .map(|p| (e.id.clone(), p.combined_importance))
            })
            .collect();

        // Apply importance threshold if set
        if let Some(threshold) = self.min_importance {
            candidates.retain(|(_, importance)| *importance < threshold);
        }

        // Sort by importance (lowest first)
        candidates.sort_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take up to count entities
        let result: Vec<String> = candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect();

        // Postconditions
        assert!(
            result.len() <= count,
            "returned more than requested count"
        );
        assert!(
            result.iter().all(|id| {
                core_entities
                    .iter()
                    .find(|e| &e.id == id)
                    .map_or(false, |e| !self.is_protected(e.entity_type))
            }),
            "protected entities in eviction list"
        );

        result
    }
}

/// Hybrid eviction policy combining importance and recency.
///
/// Calculates eviction score as weighted combination of importance and recency.
/// Lower scores are evicted first.
/// Self_ entities are never evicted (protected).
#[derive(Debug, Clone)]
pub struct HybridEvictionPolicy {
    /// Weight for importance score
    weight_importance: f64,
    /// Weight for recency score
    weight_recency: f64,
    /// Optional minimum importance threshold
    min_importance: Option<f64>,
}

impl HybridEvictionPolicy {
    /// Create a new hybrid eviction policy with default weights.
    ///
    /// # Postconditions
    /// - Weights sum to 1.0
    /// - No importance threshold
    #[must_use]
    pub fn new() -> Self {
        Self {
            weight_importance: EVICTION_WEIGHT_IMPORTANCE,
            weight_recency: EVICTION_WEIGHT_RECENCY,
            min_importance: None,
        }
    }

    /// Create a new hybrid eviction policy with custom weights.
    ///
    /// # Arguments
    /// * `weight_importance` - Weight for importance score
    /// * `weight_recency` - Weight for recency score
    ///
    /// # Preconditions
    /// - All weights must be >= 0.0
    /// - Weights should sum to approximately 1.0
    ///
    /// # Postconditions
    /// - Returns policy with validated weights
    #[must_use]
    pub fn with_weights(weight_importance: f64, weight_recency: f64) -> Self {
        // Preconditions
        assert!(
            weight_importance >= 0.0,
            "weight_importance must be >= 0.0"
        );
        assert!(weight_recency >= 0.0, "weight_recency must be >= 0.0");

        let weight_sum = weight_importance + weight_recency;
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "weights should sum to ~1.0, got {}",
            weight_sum
        );

        Self {
            weight_importance,
            weight_recency,
            min_importance: None,
        }
    }

    /// Create a new hybrid eviction policy with minimum importance threshold.
    ///
    /// # Arguments
    /// * `min_importance` - Never evict entities above this importance
    ///
    /// # Preconditions
    /// - `min_importance` must be in range [0.0, 1.0]
    ///
    /// # Postconditions
    /// - Returns policy with validated threshold and default weights
    #[must_use]
    pub fn with_threshold(min_importance: f64) -> Self {
        // Preconditions
        assert!(
            min_importance >= 0.0 && min_importance <= 1.0,
            "min_importance must be in range [0.0, 1.0], got {}",
            min_importance
        );

        Self {
            weight_importance: EVICTION_WEIGHT_IMPORTANCE,
            weight_recency: EVICTION_WEIGHT_RECENCY,
            min_importance: Some(min_importance),
        }
    }

    /// Check if entity type is protected from eviction.
    fn is_protected(&self, entity_type: EntityType) -> bool {
        matches!(entity_type, EntityType::Self_)
    }

    /// Calculate eviction score (lower = more likely to evict).
    fn calculate_eviction_score(&self, importance: f64, recency: f64) -> f64 {
        // Preconditions
        assert!(
            importance >= 0.0 && importance <= 1.0,
            "importance must be in [0.0, 1.0]"
        );
        assert!(
            recency >= 0.0 && recency <= 1.0,
            "recency must be in [0.0, 1.0]"
        );

        let score = self.weight_importance * importance + self.weight_recency * recency;

        // Postconditions
        assert!(score >= 0.0 && score <= 1.0, "score must be in [0.0, 1.0]");

        score
    }
}

impl Default for HybridEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for HybridEvictionPolicy {
    fn select_eviction_candidates(
        &self,
        core_entities: &[Entity],
        access_tracker: &AccessTracker,
        count: usize,
    ) -> Vec<String> {
        // Preconditions
        assert!(count > 0, "count must be > 0");

        if core_entities.is_empty() {
            return Vec::new();
        }

        // Filter out protected entities and calculate eviction scores
        let mut candidates: Vec<(String, f64)> = core_entities
            .iter()
            .filter(|e| !self.is_protected(e.entity_type))
            .filter_map(|e| {
                access_tracker.get_access_pattern(&e.id).map(|p| {
                    let score =
                        self.calculate_eviction_score(p.combined_importance, p.recency_score);
                    (e.id.clone(), score)
                })
            })
            .collect();

        // Apply importance threshold if set
        if let Some(threshold) = self.min_importance {
            candidates.retain(|(id, _)| {
                if let Some(entity) = core_entities.iter().find(|e| &e.id == id) {
                    if let Some(pattern) = access_tracker.get_access_pattern(&entity.id) {
                        return pattern.combined_importance < threshold;
                    }
                }
                false
            });
        }

        // Sort by eviction score (lowest first = highest priority to evict)
        candidates.sort_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take up to count entities
        let result: Vec<String> = candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect();

        // Postconditions
        assert!(
            result.len() <= count,
            "returned more than requested count"
        );
        assert!(
            result.iter().all(|id| {
                core_entities
                    .iter()
                    .find(|e| &e.id == id)
                    .map_or(false, |e| !self.is_protected(e.entity_type))
            }),
            "protected entities in eviction list"
        );

        result
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::SimClock;
    use crate::orchestration::AccessTracker;
    use crate::storage::EntityBuilder;

    const ONE_DAY_MS: u64 = 24 * 60 * 60 * 1000;

    /// Helper: Create test entity
    fn create_test_entity(entity_type: EntityType, id: &str) -> Entity {
        let mut entity =
            EntityBuilder::new(entity_type, "Test Entity".to_string(), "Test content".to_string())
                .build();
        entity.id = id.to_string(); // Override ID for testing
        entity
    }

    /// Helper: Record access and advance time
    fn record_access_and_advance(
        tracker: &mut AccessTracker,
        clock: &SimClock,
        entity_id: &str,
        importance: f64,
        advance_days: u64,
    ) {
        tracker.record_access(entity_id, importance);
        if advance_days > 0 {
            // Advance in daily increments to respect DST limits
            for _ in 0..advance_days {
                clock.advance_ms(ONE_DAY_MS);
            }
        }
    }

    #[test]
    fn test_lru_eviction_basic() {
        let policy = LRUEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Create entities with different access times
        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");
        let entity3 = create_test_entity(EntityType::Note, "note3");

        // Access in order: note1, note2, note3
        record_access_and_advance(&mut tracker, &clock, "note1", 0.5, 1);
        record_access_and_advance(&mut tracker, &clock, "note2", 0.5, 1);
        record_access_and_advance(&mut tracker, &clock, "note3", 0.5, 0);

        let entities = vec![entity1, entity2, entity3];

        // Should evict note1 (oldest)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");

        // Should evict note1 and note2 (two oldest)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 2);
        assert_eq!(to_evict.len(), 2);
        assert!(to_evict.contains(&"note1".to_string()));
        assert!(to_evict.contains(&"note2".to_string()));
    }

    #[test]
    fn test_lru_eviction_protects_self() {
        let policy = LRUEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Create Self_ entity (oldest) and Note entity (newest)
        let entity_self = create_test_entity(EntityType::Self_, "self1");
        let entity_note = create_test_entity(EntityType::Note, "note1");

        record_access_and_advance(&mut tracker, &clock, "self1", 0.9, 5);
        record_access_and_advance(&mut tracker, &clock, "note1", 0.5, 0);

        let entities = vec![entity_self, entity_note];

        // Should evict note1, not self1 (even though self1 is older)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_lru_eviction_with_threshold() {
        let policy = LRUEvictionPolicy::with_threshold(3 * ONE_DAY_MS);
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");

        // note1 accessed 5 days ago (should evict)
        record_access_and_advance(&mut tracker, &clock, "note1", 0.5, 5);
        // note2 accessed 1 day ago (should NOT evict - within threshold)
        record_access_and_advance(&mut tracker, &clock, "note2", 0.5, 0);

        let entities = vec![entity1, entity2];

        // Should only evict note1 (outside threshold)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 2);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_importance_eviction_basic() {
        let policy = ImportanceEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");
        let entity3 = create_test_entity(EntityType::Note, "note3");

        // Different importance levels
        tracker.record_access("note1", 0.3); // Lowest
        tracker.record_access("note2", 0.7); // Highest
        tracker.record_access("note3", 0.5); // Middle

        let entities = vec![entity1, entity2, entity3];

        // Should evict note1 (lowest importance)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");

        // Should evict note1 and note3 (two lowest)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 2);
        assert_eq!(to_evict.len(), 2);
        assert!(to_evict.contains(&"note1".to_string()));
        assert!(to_evict.contains(&"note3".to_string()));
    }

    #[test]
    fn test_importance_eviction_protects_self() {
        let policy = ImportanceEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity_self = create_test_entity(EntityType::Self_, "self1");
        let entity_note = create_test_entity(EntityType::Note, "note1");

        // Self_ has low importance, but should be protected
        tracker.record_access("self1", 0.2);
        tracker.record_access("note1", 0.8);

        let entities = vec![entity_self, entity_note];

        // Should evict note1, not self1 (even though note1 has higher importance)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_importance_eviction_with_threshold() {
        let policy = ImportanceEvictionPolicy::with_threshold(0.6);
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");

        // combined_importance = 0.5 * base + 0.3 * recency + 0.2 * frequency
        // For freshly accessed: recency=1.0, frequency=0.5
        // note1: base 0.0 -> combined = 0.0 + 0.3 + 0.1 = 0.4 (below 0.6)
        // note2: base 1.0 -> combined = 0.5 + 0.3 + 0.1 = 0.9 (above 0.6)
        tracker.record_access("note1", 0.0); // Combined ~0.4 (below threshold)
        tracker.record_access("note2", 1.0); // Combined ~0.9 (above threshold, protected)

        let entities = vec![entity1, entity2];

        // Should only evict note1 (below threshold)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 2);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_hybrid_eviction_basic() {
        let policy = HybridEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");

        // note1: low importance, old access
        record_access_and_advance(&mut tracker, &clock, "note1", 0.3, 5);
        // note2: high importance, recent access
        record_access_and_advance(&mut tracker, &clock, "note2", 0.8, 0);

        let entities = vec![entity1, entity2];

        // Should evict note1 (lower combined score)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_hybrid_eviction_protects_self() {
        let policy = HybridEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity_self = create_test_entity(EntityType::Self_, "self1");
        let entity_note = create_test_entity(EntityType::Note, "note1");

        // Both have low scores, but Self_ is protected
        record_access_and_advance(&mut tracker, &clock, "self1", 0.2, 10);
        record_access_and_advance(&mut tracker, &clock, "note1", 0.8, 0);

        let entities = vec![entity_self, entity_note];

        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_hybrid_eviction_custom_weights() {
        // 100% weight on importance, 0% on recency
        let policy = HybridEvictionPolicy::with_weights(1.0, 0.0);
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");

        // note1: low importance, recent access
        tracker.record_access("note1", 0.2);
        // note2: high importance, old access
        record_access_and_advance(&mut tracker, &clock, "note2", 0.9, 10);

        let entities = vec![entity1, entity2];

        // Should evict note1 (lower importance, recency ignored)
        let to_evict = policy.select_eviction_candidates(&entities, &tracker, 1);
        assert_eq!(to_evict.len(), 1);
        assert_eq!(to_evict[0], "note1");
    }

    #[test]
    fn test_eviction_empty_entities() {
        let lru_policy = LRUEvictionPolicy::new();
        let importance_policy = ImportanceEvictionPolicy::new();
        let hybrid_policy = HybridEvictionPolicy::new();
        let clock = SimClock::new();
        let tracker = AccessTracker::new(clock);

        let entities = Vec::new();

        // All policies should return empty for empty input
        assert!(lru_policy
            .select_eviction_candidates(&entities, &tracker, 1)
            .is_empty());
        assert!(importance_policy
            .select_eviction_candidates(&entities, &tracker, 1)
            .is_empty());
        assert!(hybrid_policy
            .select_eviction_candidates(&entities, &tracker, 1)
            .is_empty());
    }

    #[test]
    fn test_eviction_all_protected() {
        let lru_policy = LRUEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Self_, "self1");
        let entity2 = create_test_entity(EntityType::Self_, "self2");

        tracker.record_access("self1", 0.5);
        tracker.record_access("self2", 0.5);

        let entities = vec![entity1, entity2];

        // Should return empty (all protected)
        let to_evict = lru_policy.select_eviction_candidates(&entities, &tracker, 2);
        assert!(to_evict.is_empty());
    }

    #[test]
    #[should_panic(expected = "threshold_ms must be > 0")]
    fn test_lru_invalid_threshold() {
        LRUEvictionPolicy::with_threshold(0);
    }

    #[test]
    #[should_panic(expected = "min_importance must be in range")]
    fn test_importance_invalid_threshold_high() {
        ImportanceEvictionPolicy::with_threshold(1.5);
    }

    #[test]
    #[should_panic(expected = "min_importance must be in range")]
    fn test_importance_invalid_threshold_low() {
        ImportanceEvictionPolicy::with_threshold(-0.1);
    }

    #[test]
    #[should_panic(expected = "weight_importance must be >= 0.0")]
    fn test_hybrid_invalid_weight_importance() {
        HybridEvictionPolicy::with_weights(-0.1, 1.1);
    }

    #[test]
    #[should_panic(expected = "weights should sum to ~1.0")]
    fn test_hybrid_weights_dont_sum_to_one() {
        HybridEvictionPolicy::with_weights(0.5, 0.3);
    }

    #[test]
    fn test_eviction_determinism() {
        // Same inputs should produce same outputs
        let policy = HybridEvictionPolicy::new();
        let clock = SimClock::new();
        let mut tracker1 = AccessTracker::new(clock.clone());
        let mut tracker2 = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Note, "note1");
        let entity2 = create_test_entity(EntityType::Note, "note2");

        // Record same accesses in both trackers
        tracker1.record_access("note1", 0.3);
        tracker1.record_access("note2", 0.8);
        tracker2.record_access("note1", 0.3);
        tracker2.record_access("note2", 0.8);

        let entities = vec![entity1.clone(), entity2.clone()];

        let to_evict1 = policy.select_eviction_candidates(&entities, &tracker1, 1);
        let to_evict2 = policy.select_eviction_candidates(&entities, &tracker2, 1);

        // Should be deterministic
        assert_eq!(to_evict1, to_evict2);
    }
}
