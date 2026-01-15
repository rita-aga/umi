//! Promotion policies for automatic Core Memory management.
//!
//! `TigerStyle`: Explicit thresholds, configurable weights, deterministic decisions.
//!
//! This module defines policies that determine when entities should be promoted
//! from archival storage to Core Memory based on importance, recency, access
//! frequency, and entity type.

use crate::constants::*;
use crate::orchestration::AccessPattern;
use crate::storage::{Entity, EntityType};

/// Policy for determining entity promotion to Core Memory.
///
/// Implementations decide which entities should be promoted from archival
/// storage to Core Memory based on access patterns and entity metadata.
pub trait PromotionPolicy: Send + Sync {
    /// Check if an entity should be promoted to Core Memory.
    ///
    /// # Arguments
    /// * `entity` - The entity being considered for promotion
    /// * `access_pattern` - Access statistics for this entity
    ///
    /// # Returns
    /// `true` if the entity should be promoted, `false` otherwise
    fn should_promote(&self, entity: &Entity, access_pattern: &AccessPattern) -> bool;

    /// Calculate promotion priority score (higher = more important to promote).
    ///
    /// # Arguments
    /// * `entity` - The entity to calculate priority for
    /// * `access_pattern` - Access statistics for this entity
    ///
    /// # Returns
    /// Priority score in range [0.0, 1.0]
    fn calculate_priority(&self, entity: &Entity, access_pattern: &AccessPattern) -> f64;
}

/// Simple importance-based promotion policy.
///
/// Promotes entities if their combined importance score exceeds a threshold.
/// This is the simplest policy and easiest to understand.
#[derive(Debug, Clone)]
pub struct ImportanceBasedPolicy {
    /// Importance threshold for promotion (0.0 to 1.0)
    threshold: f64,
}

impl ImportanceBasedPolicy {
    /// Create a new importance-based policy with default threshold.
    ///
    /// # Preconditions
    /// - None (uses validated default constant)
    ///
    /// # Postconditions
    /// - Threshold is in valid range [0.0, 1.0]
    #[must_use]
    pub fn new() -> Self {
        Self::with_threshold(PROMOTION_IMPORTANCE_THRESHOLD_DEFAULT)
    }

    /// Create a new importance-based policy with custom threshold.
    ///
    /// # Arguments
    /// * `threshold` - Importance threshold for promotion (0.0 to 1.0)
    ///
    /// # Preconditions
    /// - `threshold` must be in range [0.0, 1.0]
    ///
    /// # Postconditions
    /// - Returns policy with validated threshold
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        // Preconditions
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );

        Self { threshold }
    }
}

impl Default for ImportanceBasedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl PromotionPolicy for ImportanceBasedPolicy {
    fn should_promote(&self, _entity: &Entity, access_pattern: &AccessPattern) -> bool {
        // Preconditions
        assert!(
            self.threshold >= 0.0 && self.threshold <= 1.0,
            "threshold invariant violated"
        );

        let importance = access_pattern.combined_importance;

        // Postconditions
        assert!(
            importance >= 0.0 && importance <= 1.0,
            "importance must be in range [0.0, 1.0]"
        );

        importance >= self.threshold
    }

    fn calculate_priority(&self, _entity: &Entity, access_pattern: &AccessPattern) -> f64 {
        // Preconditions
        assert!(
            self.threshold >= 0.0 && self.threshold <= 1.0,
            "threshold invariant violated"
        );

        let priority = access_pattern.combined_importance;

        // Postconditions
        assert!(
            priority >= 0.0 && priority <= 1.0,
            "priority must be in range [0.0, 1.0]"
        );

        priority
    }
}

/// Hybrid promotion policy combining multiple factors.
///
/// Calculates a weighted score from:
/// - Base importance (40%)
/// - Recency score (30%)
/// - Frequency score (20%)
/// - Entity type priority (10%)
///
/// This policy is more sophisticated and considers multiple dimensions.
#[derive(Debug, Clone)]
pub struct HybridPolicy {
    /// Score threshold for promotion (0.0 to 1.0)
    threshold: f64,
    /// Weight for base importance
    weight_importance: f64,
    /// Weight for recency score
    weight_recency: f64,
    /// Weight for frequency score
    weight_frequency: f64,
    /// Weight for entity type priority
    weight_type_priority: f64,
}

impl HybridPolicy {
    /// Create a new hybrid policy with default configuration.
    ///
    /// # Preconditions
    /// - None (uses validated default constants)
    ///
    /// # Postconditions
    /// - All weights sum to approximately 1.0
    /// - Threshold is in valid range [0.0, 1.0]
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: PROMOTION_SCORE_THRESHOLD_DEFAULT,
            weight_importance: PROMOTION_WEIGHT_IMPORTANCE,
            weight_recency: PROMOTION_WEIGHT_RECENCY,
            weight_frequency: PROMOTION_WEIGHT_FREQUENCY,
            weight_type_priority: PROMOTION_WEIGHT_TYPE_PRIORITY,
        }
    }

    /// Create a new hybrid policy with custom threshold.
    ///
    /// # Arguments
    /// * `threshold` - Score threshold for promotion (0.0 to 1.0)
    ///
    /// # Preconditions
    /// - `threshold` must be in range [0.0, 1.0]
    ///
    /// # Postconditions
    /// - Returns policy with validated threshold and default weights
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        // Preconditions
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );

        Self {
            threshold,
            weight_importance: PROMOTION_WEIGHT_IMPORTANCE,
            weight_recency: PROMOTION_WEIGHT_RECENCY,
            weight_frequency: PROMOTION_WEIGHT_FREQUENCY,
            weight_type_priority: PROMOTION_WEIGHT_TYPE_PRIORITY,
        }
    }

    /// Create a new hybrid policy with custom weights.
    ///
    /// # Arguments
    /// * `threshold` - Score threshold for promotion (0.0 to 1.0)
    /// * `weight_importance` - Weight for base importance
    /// * `weight_recency` - Weight for recency score
    /// * `weight_frequency` - Weight for frequency score
    /// * `weight_type_priority` - Weight for entity type priority
    ///
    /// # Preconditions
    /// - All weights must be >= 0.0
    /// - Weights should sum to approximately 1.0
    /// - Threshold must be in range [0.0, 1.0]
    ///
    /// # Postconditions
    /// - Returns policy with validated parameters
    #[must_use]
    pub fn with_custom_weights(
        threshold: f64,
        weight_importance: f64,
        weight_recency: f64,
        weight_frequency: f64,
        weight_type_priority: f64,
    ) -> Self {
        // Preconditions
        assert!(
            threshold >= 0.0 && threshold <= 1.0,
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );
        assert!(weight_importance >= 0.0, "weight_importance must be >= 0.0");
        assert!(weight_recency >= 0.0, "weight_recency must be >= 0.0");
        assert!(weight_frequency >= 0.0, "weight_frequency must be >= 0.0");
        assert!(
            weight_type_priority >= 0.0,
            "weight_type_priority must be >= 0.0"
        );

        let weight_sum =
            weight_importance + weight_recency + weight_frequency + weight_type_priority;
        assert!(
            (weight_sum - 1.0).abs() < 0.01,
            "weights should sum to ~1.0, got {}",
            weight_sum
        );

        Self {
            threshold,
            weight_importance,
            weight_recency,
            weight_frequency,
            weight_type_priority,
        }
    }

    /// Get entity type priority score.
    ///
    /// # Arguments
    /// * `entity_type` - The entity type to get priority for
    ///
    /// # Returns
    /// Priority score in range [0.0, 1.0]
    ///
    /// # Postconditions
    /// - Returns value in range [0.0, 1.0]
    fn entity_type_priority(&self, entity_type: EntityType) -> f64 {
        let priority = match entity_type {
            EntityType::Self_ => ENTITY_TYPE_PRIORITY_SELF,
            EntityType::Project => ENTITY_TYPE_PRIORITY_PROJECT,
            EntityType::Task => ENTITY_TYPE_PRIORITY_TASK,
            EntityType::Person => ENTITY_TYPE_PRIORITY_PERSON,
            EntityType::Topic => ENTITY_TYPE_PRIORITY_TOPIC,
            EntityType::Note => ENTITY_TYPE_PRIORITY_NOTE,
        };

        // Postconditions
        assert!(
            priority >= 0.0 && priority <= 1.0,
            "priority must be in range [0.0, 1.0]"
        );

        priority
    }
}

impl Default for HybridPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl PromotionPolicy for HybridPolicy {
    fn should_promote(&self, entity: &Entity, access_pattern: &AccessPattern) -> bool {
        // Preconditions
        assert!(
            self.threshold >= 0.0 && self.threshold <= 1.0,
            "threshold invariant violated"
        );

        let score = self.calculate_priority(entity, access_pattern);

        // Postconditions
        assert!(
            score >= 0.0 && score <= 1.0,
            "score must be in range [0.0, 1.0]"
        );

        score >= self.threshold
    }

    fn calculate_priority(&self, entity: &Entity, access_pattern: &AccessPattern) -> f64 {
        // Preconditions
        assert!(
            self.threshold >= 0.0 && self.threshold <= 1.0,
            "threshold invariant violated"
        );
        assert!(
            self.weight_importance >= 0.0,
            "weight_importance invariant violated"
        );
        assert!(
            self.weight_recency >= 0.0,
            "weight_recency invariant violated"
        );
        assert!(
            self.weight_frequency >= 0.0,
            "weight_frequency invariant violated"
        );
        assert!(
            self.weight_type_priority >= 0.0,
            "weight_type_priority invariant violated"
        );

        let importance = access_pattern.combined_importance;
        let recency = access_pattern.recency_score;
        let frequency = access_pattern.frequency_score;
        let type_priority = self.entity_type_priority(entity.entity_type);

        let score = self.weight_importance * importance
            + self.weight_recency * recency
            + self.weight_frequency * frequency
            + self.weight_type_priority * type_priority;

        // Postconditions
        assert!(
            score >= 0.0 && score <= 1.0,
            "score must be in range [0.0, 1.0]"
        );

        score
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

    /// Helper: Create a test entity
    fn create_test_entity(entity_type: EntityType) -> Entity {
        EntityBuilder::new(
            entity_type,
            "Test Entity".to_string(),
            "Test content".to_string(),
        )
        .build()
    }

    /// Helper: Create access pattern with specific scores
    fn create_access_pattern(
        _clock: &SimClock,
        tracker: &mut AccessTracker,
        entity_id: &str,
        importance: f64,
    ) -> AccessPattern {
        // Record access with the specified importance
        tracker.record_access(entity_id, importance);

        tracker
            .get_access_pattern(entity_id)
            .expect("Failed to get access pattern")
    }

    #[test]
    fn test_importance_based_policy_default_threshold() {
        let policy = ImportanceBasedPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Entity with high importance (should promote)
        let entity_high = create_test_entity(EntityType::Person);
        let pattern_high = create_access_pattern(&clock, &mut tracker, "entity_high", 0.8);
        assert!(policy.should_promote(&entity_high, &pattern_high));

        // Entity with low importance (should NOT promote)
        let entity_low = create_test_entity(EntityType::Note);
        let pattern_low = create_access_pattern(&clock, &mut tracker, "entity_low", 0.5);
        assert!(!policy.should_promote(&entity_low, &pattern_low));
    }

    #[test]
    fn test_importance_based_policy_custom_threshold() {
        let policy = ImportanceBasedPolicy::with_threshold(0.9);
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Entity with base 0.85 -> combined ~0.78 (should NOT promote)
        // combined = 0.5 * 0.85 + 0.3 * 1.0 + 0.2 * 0.5 = 0.425 + 0.3 + 0.1 = 0.825
        let entity_below = create_test_entity(EntityType::Person);
        let pattern_below = create_access_pattern(&clock, &mut tracker, "entity_below", 0.85);
        assert!(!policy.should_promote(&entity_below, &pattern_below));
        assert!(pattern_below.combined_importance < 0.9);

        // Entity with base 1.0 -> combined ~0.90 (should promote)
        // combined = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.5 = 0.5 + 0.3 + 0.1 = 0.9
        let entity_at = create_test_entity(EntityType::Person);
        let pattern_at = create_access_pattern(&clock, &mut tracker, "entity_at", 1.0);
        assert!(policy.should_promote(&entity_at, &pattern_at));
        assert!((pattern_at.combined_importance - 0.9).abs() < 0.01);

        // Entity with base 1.0 and high importance (should definitely promote)
        let entity_above = create_test_entity(EntityType::Person);
        let pattern_above = create_access_pattern(&clock, &mut tracker, "entity_above", 1.0);
        assert!(policy.should_promote(&entity_above, &pattern_above));
        assert!(pattern_above.combined_importance >= 0.9);
    }

    #[test]
    fn test_importance_based_policy_calculate_priority() {
        let policy = ImportanceBasedPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        let entity = create_test_entity(EntityType::Person);
        let pattern = create_access_pattern(&clock, &mut tracker, "entity", 0.8);

        let priority = policy.calculate_priority(&entity, &pattern);

        // Priority should equal combined importance
        assert!((priority - 0.8).abs() < 0.01);
        assert!(priority >= 0.0 && priority <= 1.0);
    }

    #[test]
    #[should_panic(expected = "threshold must be in range")]
    fn test_importance_based_policy_invalid_threshold_high() {
        ImportanceBasedPolicy::with_threshold(1.5);
    }

    #[test]
    #[should_panic(expected = "threshold must be in range")]
    fn test_importance_based_policy_invalid_threshold_low() {
        ImportanceBasedPolicy::with_threshold(-0.1);
    }

    #[test]
    fn test_hybrid_policy_default() {
        let policy = HybridPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Self_ entities should have highest priority due to entity type
        let entity_self = create_test_entity(EntityType::Self_);
        let pattern_self = create_access_pattern(&clock, &mut tracker, "entity_self", 0.7);
        let priority_self = policy.calculate_priority(&entity_self, &pattern_self);

        // Note entities should have lower priority due to entity type
        let entity_note = create_test_entity(EntityType::Note);
        let pattern_note = create_access_pattern(&clock, &mut tracker, "entity_note", 0.7);
        let priority_note = policy.calculate_priority(&entity_note, &pattern_note);

        // Self_ should have higher priority than Note with same importance
        assert!(priority_self > priority_note);
        assert!(priority_self >= 0.0 && priority_self <= 1.0);
        assert!(priority_note >= 0.0 && priority_note <= 1.0);
    }

    #[test]
    fn test_hybrid_policy_entity_type_priorities() {
        let policy = HybridPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // All entities with same importance
        let importance = 0.5_f64;

        let types_and_priorities = vec![
            (EntityType::Self_, ENTITY_TYPE_PRIORITY_SELF),
            (EntityType::Project, ENTITY_TYPE_PRIORITY_PROJECT),
            (EntityType::Task, ENTITY_TYPE_PRIORITY_TASK),
            (EntityType::Person, ENTITY_TYPE_PRIORITY_PERSON),
            (EntityType::Topic, ENTITY_TYPE_PRIORITY_TOPIC),
            (EntityType::Note, ENTITY_TYPE_PRIORITY_NOTE),
        ];

        let mut priorities = Vec::new();
        for (entity_type, _expected_type_priority) in &types_and_priorities {
            let entity = create_test_entity(*entity_type);
            let pattern = create_access_pattern(
                &clock,
                &mut tracker,
                &format!("{:?}", entity_type),
                importance,
            );
            let priority = policy.calculate_priority(&entity, &pattern);
            priorities.push(priority);
        }

        // Priorities should be in descending order (Self_ > Project > Task > Person > Topic > Note)
        for i in 0..priorities.len() - 1 {
            assert!(
                priorities[i] >= priorities[i + 1],
                "Priority order violated: {} < {}",
                priorities[i],
                priorities[i + 1]
            );
        }
    }

    #[test]
    fn test_hybrid_policy_custom_threshold() {
        let policy = HybridPolicy::with_threshold(0.9);
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // High importance Self_ entity
        let entity = create_test_entity(EntityType::Self_);
        let pattern = create_access_pattern(&clock, &mut tracker, "entity", 0.95);

        // Even Self_ with high importance might not reach 0.9 threshold
        // due to weighted calculation
        let should_promote = policy.should_promote(&entity, &pattern);
        let priority = policy.calculate_priority(&entity, &pattern);

        assert!(priority >= 0.0 && priority <= 1.0);
        // Should promote only if calculated score >= threshold
        assert_eq!(should_promote, priority >= 0.9);
    }

    #[test]
    fn test_hybrid_policy_custom_weights() {
        // Create policy with 100% weight on entity type
        let policy = HybridPolicy::with_custom_weights(0.5, 0.0, 0.0, 0.0, 1.0);
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Self_ should always get high score
        let entity_self = create_test_entity(EntityType::Self_);
        let pattern_self = create_access_pattern(&clock, &mut tracker, "entity_self", 0.1);
        let priority_self = policy.calculate_priority(&entity_self, &pattern_self);

        // Should be close to ENTITY_TYPE_PRIORITY_SELF (1.0) since weight is 100% on type
        assert!((priority_self - ENTITY_TYPE_PRIORITY_SELF).abs() < 0.01);

        // Note should get low score
        let entity_note = create_test_entity(EntityType::Note);
        let pattern_note = create_access_pattern(&clock, &mut tracker, "entity_note", 0.9);
        let priority_note = policy.calculate_priority(&entity_note, &pattern_note);

        // Should be close to ENTITY_TYPE_PRIORITY_NOTE (0.4) regardless of importance
        assert!((priority_note - ENTITY_TYPE_PRIORITY_NOTE).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "threshold must be in range")]
    fn test_hybrid_policy_invalid_threshold() {
        HybridPolicy::with_threshold(1.5);
    }

    #[test]
    #[should_panic(expected = "weight_importance must be >= 0.0")]
    fn test_hybrid_policy_invalid_weight_importance() {
        HybridPolicy::with_custom_weights(0.5, -0.1, 0.0, 0.0, 1.1);
    }

    #[test]
    #[should_panic(expected = "weights should sum to ~1.0")]
    fn test_hybrid_policy_weights_dont_sum_to_one() {
        HybridPolicy::with_custom_weights(0.5, 0.5, 0.5, 0.5, 0.5);
    }

    #[test]
    fn test_promotion_determinism() {
        // Same inputs should produce same outputs
        let policy = HybridPolicy::new();
        let clock = SimClock::new();
        let mut tracker1 = AccessTracker::new(clock.clone());
        let mut tracker2 = AccessTracker::new(clock.clone());

        let entity1 = create_test_entity(EntityType::Person);
        let entity2 = create_test_entity(EntityType::Person);

        let pattern1 = create_access_pattern(&clock, &mut tracker1, "entity", 0.8);
        let pattern2 = create_access_pattern(&clock, &mut tracker2, "entity", 0.8);

        let priority1 = policy.calculate_priority(&entity1, &pattern1);
        let priority2 = policy.calculate_priority(&entity2, &pattern2);

        // Should be deterministic
        assert_eq!(priority1, priority2);
    }

    #[test]
    fn test_promotion_priority_bounds() {
        // All policies should return priorities in [0.0, 1.0]
        let importance_policy = ImportanceBasedPolicy::new();
        let hybrid_policy = HybridPolicy::new();
        let clock = SimClock::new();
        let mut tracker = AccessTracker::new(clock.clone());

        // Test with various entity types and importance levels
        for entity_type in [
            EntityType::Self_,
            EntityType::Person,
            EntityType::Project,
            EntityType::Task,
            EntityType::Topic,
            EntityType::Note,
        ] {
            for importance in [0.0, 0.5, 1.0] {
                let entity = create_test_entity(entity_type);
                let pattern = create_access_pattern(
                    &clock,
                    &mut tracker,
                    &format!("{:?}_{}", entity_type, importance),
                    importance,
                );

                let priority_importance = importance_policy.calculate_priority(&entity, &pattern);
                let priority_hybrid = hybrid_policy.calculate_priority(&entity, &pattern);

                assert!(
                    priority_importance >= 0.0 && priority_importance <= 1.0,
                    "Importance policy priority out of bounds: {}",
                    priority_importance
                );
                assert!(
                    priority_hybrid >= 0.0 && priority_hybrid <= 1.0,
                    "Hybrid policy priority out of bounds: {}",
                    priority_hybrid
                );
            }
        }
    }
}
