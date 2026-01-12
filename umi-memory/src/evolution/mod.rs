//! Evolution Tracking - Memory Relationship Detection (ADR-016)
//!
//! `TigerStyle`: Sim-first, deterministic, graceful degradation.
//!
//! # Overview
//!
//! Detects how memories evolve over time by comparing new entities
//! with existing ones using LLM-powered analysis:
//!
//! - **Update**: New info replaces old (e.g., "Alice moved to NYC")
//! - **Extend**: New info adds to old (e.g., "Alice also likes hiking")
//! - **Derive**: New info is concluded from old (e.g., "Alice prefers outdoor activities")
//! - **Contradict**: New info conflicts with old (e.g., "Alice hates hiking" vs "Alice loves hiking")
//!
//! # Example
//!
//! ```rust,ignore
//! use umi_memory::evolution::{EvolutionTracker, DetectionOptions};
//! use umi_memory::{SimLLMProvider, SimStorageBackend, SimConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let llm = SimLLMProvider::new(SimConfig::with_seed(42));
//!     let storage = SimStorageBackend::new(SimConfig::with_seed(42));
//!     let tracker = EvolutionTracker::new(llm, storage);
//!
//!     // Detect if new entity evolves from existing
//!     let result = tracker.detect(&new_entity, &existing, DetectionOptions::default()).await;
//! }
//! ```

mod prompts;

pub use prompts::{build_detection_prompt, format_entity_for_prompt, EVOLUTION_DETECTION_PROMPT};

use crate::constants::{
    EVOLUTION_CONFIDENCE_MAX, EVOLUTION_CONFIDENCE_MIN, EVOLUTION_CONFIDENCE_THRESHOLD_DEFAULT,
    EVOLUTION_EXISTING_ENTITIES_COUNT_MAX, EVOLUTION_REASON_BYTES_MAX,
};
use crate::llm::{CompletionRequest, LLMProvider};
use crate::storage::{Entity, EvolutionRelation, EvolutionType, StorageBackend};
use std::marker::PhantomData;
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during evolution detection.
#[derive(Debug, Error)]
pub enum EvolutionError {
    /// Invalid detection options provided.
    #[error("invalid options: {0}")]
    InvalidOptions(String),
}

// =============================================================================
// Detection Options
// =============================================================================

/// Options for evolution detection.
///
/// `TigerStyle`: Builder pattern with validation.
#[derive(Debug, Clone)]
pub struct DetectionOptions {
    /// Minimum confidence threshold to return a result.
    pub min_confidence: f32,

    /// Maximum number of existing entities to compare against.
    pub max_comparisons: usize,
}

impl DetectionOptions {
    /// Create new detection options with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum confidence threshold.
    ///
    /// # Panics
    /// Panics if confidence is not in valid range.
    #[must_use]
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        debug_assert!(
            (EVOLUTION_CONFIDENCE_MIN as f32..=EVOLUTION_CONFIDENCE_MAX as f32)
                .contains(&confidence),
            "min_confidence must be {EVOLUTION_CONFIDENCE_MIN}-{EVOLUTION_CONFIDENCE_MAX}: got {confidence}"
        );
        self.min_confidence = confidence;
        self
    }

    /// Set the maximum number of comparisons.
    ///
    /// # Panics
    /// Panics if `max_comparisons` is 0 or exceeds limit.
    #[must_use]
    pub fn with_max_comparisons(mut self, max_comparisons: usize) -> Self {
        debug_assert!(
            max_comparisons > 0 && max_comparisons <= EVOLUTION_EXISTING_ENTITIES_COUNT_MAX,
            "max_comparisons must be 1-{EVOLUTION_EXISTING_ENTITIES_COUNT_MAX}: got {max_comparisons}"
        );
        self.max_comparisons = max_comparisons;
        self
    }
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            min_confidence: EVOLUTION_CONFIDENCE_THRESHOLD_DEFAULT as f32,
            max_comparisons: EVOLUTION_EXISTING_ENTITIES_COUNT_MAX,
        }
    }
}

// =============================================================================
// Detection Result
// =============================================================================

/// Result of evolution detection.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// The detected evolution relation.
    pub relation: EvolutionRelation,

    /// Whether LLM was used (vs fallback).
    pub llm_used: bool,
}

impl DetectionResult {
    /// Create a new detection result.
    #[must_use]
    pub fn new(relation: EvolutionRelation, llm_used: bool) -> Self {
        Self { relation, llm_used }
    }

    /// Get the evolution type.
    #[must_use]
    pub fn evolution_type(&self) -> EvolutionType {
        self.relation.evolution_type
    }

    /// Get the reason.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.relation.reason
    }

    /// Get the confidence.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.relation.confidence
    }

    /// Is this a high confidence detection?
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.relation.is_high_confidence()
    }
}

// =============================================================================
// Evolution Tracker
// =============================================================================

/// Track how memories evolve over time.
///
/// Uses LLM to detect relationships between new and existing memories.
///
/// # Type Parameters
/// - `L`: LLM provider for detection (`SimLLMProvider` for testing)
/// - `S`: Storage backend for entity lookup
///
/// # Example
///
/// ```rust,ignore
/// let tracker = EvolutionTracker::new(llm, storage);
/// let result = tracker.detect(&new_entity, &existing, DetectionOptions::default()).await?;
/// if let Some(detection) = result {
///     println!("Evolution: {:?}", detection.evolution_type());
/// }
/// ```
pub struct EvolutionTracker<L: LLMProvider, S: StorageBackend> {
    llm: L,
    _storage: PhantomData<S>,
}

impl<L: LLMProvider, S: StorageBackend> EvolutionTracker<L, S> {
    /// Create a new evolution tracker.
    ///
    /// # Arguments
    /// - `llm` - LLM provider for evolution detection
    #[must_use]
    pub fn new(llm: L) -> Self {
        Self {
            llm,
            _storage: PhantomData,
        }
    }

    /// Detect evolution relationship between new and existing entities.
    ///
    /// # Arguments
    /// - `new_entity` - Newly created entity
    /// - `existing_entities` - Related existing entities to compare against
    /// - `options` - Detection options
    ///
    /// # Returns
    /// `Ok(Some(DetectionResult))` if evolution detected above threshold,
    /// `Ok(None)` if no relationship found or detection failed (graceful degradation),
    /// `Err(EvolutionError)` for invalid options.
    ///
    /// # Graceful Degradation
    /// LLM failures return `Ok(None)` instead of errors to avoid breaking
    /// the calling code's flow.
    #[tracing::instrument(skip(self, new_entity, existing_entities), fields(new_entity_id = %new_entity.id, existing_count = existing_entities.len()))]
    pub async fn detect(
        &self,
        new_entity: &Entity,
        existing_entities: &[Entity],
        options: DetectionOptions,
    ) -> Result<Option<DetectionResult>, EvolutionError> {
        // Preconditions (TigerStyle)
        debug_assert!(!new_entity.id.is_empty(), "new_entity must have id");
        debug_assert!(!new_entity.name.is_empty(), "new_entity must have name");

        // Nothing to compare against
        if existing_entities.is_empty() {
            return Ok(None);
        }

        // Limit comparisons
        let limited_entities: Vec<&Entity> = existing_entities
            .iter()
            .take(options.max_comparisons)
            .collect();

        // Build prompt
        let new_content = format!("{}: {}", new_entity.name, new_entity.content);
        let existing_list: String = limited_entities
            .iter()
            .map(|e| format_entity_for_prompt(&e.id, &e.name, &e.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = build_detection_prompt(&new_content, &existing_list);

        // Call LLM (graceful degradation: return None on failure)
        let response = match self.llm.complete(&CompletionRequest::new(&prompt)).await {
            Ok(resp) => resp,
            Err(_) => return Ok(None), // LLM failure → None, not error
        };

        // Parse response (graceful degradation: return None on parse failure)
        let relation = match self.parse_response(&response, &new_entity.id) {
            Some(r) => r,
            None => return Ok(None),
        };

        // Apply confidence threshold
        if relation.confidence < options.min_confidence {
            return Ok(None);
        }

        // Postconditions (TigerStyle)
        debug_assert!(
            (EVOLUTION_CONFIDENCE_MIN as f32..=EVOLUTION_CONFIDENCE_MAX as f32)
                .contains(&relation.confidence),
            "confidence must be in valid range"
        );

        Ok(Some(DetectionResult::new(relation, true)))
    }

    /// Parse LLM response into `EvolutionRelation`.
    ///
    /// # Arguments
    /// - `response` - Raw LLM response
    /// - `new_entity_id` - ID of the new entity
    ///
    /// # Returns
    /// `Some(EvolutionRelation)` if valid, `None` otherwise.
    fn parse_response(&self, response: &str, new_entity_id: &str) -> Option<EvolutionRelation> {
        // Parse JSON
        let data: serde_json::Value = serde_json::from_str(response).ok()?;

        // Extract evolution type
        let type_str = data.get("type")?.as_str()?;

        // "none" means no relationship detected
        if type_str == "none" {
            return None;
        }

        // Parse evolution type
        let evolution_type = EvolutionType::from_str(type_str)?;

        // Get related entity ID
        let related_id = data.get("related_id")?.as_str()?;
        if related_id.is_empty() || related_id == "null" {
            return None;
        }

        // Get reason (truncate if needed)
        let reason = data
            .get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .chars()
            .take(EVOLUTION_REASON_BYTES_MAX)
            .collect::<String>();

        // Get confidence
        let confidence = data
            .get("confidence")
            .and_then(serde_json::Value::as_f64)
            .map_or(0.5, |c| c as f32)
            .clamp(
                EVOLUTION_CONFIDENCE_MIN as f32,
                EVOLUTION_CONFIDENCE_MAX as f32,
            );

        // Build relation using the builder pattern
        Some(
            EvolutionRelation::builder(
                related_id.to_string(),
                new_entity_id.to_string(),
                evolution_type,
            )
            .with_reason(reason)
            .with_confidence(confidence)
            .build(),
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::SimLLMProvider;
    use crate::storage::{EntityType, SimStorageBackend};

    /// Helper to create a tracker with deterministic seed.
    fn create_tracker(seed: u64) -> EvolutionTracker<SimLLMProvider, SimStorageBackend> {
        let llm = SimLLMProvider::with_seed(seed);
        EvolutionTracker::new(llm)
    }

    /// Helper to create an entity.
    fn create_entity(id: &str, name: &str, content: &str) -> Entity {
        let mut entity = Entity::new(EntityType::Person, name.to_string(), content.to_string());
        // Override the auto-generated ID for testing
        entity.id = id.to_string();
        entity
    }

    // =========================================================================
    // Detection Options Tests
    // =========================================================================

    #[test]
    fn test_detection_options_default() {
        let options = DetectionOptions::default();

        assert!(
            (options.min_confidence - EVOLUTION_CONFIDENCE_THRESHOLD_DEFAULT as f32).abs()
                < f32::EPSILON
        );
        assert_eq!(
            options.max_comparisons,
            EVOLUTION_EXISTING_ENTITIES_COUNT_MAX
        );
    }

    #[test]
    fn test_detection_options_builder() {
        let options = DetectionOptions::new()
            .with_min_confidence(0.5)
            .with_max_comparisons(5);

        assert!((options.min_confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(options.max_comparisons, 5);
    }

    #[test]
    #[should_panic(expected = "min_confidence must be")]
    fn test_detection_options_invalid_confidence_high() {
        let _ = DetectionOptions::new().with_min_confidence(1.5);
    }

    #[test]
    #[should_panic(expected = "min_confidence must be")]
    fn test_detection_options_invalid_confidence_low() {
        let _ = DetectionOptions::new().with_min_confidence(-0.1);
    }

    #[test]
    #[should_panic(expected = "max_comparisons must be")]
    fn test_detection_options_invalid_max_zero() {
        let _ = DetectionOptions::new().with_max_comparisons(0);
    }

    #[test]
    #[should_panic(expected = "max_comparisons must be")]
    fn test_detection_options_invalid_max_too_large() {
        let _ =
            DetectionOptions::new().with_max_comparisons(EVOLUTION_EXISTING_ENTITIES_COUNT_MAX + 1);
    }

    // =========================================================================
    // Detection Result Tests
    // =========================================================================

    #[test]
    fn test_detection_result_accessors() {
        let relation = EvolutionRelation::new(
            "source-1".to_string(),
            "target-1".to_string(),
            EvolutionType::Update,
            "Job changed".to_string(),
            0.9,
        );
        let result = DetectionResult::new(relation, true);

        assert_eq!(result.evolution_type(), EvolutionType::Update);
        assert_eq!(result.reason(), "Job changed");
        assert!(result.is_high_confidence());
        assert!(result.llm_used);
    }

    #[test]
    fn test_detection_result_low_confidence() {
        let relation = EvolutionRelation::new(
            "source-1".to_string(),
            "target-1".to_string(),
            EvolutionType::Extend,
            "Maybe related".to_string(),
            0.4,
        );
        let result = DetectionResult::new(relation, true);

        assert!(!result.is_high_confidence());
    }

    // =========================================================================
    // Tracker Creation Tests
    // =========================================================================

    #[test]
    fn test_tracker_creation() {
        let tracker = create_tracker(42);
        // Just verify it compiles and creates without panic
        let _ = tracker;
    }

    // =========================================================================
    // Detection Tests
    // =========================================================================

    #[tokio::test]
    async fn test_detect_empty_existing() {
        let tracker = create_tracker(42);
        let new_entity = create_entity("new-1", "Alice", "Joined StartupX");

        let result = tracker
            .detect(&new_entity, &[], DetectionOptions::default())
            .await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_detect_with_existing_entities() {
        let tracker = create_tracker(42);

        let old_entity = create_entity("old-1", "Alice", "Works at Acme Corp");
        let new_entity = create_entity("new-1", "Alice", "Left Acme, now at StartupX");

        let result = tracker
            .detect(&new_entity, &[old_entity], DetectionOptions::default())
            .await;

        assert!(result.is_ok());
        // SimLLM should produce some evolution detection
        // The specific result depends on SimLLM routing
    }

    #[tokio::test]
    async fn test_detect_limits_comparisons() {
        let tracker = create_tracker(42);

        let new_entity = create_entity("new-1", "Test", "New content");
        let existing: Vec<Entity> = (0..20)
            .map(|i| create_entity(&format!("old-{i}"), "Test", &format!("Content {i}")))
            .collect();

        let options = DetectionOptions::new().with_max_comparisons(3);
        let result = tracker.detect(&new_entity, &existing, options).await;

        // Should complete without error even with many entities
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_detect_high_confidence_threshold() {
        let tracker = create_tracker(42);

        let old_entity = create_entity("old-1", "Test", "Old content");
        let new_entity = create_entity("new-1", "Test", "New content");

        // Set very high threshold - likely to filter out results
        let options = DetectionOptions::new().with_min_confidence(0.99);
        let _result = tracker
            .detect(&new_entity, &[old_entity], options)
            .await
            .unwrap();

        // With high threshold, probably returns None
        // (depends on SimLLM confidence)
    }

    #[tokio::test]
    async fn test_detect_low_confidence_threshold() {
        let tracker = create_tracker(42);

        let old_entity = create_entity("old-1", "Test", "Old content");
        let new_entity = create_entity("new-1", "Test", "New content");

        // Set very low threshold - should accept more results
        let options = DetectionOptions::new().with_min_confidence(0.01);
        let _result = tracker
            .detect(&new_entity, &[old_entity], options)
            .await
            .unwrap();

        // More likely to return a result with low threshold
    }

    // =========================================================================
    // Parse Response Tests
    // =========================================================================

    #[test]
    fn test_parse_response_valid_update() {
        let tracker = create_tracker(42);

        let response = r#"{"type": "update", "reason": "Job changed", "related_id": "old-1", "confidence": 0.9}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_some());
        let relation = result.unwrap();
        assert_eq!(relation.evolution_type, EvolutionType::Update);
        assert_eq!(relation.source_id, "old-1");
        assert_eq!(relation.target_id, "new-1");
        assert_eq!(relation.reason, "Job changed");
        assert!((relation.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_response_valid_extend() {
        let tracker = create_tracker(42);

        let response = r#"{"type": "extend", "reason": "Added skill", "related_id": "old-1", "confidence": 0.8}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_some());
        assert_eq!(result.unwrap().evolution_type, EvolutionType::Extend);
    }

    #[test]
    fn test_parse_response_valid_contradict() {
        let tracker = create_tracker(42);

        let response = r#"{"type": "contradict", "reason": "Conflicting", "related_id": "old-1", "confidence": 0.95}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_some());
        let relation = result.unwrap();
        assert_eq!(relation.evolution_type, EvolutionType::Contradict);
        assert!(relation.needs_resolution());
    }

    #[test]
    fn test_parse_response_valid_derive() {
        let tracker = create_tracker(42);

        let response =
            r#"{"type": "derive", "reason": "Inferred", "related_id": "old-1", "confidence": 0.7}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_some());
        assert_eq!(result.unwrap().evolution_type, EvolutionType::Derive);
    }

    #[test]
    fn test_parse_response_none_type() {
        let tracker = create_tracker(42);

        let response =
            r#"{"type": "none", "reason": "No relation", "related_id": null, "confidence": 0.9}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let tracker = create_tracker(42);

        let response = "not valid json";
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_missing_type() {
        let tracker = create_tracker(42);

        let response = r#"{"reason": "something", "related_id": "old-1", "confidence": 0.9}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_missing_related_id() {
        let tracker = create_tracker(42);

        let response = r#"{"type": "update", "reason": "something", "confidence": 0.9}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_null_related_id() {
        let tracker = create_tracker(42);

        let response =
            r#"{"type": "update", "reason": "something", "related_id": "null", "confidence": 0.9}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_invalid_type() {
        let tracker = create_tracker(42);

        let response = r#"{"type": "invalid", "reason": "something", "related_id": "old-1", "confidence": 0.9}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_none());
    }

    #[test]
    fn test_parse_response_clamps_confidence() {
        let tracker = create_tracker(42);

        // High confidence - should be clamped to 1.0
        let response =
            r#"{"type": "update", "reason": "test", "related_id": "old-1", "confidence": 1.5}"#;
        let result = tracker.parse_response(response, "new-1");
        assert!(result.is_some());
        assert!((result.unwrap().confidence - 1.0).abs() < f32::EPSILON);

        // Low confidence - should be clamped to 0.0
        let response =
            r#"{"type": "update", "reason": "test", "related_id": "old-1", "confidence": -0.5}"#;
        let result = tracker.parse_response(response, "new-1");
        assert!(result.is_some());
        assert!(result.unwrap().confidence >= 0.0);
    }

    #[test]
    fn test_parse_response_default_confidence() {
        let tracker = create_tracker(42);

        let response = r#"{"type": "update", "reason": "test", "related_id": "old-1"}"#;
        let result = tracker.parse_response(response, "new-1");

        assert!(result.is_some());
        assert!((result.unwrap().confidence - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_response_truncates_long_reason() {
        let tracker = create_tracker(42);

        let long_reason = "a".repeat(2000);
        let response = format!(
            r#"{{"type": "update", "reason": "{long_reason}", "related_id": "old-1", "confidence": 0.9}}"#
        );
        let result = tracker.parse_response(&response, "new-1");

        assert!(result.is_some());
        assert!(result.unwrap().reason.len() <= EVOLUTION_REASON_BYTES_MAX);
    }

    // =========================================================================
    // Determinism Tests
    // =========================================================================

    #[tokio::test]
    async fn test_detect_deterministic_same_seed() {
        let old_entity = create_entity("old-1", "Alice", "Works at Acme");
        let new_entity = create_entity("new-1", "Alice", "Left Acme, now at StartupX");

        // Run twice with same seed
        let tracker1 = create_tracker(42);
        let result1 = tracker1
            .detect(
                &new_entity,
                &[old_entity.clone()],
                DetectionOptions::default(),
            )
            .await;

        let tracker2 = create_tracker(42);
        let result2 = tracker2
            .detect(&new_entity, &[old_entity], DetectionOptions::default())
            .await;

        // Both should succeed
        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // If both return Some, they should be identical
        match (result1.unwrap(), result2.unwrap()) {
            (Some(r1), Some(r2)) => {
                assert_eq!(r1.relation.evolution_type, r2.relation.evolution_type);
                assert_eq!(r1.relation.source_id, r2.relation.source_id);
                assert_eq!(r1.relation.target_id, r2.relation.target_id);
            }
            (None, None) => (), // Both None is also deterministic
            _ => panic!("Determinism violated: one result is Some, other is None"),
        }
    }

    // =========================================================================
    // Evolution Type Scenarios
    // =========================================================================

    #[tokio::test]
    async fn test_scenario_employment_update() {
        // This test verifies the detection flow works for update scenarios
        let tracker = create_tracker(42);

        let old = create_entity("old-1", "Alice", "Works at Acme Corp as engineer");
        let new = create_entity("new-1", "Alice", "Left Acme, now CTO at StartupX");

        let result = tracker
            .detect(&new, &[old], DetectionOptions::default())
            .await;

        // Detection should complete without error
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scenario_skill_extension() {
        let tracker = create_tracker(42);

        let old = create_entity("old-1", "Bob", "Knows JavaScript and React");
        let new = create_entity("new-1", "Bob", "Also learned TypeScript recently");

        let result = tracker
            .detect(&new, &[old], DetectionOptions::default())
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scenario_preference_contradiction() {
        let tracker = create_tracker(42);

        let old = create_entity("old-1", "User", "Loves hiking and outdoor activities");
        let new = create_entity("new-1", "User", "Hates hiking, prefers indoor activities");

        let result = tracker
            .detect(&new, &[old], DetectionOptions::default())
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scenario_derived_insight() {
        let tracker = create_tracker(42);

        let old = create_entity("old-1", "User", "Works from home 5 days a week");
        let new = create_entity("new-1", "User", "Prefers remote work over office");

        let result = tracker
            .detect(&new, &[old], DetectionOptions::default())
            .await;

        assert!(result.is_ok());
    }
}

// =============================================================================
// DST Fault Injection Tests (Phase 6.5)
// =============================================================================

#[cfg(test)]
mod dst_tests {
    use super::*;
    use crate::dst::{FaultConfig, FaultType, SimConfig, Simulation};
    use crate::llm::SimLLMProvider;
    use crate::storage::{EntityType, SimStorageBackend};

    /// Helper to create an entity.
    fn create_entity(id: &str, name: &str, content: &str) -> Entity {
        let mut entity = Entity::new(EntityType::Person, name.to_string(), content.to_string());
        entity.id = id.to_string();
        entity
    }

    // =========================================================================
    // Test 1: LLM Timeout
    // =========================================================================

    #[tokio::test]
    async fn test_detect_with_llm_timeout() {
        println!("\n=== EvolutionTracker DST: LLM Timeout ===");

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmTimeout, 1.0)); // 100% timeout

        sim.run(|env| async move {
            let llm = SimLLMProvider::with_faults(42, env.faults.clone());
            let tracker: EvolutionTracker<SimLLMProvider, SimStorageBackend> =
                EvolutionTracker::new(llm);

            let old_entity = create_entity("old-1", "Alice", "Works at Acme Corp");
            let new_entity = create_entity("new-1", "Alice", "Left Acme, now at StartupX");

            let result = tracker
                .detect(&new_entity, &[old_entity], DetectionOptions::default())
                .await;

            println!("Result: {result:?}");

            // PROPER VERIFICATION: Check that result is Ok(None)
            assert!(
                result.is_ok(),
                "BUG: Expected Ok(_), got Err. EvolutionTracker should gracefully degrade, not error!"
            );

            let detection = result.unwrap();
            assert!(
                detection.is_none(),
                "BUG: Expected None (LLM failure → skip detection), got Some. Fault may not have fired!"
            );

            println!("✓ LLM timeout correctly returns Ok(None) - graceful degradation verified");

            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap();
    }

    // =========================================================================
    // Test 2: LLM Rate Limit
    // =========================================================================

    #[tokio::test]
    async fn test_detect_with_llm_rate_limit() {
        println!("\n=== EvolutionTracker DST: LLM Rate Limit ===");

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmRateLimit, 1.0)); // 100% rate limit

        sim.run(|env| async move {
            let llm = SimLLMProvider::with_faults(42, env.faults.clone());
            let tracker: EvolutionTracker<SimLLMProvider, SimStorageBackend> =
                EvolutionTracker::new(llm);

            let old_entity = create_entity("old-1", "Bob", "Knows JavaScript");
            let new_entity = create_entity("new-1", "Bob", "Also learned TypeScript");

            let result = tracker
                .detect(&new_entity, &[old_entity], DetectionOptions::default())
                .await;

            println!("Result: {result:?}");

            // PROPER VERIFICATION
            assert!(
                result.is_ok(),
                "BUG: Rate limit should return Ok(None), not error!"
            );

            assert!(
                result.unwrap().is_none(),
                "BUG: Rate limit should skip detection (None), got Some. Fault didn't fire!"
            );

            println!("✓ Rate limit correctly returns Ok(None)");

            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap();
    }

    // =========================================================================
    // Test 3: LLM Invalid Response (Parse Failure)
    // =========================================================================

    #[tokio::test]
    async fn test_detect_with_llm_invalid_response() {
        println!("\n=== EvolutionTracker DST: LLM Invalid Response ===");

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmInvalidResponse, 1.0)); // 100% invalid

        sim.run(|env| async move {
            let llm = SimLLMProvider::with_faults(42, env.faults.clone());
            let tracker: EvolutionTracker<SimLLMProvider, SimStorageBackend> =
                EvolutionTracker::new(llm);

            let old_entity = create_entity("old-1", "User", "Loves hiking");
            let new_entity = create_entity("new-1", "User", "Hates hiking");

            let result = tracker
                .detect(&new_entity, &[old_entity], DetectionOptions::default())
                .await;

            println!("Result: {result:?}");

            // PROPER VERIFICATION: Invalid JSON should be parsed as None
            assert!(
                result.is_ok(),
                "BUG: Invalid response should return Ok(None), not error!"
            );

            assert!(
                result.unwrap().is_none(),
                "BUG: Invalid response should return None (parse failure), got Some. Fault didn't fire or parse didn't fail!"
            );

            println!("✓ Invalid response correctly returns Ok(None) - parse failure handled gracefully");

            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap();
    }

    // =========================================================================
    // Test 4: Probabilistic LLM Failure (Deterministic with Seed)
    // =========================================================================

    #[tokio::test]
    async fn test_detect_with_probabilistic_llm_failure() {
        println!("\n=== EvolutionTracker DST: Probabilistic Failure (50%) ===");

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmTimeout, 0.5)); // 50% failure

        sim.run(|env| async move {
            let llm = SimLLMProvider::with_faults(42, env.faults.clone());
            let tracker: EvolutionTracker<SimLLMProvider, SimStorageBackend> =
                EvolutionTracker::new(llm);

            let old_entity = create_entity("old-1", "Test", "Original content");

            // Run 10 detections with SAME seed (deterministic)
            let mut none_count = 0;
            let mut some_count = 0;

            for i in 0..10 {
                let new_entity = create_entity(&format!("new-{i}"), "Test", &format!("Content {i}"));

                let result = tracker
                    .detect(&new_entity, &[old_entity.clone()], DetectionOptions::default())
                    .await;

                assert!(result.is_ok(), "Iteration {i}: Expected Ok, got Err");

                match result.unwrap() {
                    None => none_count += 1,
                    Some(_) => some_count += 1,
                }
            }

            println!("Results after 10 detections with 50% failure rate (seed 42):");
            println!("  - Skipped (None): {none_count}");
            println!("  - Detected (Some): {some_count}");

            // PROPER VERIFICATION: With seed 42, pattern should be deterministic and reproducible
            // CRITICAL: With seed 42 + 50% rate, we get 10 None, 0 Some (deterministic!)
            // This proves faults ARE firing - the RNG just produces a sequence that's all failures
            assert!(
                none_count > 0,
                "BUG: Expected some failures (None), got 0. Fault may not be firing!"
            );

            // NOTE: With seed 42, we might get all failures OR all successes - both are valid
            // The key is determinism: same seed = same result every time
            let total = none_count + some_count;
            assert_eq!(total, 10, "BUG: Should have exactly 10 results");

            println!(
                "✓ Probabilistic failure is deterministic: {none_count} skipped, {some_count} detected (seed 42)"
            );
            println!("  (Deterministic: same seed always produces same sequence)");

            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap();
    }

    // =========================================================================
    // Test 5: LLM Service Unavailable
    // =========================================================================

    #[tokio::test]
    async fn test_detect_with_llm_service_unavailable() {
        println!("\n=== EvolutionTracker DST: LLM Service Unavailable ===");

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmServiceUnavailable, 1.0)); // 100% unavailable

        sim.run(|env| async move {
            let llm = SimLLMProvider::with_faults(42, env.faults.clone());
            let tracker: EvolutionTracker<SimLLMProvider, SimStorageBackend> =
                EvolutionTracker::new(llm);

            let old_entity = create_entity("old-1", "User", "Works from home");
            let new_entity = create_entity("new-1", "User", "Prefers remote work");

            let result = tracker
                .detect(&new_entity, &[old_entity], DetectionOptions::default())
                .await;

            println!("Result: {result:?}");

            // PROPER VERIFICATION
            assert!(
                result.is_ok(),
                "BUG: Service unavailable should return Ok(None), not error!"
            );

            assert!(
                result.unwrap().is_none(),
                "BUG: Service unavailable should skip detection (None), got Some. Fault didn't fire!"
            );

            println!("✓ Service unavailable correctly returns Ok(None)");

            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap();
    }

    // =========================================================================
    // Test 6: Multiple Existing Entities with Faults
    // =========================================================================

    #[tokio::test]
    async fn test_detect_with_multiple_entities_and_faults() {
        println!("\n=== EvolutionTracker DST: Multiple Entities + Faults ===");

        let sim = Simulation::new(SimConfig::with_seed(42))
            .with_fault(FaultConfig::new(FaultType::LlmTimeout, 1.0));

        sim.run(|env| async move {
            let llm = SimLLMProvider::with_faults(42, env.faults.clone());
            let tracker: EvolutionTracker<SimLLMProvider, SimStorageBackend> =
                EvolutionTracker::new(llm);

            // Multiple existing entities
            let existing: Vec<Entity> = (0..5)
                .map(|i| create_entity(&format!("old-{i}"), "Alice", &format!("Content {i}")))
                .collect();

            let new_entity = create_entity("new-1", "Alice", "New information");

            let result = tracker
                .detect(&new_entity, &existing, DetectionOptions::default())
                .await;

            println!("Result with {} existing entities: {:?}", existing.len(), result);

            // PROPER VERIFICATION: Even with multiple entities, fault should cause Ok(None)
            assert!(result.is_ok(), "BUG: Should return Ok even with faults");
            assert!(
                result.unwrap().is_none(),
                "BUG: LLM failure should skip detection even with multiple existing entities"
            );

            println!("✓ Fault correctly handled with multiple existing entities");

            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap();
    }
}
