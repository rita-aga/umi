//! Evolution - Memory Evolution Tracking (ADR-006)
//!
//! `TigerStyle`: Explicit types, comprehensive testing.
//!
//! # Overview
//!
//! Tracks how memories evolve over time. When new information is stored,
//! we detect its relationship to existing memories:
//!
//! - **Update**: New info replaces/corrects old (e.g., "Alice moved to NYC")
//! - **Extend**: New info adds to old (e.g., "Alice also likes hiking")
//! - **Derive**: New info is concluded from old (e.g., "Alice prefers outdoor activities")
//! - **Contradict**: New info conflicts with old (e.g., "Alice hates hiking" vs "Alice loves hiking")
//!
//! # Example
//!
//! ```text
//! Memory 1: "Alice works at Acme Corp"
//! Memory 2: "Alice left Acme, now at StartupX"
//!
//! EvolutionRelation {
//!     source_id: memory_1.id,
//!     target_id: memory_2.id,
//!     evolution_type: Update,
//!     reason: "Employment changed",
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::constants::EVOLUTION_REASON_BYTES_MAX;

// =============================================================================
// Evolution Type
// =============================================================================

/// Types of evolution relationships between memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionType {
    /// New info replaces/corrects old.
    /// Example: "Alice moved to NYC" updates "Alice lives in SF"
    Update,

    /// New info adds to old.
    /// Example: "Alice also speaks French" extends "Alice speaks English"
    Extend,

    /// New info is concluded from old.
    /// Example: "Alice prefers remote work" derived from multiple WFH mentions
    Derive,

    /// New info conflicts with old.
    /// Example: "Alice hates hiking" contradicts "Alice loves hiking"
    Contradict,
}

impl EvolutionType {
    /// Get string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Extend => "extend",
            Self::Derive => "derive",
            Self::Contradict => "contradict",
        }
    }

    /// Parse from string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "update" => Some(Self::Update),
            "extend" => Some(Self::Extend),
            "derive" => Some(Self::Derive),
            "contradict" => Some(Self::Contradict),
            _ => None,
        }
    }

    /// Get all evolution types.
    #[must_use]
    pub fn all() -> &'static [EvolutionType] {
        &[Self::Update, Self::Extend, Self::Derive, Self::Contradict]
    }

    /// Is this type a conflict?
    #[must_use]
    pub fn is_conflict(&self) -> bool {
        matches!(self, Self::Contradict)
    }

    /// Is this type additive (doesn't invalidate old)?
    #[must_use]
    pub fn is_additive(&self) -> bool {
        matches!(self, Self::Extend | Self::Derive)
    }
}

impl std::fmt::Display for EvolutionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Evolution Relation
// =============================================================================

/// A relationship between two memories showing how one evolved from another.
///
/// The relation is directional: source is the older memory, target is the newer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRelation {
    /// Unique identifier (UUID v4)
    pub id: String,
    /// ID of the older/source memory
    pub source_id: String,
    /// ID of the newer/target memory
    pub target_id: String,
    /// Type of evolution relationship
    pub evolution_type: EvolutionType,
    /// Human-readable reason for the relationship
    pub reason: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// When this relation was detected
    pub created_at: DateTime<Utc>,
}

impl EvolutionRelation {
    /// Create a new evolution relation.
    ///
    /// # Panics
    /// Panics if reason exceeds limit or confidence is out of range.
    #[must_use]
    pub fn new(
        source_id: String,
        target_id: String,
        evolution_type: EvolutionType,
        reason: String,
        confidence: f32,
    ) -> Self {
        // Preconditions (TigerStyle)
        assert!(!source_id.is_empty(), "source_id must not be empty");
        assert!(!target_id.is_empty(), "target_id must not be empty");
        assert!(
            source_id != target_id,
            "source_id and target_id must be different"
        );
        assert!(
            reason.len() <= EVOLUTION_REASON_BYTES_MAX,
            "reason {} bytes exceeds max {}",
            reason.len(),
            EVOLUTION_REASON_BYTES_MAX
        );
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence {confidence} must be between 0.0 and 1.0"
        );

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            source_id,
            target_id,
            evolution_type,
            reason,
            confidence,
            created_at: Utc::now(),
        }
    }

    /// Create a builder for more complex construction.
    #[must_use]
    pub fn builder(
        source_id: String,
        target_id: String,
        evolution_type: EvolutionType,
    ) -> EvolutionRelationBuilder {
        EvolutionRelationBuilder::new(source_id, target_id, evolution_type)
    }

    /// Is this a high-confidence relation?
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Is this a conflict that needs resolution?
    #[must_use]
    pub fn needs_resolution(&self) -> bool {
        self.evolution_type.is_conflict() && self.is_high_confidence()
    }
}

// =============================================================================
// Evolution Relation Builder
// =============================================================================

/// Builder for `EvolutionRelation` with fluent API.
#[derive(Debug)]
pub struct EvolutionRelationBuilder {
    source_id: String,
    target_id: String,
    evolution_type: EvolutionType,
    id: Option<String>,
    reason: String,
    confidence: f32,
    created_at: Option<DateTime<Utc>>,
}

impl EvolutionRelationBuilder {
    /// Create a new builder with required fields.
    #[must_use]
    pub fn new(source_id: String, target_id: String, evolution_type: EvolutionType) -> Self {
        Self {
            source_id,
            target_id,
            evolution_type,
            id: None,
            reason: String::new(),
            confidence: 0.5, // Default medium confidence
            created_at: None,
        }
    }

    /// Set custom ID.
    #[must_use]
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Set reason.
    #[must_use]
    pub fn with_reason(mut self, reason: String) -> Self {
        self.reason = reason;
        self
    }

    /// Set confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set `created_at` (for DST).
    #[must_use]
    pub fn with_created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = Some(created_at);
        self
    }

    /// Build the evolution relation.
    ///
    /// # Panics
    /// Panics if preconditions are violated.
    #[must_use]
    pub fn build(self) -> EvolutionRelation {
        // Preconditions
        assert!(!self.source_id.is_empty(), "source_id must not be empty");
        assert!(!self.target_id.is_empty(), "target_id must not be empty");
        assert!(
            self.source_id != self.target_id,
            "source_id and target_id must be different"
        );
        assert!(
            self.reason.len() <= EVOLUTION_REASON_BYTES_MAX,
            "reason {} bytes exceeds max {}",
            self.reason.len(),
            EVOLUTION_REASON_BYTES_MAX
        );
        assert!(
            (0.0..=1.0).contains(&self.confidence),
            "confidence {} must be between 0.0 and 1.0",
            self.confidence
        );

        EvolutionRelation {
            id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            source_id: self.source_id,
            target_id: self.target_id,
            evolution_type: self.evolution_type,
            reason: self.reason,
            confidence: self.confidence,
            created_at: self.created_at.unwrap_or_else(Utc::now),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EvolutionType Tests
    // =========================================================================

    #[test]
    fn test_evolution_type_as_str() {
        assert_eq!(EvolutionType::Update.as_str(), "update");
        assert_eq!(EvolutionType::Extend.as_str(), "extend");
        assert_eq!(EvolutionType::Derive.as_str(), "derive");
        assert_eq!(EvolutionType::Contradict.as_str(), "contradict");
    }

    #[test]
    fn test_evolution_type_from_str() {
        assert_eq!(
            EvolutionType::from_str("update"),
            Some(EvolutionType::Update)
        );
        assert_eq!(
            EvolutionType::from_str("EXTEND"),
            Some(EvolutionType::Extend)
        );
        assert_eq!(
            EvolutionType::from_str("Derive"),
            Some(EvolutionType::Derive)
        );
        assert_eq!(
            EvolutionType::from_str("contradict"),
            Some(EvolutionType::Contradict)
        );
        assert_eq!(EvolutionType::from_str("unknown"), None);
    }

    #[test]
    fn test_evolution_type_is_conflict() {
        assert!(!EvolutionType::Update.is_conflict());
        assert!(!EvolutionType::Extend.is_conflict());
        assert!(!EvolutionType::Derive.is_conflict());
        assert!(EvolutionType::Contradict.is_conflict());
    }

    #[test]
    fn test_evolution_type_is_additive() {
        assert!(!EvolutionType::Update.is_additive());
        assert!(EvolutionType::Extend.is_additive());
        assert!(EvolutionType::Derive.is_additive());
        assert!(!EvolutionType::Contradict.is_additive());
    }

    // =========================================================================
    // EvolutionRelation Tests
    // =========================================================================

    #[test]
    fn test_evolution_relation_new() {
        let relation = EvolutionRelation::new(
            "source-123".to_string(),
            "target-456".to_string(),
            EvolutionType::Update,
            "Employment changed".to_string(),
            0.9,
        );

        assert!(!relation.id.is_empty());
        assert_eq!(relation.source_id, "source-123");
        assert_eq!(relation.target_id, "target-456");
        assert_eq!(relation.evolution_type, EvolutionType::Update);
        assert_eq!(relation.reason, "Employment changed");
        assert!((relation.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_evolution_relation_builder() {
        let relation =
            EvolutionRelation::builder("src".to_string(), "tgt".to_string(), EvolutionType::Extend)
                .with_id("custom-id".to_string())
                .with_reason("Added new skill".to_string())
                .with_confidence(0.85)
                .build();

        assert_eq!(relation.id, "custom-id");
        assert_eq!(relation.evolution_type, EvolutionType::Extend);
        assert_eq!(relation.reason, "Added new skill");
        assert!((relation.confidence - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_evolution_relation_is_high_confidence() {
        let high = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Update,
            String::new(),
            0.9,
        );
        let low = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Update,
            String::new(),
            0.5,
        );

        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_evolution_relation_needs_resolution() {
        // High confidence contradiction - needs resolution
        let conflict = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Contradict,
            "Conflicting info".to_string(),
            0.95,
        );
        assert!(conflict.needs_resolution());

        // Low confidence contradiction - doesn't need resolution
        let low_conflict = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Contradict,
            "Maybe conflicting".to_string(),
            0.3,
        );
        assert!(!low_conflict.needs_resolution());

        // High confidence update - doesn't need resolution (not a conflict)
        let update = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Update,
            "Updated".to_string(),
            0.95,
        );
        assert!(!update.needs_resolution());
    }

    // =========================================================================
    // Precondition Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "source_id must not be empty")]
    fn test_evolution_relation_empty_source() {
        let _ = EvolutionRelation::new(
            String::new(),
            "target".to_string(),
            EvolutionType::Update,
            String::new(),
            0.5,
        );
    }

    #[test]
    #[should_panic(expected = "target_id must not be empty")]
    fn test_evolution_relation_empty_target() {
        let _ = EvolutionRelation::new(
            "source".to_string(),
            String::new(),
            EvolutionType::Update,
            String::new(),
            0.5,
        );
    }

    #[test]
    #[should_panic(expected = "source_id and target_id must be different")]
    fn test_evolution_relation_same_source_target() {
        let _ = EvolutionRelation::new(
            "same-id".to_string(),
            "same-id".to_string(),
            EvolutionType::Update,
            String::new(),
            0.5,
        );
    }

    #[test]
    #[should_panic(expected = "confidence")]
    fn test_evolution_relation_invalid_confidence_high() {
        let _ = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Update,
            String::new(),
            1.5, // Too high
        );
    }

    #[test]
    #[should_panic(expected = "confidence")]
    fn test_evolution_relation_invalid_confidence_low() {
        let _ = EvolutionRelation::new(
            "a".to_string(),
            "b".to_string(),
            EvolutionType::Update,
            String::new(),
            -0.1, // Too low
        );
    }

    // =========================================================================
    // Scenario Tests (ADR-006)
    // =========================================================================

    #[test]
    fn test_scenario_employment_update() {
        // "Alice works at Acme" -> "Alice left Acme, now at StartupX"
        let relation = EvolutionRelation::builder(
            "memory-alice-acme".to_string(),
            "memory-alice-startupx".to_string(),
            EvolutionType::Update,
        )
        .with_reason("Employment changed from Acme to StartupX".to_string())
        .with_confidence(0.95)
        .build();

        assert_eq!(relation.evolution_type, EvolutionType::Update);
        assert!(relation.is_high_confidence());
        assert!(!relation.needs_resolution()); // Updates don't need resolution
    }

    #[test]
    fn test_scenario_preference_contradiction() {
        // "User likes Python" -> "User says they hate Python"
        let relation = EvolutionRelation::builder(
            "memory-likes-python".to_string(),
            "memory-hates-python".to_string(),
            EvolutionType::Contradict,
        )
        .with_reason("Conflicting statements about Python preference".to_string())
        .with_confidence(0.9)
        .build();

        assert!(relation.evolution_type.is_conflict());
        assert!(relation.needs_resolution());
    }

    #[test]
    fn test_scenario_skill_extension() {
        // "Bob knows JavaScript" -> "Bob also knows TypeScript"
        let relation = EvolutionRelation::builder(
            "memory-bob-js".to_string(),
            "memory-bob-ts".to_string(),
            EvolutionType::Extend,
        )
        .with_reason("Additional programming language skill".to_string())
        .with_confidence(0.85)
        .build();

        assert!(relation.evolution_type.is_additive());
        assert!(!relation.needs_resolution());
    }

    #[test]
    fn test_scenario_derived_insight() {
        // Multiple WFH mentions -> "User prefers remote work"
        let relation = EvolutionRelation::builder(
            "memory-wfh-mentions".to_string(),
            "memory-prefers-remote".to_string(),
            EvolutionType::Derive,
        )
        .with_reason("Derived from multiple work-from-home mentions".to_string())
        .with_confidence(0.7)
        .build();

        assert!(relation.evolution_type.is_additive());
        assert_eq!(relation.evolution_type, EvolutionType::Derive);
    }
}
