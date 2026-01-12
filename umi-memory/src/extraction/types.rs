//! Extraction Types - Entity and Relation Data Structures
//!
//! `TigerStyle`: Type-safe enums, explicit validation, no invalid states.

use serde::{Deserialize, Serialize};

use crate::constants::{
    EXTRACTION_CONFIDENCE_DEFAULT, EXTRACTION_CONFIDENCE_MAX, EXTRACTION_CONFIDENCE_MIN,
    EXTRACTION_ENTITY_CONTENT_BYTES_MAX, EXTRACTION_ENTITY_NAME_BYTES_MAX,
};

// =============================================================================
// Entity Types
// =============================================================================

/// Types of entities that can be extracted.
///
/// `TigerStyle`: Exhaustive enum prevents invalid states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum EntityType {
    /// Person mentioned in text
    Person,
    /// Organization or company
    #[serde(alias = "org")]
    Organization,
    /// Project or initiative
    Project,
    /// Topic or concept
    Topic,
    /// User preference
    Preference,
    /// Task or action item
    Task,
    /// Event or meeting
    Event,
    /// Fallback for unstructured content
    #[default]
    Note,
}

impl EntityType {
    /// Get all entity types in order.
    #[must_use]
    pub fn all() -> &'static [EntityType] {
        &[
            EntityType::Person,
            EntityType::Organization,
            EntityType::Project,
            EntityType::Topic,
            EntityType::Preference,
            EntityType::Task,
            EntityType::Event,
            EntityType::Note,
        ]
    }

    /// Get the string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Person => "person",
            EntityType::Organization => "organization",
            EntityType::Project => "project",
            EntityType::Topic => "topic",
            EntityType::Preference => "preference",
            EntityType::Task => "task",
            EntityType::Event => "event",
            EntityType::Note => "note",
        }
    }

    /// Parse from string, defaulting to Note for unknown types.
    #[must_use]
    pub fn from_str_or_note(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "person" => EntityType::Person,
            "org" | "organization" => EntityType::Organization,
            "project" => EntityType::Project,
            "topic" => EntityType::Topic,
            "preference" => EntityType::Preference,
            "task" => EntityType::Task,
            "event" => EntityType::Event,
            _ => EntityType::Note,
        }
    }
}


impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Relation Types
// =============================================================================

/// Types of relations between entities.
///
/// `TigerStyle`: Exhaustive enum prevents invalid states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum RelationType {
    /// Person works at organization
    WorksAt,
    /// Person knows person
    Knows,
    /// Person manages project
    Manages,
    /// Generic relation
    #[default]
    RelatesTo,
    /// User prefers something
    Prefers,
    /// Entity is part of another
    PartOf,
}

impl RelationType {
    /// Get all relation types.
    #[must_use]
    pub fn all() -> &'static [RelationType] {
        &[
            RelationType::WorksAt,
            RelationType::Knows,
            RelationType::Manages,
            RelationType::RelatesTo,
            RelationType::Prefers,
            RelationType::PartOf,
        ]
    }

    /// Get the string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationType::WorksAt => "works_at",
            RelationType::Knows => "knows",
            RelationType::Manages => "manages",
            RelationType::RelatesTo => "relates_to",
            RelationType::Prefers => "prefers",
            RelationType::PartOf => "part_of",
        }
    }

    /// Parse from string, defaulting to `RelatesTo` for unknown types.
    #[must_use]
    pub fn from_str_or_relates_to(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "works_at" => RelationType::WorksAt,
            "knows" => RelationType::Knows,
            "manages" => RelationType::Manages,
            "relates_to" => RelationType::RelatesTo,
            "prefers" => RelationType::Prefers,
            "part_of" => RelationType::PartOf,
            _ => RelationType::RelatesTo,
        }
    }
}


impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Extracted Entity
// =============================================================================

/// Entity extracted from text by LLM.
///
/// `TigerStyle`: Immutable after creation, validated on construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity name/identifier
    pub name: String,
    /// Type of entity
    #[serde(rename = "type")]
    pub entity_type: EntityType,
    /// Brief description or context
    pub content: String,
    /// Extraction confidence (0.0-1.0)
    pub confidence: f64,
}

impl ExtractedEntity {
    /// Create a new extracted entity with validation.
    ///
    /// # Panics
    /// Panics if name is empty or confidence is out of range.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        entity_type: EntityType,
        content: impl Into<String>,
        confidence: f64,
    ) -> Self {
        let name = name.into();
        let content = content.into();

        // TigerStyle: Preconditions
        assert!(!name.is_empty(), "entity name must not be empty");
        assert!(
            name.len() <= EXTRACTION_ENTITY_NAME_BYTES_MAX,
            "entity name too long"
        );
        assert!(
            content.len() <= EXTRACTION_ENTITY_CONTENT_BYTES_MAX,
            "entity content too long"
        );
        assert!(
            (EXTRACTION_CONFIDENCE_MIN..=EXTRACTION_CONFIDENCE_MAX).contains(&confidence),
            "confidence must be {EXTRACTION_CONFIDENCE_MIN}-{EXTRACTION_CONFIDENCE_MAX}, got {confidence}"
        );

        Self {
            name,
            entity_type,
            content,
            confidence,
        }
    }

    /// Create a new entity with default confidence.
    #[must_use]
    pub fn with_default_confidence(
        name: impl Into<String>,
        entity_type: EntityType,
        content: impl Into<String>,
    ) -> Self {
        Self::new(name, entity_type, content, EXTRACTION_CONFIDENCE_DEFAULT)
    }

    /// Check if this is a high confidence extraction (>= 0.8).
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Check if this is a fallback note entity.
    #[must_use]
    pub fn is_fallback(&self) -> bool {
        self.entity_type == EntityType::Note
    }
}

// =============================================================================
// Extracted Relation
// =============================================================================

/// Relation between two entities extracted from text.
///
/// `TigerStyle`: Immutable after creation, validated on construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedRelation {
    /// Source entity name
    pub source: String,
    /// Target entity name
    pub target: String,
    /// Type of relation
    #[serde(rename = "type")]
    pub relation_type: RelationType,
    /// Extraction confidence (0.0-1.0)
    pub confidence: f64,
}

impl ExtractedRelation {
    /// Create a new extracted relation with validation.
    ///
    /// # Panics
    /// Panics if source/target are empty or confidence is out of range.
    #[must_use]
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        relation_type: RelationType,
        confidence: f64,
    ) -> Self {
        let source = source.into();
        let target = target.into();

        // TigerStyle: Preconditions
        assert!(!source.is_empty(), "relation source must not be empty");
        assert!(!target.is_empty(), "relation target must not be empty");
        assert!(
            (EXTRACTION_CONFIDENCE_MIN..=EXTRACTION_CONFIDENCE_MAX).contains(&confidence),
            "confidence must be {EXTRACTION_CONFIDENCE_MIN}-{EXTRACTION_CONFIDENCE_MAX}, got {confidence}"
        );

        Self {
            source,
            target,
            relation_type,
            confidence,
        }
    }

    /// Create a new relation with default confidence.
    #[must_use]
    pub fn with_default_confidence(
        source: impl Into<String>,
        target: impl Into<String>,
        relation_type: RelationType,
    ) -> Self {
        Self::new(source, target, relation_type, EXTRACTION_CONFIDENCE_DEFAULT)
    }

    /// Check if this is a high confidence extraction (>= 0.8).
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
}

// =============================================================================
// Extraction Result
// =============================================================================

/// Result of entity extraction from text.
///
/// Contains extracted entities, relations, and the original text.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
    /// Extracted relations
    pub relations: Vec<ExtractedRelation>,
    /// Original input text
    pub raw_text: String,
}

impl ExtractionResult {
    /// Create a new extraction result.
    #[must_use]
    pub fn new(
        entities: Vec<ExtractedEntity>,
        relations: Vec<ExtractedRelation>,
        raw_text: impl Into<String>,
    ) -> Self {
        Self {
            entities,
            relations,
            raw_text: raw_text.into(),
        }
    }

    /// Create an empty result (no entities or relations).
    #[must_use]
    pub fn empty(raw_text: impl Into<String>) -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            raw_text: raw_text.into(),
        }
    }

    /// Check if the result is empty (no entities).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the number of entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get the number of relations.
    #[must_use]
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Get entities filtered by type.
    #[must_use]
    pub fn entities_of_type(&self, entity_type: EntityType) -> Vec<&ExtractedEntity> {
        self.entities
            .iter()
            .filter(|e| e.entity_type == entity_type)
            .collect()
    }

    /// Get entities with confidence above threshold.
    #[must_use]
    pub fn entities_above_confidence(&self, min_confidence: f64) -> Vec<&ExtractedEntity> {
        self.entities
            .iter()
            .filter(|e| e.confidence >= min_confidence)
            .collect()
    }
}

// =============================================================================
// Extraction Options
// =============================================================================

/// Options for entity extraction.
#[derive(Debug, Clone, Default)]
pub struct ExtractionOptions {
    /// Known entity names for context
    pub existing_entities: Vec<String>,
    /// Minimum confidence threshold (0.0-1.0)
    pub min_confidence: f64,
}

impl ExtractionOptions {
    /// Create new extraction options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add existing entities for context.
    #[must_use]
    pub fn with_existing_entities(mut self, entities: Vec<String>) -> Self {
        self.existing_entities = entities;
        self
    }

    /// Set minimum confidence threshold.
    ///
    /// # Panics
    /// Panics if confidence is not in [0.0, 1.0].
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        assert!(
            (EXTRACTION_CONFIDENCE_MIN..=EXTRACTION_CONFIDENCE_MAX).contains(&min_confidence),
            "min_confidence must be {EXTRACTION_CONFIDENCE_MIN}-{EXTRACTION_CONFIDENCE_MAX}"
        );
        self.min_confidence = min_confidence;
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_as_str() {
        assert_eq!(EntityType::Person.as_str(), "person");
        assert_eq!(EntityType::Organization.as_str(), "organization");
        assert_eq!(EntityType::Note.as_str(), "note");
    }

    #[test]
    fn test_entity_type_from_str() {
        assert_eq!(EntityType::from_str_or_note("person"), EntityType::Person);
        assert_eq!(
            EntityType::from_str_or_note("org"),
            EntityType::Organization
        );
        assert_eq!(
            EntityType::from_str_or_note("organization"),
            EntityType::Organization
        );
        assert_eq!(EntityType::from_str_or_note("unknown"), EntityType::Note);
    }

    #[test]
    fn test_relation_type_as_str() {
        assert_eq!(RelationType::WorksAt.as_str(), "works_at");
        assert_eq!(RelationType::Knows.as_str(), "knows");
        assert_eq!(RelationType::RelatesTo.as_str(), "relates_to");
    }

    #[test]
    fn test_relation_type_from_str() {
        assert_eq!(
            RelationType::from_str_or_relates_to("works_at"),
            RelationType::WorksAt
        );
        assert_eq!(
            RelationType::from_str_or_relates_to("unknown"),
            RelationType::RelatesTo
        );
    }

    #[test]
    fn test_extracted_entity_new() {
        let entity = ExtractedEntity::new("Alice", EntityType::Person, "A person", 0.9);
        assert_eq!(entity.name, "Alice");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert_eq!(entity.content, "A person");
        assert_eq!(entity.confidence, 0.9);
    }

    #[test]
    fn test_extracted_entity_default_confidence() {
        let entity =
            ExtractedEntity::with_default_confidence("Bob", EntityType::Person, "Another person");
        assert_eq!(entity.confidence, EXTRACTION_CONFIDENCE_DEFAULT);
    }

    #[test]
    fn test_extracted_entity_high_confidence() {
        let high = ExtractedEntity::new("Alice", EntityType::Person, "test", 0.9);
        let low = ExtractedEntity::new("Bob", EntityType::Person, "test", 0.5);
        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_extracted_entity_is_fallback() {
        let note = ExtractedEntity::new("Note", EntityType::Note, "content", 0.5);
        let person = ExtractedEntity::new("Alice", EntityType::Person, "content", 0.5);
        assert!(note.is_fallback());
        assert!(!person.is_fallback());
    }

    #[test]
    #[should_panic(expected = "entity name must not be empty")]
    fn test_extracted_entity_empty_name() {
        let _ = ExtractedEntity::new("", EntityType::Person, "content", 0.5);
    }

    #[test]
    #[should_panic(expected = "confidence must be")]
    fn test_extracted_entity_invalid_confidence() {
        let _ = ExtractedEntity::new("Alice", EntityType::Person, "content", 1.5);
    }

    #[test]
    fn test_extracted_relation_new() {
        let relation = ExtractedRelation::new("Alice", "Acme", RelationType::WorksAt, 0.9);
        assert_eq!(relation.source, "Alice");
        assert_eq!(relation.target, "Acme");
        assert_eq!(relation.relation_type, RelationType::WorksAt);
        assert_eq!(relation.confidence, 0.9);
    }

    #[test]
    #[should_panic(expected = "relation source must not be empty")]
    fn test_extracted_relation_empty_source() {
        let _ = ExtractedRelation::new("", "Acme", RelationType::WorksAt, 0.5);
    }

    #[test]
    fn test_extraction_result_empty() {
        let result = ExtractionResult::empty("test text");
        assert!(result.is_empty());
        assert_eq!(result.entity_count(), 0);
        assert_eq!(result.relation_count(), 0);
        assert_eq!(result.raw_text, "test text");
    }

    #[test]
    fn test_extraction_result_entities_of_type() {
        let entities = vec![
            ExtractedEntity::new("Alice", EntityType::Person, "", 0.9),
            ExtractedEntity::new("Acme", EntityType::Organization, "", 0.8),
            ExtractedEntity::new("Bob", EntityType::Person, "", 0.7),
        ];
        let result = ExtractionResult::new(entities, vec![], "text");

        let people = result.entities_of_type(EntityType::Person);
        assert_eq!(people.len(), 2);
        assert_eq!(people[0].name, "Alice");
        assert_eq!(people[1].name, "Bob");
    }

    #[test]
    fn test_extraction_result_entities_above_confidence() {
        let entities = vec![
            ExtractedEntity::new("Alice", EntityType::Person, "", 0.9),
            ExtractedEntity::new("Bob", EntityType::Person, "", 0.5),
        ];
        let result = ExtractionResult::new(entities, vec![], "text");

        let high = result.entities_above_confidence(0.8);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].name, "Alice");
    }

    #[test]
    fn test_extraction_options_builder() {
        let options = ExtractionOptions::new()
            .with_existing_entities(vec!["Alice".into()])
            .with_min_confidence(0.5);

        assert_eq!(options.existing_entities, vec!["Alice"]);
        assert_eq!(options.min_confidence, 0.5);
    }

    #[test]
    #[should_panic(expected = "min_confidence must be")]
    fn test_extraction_options_invalid_confidence() {
        let _ = ExtractionOptions::new().with_min_confidence(1.5);
    }

    #[test]
    fn test_entity_type_serde() {
        let json = serde_json::to_string(&EntityType::Person).unwrap();
        assert_eq!(json, r#""person""#);

        let parsed: EntityType = serde_json::from_str(r#""organization""#).unwrap();
        assert_eq!(parsed, EntityType::Organization);

        // Test alias
        let parsed: EntityType = serde_json::from_str(r#""org""#).unwrap();
        assert_eq!(parsed, EntityType::Organization);
    }

    #[test]
    fn test_relation_type_serde() {
        let json = serde_json::to_string(&RelationType::WorksAt).unwrap();
        assert_eq!(json, r#""works_at""#);

        let parsed: RelationType = serde_json::from_str(r#""works_at""#).unwrap();
        assert_eq!(parsed, RelationType::WorksAt);
    }
}
