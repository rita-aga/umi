//! Entity Extraction - LLM-powered entity and relation extraction
//!
//! TigerStyle: Sim-first, deterministic, graceful degradation.
//!
//! See ADR-014 for design rationale.
//!
//! # Architecture
//!
//! ```text
//! EntityExtractor<P: LLMProvider>
//! ├── extract()         → ExtractionResult
//! ├── extract_entities_only() → Vec<ExtractedEntity>
//! └── Uses prompts::build_extraction_prompt()
//! ```
//!
//! # Usage
//!
//! ```rust
//! use umi_memory::extraction::{EntityExtractor, ExtractionOptions};
//! use umi_memory::llm::SimLLMProvider;
//!
//! #[tokio::main]
//! async fn main() {
//!     let provider = SimLLMProvider::with_seed(42);
//!     let extractor = EntityExtractor::new(provider);
//!
//!     let result = extractor.extract("Alice works at Acme Corp", ExtractionOptions::default()).await.unwrap();
//!     println!("Found {} entities", result.entity_count());
//! }
//! ```

mod prompts;
mod types;

pub use prompts::build_extraction_prompt;
pub use types::{
    EntityType, ExtractedEntity, ExtractedRelation, ExtractionOptions, ExtractionResult,
    RelationType,
};

use serde::Deserialize;

use crate::constants::{
    EXTRACTION_CONFIDENCE_DEFAULT, EXTRACTION_CONFIDENCE_MAX, EXTRACTION_CONFIDENCE_MIN,
    EXTRACTION_ENTITIES_COUNT_MAX, EXTRACTION_RELATIONS_COUNT_MAX, EXTRACTION_TEXT_BYTES_MAX,
};
use crate::llm::{CompletionRequest, LLMProvider, ProviderError};

// =============================================================================
// Error Types
// =============================================================================

/// Errors from entity extraction.
///
/// Note: LLM errors result in graceful degradation (fallback entity),
/// not an error return.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExtractionError {
    /// Input text is empty
    #[error("Text is empty")]
    EmptyText,

    /// Input text exceeds size limit
    #[error("Text too long: {len} bytes (max {max})")]
    TextTooLong {
        /// Actual length
        len: usize,
        /// Maximum allowed
        max: usize,
    },

    /// Invalid confidence threshold
    #[error("Invalid confidence: {value} (must be {min}-{max})")]
    InvalidConfidence {
        /// Provided value
        value: f64,
        /// Minimum allowed
        min: f64,
        /// Maximum allowed
        max: f64,
    },
}

// =============================================================================
// LLM Response Types (for parsing)
// =============================================================================

/// Raw LLM response structure.
#[derive(Debug, Deserialize, Default)]
struct LLMExtractionResponse {
    #[serde(default)]
    entities: Vec<RawEntity>,
    #[serde(default)]
    relations: Vec<RawRelation>,
}

#[derive(Debug, Deserialize)]
struct RawEntity {
    name: Option<String>,
    #[serde(rename = "type")]
    entity_type: Option<String>,
    content: Option<String>,
    confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawRelation {
    source: Option<String>,
    target: Option<String>,
    #[serde(rename = "type")]
    relation_type: Option<String>,
    confidence: Option<f64>,
}

// =============================================================================
// EntityExtractor
// =============================================================================

/// Entity extractor using LLM.
///
/// TigerStyle: Generic over provider for sim/production flexibility.
///
/// # Example
///
/// ```rust
/// use umi_memory::extraction::{EntityExtractor, ExtractionOptions};
/// use umi_memory::llm::SimLLMProvider;
///
/// #[tokio::main]
/// async fn main() {
///     // Simulation provider for testing
///     let provider = SimLLMProvider::with_seed(42);
///     let extractor = EntityExtractor::new(provider);
///
///     let result = extractor
///         .extract("Alice works at Acme Corp", ExtractionOptions::default())
///         .await
///         .unwrap();
///
///     // Result contains entities and relations
///     assert!(!result.is_empty());
/// }
/// ```
#[derive(Debug)]
pub struct EntityExtractor<P: LLMProvider> {
    provider: P,
}

impl<P: LLMProvider> EntityExtractor<P> {
    /// Create a new entity extractor with the given LLM provider.
    #[must_use]
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Extract entities and relations from text.
    ///
    /// # Arguments
    /// - `text` - Text to extract from
    /// - `options` - Extraction options (existing entities, min confidence)
    ///
    /// # Returns
    /// `ExtractionResult` with entities, relations, and raw text.
    ///
    /// # Errors
    /// Returns `ExtractionError` if text is empty or too long.
    ///
    /// # Graceful Degradation
    /// If the LLM fails or returns invalid JSON, a fallback "note" entity
    /// is created from the input text. This ensures extraction never fails
    /// due to LLM issues.
    pub async fn extract(
        &self,
        text: &str,
        options: ExtractionOptions,
    ) -> Result<ExtractionResult, ExtractionError> {
        // TigerStyle: Preconditions
        if text.is_empty() {
            return Err(ExtractionError::EmptyText);
        }
        if text.len() > EXTRACTION_TEXT_BYTES_MAX {
            return Err(ExtractionError::TextTooLong {
                len: text.len(),
                max: EXTRACTION_TEXT_BYTES_MAX,
            });
        }
        if !(EXTRACTION_CONFIDENCE_MIN..=EXTRACTION_CONFIDENCE_MAX)
            .contains(&options.min_confidence)
        {
            return Err(ExtractionError::InvalidConfidence {
                value: options.min_confidence,
                min: EXTRACTION_CONFIDENCE_MIN,
                max: EXTRACTION_CONFIDENCE_MAX,
            });
        }

        // Build prompt
        let existing = if options.existing_entities.is_empty() {
            None
        } else {
            Some(options.existing_entities.as_slice())
        };
        let prompt = build_extraction_prompt(text, existing);

        // Call LLM
        let (entities, relations) = match self.call_llm(&prompt, text).await {
            Ok((e, r)) => (e, r),
            Err(_) => {
                // Graceful degradation: return fallback
                (self.create_fallback_entity(text), Vec::new())
            }
        };

        // Filter by confidence
        let entities: Vec<_> = if options.min_confidence > 0.0 {
            entities
                .into_iter()
                .filter(|e| e.confidence >= options.min_confidence)
                .collect()
        } else {
            entities
        };

        let relations: Vec<_> = if options.min_confidence > 0.0 {
            relations
                .into_iter()
                .filter(|r| r.confidence >= options.min_confidence)
                .collect()
        } else {
            relations
        };

        // Apply limits
        let entities: Vec<_> = entities
            .into_iter()
            .take(EXTRACTION_ENTITIES_COUNT_MAX)
            .collect();
        let relations: Vec<_> = relations
            .into_iter()
            .take(EXTRACTION_RELATIONS_COUNT_MAX)
            .collect();

        let result = ExtractionResult::new(entities, relations, text);

        // TigerStyle: Postconditions
        debug_assert!(
            result.entity_count() <= EXTRACTION_ENTITIES_COUNT_MAX,
            "too many entities"
        );
        debug_assert!(
            result.relation_count() <= EXTRACTION_RELATIONS_COUNT_MAX,
            "too many relations"
        );

        Ok(result)
    }

    /// Extract only entities (convenience method).
    ///
    /// # Arguments
    /// - `text` - Text to extract from
    ///
    /// # Returns
    /// Vector of extracted entities.
    pub async fn extract_entities_only(
        &self,
        text: &str,
    ) -> Result<Vec<ExtractedEntity>, ExtractionError> {
        let result = self.extract(text, ExtractionOptions::default()).await?;
        Ok(result.entities)
    }

    /// Call LLM and parse response.
    async fn call_llm(
        &self,
        prompt: &str,
        original_text: &str,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>), ProviderError> {
        let request = CompletionRequest::new(prompt).with_json_mode();
        let response = self.provider.complete(&request).await?;

        // Parse response
        let parsed = self.parse_response(&response, original_text);
        Ok(parsed)
    }

    /// Parse LLM response into entities and relations.
    fn parse_response(
        &self,
        response: &str,
        original_text: &str,
    ) -> (Vec<ExtractedEntity>, Vec<ExtractedRelation>) {
        // Try to parse as JSON
        let data: LLMExtractionResponse = match serde_json::from_str(response) {
            Ok(d) => d,
            Err(_) => {
                // Fallback on parse error
                return (self.create_fallback_entity(original_text), Vec::new());
            }
        };

        let entities = self.parse_entities(&data.entities, original_text);
        let relations = self.parse_relations(&data.relations);

        // If no valid entities, create fallback
        if entities.is_empty() {
            return (self.create_fallback_entity(original_text), relations);
        }

        (entities, relations)
    }

    /// Parse raw entities into validated entities.
    fn parse_entities(
        &self,
        raw_entities: &[RawEntity],
        original_text: &str,
    ) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        for raw in raw_entities {
            // Extract name (required)
            let name = match &raw.name {
                Some(n) if !n.trim().is_empty() => n.trim().to_string(),
                _ => continue,
            };

            // Truncate if needed
            let name = if name.len() > crate::constants::EXTRACTION_ENTITY_NAME_BYTES_MAX {
                name[..crate::constants::EXTRACTION_ENTITY_NAME_BYTES_MAX].to_string()
            } else {
                name
            };

            // Parse entity type
            let entity_type = raw
                .entity_type
                .as_deref()
                .map(EntityType::from_str_or_note)
                .unwrap_or(EntityType::Note);

            // Get content
            let content = raw
                .content
                .as_deref()
                .unwrap_or(&original_text[..200.min(original_text.len())])
                .to_string();

            // Truncate content if needed
            let content = if content.len() > crate::constants::EXTRACTION_ENTITY_CONTENT_BYTES_MAX {
                content[..crate::constants::EXTRACTION_ENTITY_CONTENT_BYTES_MAX].to_string()
            } else {
                content
            };

            // Parse confidence
            let confidence = raw
                .confidence
                .map(|c| c.clamp(EXTRACTION_CONFIDENCE_MIN, EXTRACTION_CONFIDENCE_MAX))
                .unwrap_or(EXTRACTION_CONFIDENCE_DEFAULT);

            entities.push(ExtractedEntity::new(name, entity_type, content, confidence));
        }

        entities
    }

    /// Parse raw relations into validated relations.
    fn parse_relations(&self, raw_relations: &[RawRelation]) -> Vec<ExtractedRelation> {
        let mut relations = Vec::new();

        for raw in raw_relations {
            // Extract source and target (required)
            let source = match &raw.source {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => continue,
            };

            let target = match &raw.target {
                Some(t) if !t.trim().is_empty() => t.trim().to_string(),
                _ => continue,
            };

            // Parse relation type
            let relation_type = raw
                .relation_type
                .as_deref()
                .map(RelationType::from_str_or_relates_to)
                .unwrap_or(RelationType::RelatesTo);

            // Parse confidence
            let confidence = raw
                .confidence
                .map(|c| c.clamp(EXTRACTION_CONFIDENCE_MIN, EXTRACTION_CONFIDENCE_MAX))
                .unwrap_or(EXTRACTION_CONFIDENCE_DEFAULT);

            relations.push(ExtractedRelation::new(
                source,
                target,
                relation_type,
                confidence,
            ));
        }

        relations
    }

    /// Create fallback note entity from text.
    fn create_fallback_entity(&self, text: &str) -> Vec<ExtractedEntity> {
        let name = format!("Note: {}", &text[..50.min(text.len())]);
        let content = text[..500.min(text.len())].to_string();

        vec![ExtractedEntity::new(
            name,
            EntityType::Note,
            content,
            EXTRACTION_CONFIDENCE_DEFAULT,
        )]
    }

    /// Get a reference to the underlying provider.
    #[must_use]
    pub fn provider(&self) -> &P {
        &self.provider
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::SimLLMProvider;

    fn create_test_extractor(seed: u64) -> EntityExtractor<SimLLMProvider> {
        EntityExtractor::new(SimLLMProvider::with_seed(seed))
    }

    #[tokio::test]
    async fn test_basic_extraction() {
        let extractor = create_test_extractor(42);

        let result = extractor
            .extract("Alice works at Acme Corp", ExtractionOptions::default())
            .await
            .unwrap();

        // Should have at least one entity (from SimLLM's entity extraction routing)
        assert!(!result.is_empty());
        assert_eq!(result.raw_text, "Alice works at Acme Corp");
    }

    #[tokio::test]
    async fn test_extraction_with_existing_entities() {
        let extractor = create_test_extractor(42);

        let options = ExtractionOptions::new()
            .with_existing_entities(vec!["Alice".to_string(), "Acme".to_string()]);

        let result = extractor
            .extract("She joined last month", options)
            .await
            .unwrap();

        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_extraction_entities_only() {
        let extractor = create_test_extractor(42);

        let entities = extractor
            .extract_entities_only("Bob met Charlie at Google")
            .await
            .unwrap();

        assert!(!entities.is_empty());
    }

    #[tokio::test]
    async fn test_extraction_with_min_confidence() {
        let extractor = create_test_extractor(42);

        let options = ExtractionOptions::new().with_min_confidence(0.9);

        let result = extractor
            .extract("Alice works at Acme", options)
            .await
            .unwrap();

        // All entities should have confidence >= 0.9
        for entity in &result.entities {
            assert!(entity.confidence >= 0.9);
        }
    }

    #[tokio::test]
    async fn test_empty_text_error() {
        let extractor = create_test_extractor(42);

        let result = extractor.extract("", ExtractionOptions::default()).await;

        assert!(matches!(result, Err(ExtractionError::EmptyText)));
    }

    #[tokio::test]
    async fn test_text_too_long_error() {
        let extractor = create_test_extractor(42);

        let long_text = "x".repeat(EXTRACTION_TEXT_BYTES_MAX + 1);
        let result = extractor
            .extract(&long_text, ExtractionOptions::default())
            .await;

        assert!(matches!(result, Err(ExtractionError::TextTooLong { .. })));
    }

    #[tokio::test]
    async fn test_invalid_confidence_error() {
        let extractor = create_test_extractor(42);

        let result = extractor
            .extract(
                "test",
                ExtractionOptions {
                    existing_entities: vec![],
                    min_confidence: 1.5,
                },
            )
            .await;

        assert!(matches!(
            result,
            Err(ExtractionError::InvalidConfidence { .. })
        ));
    }

    #[tokio::test]
    async fn test_determinism() {
        let extractor1 = create_test_extractor(42);
        let extractor2 = create_test_extractor(42);

        let result1 = extractor1
            .extract("Alice works at Microsoft", ExtractionOptions::default())
            .await
            .unwrap();

        let result2 = extractor2
            .extract("Alice works at Microsoft", ExtractionOptions::default())
            .await
            .unwrap();

        // Same seed should produce same results
        assert_eq!(result1.entity_count(), result2.entity_count());
        assert_eq!(result1.relation_count(), result2.relation_count());
    }

    #[test]
    fn test_parse_entities_with_valid_data() {
        let extractor = create_test_extractor(42);

        let raw = vec![
            RawEntity {
                name: Some("Alice".to_string()),
                entity_type: Some("person".to_string()),
                content: Some("A person".to_string()),
                confidence: Some(0.9),
            },
            RawEntity {
                name: Some("Acme".to_string()),
                entity_type: Some("org".to_string()),
                content: Some("A company".to_string()),
                confidence: Some(0.8),
            },
        ];

        let entities = extractor.parse_entities(&raw, "original text");

        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[0].entity_type, EntityType::Person);
        assert_eq!(entities[1].name, "Acme");
        assert_eq!(entities[1].entity_type, EntityType::Organization);
    }

    #[test]
    fn test_parse_entities_with_invalid_data() {
        let extractor = create_test_extractor(42);

        let raw = vec![
            RawEntity {
                name: None, // Missing name - should be skipped
                entity_type: Some("person".to_string()),
                content: None,
                confidence: None,
            },
            RawEntity {
                name: Some("  ".to_string()), // Empty name - should be skipped
                entity_type: None,
                content: None,
                confidence: None,
            },
        ];

        let entities = extractor.parse_entities(&raw, "original text");

        // Both should be skipped
        assert!(entities.is_empty());
    }

    #[test]
    fn test_parse_entities_with_unknown_type() {
        let extractor = create_test_extractor(42);

        let raw = vec![RawEntity {
            name: Some("Unknown".to_string()),
            entity_type: Some("unknown_type".to_string()),
            content: None,
            confidence: None,
        }];

        let entities = extractor.parse_entities(&raw, "original text");

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, EntityType::Note); // Falls back to Note
    }

    #[test]
    fn test_parse_relations_with_valid_data() {
        let extractor = create_test_extractor(42);

        let raw = vec![RawRelation {
            source: Some("Alice".to_string()),
            target: Some("Acme".to_string()),
            relation_type: Some("works_at".to_string()),
            confidence: Some(0.9),
        }];

        let relations = extractor.parse_relations(&raw);

        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].source, "Alice");
        assert_eq!(relations[0].target, "Acme");
        assert_eq!(relations[0].relation_type, RelationType::WorksAt);
    }

    #[test]
    fn test_parse_relations_with_missing_fields() {
        let extractor = create_test_extractor(42);

        let raw = vec![
            RawRelation {
                source: None,
                target: Some("Acme".to_string()),
                relation_type: None,
                confidence: None,
            },
            RawRelation {
                source: Some("Alice".to_string()),
                target: None,
                relation_type: None,
                confidence: None,
            },
        ];

        let relations = extractor.parse_relations(&raw);

        // Both should be skipped due to missing source/target
        assert!(relations.is_empty());
    }

    #[test]
    fn test_create_fallback_entity() {
        let extractor = create_test_extractor(42);

        let fallback = extractor.create_fallback_entity("This is some text for testing");

        assert_eq!(fallback.len(), 1);
        assert!(fallback[0].name.starts_with("Note: "));
        assert_eq!(fallback[0].entity_type, EntityType::Note);
        assert_eq!(fallback[0].confidence, EXTRACTION_CONFIDENCE_DEFAULT);
    }

    #[test]
    fn test_provider_accessor() {
        let provider = SimLLMProvider::with_seed(42);
        let extractor = EntityExtractor::new(provider);

        assert!(extractor.provider().is_simulation());
    }
}
