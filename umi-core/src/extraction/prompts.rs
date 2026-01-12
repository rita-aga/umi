//! Extraction Prompts - LLM Prompt Templates
//!
//! TigerStyle: Structured prompts with clear output format.

/// Extraction prompt template.
///
/// Placeholders:
/// - `{text}` - The text to extract from
/// - `{context_section}` - Optional existing entity context
pub const EXTRACTION_PROMPT: &str = r#"Extract entities and relationships from this text.

Text: {text}

{context_section}

Return JSON with this exact structure:
{
  "entities": [
    {"name": "entity name", "type": "person|organization|project|topic|preference|task|event|note", "content": "brief description", "confidence": 0.0-1.0}
  ],
  "relations": [
    {"source": "entity1 name", "target": "entity2 name", "type": "works_at|knows|manages|relates_to|prefers|part_of", "confidence": 0.0-1.0}
  ]
}

Rules:
- Only extract clear, factual entities
- Use confidence 0.9+ for explicit mentions, 0.5-0.8 for inferred
- Skip uncertain entities
- Return empty arrays if no entities found

Only return the JSON, nothing else."#;

/// Context section template for existing entities.
pub const CONTEXT_SECTION_TEMPLATE: &str = "Known entities (for context): {entities}";

/// Build the full extraction prompt.
///
/// # Arguments
/// - `text` - The text to extract from
/// - `existing_entities` - Optional list of known entity names
///
/// # Returns
/// The formatted prompt string.
#[must_use]
pub fn build_extraction_prompt(text: &str, existing_entities: Option<&[String]>) -> String {
    let context_section = match existing_entities {
        Some(entities) if !entities.is_empty() => {
            CONTEXT_SECTION_TEMPLATE.replace("{entities}", &entities.join(", "))
        }
        _ => String::new(),
    };

    EXTRACTION_PROMPT
        .replace("{text}", text)
        .replace("{context_section}", &context_section)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_without_context() {
        let prompt = build_extraction_prompt("Alice works at Acme", None);

        assert!(prompt.contains("Alice works at Acme"));
        assert!(!prompt.contains("Known entities"));
    }

    #[test]
    fn test_build_prompt_with_empty_context() {
        let prompt = build_extraction_prompt("Alice works at Acme", Some(&[]));

        assert!(prompt.contains("Alice works at Acme"));
        assert!(!prompt.contains("Known entities"));
    }

    #[test]
    fn test_build_prompt_with_context() {
        let entities = vec!["Alice".to_string(), "Acme Corp".to_string()];
        let prompt = build_extraction_prompt("She joined last month", Some(&entities));

        assert!(prompt.contains("She joined last month"));
        assert!(prompt.contains("Known entities (for context): Alice, Acme Corp"));
    }

    #[test]
    fn test_prompt_structure() {
        let prompt = build_extraction_prompt("test", None);

        // Verify key structural elements
        assert!(prompt.contains("Extract entities and relationships"));
        assert!(prompt.contains("Text: test"));
        assert!(prompt.contains(r#""entities""#));
        assert!(prompt.contains(r#""relations""#));
        assert!(prompt.contains("Only return the JSON"));
    }
}
