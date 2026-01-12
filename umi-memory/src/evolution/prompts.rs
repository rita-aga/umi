//! Evolution Detection Prompts - LLM Prompt Templates
//!
//! TigerStyle: Structured prompts with clear output format.

/// Evolution detection prompt template.
///
/// Placeholders:
/// - `{new_content}` - The new entity content to analyze
/// - `{existing_list}` - Formatted list of existing entities
pub const EVOLUTION_DETECTION_PROMPT: &str = r#"Compare new information with existing memories and determine the relationship.

New information:
{new_content}

Existing memories:
{existing_list}

What is the relationship between the new information and existing memories?
- "update": New info replaces/corrects old (e.g., changed job, moved address)
- "extend": New info adds to old (e.g., more details, clarification)
- "derive": New info is conclusion from old (e.g., inference, deduction)
- "contradict": New info conflicts with old (e.g., disagreement, correction)
- "none": No significant relationship

Return JSON with this exact structure:
{"type": "update|extend|derive|contradict|none", "reason": "brief explanation", "related_id": "id of most related memory or null", "confidence": 0.0-1.0}

Only return the JSON, nothing else."#;

/// Build the evolution detection prompt.
///
/// # Arguments
/// - `new_content` - The new entity content (name: content format)
/// - `existing_list` - Formatted string of existing entities
///
/// # Returns
/// The formatted prompt string.
#[must_use]
pub fn build_detection_prompt(new_content: &str, existing_list: &str) -> String {
    EVOLUTION_DETECTION_PROMPT
        .replace("{new_content}", new_content)
        .replace("{existing_list}", existing_list)
}

/// Format an entity for the existing memories list.
///
/// # Arguments
/// - `id` - Entity ID
/// - `name` - Entity name
/// - `content` - Entity content (truncated to 200 chars)
///
/// # Returns
/// Formatted string: `[id] name: content`
#[must_use]
pub fn format_entity_for_prompt(id: &str, name: &str, content: &str) -> String {
    let content_preview = if content.len() > 200 {
        format!("{}...", &content[..200])
    } else {
        content.to_string()
    };
    format!("[{}] {}: {}", id, name, content_preview)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_detection_prompt() {
        let prompt = build_detection_prompt(
            "Alice: Left Acme and joined StartupX",
            "[entity-1] Alice: Works at Acme Corp",
        );

        assert!(prompt.contains("Left Acme and joined StartupX"));
        assert!(prompt.contains("[entity-1] Alice: Works at Acme Corp"));
        assert!(prompt.contains("What is the relationship"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_prompt_structure() {
        let prompt = build_detection_prompt("test content", "test list");

        // Verify key structural elements
        assert!(prompt.contains("Compare new information"));
        assert!(prompt.contains("New information:"));
        assert!(prompt.contains("Existing memories:"));
        assert!(prompt.contains("update"));
        assert!(prompt.contains("extend"));
        assert!(prompt.contains("derive"));
        assert!(prompt.contains("contradict"));
        assert!(prompt.contains("none"));
        assert!(prompt.contains("Only return the JSON"));
    }

    #[test]
    fn test_format_entity_for_prompt() {
        let formatted = format_entity_for_prompt("id-123", "Alice", "Works at Acme Corp");
        assert_eq!(formatted, "[id-123] Alice: Works at Acme Corp");
    }

    #[test]
    fn test_format_entity_truncates_long_content() {
        let long_content = "a".repeat(300);
        let formatted = format_entity_for_prompt("id-123", "Test", &long_content);

        assert!(formatted.len() < 300);
        assert!(formatted.ends_with("..."));
        assert!(formatted.starts_with("[id-123] Test: "));
    }

    #[test]
    fn test_format_entity_short_content_unchanged() {
        let short_content = "Short content";
        let formatted = format_entity_for_prompt("id-1", "Name", short_content);

        assert_eq!(formatted, "[id-1] Name: Short content");
        assert!(!formatted.ends_with("..."));
    }
}
