//! Retrieval Prompts - LLM Prompt Templates
//!
//! `TigerStyle`: Structured prompts with clear output format.

/// Query rewrite prompt template.
///
/// Placeholder:
/// - `{query}` - The search query to rewrite
pub const QUERY_REWRITE_PROMPT: &str = r#"Rewrite this search query into 2-3 variations that would help find relevant memories.

Query: {query}

Return as JSON array of strings. Example: ["variation 1", "variation 2", "variation 3"]
Only return the JSON array, nothing else."#;

/// Build the query rewrite prompt.
///
/// # Arguments
/// - `query` - The search query to rewrite
///
/// # Returns
/// The formatted prompt string.
#[must_use]
pub fn build_query_rewrite_prompt(query: &str) -> String {
    QUERY_REWRITE_PROMPT.replace("{query}", query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let prompt = build_query_rewrite_prompt("Who works at Acme?");

        assert!(prompt.contains("Who works at Acme?"));
        assert!(prompt.contains("Query:"));
        assert!(prompt.contains("JSON array"));
    }

    #[test]
    fn test_prompt_structure() {
        let prompt = build_query_rewrite_prompt("test");

        // Verify key structural elements
        assert!(prompt.contains("Rewrite this search query"));
        assert!(prompt.contains("Query: test"));
        assert!(prompt.contains("2-3 variations"));
        assert!(prompt.contains("Only return the JSON"));
    }
}
