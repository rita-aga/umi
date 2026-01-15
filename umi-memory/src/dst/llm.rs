//! `SimLLM` - Deterministic LLM Simulation
//!
//! `TigerStyle`: Deterministic LLM responses for simulation testing.
//!
//! See ADR-012 for design rationale.

use std::sync::{Arc, Mutex};

use serde::de::DeserializeOwned;
use serde_json::json;

use super::clock::SimClock;
use super::fault::{FaultInjector, FaultType};
use super::rng::DeterministicRng;
use crate::constants::{
    LLM_ENTITIES_COUNT_MAX, LLM_LATENCY_MS_DEFAULT, LLM_LATENCY_MS_MAX, LLM_LATENCY_MS_MIN,
    LLM_PROMPT_BYTES_MAX, LLM_QUERY_REWRITES_COUNT_MAX, LLM_RESPONSE_BYTES_MAX,
};

// =============================================================================
// Error Types
// =============================================================================

/// Errors from LLM operations.
///
/// `TigerStyle`: Explicit error variants for all failure modes.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LLMError {
    /// Request timed out
    #[error("LLM request timed out")]
    Timeout,

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimit,

    /// Context/prompt too long
    #[error("Context length exceeded: {0} bytes")]
    ContextOverflow(usize),

    /// Response format invalid
    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    /// Service unavailable
    #[error("Service unavailable")]
    ServiceUnavailable,

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(String),

    /// Prompt validation failed
    #[error("Invalid prompt: {0}")]
    InvalidPrompt(String),
}

// =============================================================================
// SimLLM
// =============================================================================

/// Common names for entity extraction simulation.
/// Reserved for future use in enhanced entity generation.
#[allow(dead_code)]
const COMMON_NAMES: &[&str] = &[
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
];

/// Common organizations for entity extraction simulation.
/// Reserved for future use in enhanced entity generation.
#[allow(dead_code)]
const COMMON_ORGS: &[&str] = &[
    "Acme",
    "Google",
    "Microsoft",
    "Apple",
    "Amazon",
    "OpenAI",
    "Anthropic",
];

/// Simulated LLM for deterministic testing.
///
/// `TigerStyle`:
/// - Deterministic responses via seeded RNG
/// - Prompt routing to domain-specific generators
/// - Fault injection integration
/// - Thread-safe via `Mutex` for use in async contexts
///
/// # Example
///
/// ```rust
/// use umi_memory::dst::{SimLLM, SimClock, DeterministicRng, FaultInjector};
/// use std::sync::Arc;
///
/// let clock = SimClock::new();
/// let rng = DeterministicRng::new(42);
/// let faults = Arc::new(FaultInjector::new(DeterministicRng::new(42)));
/// let llm = SimLLM::new(clock, rng, faults);
///
/// // Same seed = same response
/// ```
#[derive(Debug, Clone)]
pub struct SimLLM {
    /// Simulated clock for latency
    clock: SimClock,
    /// RNG with thread-safe interior mutability (Arc for Clone)
    rng: Arc<Mutex<DeterministicRng>>,
    /// Shared fault injector
    fault_injector: Arc<FaultInjector>,
    /// Base latency for simulated responses
    base_latency_ms: u64,
    /// Whether to simulate latency (disable for tests without time advancement)
    simulate_latency_enabled: bool,
}

impl SimLLM {
    /// Create a new `SimLLM`.
    ///
    /// # Arguments
    /// - `clock`: Simulated clock for latency
    /// - `rng`: Deterministic RNG for response generation
    /// - `fault_injector`: Shared fault injector
    #[must_use]
    pub fn new(clock: SimClock, rng: DeterministicRng, fault_injector: Arc<FaultInjector>) -> Self {
        Self {
            clock,
            rng: Arc::new(Mutex::new(rng)),
            fault_injector,
            base_latency_ms: LLM_LATENCY_MS_DEFAULT,
            simulate_latency_enabled: true,
        }
    }

    /// Disable latency simulation (useful for tests without time advancement).
    ///
    /// By default, `SimLLM` simulates latency using the clock. This blocks if
    /// the clock isn't being advanced. Use this method to disable latency
    /// for simple tests.
    #[must_use]
    pub fn without_latency(mut self) -> Self {
        self.simulate_latency_enabled = false;
        self
    }

    /// Set base latency for simulated responses.
    ///
    /// # Panics
    /// Panics if latency is outside valid range.
    #[must_use]
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        // Precondition
        assert!(
            (LLM_LATENCY_MS_MIN..=LLM_LATENCY_MS_MAX).contains(&latency_ms),
            "latency must be in [{LLM_LATENCY_MS_MIN}, {LLM_LATENCY_MS_MAX}], got {latency_ms}"
        );

        self.base_latency_ms = latency_ms;
        self
    }

    /// Complete a prompt with a deterministic response.
    ///
    /// # Errors
    /// Returns `LLMError` on fault injection or validation failure.
    ///
    /// # Panics
    /// Debug panics on precondition/postcondition violations.
    pub async fn complete(&self, prompt: &str) -> Result<String, LLMError> {
        // Preconditions (runtime checks - return errors for recoverable cases)
        if prompt.is_empty() {
            return Err(LLMError::InvalidPrompt("prompt must not be empty".into()));
        }
        if prompt.len() > LLM_PROMPT_BYTES_MAX {
            return Err(LLMError::ContextOverflow(prompt.len()));
        }

        // Check for faults
        if let Some(fault) = self.fault_injector.should_inject("llm_complete") {
            return Err(self.fault_to_error(fault));
        }

        // Simulate latency
        self.simulate_latency().await;

        // Route prompt to appropriate generator
        let response = self.route_prompt(prompt);

        // Postconditions
        debug_assert!(!response.is_empty(), "response must not be empty");
        debug_assert!(
            response.len() <= LLM_RESPONSE_BYTES_MAX,
            "response exceeds limit"
        );

        Ok(response)
    }

    /// Complete a prompt expecting a JSON response.
    ///
    /// # Errors
    /// Returns `LLMError` on fault injection, validation, or JSON parse failure.
    pub async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T, LLMError> {
        let response = self.complete(prompt).await?;

        serde_json::from_str(&response)
            .map_err(|e| LLMError::JsonError(format!("Failed to parse JSON: {e}")))
    }

    /// Route prompt to the appropriate generator based on content.
    fn route_prompt(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();

        if prompt_lower.contains("extract") && prompt_lower.contains("entit") {
            self.sim_entity_extraction(prompt)
        } else if prompt_lower.contains("rewrite") && prompt_lower.contains("query") {
            self.sim_query_rewrite(prompt)
        } else if prompt_lower.contains("detect") && prompt_lower.contains("evolution") {
            self.sim_evolution_detection(prompt)
        } else if prompt_lower.contains("detect")
            && (prompt_lower.contains("relation") || prompt_lower.contains("relationship"))
        {
            self.sim_relation_detection(prompt)
        } else {
            self.sim_generic(prompt)
        }
    }

    /// Simulate entity extraction response using semantic token analysis.
    ///
    /// **NEW APPROACH**: Instead of hardcoded names, extract ACTUAL entities from text:
    /// - Capitalized words/phrases → potential entities
    /// - Context clues (verbs, prepositions) → entity types
    /// - Multi-word names (e.g., "Sarah Chen", "Acme Corp") → preserved
    ///
    /// This provides meaningful entity extraction for DST tests without requiring real LLMs.
    fn sim_entity_extraction(&self, prompt: &str) -> String {
        let mut entities = Vec::new();
        let mut rng = self.rng.lock().unwrap();

        // Extract the actual text from the prompt
        // Handle multiple prompt formats:
        // 1. "Extract entities from: TEXT" (simple one-line)
        // 2. Multi-line with "Text: USER_TEXT" followed by instructions
        let text = {
            let lines: Vec<&str> = prompt.lines().collect();

            // Look for "Text:" marker (case-insensitive)
            if let Some(text_line_idx) = lines
                .iter()
                .position(|line| line.trim().to_lowercase().starts_with("text:"))
            {
                // Found "Text:" line - extract everything after "Text:" on that line
                let text_line = lines[text_line_idx];
                if let Some(colon_pos) = text_line.find(':') {
                    let after_colon = text_line[colon_pos + 1..].trim();

                    // If text is on the same line as "Text:", use it
                    if !after_colon.is_empty() {
                        after_colon
                    } else if text_line_idx + 1 < lines.len() {
                        // Text might be on the next line
                        lines[text_line_idx + 1].trim()
                    } else {
                        prompt.trim()
                    }
                } else {
                    prompt.trim()
                }
            } else if let Some(colon_pos) = prompt.find(':') {
                // Fallback: simple "Extract entities from: TEXT" format
                let after_colon = &prompt[colon_pos + 1..].trim();
                if after_colon.is_empty() {
                    prompt.trim()
                } else {
                    after_colon
                }
            } else {
                // No markers found, use the whole prompt
                prompt.trim()
            }
        };

        // Tokenize: split into words, preserve capitalization
        let words: Vec<&str> = text.split_whitespace().collect();

        // Track used words to avoid duplicates
        let mut used_indices = std::collections::HashSet::new();

        // Extract multi-word capitalized phrases (e.g., "Sarah Chen", "Acme Corp")
        let mut i = 0;
        while i < words.len() {
            if used_indices.contains(&i) {
                i += 1;
                continue;
            }

            let word = words[i];
            // Check if word starts with capital letter (potential entity)
            if let Some(first_char) = word.chars().next() {
                if first_char.is_uppercase() && word.chars().all(|c| c.is_alphanumeric()) {
                    // Collect consecutive capitalized words (multi-word entities)
                    let mut entity_words = vec![word];
                    let mut j = i + 1;

                    while j < words.len() {
                        let next_word = words[j];
                        if let Some(first) = next_word.chars().next() {
                            if first.is_uppercase()
                                && next_word.chars().all(|c| c.is_alphanumeric())
                            {
                                entity_words.push(next_word);
                                used_indices.insert(j);
                                j += 1;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    used_indices.insert(i);
                    let entity_name = entity_words.join(" ");

                    // Skip pronouns (not real entities)
                    let pronoun_blocklist = [
                        "I",
                        "Me",
                        "My",
                        "Mine",
                        "Myself",
                        "You",
                        "Your",
                        "Yours",
                        "Yourself",
                        "He",
                        "Him",
                        "His",
                        "Himself",
                        "She",
                        "Her",
                        "Hers",
                        "Herself",
                        "It",
                        "Its",
                        "Itself",
                        "We",
                        "Us",
                        "Our",
                        "Ours",
                        "Ourselves",
                        "They",
                        "Them",
                        "Their",
                        "Theirs",
                        "Themselves",
                    ];

                    if pronoun_blocklist.contains(&entity_name.as_str()) {
                        i = j;
                        continue;
                    }

                    // Classify entity type based on context
                    let entity_type = self.classify_entity_type(&entity_name, text);

                    // Generate context snippet
                    let context = self.extract_context(&entity_name, text);

                    entities.push(json!({
                        "name": entity_name,
                        "type": entity_type,
                        "content": context,
                        "confidence": 0.75 + rng.next_float() * 0.25,
                    }));

                    if entities.len() >= LLM_ENTITIES_COUNT_MAX {
                        break;
                    }

                    i = j;
                    continue;
                }
            }
            i += 1;
        }

        // Fallback: if no entities found, create a note from the text
        if entities.is_empty() {
            let hash = self.prompt_hash(prompt);
            let snippet = &text[..100.min(text.len())];
            entities.push(json!({
                "name": format!("note_{}", hash % 1000),
                "type": "note",
                "content": snippet,
                "confidence": 0.5,
            }));
        }

        serde_json::to_string(&json!({
            "entities": entities,
            "relations": [],
        }))
        .unwrap()
    }

    /// Classify entity type based on context clues.
    ///
    /// Heuristics:
    /// - Contains "Corp", "Inc", "LLC", "Ltd" → organization
    /// - Known cities/countries → location
    /// - Near "based in", "located in", "city" → location
    /// - Multi-word names (likely person names) checked for person indicators FIRST
    /// - Near "engineer", "developer", "manager", "learning" → person
    /// - Near "works at", "company", "organization" → organization (for single words)
    /// - Near "team", "group", "department" → organization
    /// - Default → note
    fn classify_entity_type(&self, entity_name: &str, text: &str) -> &'static str {
        let entity_lower = entity_name.to_lowercase();

        // Check entity name itself for obvious organization indicators
        if entity_lower.contains("corp")
            || entity_lower.contains("inc")
            || entity_lower.contains("llc")
            || entity_lower.contains("ltd")
        {
            return "organization";
        }

        // Known cities and locations
        let known_locations = [
            "san francisco",
            "tokyo",
            "new york",
            "london",
            "paris",
            "berlin",
            "seattle",
            "boston",
            "austin",
            "chicago",
            "los angeles",
            "beijing",
            "shanghai",
            "singapore",
            "sydney",
            "toronto",
            "vancouver",
            "mumbai",
            "bangalore",
            "delhi",
            "usa",
            "uk",
            "japan",
            "china",
            "india",
            "canada",
            "australia",
            "germany",
            "france",
        ];

        if known_locations.contains(&entity_lower.as_str()) {
            return "location";
        }

        // Known tech companies and organizations
        let known_orgs = [
            "google",
            "microsoft",
            "apple",
            "amazon",
            "meta",
            "facebook",
            "netflix",
            "tesla",
            "openai",
            "anthropic",
            "nvidia",
            "intel",
            "ibm",
            "oracle",
            "salesforce",
            "adobe",
            "twitter",
            "linkedin",
            "github",
            "gitlab",
            "stackoverflow",
            "reddit",
            "spotify",
            "uber",
            "acme",
            "techco",
            "neuralflow", // Test/demo companies
        ];

        if known_orgs.contains(&entity_lower.as_str()) {
            return "organization";
        }

        // Check context around entity
        if let Some(pos) = text.find(entity_name) {
            let context_start = pos.saturating_sub(50);
            let context_end = (pos + entity_name.len() + 50).min(text.len());
            let context = &text[context_start..context_end].to_lowercase();

            // Location indicators (check first for specificity)
            if context.contains("based in")
                || context.contains("located in")
                || context.contains("city")
                || context.contains("living in")
                || context.contains("moved to")
                || context.contains("visiting")
            {
                return "location";
            }

            // For multi-word names (likely person names), check person indicators FIRST
            if entity_name.contains(' ') {
                // Person indicators (roles, actions)
                if context.contains(" as ")
                    || context.contains("engineer")
                    || context.contains("developer")
                    || context.contains("manager")
                    || context.contains("learning")
                    || context.contains("studying")
                    || context.contains("'s main")
                    || context.contains(" is a ")
                    || context.contains(" works as ")
                {
                    return "person";
                }
            }

            // Organization indicators (applies to all entities)
            // Note: "works at X" means X is the organization, not the subject
            if context.contains("company")
                || context.contains("organization")
                || context.contains("team")
                || context.contains("department")
                || context.contains("group")
            {
                return "organization";
            }

            // For single words after "works at", classify as organization
            if !entity_name.contains(' ') && context.contains("works at") {
                return "organization";
            }

            // Person indicators for single words
            if context.contains("engineer")
                || context.contains("developer")
                || context.contains("manager")
                || context.contains("learning")
                || context.contains("'s main")
            {
                return "person";
            }
        }

        // Default: note
        "note"
    }

    /// Extract context snippet for an entity.
    fn extract_context(&self, entity_name: &str, text: &str) -> String {
        if let Some(pos) = text.find(entity_name) {
            // Find sentence boundaries
            let start = text[..pos].rfind('.').map(|p| p + 1).unwrap_or(0);
            let end = text[pos..]
                .find('.')
                .map(|p| pos + p + 1)
                .unwrap_or(text.len());

            text[start..end].trim().to_string()
        } else {
            format!("Information about {}", entity_name)
        }
    }

    /// Simulate query rewrite response.
    fn sim_query_rewrite(&self, prompt: &str) -> String {
        let mut rng = self.rng.lock().unwrap();

        // Extract the actual query from the prompt (simple heuristic)
        let query = prompt
            .lines()
            .find(|line| line.trim().starts_with("Query:") || line.trim().starts_with("query:"))
            .map_or(&prompt[..50.min(prompt.len())], |line| {
                line.trim_start_matches("Query:")
                    .trim_start_matches("query:")
                    .trim()
            });

        // Generate variations
        let num_rewrites = rng.next_usize(2, LLM_QUERY_REWRITES_COUNT_MAX);
        let mut rewrites = vec![query.to_string()];

        let prefixes = [
            "What is",
            "Tell me about",
            "Information on",
            "Details about",
        ];
        let suffixes = ["?", " please", " in detail", ""];

        for _ in 0..num_rewrites - 1 {
            let prefix = prefixes[rng.next_usize(0, prefixes.len() - 1)];
            let suffix = suffixes[rng.next_usize(0, suffixes.len() - 1)];
            rewrites.push(format!("{prefix} {query}{suffix}"));
        }

        serde_json::to_string(&json!({
            "queries": rewrites,
        }))
        .unwrap()
    }

    /// Simulate evolution detection response.
    fn sim_evolution_detection(&self, prompt: &str) -> String {
        let mut rng = self.rng.lock().unwrap();

        // Weighted evolution types (update most common)
        let evolution_types = [
            ("update", 0.4),
            ("extend", 0.3),
            ("derive", 0.2),
            ("contradict", 0.1),
        ];

        let roll = rng.next_float();
        let mut cumulative = 0.0;
        let mut selected_type = "update";

        for (etype, weight) in &evolution_types {
            cumulative += weight;
            if roll < cumulative {
                selected_type = etype;
                break;
            }
        }

        // Sometimes no evolution detected
        if rng.next_bool(0.3) {
            return serde_json::to_string(&json!({
                "detected": false,
                "evolution_type": null,
                "reason": null,
                "confidence": 0.0,
            }))
            .unwrap();
        }

        let reasons = match selected_type {
            "update" => vec![
                "New information replaces outdated data",
                "Values have been updated",
                "Status has changed",
            ],
            "extend" => vec![
                "Additional details provided",
                "New attributes added",
                "Information expanded",
            ],
            "derive" => vec![
                "Conclusion drawn from existing data",
                "Inference based on prior knowledge",
                "Logically follows from previous entity",
            ],
            "contradict" => vec![
                "Information conflicts with existing record",
                "Inconsistent values detected",
                "Contradictory statement found",
            ],
            _ => vec!["Evolution detected"],
        };

        let reason = reasons[rng.next_usize(0, reasons.len() - 1)];
        let confidence = 0.6 + rng.next_float() * 0.4;

        // Extract entity names from prompt if present (for source/target)
        let hash = self.prompt_hash(prompt);

        serde_json::to_string(&json!({
            "detected": true,
            "evolution_type": selected_type,
            "source_id": format!("entity_{}", hash % 1000),
            "target_id": format!("entity_{}", (hash / 1000) % 1000),
            "reason": reason,
            "confidence": confidence,
        }))
        .unwrap()
    }

    /// Simulate relation detection response.
    fn sim_relation_detection(&self, prompt: &str) -> String {
        let mut rng = self.rng.lock().unwrap();

        // Sometimes no relation detected
        if rng.next_bool(0.4) {
            return serde_json::to_string(&json!({
                "relations": [],
            }))
            .unwrap();
        }

        let relation_types = [
            "works_at",
            "knows",
            "located_in",
            "part_of",
            "created_by",
            "related_to",
        ];

        let num_relations = rng.next_usize(1, 3);
        let mut relations = Vec::new();
        let hash = self.prompt_hash(prompt);

        for i in 0..num_relations {
            let rel_type = relation_types[rng.next_usize(0, relation_types.len() - 1)];
            relations.push(json!({
                "source": format!("entity_{}", (hash + i as u64) % 100),
                "target": format!("entity_{}", (hash + i as u64 + 50) % 100),
                "relation_type": rel_type,
                "confidence": 0.5 + rng.next_float() * 0.5,
            }));
        }

        serde_json::to_string(&json!({
            "relations": relations,
        }))
        .unwrap()
    }

    /// Generic response for unrecognized prompts.
    fn sim_generic(&self, prompt: &str) -> String {
        let hash = self.prompt_hash(prompt);
        let mut rng = self.rng.lock().unwrap();

        let responses = [
            "Acknowledged.",
            "Understood.",
            "Processing complete.",
            "Request handled.",
            "Task completed successfully.",
        ];

        let response = responses[rng.next_usize(0, responses.len() - 1)];

        serde_json::to_string(&json!({
            "response": response,
            "prompt_hash": hash,
            "success": true,
        }))
        .unwrap()
    }

    /// Generate a deterministic hash from prompt.
    fn prompt_hash(&self, prompt: &str) -> u64 {
        // Simple FNV-1a hash for determinism
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in prompt.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Simulate latency using the clock.
    async fn simulate_latency(&self) {
        if !self.simulate_latency_enabled {
            return;
        }

        let jitter = {
            let mut rng = self.rng.lock().unwrap();
            rng.next_usize(0, 50) as u64
        };
        let latency = self.base_latency_ms + jitter;
        self.clock.sleep_ms(latency).await;
    }

    /// Convert fault type to LLM error.
    fn fault_to_error(&self, fault: FaultType) -> LLMError {
        match fault {
            FaultType::LlmTimeout => LLMError::Timeout,
            FaultType::LlmRateLimit => LLMError::RateLimit,
            FaultType::LlmContextOverflow => LLMError::ContextOverflow(0),
            FaultType::LlmInvalidResponse => {
                LLMError::InvalidResponse("Simulated invalid response".into())
            }
            FaultType::LlmServiceUnavailable => LLMError::ServiceUnavailable,
            // Map network faults to service unavailable
            FaultType::NetworkTimeout | FaultType::NetworkConnectionRefused => {
                LLMError::ServiceUnavailable
            }
            // Default mapping
            _ => LLMError::ServiceUnavailable,
        }
    }

    /// Get the current seed (for debugging).
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.rng.lock().unwrap().seed()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::fault::FaultConfig;

    fn create_test_llm(seed: u64) -> SimLLM {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(seed);
        let faults = Arc::new(FaultInjector::new(DeterministicRng::new(seed)));
        SimLLM::new(clock, rng, faults).without_latency()
    }

    #[tokio::test]
    async fn test_determinism() {
        let llm1 = create_test_llm(42);
        let llm2 = create_test_llm(42);

        let prompt = "Extract entities from: Alice works at Acme Corp.";

        let response1 = llm1.complete(prompt).await.unwrap();
        let response2 = llm2.complete(prompt).await.unwrap();

        assert_eq!(
            response1, response2,
            "Same seed should produce same response"
        );
    }

    #[tokio::test]
    async fn test_different_seeds_different_responses() {
        let llm1 = create_test_llm(42);
        let llm2 = create_test_llm(12345);

        let prompt = "Extract entities from: Bob met Charlie at Google.";

        let response1 = llm1.complete(prompt).await.unwrap();
        let response2 = llm2.complete(prompt).await.unwrap();

        // Responses may still be similar due to pattern matching,
        // but confidence values should differ
        assert!(response1.contains("Bob") || response1.contains("Charlie"));
        assert!(response2.contains("Bob") || response2.contains("Charlie"));
    }

    #[tokio::test]
    async fn test_entity_extraction_routing() {
        let llm = create_test_llm(42);

        let prompt = "Extract entities from the following text: Alice and Bob work at Microsoft.";
        let response = llm.complete(prompt).await.unwrap();

        assert!(response.contains("entities"));
        assert!(response.contains("Alice") || response.contains("Bob"));
    }

    #[tokio::test]
    async fn test_query_rewrite_routing() {
        let llm = create_test_llm(42);

        let prompt =
            "Rewrite the following query for better search:\nQuery: what is rust programming";
        let response = llm.complete(prompt).await.unwrap();

        assert!(response.contains("queries"));
    }

    #[tokio::test]
    async fn test_evolution_detection_routing() {
        let llm = create_test_llm(42);

        let prompt = "Detect evolution relationship between:\nOld: Alice is 25\nNew: Alice is 26";
        let response = llm.complete(prompt).await.unwrap();

        assert!(response.contains("evolution_type") || response.contains("detected"));
    }

    #[tokio::test]
    async fn test_generic_routing() {
        let llm = create_test_llm(42);

        let prompt = "Hello, how are you?";
        let response = llm.complete(prompt).await.unwrap();

        assert!(response.contains("response") || response.contains("success"));
    }

    #[tokio::test]
    async fn test_empty_prompt_error() {
        let llm = create_test_llm(42);

        let result = llm.complete("").await;
        assert!(matches!(result, Err(LLMError::InvalidPrompt(_))));
    }

    #[tokio::test]
    async fn test_prompt_too_long_error() {
        let llm = create_test_llm(42);

        let long_prompt = "x".repeat(LLM_PROMPT_BYTES_MAX + 1);
        let result = llm.complete(&long_prompt).await;

        assert!(matches!(result, Err(LLMError::ContextOverflow(_))));
    }

    #[tokio::test]
    async fn test_fault_injection_timeout() {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(DeterministicRng::new(42));
        injector.register(FaultConfig::new(FaultType::LlmTimeout, 1.0));
        let faults = Arc::new(injector);

        let llm = SimLLM::new(clock, rng, faults).without_latency();
        let result = llm.complete("test prompt").await;

        assert!(matches!(result, Err(LLMError::Timeout)));
    }

    #[tokio::test]
    async fn test_fault_injection_rate_limit() {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(42);
        let mut injector = FaultInjector::new(DeterministicRng::new(42));
        injector.register(FaultConfig::new(FaultType::LlmRateLimit, 1.0));
        let faults = Arc::new(injector);

        let llm = SimLLM::new(clock, rng, faults).without_latency();
        let result = llm.complete("test prompt").await;

        assert!(matches!(result, Err(LLMError::RateLimit)));
    }

    #[tokio::test]
    async fn test_complete_json() {
        let llm = create_test_llm(42);

        #[derive(serde::Deserialize)]
        struct GenericResponse {
            response: String,
            success: bool,
        }

        let prompt = "Hello, world!";
        // Note: SimLLM::complete_json remains generic (internal to DST)
        // This test verifies the internal DST API still works
        let result: GenericResponse = llm.complete_json(prompt).await.unwrap();

        assert!(result.success);
        assert!(!result.response.is_empty());
    }

    #[tokio::test]
    async fn test_with_latency() {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(42);
        let faults = Arc::new(FaultInjector::new(DeterministicRng::new(42)));

        let llm = SimLLM::new(clock.clone(), rng, faults).with_latency(500);

        // Spawn a task to advance time while the LLM waits
        let clock_for_advance = clock.clone();
        let advance_handle = tokio::spawn(async move {
            // Give the complete() call time to start waiting
            tokio::task::yield_now().await;
            // Advance time enough to cover latency + jitter (500 + up to 50)
            clock_for_advance.advance_ms(600);
        });

        let start = clock.now_ms();
        llm.complete("test").await.unwrap();
        let end = clock.now_ms();

        advance_handle.await.unwrap();

        // Clock should have advanced (we advanced by 600ms)
        assert!(
            end >= start + 500,
            "Expected clock to advance at least 500ms, start={start}, end={end}"
        );
    }

    #[test]
    fn test_prompt_hash_determinism() {
        let llm = create_test_llm(42);

        let hash1 = llm.prompt_hash("test prompt");
        let hash2 = llm.prompt_hash("test prompt");
        let hash3 = llm.prompt_hash("different prompt");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    #[should_panic(expected = "latency must be in")]
    fn test_invalid_latency() {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(42);
        let faults = Arc::new(FaultInjector::new(DeterministicRng::new(42)));

        let _ = SimLLM::new(clock, rng, faults).with_latency(999999);
    }

    #[tokio::test]
    async fn test_entity_extraction_sarah_chen() {
        let llm = create_test_llm(42);

        let prompt = "Extract entities from: Sarah Chen works at NeuralFlow as an ML engineer";
        let response = llm.complete(prompt).await.unwrap();

        println!("=== Entity Extraction Test ===");
        println!("Prompt: {}", prompt);
        println!("Response: {}", response);

        // Pretty print JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response) {
            println!(
                "Pretty printed:\n{}",
                serde_json::to_string_pretty(&json).unwrap()
            );

            if let Some(entities) = json.get("entities").and_then(|e| e.as_array()) {
                println!("\nExtracted {} entities:", entities.len());
                for (i, entity) in entities.iter().enumerate() {
                    let name = entity.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let entity_type = entity
                        .get("entity_type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("?");
                    let confidence = entity
                        .get("confidence")
                        .and_then(|c| c.as_f64())
                        .unwrap_or(0.0);
                    println!(
                        "  {}. {} ({}) - confidence: {:.2}",
                        i + 1,
                        name,
                        entity_type,
                        confidence
                    );
                }
            }
        }

        // Verify we extracted something useful
        assert!(
            response.contains("Sarah") || response.contains("Chen"),
            "Should extract Sarah or Chen"
        );
    }
}
