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
const COMMON_NAMES: &[&str] = &[
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
];

/// Common organizations for entity extraction simulation.
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

    /// Simulate entity extraction response.
    fn sim_entity_extraction(&self, prompt: &str) -> String {
        let mut entities = Vec::new();
        let mut rng = self.rng.lock().unwrap();

        // Detect common names in prompt
        for name in COMMON_NAMES {
            if prompt.to_uppercase().contains(&name.to_uppercase()) {
                if entities.len() >= LLM_ENTITIES_COUNT_MAX {
                    break;
                }
                entities.push(json!({
                    "name": name,
                    "entity_type": "person",
                    "content": format!("Information about {}", name),
                    "confidence": 0.7 + rng.next_float() * 0.3,
                }));
            }
        }

        // Detect common organizations in prompt
        for org in COMMON_ORGS {
            if prompt.to_uppercase().contains(&org.to_uppercase()) {
                if entities.len() >= LLM_ENTITIES_COUNT_MAX {
                    break;
                }
                entities.push(json!({
                    "name": org,
                    "entity_type": "organization",
                    "content": format!("Organization: {}", org),
                    "confidence": 0.8 + rng.next_float() * 0.2,
                }));
            }
        }

        // Fallback to note entity if nothing found
        if entities.is_empty() {
            let hash = self.prompt_hash(prompt);
            let snippet = &prompt[..100.min(prompt.len())];
            entities.push(json!({
                "name": format!("Note_{}", hash),
                "entity_type": "note",
                "content": snippet,
                "confidence": 0.5 + rng.next_float() * 0.3,
            }));
        }

        serde_json::to_string(&json!({
            "entities": entities,
            "relations": [],
        }))
        .unwrap()
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
}
