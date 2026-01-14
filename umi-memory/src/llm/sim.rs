//! `SimLLMProvider` - Simulation-First LLM Provider
//!
//! `TigerStyle`: Primary implementation, wraps DST `SimLLM`.
//!
//! This is the DEFAULT provider for all tests and development.
//! Real providers (Anthropic, `OpenAI`) are secondary.

use std::sync::Arc;

use async_trait::async_trait;

use super::{CompletionRequest, LLMProvider, ProviderError};
use crate::dst::{DeterministicRng, FaultInjector, LLMError, SimClock, SimLLM};

// =============================================================================
// SimLLMProvider
// =============================================================================

/// Simulation LLM provider wrapping DST `SimLLM`.
///
/// `TigerStyle`: Primary implementation, always available.
///
/// This provider wraps the deterministic `SimLLM` from the DST module,
/// providing the same interface as production providers but with:
/// - Deterministic responses (same seed = same output)
/// - Fault injection support
/// - No external dependencies
///
/// # Example
///
/// ```rust
/// use umi_memory::llm::{SimLLMProvider, CompletionRequest, LLMProvider};
///
/// #[tokio::main]
/// async fn main() {
///     // Create with explicit seed for reproducibility
///     let provider = SimLLMProvider::with_seed(42);
///
///     let request = CompletionRequest::new("Extract entities from: Alice works at Acme.");
///     let response = provider.complete(&request).await.unwrap();
///
///     // Same seed always produces same response
///     let provider2 = SimLLMProvider::with_seed(42);
///     let response2 = provider2.complete(&request).await.unwrap();
///     assert_eq!(response, response2);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SimLLMProvider {
    /// The underlying `SimLLM` from DST
    inner: SimLLM,
}

impl SimLLMProvider {
    /// Create a new `SimLLMProvider` from an existing `SimLLM`.
    ///
    /// Use this when you already have a `SimLLM` from `SimEnvironment`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use umi_memory::dst::{Simulation, SimConfig};
    /// use umi_memory::llm::SimLLMProvider;
    ///
    /// let sim = Simulation::new(SimConfig::with_seed(42));
    /// let env = sim.build();
    ///
    /// // Note: In practice, you'd typically use env.llm directly
    /// // This is for cases where you need the LLMProvider trait
    /// ```
    #[must_use]
    pub fn from_sim_llm(sim_llm: SimLLM) -> Self {
        Self { inner: sim_llm }
    }

    /// Create a new standalone `SimLLMProvider` with the given seed.
    ///
    /// This is the most common way to create a `SimLLMProvider` for testing.
    ///
    /// # Example
    ///
    /// ```rust
    /// use umi_memory::llm::SimLLMProvider;
    ///
    /// let provider = SimLLMProvider::with_seed(42);
    /// ```
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(seed);
        let faults = Arc::new(FaultInjector::new(DeterministicRng::new(seed)));

        // Disable latency for standalone use (no clock advancement)
        let sim_llm = SimLLM::new(clock, rng, faults).without_latency();

        Self { inner: sim_llm }
    }

    /// Create a new `SimLLMProvider` with fault injection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use umi_memory::llm::SimLLMProvider;
    /// use umi_memory::dst::{FaultConfig, FaultType, FaultInjector, DeterministicRng};
    /// use std::sync::Arc;
    ///
    /// let mut injector = FaultInjector::new(DeterministicRng::new(42));
    /// injector.register(FaultConfig::new(FaultType::LlmTimeout, 0.5));
    ///
    /// let provider = SimLLMProvider::with_faults(42, Arc::new(injector));
    /// ```
    #[must_use]
    pub fn with_faults(seed: u64, faults: Arc<FaultInjector>) -> Self {
        let clock = SimClock::new();
        let rng = DeterministicRng::new(seed);

        // Disable latency for standalone use
        let sim_llm = SimLLM::new(clock, rng, faults).without_latency();

        Self { inner: sim_llm }
    }

    /// Get the seed used by this provider (for debugging/logging).
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.inner.seed()
    }
}

#[async_trait]
impl LLMProvider for SimLLMProvider {
    #[tracing::instrument(skip(self, request), fields(prompt_len = request.prompt.len()))]
    async fn complete(&self, request: &CompletionRequest) -> Result<String, ProviderError> {
        // Build the full prompt (system + user prompt)
        let full_prompt = match &request.system {
            Some(system) => format!("{}\n\n{}", system, request.prompt),
            None => request.prompt.clone(),
        };

        // Call the underlying SimLLM
        self.inner
            .complete(&full_prompt)
            .await
            .map_err(llm_error_to_provider_error)
    }

    fn name(&self) -> &'static str {
        "sim"
    }

    fn is_simulation(&self) -> bool {
        true
    }
}

/// Convert `LLMError` from DST to `ProviderError`.
fn llm_error_to_provider_error(err: LLMError) -> ProviderError {
    match err {
        LLMError::Timeout => ProviderError::Timeout,
        LLMError::RateLimit => ProviderError::rate_limit(None),
        LLMError::ContextOverflow(size) => ProviderError::context_overflow(size),
        LLMError::InvalidResponse(msg) => ProviderError::invalid_response(msg),
        LLMError::ServiceUnavailable => ProviderError::service_unavailable("service unavailable"),
        LLMError::JsonError(msg) => ProviderError::json_error(msg),
        LLMError::InvalidPrompt(msg) => ProviderError::invalid_request(msg),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::{FaultConfig, FaultType};

    #[tokio::test]
    async fn test_determinism() {
        let provider1 = SimLLMProvider::with_seed(42);
        let provider2 = SimLLMProvider::with_seed(42);

        let request = CompletionRequest::new("Extract entities from: Alice works at Acme.");

        let response1 = provider1.complete(&request).await.unwrap();
        let response2 = provider2.complete(&request).await.unwrap();

        assert_eq!(
            response1, response2,
            "Same seed should produce same response"
        );
    }

    #[tokio::test]
    async fn test_different_seeds() {
        let provider1 = SimLLMProvider::with_seed(42);
        let provider2 = SimLLMProvider::with_seed(12345);

        let request = CompletionRequest::new("Extract entities from: Bob met Charlie.");

        let response1 = provider1.complete(&request).await.unwrap();
        let response2 = provider2.complete(&request).await.unwrap();

        // Responses should be structurally similar but may have different values
        assert!(response1.contains("entities") || response1.contains("Bob"));
        assert!(response2.contains("entities") || response2.contains("Bob"));
    }

    #[tokio::test]
    async fn test_name() {
        let provider = SimLLMProvider::with_seed(42);
        assert_eq!(provider.name(), "sim");
    }

    #[tokio::test]
    async fn test_is_simulation() {
        let provider = SimLLMProvider::with_seed(42);
        assert!(provider.is_simulation());
    }

    #[tokio::test]
    async fn test_with_system_prompt() {
        let provider = SimLLMProvider::with_seed(42);

        let request = CompletionRequest::new("Extract entities from: Alice.")
            .with_system("You are an entity extractor.");

        let response = provider.complete(&request).await.unwrap();
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_complete_json() {
        let provider = SimLLMProvider::with_seed(42);

        #[derive(serde::Deserialize)]
        struct GenericResponse {
            response: String,
            success: bool,
        }

        let request = CompletionRequest::new("Hello, world!").with_json_mode();
        let value = provider.complete_json(&request).await.unwrap();
        let result: GenericResponse = serde_json::from_value(value).unwrap();

        assert!(result.success);
        assert!(!result.response.is_empty());
    }

    #[tokio::test]
    async fn test_entity_extraction_prompt() {
        let provider = SimLLMProvider::with_seed(42);

        let request =
            CompletionRequest::new("Extract entities from the text: Alice works at Microsoft.");

        let response = provider.complete(&request).await.unwrap();

        // Should route to entity extraction and contain entities
        assert!(response.contains("entities"));
        assert!(response.contains("Alice") || response.contains("Microsoft"));
    }

    #[tokio::test]
    async fn test_query_rewrite_prompt() {
        let provider = SimLLMProvider::with_seed(42);

        let request = CompletionRequest::new(
            "Rewrite this query for better search results:\nQuery: what is rust programming",
        );

        let response = provider.complete(&request).await.unwrap();

        // Should route to query rewrite
        assert!(response.contains("queries"));
    }

    #[tokio::test]
    async fn test_fault_injection_timeout() {
        let mut injector = FaultInjector::new(DeterministicRng::new(42));
        injector.register(FaultConfig::new(FaultType::LlmTimeout, 1.0));

        let provider = SimLLMProvider::with_faults(42, Arc::new(injector));
        let request = CompletionRequest::new("Test prompt");

        let result = provider.complete(&request).await;
        assert!(matches!(result, Err(ProviderError::Timeout)));
    }

    #[tokio::test]
    async fn test_fault_injection_rate_limit() {
        let mut injector = FaultInjector::new(DeterministicRng::new(42));
        injector.register(FaultConfig::new(FaultType::LlmRateLimit, 1.0));

        let provider = SimLLMProvider::with_faults(42, Arc::new(injector));
        let request = CompletionRequest::new("Test prompt");

        let result = provider.complete(&request).await;
        assert!(matches!(result, Err(ProviderError::RateLimit { .. })));
    }

    #[tokio::test]
    async fn test_seed_getter() {
        let provider = SimLLMProvider::with_seed(12345);
        assert_eq!(provider.seed(), 12345);
    }
}
