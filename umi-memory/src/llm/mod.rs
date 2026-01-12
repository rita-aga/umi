//! LLM Provider Trait - Unified Interface for Sim and Production
//!
//! TigerStyle: Simulation-first LLM abstraction.
//!
//! See ADR-013 for design rationale.
//!
//! # Architecture
//!
//! ```text
//! LLMProvider (trait)
//! ├── SimLLMProvider      (always available, wraps DST SimLLM)
//! ├── AnthropicProvider   (feature: anthropic)
//! └── OpenAIProvider      (feature: openai)
//! ```
//!
//! # Usage
//!
//! ```rust
//! use umi_memory::llm::{LLMProvider, SimLLMProvider, CompletionRequest};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Simulation (always available, no external deps)
//!     let provider = SimLLMProvider::with_seed(42);
//!
//!     let request = CompletionRequest::new("Extract entities from: Alice works at Acme.");
//!     let response = provider.complete(&request).await.unwrap();
//!     println!("Response: {}", response);
//! }
//! ```

mod sim;

#[cfg(feature = "anthropic")]
mod anthropic;

#[cfg(feature = "openai")]
mod openai;

pub use sim::SimLLMProvider;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::constants::{LLM_PROMPT_BYTES_MAX, LLM_RESPONSE_BYTES_MAX};

// =============================================================================
// Error Types
// =============================================================================

/// Unified error type for all LLM providers.
///
/// TigerStyle: Explicit variants for all failure modes.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProviderError {
    /// Request timed out
    #[error("Request timed out")]
    Timeout,

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after_secs:?}s")]
    RateLimit {
        /// Seconds until rate limit resets (if known)
        retry_after_secs: Option<u64>,
    },

    /// Context/prompt too long
    #[error("Context length exceeded: {tokens} tokens")]
    ContextOverflow {
        /// Number of tokens that exceeded the limit
        tokens: usize,
    },

    /// Invalid response from provider
    #[error("Invalid response: {message}")]
    InvalidResponse {
        /// Description of what was invalid
        message: String,
    },

    /// Service unavailable
    #[error("Service unavailable: {message}")]
    ServiceUnavailable {
        /// Reason for unavailability
        message: String,
    },

    /// Authentication failed
    #[error("Authentication failed")]
    AuthenticationFailed,

    /// JSON serialization/deserialization error
    #[error("JSON error: {message}")]
    JsonError {
        /// Description of the JSON error
        message: String,
    },

    /// Network error
    #[error("Network error: {message}")]
    NetworkError {
        /// Description of the network error
        message: String,
    },

    /// Invalid request parameters
    #[error("Invalid request: {message}")]
    InvalidRequest {
        /// Description of what was invalid
        message: String,
    },
}

impl ProviderError {
    /// Create a timeout error.
    #[must_use]
    pub fn timeout() -> Self {
        Self::Timeout
    }

    /// Create a rate limit error.
    #[must_use]
    pub fn rate_limit(retry_after_secs: Option<u64>) -> Self {
        Self::RateLimit { retry_after_secs }
    }

    /// Create a context overflow error.
    #[must_use]
    pub fn context_overflow(tokens: usize) -> Self {
        Self::ContextOverflow { tokens }
    }

    /// Create an invalid response error.
    #[must_use]
    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::InvalidResponse {
            message: message.into(),
        }
    }

    /// Create a service unavailable error.
    #[must_use]
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::ServiceUnavailable {
            message: message.into(),
        }
    }

    /// Create a JSON error.
    #[must_use]
    pub fn json_error(message: impl Into<String>) -> Self {
        Self::JsonError {
            message: message.into(),
        }
    }

    /// Create a network error.
    #[must_use]
    pub fn network_error(message: impl Into<String>) -> Self {
        Self::NetworkError {
            message: message.into(),
        }
    }

    /// Create an invalid request error.
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    /// Check if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout | Self::RateLimit { .. } | Self::ServiceUnavailable { .. }
        )
    }
}

// =============================================================================
// Request Types
// =============================================================================

/// Request for LLM completion.
///
/// TigerStyle: Explicit fields, no hidden defaults.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// The prompt text (required)
    pub prompt: String,
    /// Optional system message (for chat-style APIs)
    pub system: Option<String>,
    /// Maximum tokens to generate (provider default if None)
    pub max_tokens: Option<usize>,
    /// Temperature (0.0-1.0, provider default if None)
    pub temperature: Option<f32>,
    /// Whether to request JSON output
    pub json_mode: bool,
}

impl CompletionRequest {
    /// Create a new completion request with just a prompt.
    ///
    /// # Panics
    /// Panics if prompt is empty or exceeds `LLM_PROMPT_BYTES_MAX`.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();

        // Preconditions
        assert!(!prompt.is_empty(), "prompt must not be empty");
        assert!(
            prompt.len() <= LLM_PROMPT_BYTES_MAX,
            "prompt exceeds {} bytes",
            LLM_PROMPT_BYTES_MAX
        );

        Self {
            prompt,
            system: None,
            max_tokens: None,
            temperature: None,
            json_mode: false,
        }
    }

    /// Set the system message.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set maximum tokens to generate.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature.
    ///
    /// # Panics
    /// Panics if temperature is not in [0.0, 1.0].
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&temperature),
            "temperature must be in [0.0, 1.0]"
        );
        self.temperature = Some(temperature);
        self
    }

    /// Enable JSON mode (request structured output).
    #[must_use]
    pub fn with_json_mode(mut self) -> Self {
        self.json_mode = true;
        self
    }
}

// =============================================================================
// Provider Trait
// =============================================================================

/// Trait for LLM providers.
///
/// TigerStyle: Unified interface for simulation and production.
///
/// All providers implement this trait, allowing higher-level components
/// to work with any provider without knowing the concrete type.
///
/// # Example
///
/// ```rust
/// use umi_memory::llm::{LLMProvider, SimLLMProvider, CompletionRequest};
///
/// async fn extract_entities<P: LLMProvider>(provider: &P, text: &str) -> String {
///     let request = CompletionRequest::new(format!("Extract entities from: {}", text));
///     provider.complete(&request).await.unwrap()
/// }
/// ```
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Complete a prompt with a text response.
    ///
    /// # Errors
    /// Returns `ProviderError` on failure.
    async fn complete(&self, request: &CompletionRequest) -> Result<String, ProviderError>;

    /// Complete a prompt expecting a JSON response.
    ///
    /// This is a convenience method that calls `complete` and parses the response.
    ///
    /// # Errors
    /// Returns `ProviderError` on failure or JSON parse error.
    async fn complete_json<T: DeserializeOwned + Send>(
        &self,
        request: &CompletionRequest,
    ) -> Result<T, ProviderError> {
        let response = self.complete(request).await?;

        // Postcondition: response should be valid JSON
        debug_assert!(
            response.len() <= LLM_RESPONSE_BYTES_MAX,
            "response exceeds limit"
        );

        serde_json::from_str(&response).map_err(|e| ProviderError::json_error(e.to_string()))
    }

    /// Get the provider name for logging/debugging.
    fn name(&self) -> &'static str;

    /// Check if this is a simulation provider.
    ///
    /// Returns `true` for `SimLLMProvider`, `false` for real providers.
    fn is_simulation(&self) -> bool;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_request_new() {
        let request = CompletionRequest::new("Hello, world!");
        assert_eq!(request.prompt, "Hello, world!");
        assert!(request.system.is_none());
        assert!(request.max_tokens.is_none());
        assert!(request.temperature.is_none());
        assert!(!request.json_mode);
    }

    #[test]
    fn test_completion_request_builder() {
        let request = CompletionRequest::new("Hello")
            .with_system("You are a helpful assistant")
            .with_max_tokens(100)
            .with_temperature(0.7)
            .with_json_mode();

        assert_eq!(request.prompt, "Hello");
        assert_eq!(request.system, Some("You are a helpful assistant".into()));
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
        assert!(request.json_mode);
    }

    #[test]
    #[should_panic(expected = "prompt must not be empty")]
    fn test_completion_request_empty_prompt() {
        let _ = CompletionRequest::new("");
    }

    #[test]
    #[should_panic(expected = "temperature must be in")]
    fn test_completion_request_invalid_temperature() {
        let _ = CompletionRequest::new("Hello").with_temperature(1.5);
    }

    #[test]
    fn test_provider_error_is_retryable() {
        assert!(ProviderError::timeout().is_retryable());
        assert!(ProviderError::rate_limit(Some(60)).is_retryable());
        assert!(ProviderError::service_unavailable("down").is_retryable());
        assert!(!ProviderError::AuthenticationFailed.is_retryable());
        assert!(!ProviderError::json_error("parse failed").is_retryable());
    }

    #[test]
    fn test_provider_error_constructors() {
        let err = ProviderError::context_overflow(10000);
        assert!(matches!(
            err,
            ProviderError::ContextOverflow { tokens: 10000 }
        ));

        let err = ProviderError::invalid_response("bad format");
        assert!(matches!(err, ProviderError::InvalidResponse { .. }));

        let err = ProviderError::network_error("connection refused");
        assert!(matches!(err, ProviderError::NetworkError { .. }));
    }
}
