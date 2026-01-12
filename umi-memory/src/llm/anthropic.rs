//! AnthropicProvider - Claude API Integration
//!
//! TigerStyle: Production provider, feature-gated.
//!
//! Requires `anthropic` feature flag:
//! ```toml
//! umi-core = { version = "0.1", features = ["anthropic"] }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{CompletionRequest, LLMProvider, ProviderError};
use crate::constants::{LLM_PROMPT_BYTES_MAX, LLM_RESPONSE_BYTES_MAX};

// =============================================================================
// Constants
// =============================================================================

/// Default Anthropic API URL
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Default model
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";

/// Default max tokens
const DEFAULT_MAX_TOKENS: usize = 4096;

/// API version header
const ANTHROPIC_VERSION: &str = "2023-06-01";

// =============================================================================
// API Types
// =============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: usize,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// =============================================================================
// AnthropicProvider
// =============================================================================

/// Anthropic Claude API provider.
///
/// TigerStyle: Production provider with explicit configuration.
///
/// # Example
///
/// ```rust,ignore
/// use umi_memory::llm::{AnthropicProvider, CompletionRequest, LLMProvider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").unwrap())
///         .with_model("claude-sonnet-4-20250514");
///
///     let request = CompletionRequest::new("Hello, Claude!");
///     let response = provider.complete(&request).await.unwrap();
///     println!("{}", response);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    /// HTTP client
    client: reqwest::Client,
    /// API key
    api_key: String,
    /// Model to use
    model: String,
    /// API URL (for testing/proxies)
    api_url: String,
}

impl AnthropicProvider {
    /// Create a new `AnthropicProvider` with the given API key.
    ///
    /// Uses default model (`claude-sonnet-4-20250514`).
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            api_url: ANTHROPIC_API_URL.to_string(),
        }
    }

    /// Set the model to use.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = AnthropicProvider::new(api_key)
    ///     .with_model("claude-opus-4-20250514");
    /// ```
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set a custom API URL (for testing or proxies).
    #[must_use]
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Get the current model.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Build the request body.
    fn build_request(&self, request: &CompletionRequest) -> AnthropicRequest {
        AnthropicRequest {
            model: self.model.clone(),
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages: vec![Message {
                role: "user".to_string(),
                content: request.prompt.clone(),
            }],
            system: request.system.clone(),
            temperature: request.temperature,
        }
    }

    /// Parse error response.
    fn parse_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
        // Try to parse as Anthropic error
        if let Ok(err) = serde_json::from_str::<AnthropicError>(body) {
            return match err.error.error_type.as_str() {
                "authentication_error" => ProviderError::AuthenticationFailed,
                "rate_limit_error" => ProviderError::rate_limit(None),
                "overloaded_error" => {
                    ProviderError::service_unavailable("Anthropic API overloaded")
                }
                "invalid_request_error" => ProviderError::invalid_request(err.error.message),
                _ => ProviderError::invalid_response(err.error.message),
            };
        }

        // Fall back to status code
        match status {
            reqwest::StatusCode::UNAUTHORIZED => ProviderError::AuthenticationFailed,
            reqwest::StatusCode::TOO_MANY_REQUESTS => ProviderError::rate_limit(None),
            reqwest::StatusCode::SERVICE_UNAVAILABLE | reqwest::StatusCode::BAD_GATEWAY => {
                ProviderError::service_unavailable("Anthropic API unavailable")
            }
            reqwest::StatusCode::REQUEST_TIMEOUT | reqwest::StatusCode::GATEWAY_TIMEOUT => {
                ProviderError::Timeout
            }
            _ => ProviderError::invalid_response(format!("HTTP {}: {}", status, body)),
        }
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<String, ProviderError> {
        // Preconditions
        debug_assert!(
            request.prompt.len() <= LLM_PROMPT_BYTES_MAX,
            "prompt exceeds limit"
        );

        let body = self.build_request(request);

        let response = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout
                } else if e.is_connect() {
                    ProviderError::network_error("Connection failed")
                } else {
                    ProviderError::network_error(e.to_string())
                }
            })?;

        let status = response.status();
        let response_body = response
            .text()
            .await
            .map_err(|e| ProviderError::network_error(e.to_string()))?;

        if !status.is_success() {
            return Err(Self::parse_error(status, &response_body));
        }

        let parsed: AnthropicResponse = serde_json::from_str(&response_body)
            .map_err(|e| ProviderError::json_error(format!("Failed to parse response: {}", e)))?;

        // Extract text from content blocks
        let text = parsed
            .content
            .into_iter()
            .filter_map(|block| {
                if block.content_type == "text" {
                    block.text
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        // Postcondition
        debug_assert!(
            text.len() <= LLM_RESPONSE_BYTES_MAX,
            "response exceeds limit"
        );

        Ok(text)
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn is_simulation(&self) -> bool {
        false
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let provider = AnthropicProvider::new("test-key");
        assert_eq!(provider.model(), DEFAULT_MODEL);
        assert!(!provider.is_simulation());
        assert_eq!(provider.name(), "anthropic");
    }

    #[test]
    fn test_with_model() {
        let provider = AnthropicProvider::new("test-key").with_model("claude-opus-4-20250514");
        assert_eq!(provider.model(), "claude-opus-4-20250514");
    }

    #[test]
    fn test_with_api_url() {
        let provider =
            AnthropicProvider::new("test-key").with_api_url("http://localhost:8080/v1/messages");
        assert_eq!(provider.api_url, "http://localhost:8080/v1/messages");
    }

    #[test]
    fn test_build_request() {
        let provider = AnthropicProvider::new("test-key");

        let request = CompletionRequest::new("Hello")
            .with_system("You are helpful")
            .with_max_tokens(100)
            .with_temperature(0.5);

        let built = provider.build_request(&request);

        assert_eq!(built.model, DEFAULT_MODEL);
        assert_eq!(built.max_tokens, 100);
        assert_eq!(built.messages.len(), 1);
        assert_eq!(built.messages[0].role, "user");
        assert_eq!(built.messages[0].content, "Hello");
        assert_eq!(built.system, Some("You are helpful".to_string()));
        assert_eq!(built.temperature, Some(0.5));
    }

    #[test]
    fn test_parse_error_auth() {
        let body = r#"{"error":{"type":"authentication_error","message":"Invalid API key"}}"#;
        let err = AnthropicProvider::parse_error(reqwest::StatusCode::UNAUTHORIZED, body);
        assert!(matches!(err, ProviderError::AuthenticationFailed));
    }

    #[test]
    fn test_parse_error_rate_limit() {
        let body = r#"{"error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}"#;
        let err = AnthropicProvider::parse_error(reqwest::StatusCode::TOO_MANY_REQUESTS, body);
        assert!(matches!(err, ProviderError::RateLimit { .. }));
    }

    #[test]
    fn test_parse_error_overloaded() {
        let body = r#"{"error":{"type":"overloaded_error","message":"API overloaded"}}"#;
        let err = AnthropicProvider::parse_error(reqwest::StatusCode::SERVICE_UNAVAILABLE, body);
        assert!(matches!(err, ProviderError::ServiceUnavailable { .. }));
    }

    #[test]
    fn test_parse_error_invalid_request() {
        let body = r#"{"error":{"type":"invalid_request_error","message":"Bad prompt"}}"#;
        let err = AnthropicProvider::parse_error(reqwest::StatusCode::BAD_REQUEST, body);
        assert!(matches!(err, ProviderError::InvalidRequest { .. }));
    }

    #[test]
    fn test_parse_error_fallback() {
        let body = "Internal server error";
        let err = AnthropicProvider::parse_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, body);
        assert!(matches!(err, ProviderError::InvalidResponse { .. }));
    }
}
