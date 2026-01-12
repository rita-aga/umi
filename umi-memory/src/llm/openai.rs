//! OpenAIProvider - OpenAI API Integration
//!
//! TigerStyle: Production provider, feature-gated.
//!
//! Requires `openai` feature flag:
//! ```toml
//! umi-core = { version = "0.1", features = ["openai"] }
//! ```
//!
//! Also supports OpenAI-compatible APIs (Azure, local models, etc.)
//! via `with_base_url`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{CompletionRequest, LLMProvider, ProviderError};
use crate::constants::{LLM_PROMPT_BYTES_MAX, LLM_RESPONSE_BYTES_MAX};

// =============================================================================
// Constants
// =============================================================================

/// Default OpenAI API URL
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Default model
const DEFAULT_MODEL: &str = "gpt-4o";

/// Default max tokens
const DEFAULT_MAX_TOKENS: usize = 4096;

// =============================================================================
// API Types
// =============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
    #[serde(default)]
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

// =============================================================================
// OpenAIProvider
// =============================================================================

/// OpenAI API provider.
///
/// TigerStyle: Production provider with explicit configuration.
///
/// Supports:
/// - OpenAI API (default)
/// - Azure OpenAI (via `with_base_url`)
/// - Local models with OpenAI-compatible APIs (Ollama, vLLM, etc.)
///
/// # Example
///
/// ```rust,ignore
/// use umi_memory::llm::{OpenAIProvider, CompletionRequest, LLMProvider};
///
/// #[tokio::main]
/// async fn main() {
///     // Standard OpenAI
///     let provider = OpenAIProvider::new(std::env::var("OPENAI_API_KEY").unwrap())
///         .with_model("gpt-4o");
///
///     // Or local model with OpenAI-compatible API
///     let local = OpenAIProvider::new("not-needed")
///         .with_base_url("http://localhost:11434/v1/chat/completions")
///         .with_model("llama3.2");
///
///     let request = CompletionRequest::new("Hello!");
///     let response = provider.complete(&request).await.unwrap();
///     println!("{}", response);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    /// HTTP client
    client: reqwest::Client,
    /// API key
    api_key: String,
    /// Model to use
    model: String,
    /// API URL (for Azure, local models, proxies)
    api_url: String,
}

impl OpenAIProvider {
    /// Create a new `OpenAIProvider` with the given API key.
    ///
    /// Uses default model (`gpt-4o`).
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            api_url: OPENAI_API_URL.to_string(),
        }
    }

    /// Set the model to use.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = OpenAIProvider::new(api_key)
    ///     .with_model("gpt-4o-mini");
    /// ```
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set a custom base URL.
    ///
    /// Use this for:
    /// - Azure OpenAI
    /// - Local models (Ollama, vLLM)
    /// - Proxies
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Ollama with OpenAI-compatible API
    /// let provider = OpenAIProvider::new("not-needed")
    ///     .with_base_url("http://localhost:11434/v1/chat/completions")
    ///     .with_model("llama3.2");
    /// ```
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Get the current model.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the current API URL.
    #[must_use]
    pub fn api_url(&self) -> &str {
        &self.api_url
    }

    /// Build the request body.
    fn build_request(&self, request: &CompletionRequest) -> OpenAIRequest {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(system) = &request.system {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Add user message
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        });

        OpenAIRequest {
            model: self.model.clone(),
            messages,
            max_tokens: Some(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
            temperature: request.temperature,
            response_format: if request.json_mode {
                Some(ResponseFormat {
                    format_type: "json_object".to_string(),
                })
            } else {
                None
            },
        }
    }

    /// Parse error response.
    fn parse_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
        // Try to parse as OpenAI error
        if let Ok(err) = serde_json::from_str::<OpenAIError>(body) {
            let error_type = err.error.error_type.as_deref().unwrap_or("");
            let code = err.error.code.as_deref().unwrap_or("");

            return match (error_type, code) {
                ("invalid_api_key", _) | (_, "invalid_api_key") => {
                    ProviderError::AuthenticationFailed
                }
                ("rate_limit_exceeded", _) | (_, "rate_limit_exceeded") => {
                    ProviderError::rate_limit(None)
                }
                ("context_length_exceeded", _) | (_, "context_length_exceeded") => {
                    ProviderError::context_overflow(0)
                }
                ("server_error", _) | (_, "server_error") => {
                    ProviderError::service_unavailable("OpenAI server error")
                }
                _ => ProviderError::invalid_response(err.error.message),
            };
        }

        // Fall back to status code
        match status {
            reqwest::StatusCode::UNAUTHORIZED => ProviderError::AuthenticationFailed,
            reqwest::StatusCode::TOO_MANY_REQUESTS => ProviderError::rate_limit(None),
            reqwest::StatusCode::SERVICE_UNAVAILABLE | reqwest::StatusCode::BAD_GATEWAY => {
                ProviderError::service_unavailable("OpenAI API unavailable")
            }
            reqwest::StatusCode::REQUEST_TIMEOUT | reqwest::StatusCode::GATEWAY_TIMEOUT => {
                ProviderError::Timeout
            }
            _ => ProviderError::invalid_response(format!("HTTP {}: {}", status, body)),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
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
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
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

        let parsed: OpenAIResponse = serde_json::from_str(&response_body)
            .map_err(|e| ProviderError::json_error(format!("Failed to parse response: {}", e)))?;

        // Extract text from first choice
        let text = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default();

        // Postcondition
        debug_assert!(
            text.len() <= LLM_RESPONSE_BYTES_MAX,
            "response exceeds limit"
        );

        Ok(text)
    }

    fn name(&self) -> &'static str {
        "openai"
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
        let provider = OpenAIProvider::new("test-key");
        assert_eq!(provider.model(), DEFAULT_MODEL);
        assert_eq!(provider.api_url(), OPENAI_API_URL);
        assert!(!provider.is_simulation());
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_with_model() {
        let provider = OpenAIProvider::new("test-key").with_model("gpt-4o-mini");
        assert_eq!(provider.model(), "gpt-4o-mini");
    }

    #[test]
    fn test_with_base_url() {
        let provider = OpenAIProvider::new("test-key")
            .with_base_url("http://localhost:11434/v1/chat/completions");
        assert_eq!(
            provider.api_url(),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_request_simple() {
        let provider = OpenAIProvider::new("test-key");
        let request = CompletionRequest::new("Hello");

        let built = provider.build_request(&request);

        assert_eq!(built.model, DEFAULT_MODEL);
        assert_eq!(built.messages.len(), 1);
        assert_eq!(built.messages[0].role, "user");
        assert_eq!(built.messages[0].content, "Hello");
        assert!(built.response_format.is_none());
    }

    #[test]
    fn test_build_request_with_system() {
        let provider = OpenAIProvider::new("test-key");
        let request = CompletionRequest::new("Hello").with_system("You are helpful");

        let built = provider.build_request(&request);

        assert_eq!(built.messages.len(), 2);
        assert_eq!(built.messages[0].role, "system");
        assert_eq!(built.messages[0].content, "You are helpful");
        assert_eq!(built.messages[1].role, "user");
        assert_eq!(built.messages[1].content, "Hello");
    }

    #[test]
    fn test_build_request_with_json_mode() {
        let provider = OpenAIProvider::new("test-key");
        let request = CompletionRequest::new("Hello").with_json_mode();

        let built = provider.build_request(&request);

        assert!(built.response_format.is_some());
        assert_eq!(built.response_format.unwrap().format_type, "json_object");
    }

    #[test]
    fn test_build_request_with_options() {
        let provider = OpenAIProvider::new("test-key");
        let request = CompletionRequest::new("Hello")
            .with_max_tokens(100)
            .with_temperature(0.7);

        let built = provider.build_request(&request);

        assert_eq!(built.max_tokens, Some(100));
        assert_eq!(built.temperature, Some(0.7));
    }

    #[test]
    fn test_parse_error_auth() {
        let body = r#"{"error":{"message":"Invalid API key","type":"invalid_api_key","code":"invalid_api_key"}}"#;
        let err = OpenAIProvider::parse_error(reqwest::StatusCode::UNAUTHORIZED, body);
        assert!(matches!(err, ProviderError::AuthenticationFailed));
    }

    #[test]
    fn test_parse_error_rate_limit() {
        let body = r#"{"error":{"message":"Rate limit exceeded","type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}"#;
        let err = OpenAIProvider::parse_error(reqwest::StatusCode::TOO_MANY_REQUESTS, body);
        assert!(matches!(err, ProviderError::RateLimit { .. }));
    }

    #[test]
    fn test_parse_error_context_length() {
        let body = r#"{"error":{"message":"Context length exceeded","type":"context_length_exceeded","code":"context_length_exceeded"}}"#;
        let err = OpenAIProvider::parse_error(reqwest::StatusCode::BAD_REQUEST, body);
        assert!(matches!(err, ProviderError::ContextOverflow { .. }));
    }

    #[test]
    fn test_parse_error_server_error() {
        let body =
            r#"{"error":{"message":"Server error","type":"server_error","code":"server_error"}}"#;
        let err = OpenAIProvider::parse_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, body);
        assert!(matches!(err, ProviderError::ServiceUnavailable { .. }));
    }

    #[test]
    fn test_parse_error_fallback() {
        let body = "Internal server error";
        let err = OpenAIProvider::parse_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, body);
        assert!(matches!(err, ProviderError::InvalidResponse { .. }));
    }
}
