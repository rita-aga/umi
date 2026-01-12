//! OpenAIEmbeddingProvider - OpenAI Embeddings API Integration
//!
//! TigerStyle: Production embedding provider, feature-gated.
//!
//! Requires `embedding-openai` feature flag:
//! ```toml
//! umi-memory = { version = "0.1", features = ["embedding-openai"] }
//! ```
//!
//! Uses OpenAI's text-embedding-3-small by default (1536 dimensions).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{normalize_vector, validate_dimensions, EmbeddingError, EmbeddingProvider};
use crate::constants::{EMBEDDING_BATCH_SIZE_MAX, EMBEDDING_DIMENSIONS_COUNT};

// =============================================================================
// Constants
// =============================================================================

/// Default OpenAI Embeddings API URL
const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";

/// Default model (1536 dimensions)
const DEFAULT_MODEL: &str = "text-embedding-3-small";

/// Request timeout in seconds
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Maximum retry attempts
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Base delay for exponential backoff (milliseconds)
const RETRY_DELAY_MS_BASE: u64 = 100;

// =============================================================================
// API Types
// =============================================================================

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: EmbeddingInput,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
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
// OpenAIEmbeddingProvider
// =============================================================================

/// OpenAI embeddings API provider.
///
/// TigerStyle: Production provider with explicit configuration.
///
/// Supports:
/// - text-embedding-3-small (1536 dims, default)
/// - text-embedding-3-large (3072 dims)
/// - text-embedding-ada-002 (1536 dims, legacy)
///
/// # Example
///
/// ```rust,ignore
/// use umi_memory::embedding::{OpenAIEmbeddingProvider, EmbeddingProvider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = OpenAIEmbeddingProvider::new(
///         std::env::var("OPENAI_API_KEY").unwrap()
///     );
///
///     let embedding = provider.embed("Alice works at Acme").await.unwrap();
///     println!("Generated {} dimensional embedding", embedding.len());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIEmbeddingProvider {
    /// HTTP client
    client: reqwest::Client,
    /// API key
    api_key: String,
    /// Model to use
    model: String,
    /// API URL (for proxies or compatible APIs)
    api_url: String,
    /// Embedding dimensions
    dimensions: usize,
}

impl OpenAIEmbeddingProvider {
    /// Create a new `OpenAIEmbeddingProvider` with the given API key.
    ///
    /// Uses default model (`text-embedding-3-small`, 1536 dimensions).
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
                .build()
                .unwrap(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            api_url: OPENAI_EMBEDDINGS_URL.to_string(),
            dimensions: EMBEDDING_DIMENSIONS_COUNT,
        }
    }

    /// Set the model to use.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let provider = OpenAIEmbeddingProvider::new(api_key)
    ///     .with_model("text-embedding-3-large", 3072);
    /// ```
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>, dimensions: usize) -> Self {
        self.model = model.into();
        self.dimensions = dimensions;
        self
    }

    /// Set a custom base URL for OpenAI-compatible APIs.
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

    /// Build the request body for single text.
    fn build_request_single(&self, text: &str) -> EmbeddingRequest {
        EmbeddingRequest {
            input: EmbeddingInput::Single(text.to_string()),
            model: self.model.clone(),
            dimensions: Some(self.dimensions),
        }
    }

    /// Build the request body for batch.
    fn build_request_batch(&self, texts: &[&str]) -> EmbeddingRequest {
        EmbeddingRequest {
            input: EmbeddingInput::Batch(texts.iter().map(|s| (*s).to_string()).collect()),
            model: self.model.clone(),
            dimensions: Some(self.dimensions),
        }
    }

    /// Parse error response.
    fn parse_error(status: reqwest::StatusCode, body: &str) -> EmbeddingError {
        // Try to parse as OpenAI error
        if let Ok(err) = serde_json::from_str::<OpenAIError>(body) {
            let error_type = err.error.error_type.as_deref().unwrap_or("");
            let code = err.error.code.as_deref().unwrap_or("");

            return match (error_type, code) {
                ("invalid_api_key", _) | (_, "invalid_api_key") => {
                    EmbeddingError::AuthenticationFailed
                }
                ("rate_limit_exceeded", _) | (_, "rate_limit_exceeded") => {
                    EmbeddingError::rate_limit(None)
                }
                ("context_length_exceeded", _) | (_, "context_length_exceeded") => {
                    EmbeddingError::context_overflow(0)
                }
                ("server_error", _) | (_, "server_error") => {
                    EmbeddingError::service_unavailable("OpenAI server error")
                }
                _ => EmbeddingError::invalid_response(err.error.message),
            };
        }

        // Fall back to status code
        match status {
            reqwest::StatusCode::UNAUTHORIZED => EmbeddingError::AuthenticationFailed,
            reqwest::StatusCode::TOO_MANY_REQUESTS => EmbeddingError::rate_limit(None),
            reqwest::StatusCode::SERVICE_UNAVAILABLE | reqwest::StatusCode::BAD_GATEWAY => {
                EmbeddingError::service_unavailable("OpenAI API unavailable")
            }
            reqwest::StatusCode::REQUEST_TIMEOUT | reqwest::StatusCode::GATEWAY_TIMEOUT => {
                EmbeddingError::Timeout
            }
            _ => EmbeddingError::invalid_response(format!("HTTP {}: {}", status, body)),
        }
    }

    /// Make API request with retry logic.
    async fn make_request(
        &self,
        body: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let mut attempt = 0;
        let mut delay_ms = RETRY_DELAY_MS_BASE;

        loop {
            attempt += 1;

            let response = self
                .client
                .post(&self.api_url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| EmbeddingError::network_error(e.to_string()))?;

            let status = response.status();

            if status.is_success() {
                let response_body = response
                    .json::<EmbeddingResponse>()
                    .await
                    .map_err(|e| EmbeddingError::json_error(e.to_string()))?;
                return Ok(response_body);
            }

            // Get error body
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());

            let error = Self::parse_error(status, &error_body);

            // Retry if error is retryable and we haven't exceeded max attempts
            if error.is_retryable() && attempt < MAX_RETRY_ATTEMPTS {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                delay_ms *= 2; // Exponential backoff
                continue;
            }

            return Err(error);
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Precondition: text must not be empty
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let body = self.build_request_single(text);
        let response = self.make_request(&body).await?;

        // Extract embedding
        if response.data.is_empty() {
            return Err(EmbeddingError::invalid_response("No embeddings returned"));
        }

        let embedding = response.data[0].embedding.clone();

        // Validate dimensions
        validate_dimensions(&embedding, self.dimensions)?;

        // Ensure normalized (OpenAI should return normalized, but verify)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 0.01 {
            // Not normalized, normalize it
            Ok(normalize_vector(&embedding))
        } else {
            Ok(embedding)
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Preconditions
        if texts.is_empty() {
            return Err(EmbeddingError::invalid_request("batch cannot be empty"));
        }

        if texts.len() > EMBEDDING_BATCH_SIZE_MAX {
            return Err(EmbeddingError::invalid_request(format!(
                "batch size {} exceeds maximum {}",
                texts.len(),
                EMBEDDING_BATCH_SIZE_MAX
            )));
        }

        // Check for empty texts
        for text in texts {
            if text.is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }
        }

        let body = self.build_request_batch(texts);
        let response = self.make_request(&body).await?;

        // Validate response
        if response.data.len() != texts.len() {
            return Err(EmbeddingError::invalid_response(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                response.data.len()
            )));
        }

        // Sort by index to maintain order
        let mut sorted_data = response.data;
        sorted_data.sort_by_key(|d| d.index);

        // Extract embeddings
        let mut embeddings = Vec::with_capacity(sorted_data.len());
        for data in sorted_data {
            validate_dimensions(&data.embedding, self.dimensions)?;

            // Ensure normalized
            let norm: f32 = data.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if (norm - 1.0).abs() > 0.01 {
                embeddings.push(normalize_vector(&data.embedding));
            } else {
                embeddings.push(data.embedding);
            }
        }

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &'static str {
        "openai-embedding"
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
    fn test_openai_provider_new() {
        let provider = OpenAIEmbeddingProvider::new("test-key");

        assert_eq!(provider.model(), DEFAULT_MODEL);
        assert_eq!(provider.api_url(), OPENAI_EMBEDDINGS_URL);
        assert_eq!(provider.dimensions(), EMBEDDING_DIMENSIONS_COUNT);
        assert!(!provider.is_simulation());
    }

    #[test]
    fn test_openai_provider_with_model() {
        let provider = OpenAIEmbeddingProvider::new("test-key")
            .with_model("text-embedding-3-large", 3072);

        assert_eq!(provider.model(), "text-embedding-3-large");
        assert_eq!(provider.dimensions(), 3072);
    }

    #[test]
    fn test_openai_provider_with_base_url() {
        let provider = OpenAIEmbeddingProvider::new("test-key")
            .with_base_url("http://localhost:8080/v1/embeddings");

        assert_eq!(provider.api_url(), "http://localhost:8080/v1/embeddings");
    }

    #[test]
    fn test_build_request_single() {
        let provider = OpenAIEmbeddingProvider::new("test-key");
        let request = provider.build_request_single("test text");

        assert_eq!(request.model, DEFAULT_MODEL);
        assert!(matches!(request.input, EmbeddingInput::Single(_)));
    }

    #[test]
    fn test_build_request_batch() {
        let provider = OpenAIEmbeddingProvider::new("test-key");
        let texts = vec!["text1", "text2"];
        let request = provider.build_request_batch(&texts);

        assert_eq!(request.model, DEFAULT_MODEL);
        assert!(matches!(request.input, EmbeddingInput::Batch(_)));
    }

    #[test]
    fn test_parse_error_unauthorized() {
        let error = OpenAIEmbeddingProvider::parse_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error": {"message": "Invalid API key", "type": "invalid_api_key"}}"#,
        );

        assert!(matches!(error, EmbeddingError::AuthenticationFailed));
    }

    #[test]
    fn test_parse_error_rate_limit() {
        let error = OpenAIEmbeddingProvider::parse_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error": {"message": "Rate limit", "type": "rate_limit_exceeded"}}"#,
        );

        assert!(matches!(error, EmbeddingError::RateLimit { .. }));
        assert!(error.is_retryable());
    }

    #[test]
    fn test_parse_error_timeout() {
        let error = OpenAIEmbeddingProvider::parse_error(
            reqwest::StatusCode::REQUEST_TIMEOUT,
            "Timeout",
        );

        assert!(matches!(error, EmbeddingError::Timeout));
        assert!(error.is_retryable());
    }

    // Integration tests (require OPENAI_API_KEY environment variable)
    #[tokio::test]
    #[ignore] // Run with: cargo test --features embedding-openai -- --ignored
    async fn test_openai_embed_integration() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let provider = OpenAIEmbeddingProvider::new(api_key);

        let embedding = provider.embed("test text").await.unwrap();

        assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS_COUNT);

        // Verify normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "embedding should be normalized");
    }

    #[tokio::test]
    #[ignore]
    async fn test_openai_embed_batch_integration() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
        let provider = OpenAIEmbeddingProvider::new(api_key);

        let texts = vec!["text1", "text2", "text3"];
        let embeddings = provider.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS_COUNT);
        }
    }

    #[tokio::test]
    async fn test_openai_embed_empty_text() {
        let provider = OpenAIEmbeddingProvider::new("test-key");
        let result = provider.embed("").await;

        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[tokio::test]
    async fn test_openai_embed_batch_empty() {
        let provider = OpenAIEmbeddingProvider::new("test-key");
        let texts: Vec<&str> = vec![];
        let result = provider.embed_batch(&texts).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_openai_embed_batch_exceeds_limit() {
        let provider = OpenAIEmbeddingProvider::new("test-key");
        let texts: Vec<&str> = vec!["text"; EMBEDDING_BATCH_SIZE_MAX + 1];
        let result = provider.embed_batch(&texts).await;

        assert!(result.is_err());
    }
}
