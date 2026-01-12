//! Embedding Provider Trait - Unified Interface for Text Embeddings
//!
//! `TigerStyle`: Simulation-first embedding generation.
//!
//! See ADR-019 for design rationale.
//!
//! # Architecture
//!
//! ```text
//! EmbeddingProvider (trait)
//! ├── SimEmbeddingProvider    (always available, deterministic)
//! └── OpenAIEmbeddingProvider (feature: embedding-openai)
//! ```
//!
//! # Usage
//!
//! ```rust
//! use umi_memory::embedding::{EmbeddingProvider, SimEmbeddingProvider};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Simulation (always available, deterministic)
//!     let provider = SimEmbeddingProvider::with_seed(42);
//!
//!     let embedding = provider.embed("Alice works at Acme").await.unwrap();
//!     println!("Generated {} dimensional embedding", embedding.len());
//! }
//! ```

mod sim;

#[cfg(feature = "embedding-openai")]
mod openai;

pub use sim::SimEmbeddingProvider;

#[cfg(feature = "embedding-openai")]
pub use openai::OpenAIEmbeddingProvider;

use async_trait::async_trait;

// =============================================================================
// Error Types
// =============================================================================

/// Unified error type for all embedding providers.
///
/// `TigerStyle`: Explicit variants for all failure modes.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EmbeddingError {
    /// Request timed out
    #[error("Request timed out")]
    Timeout,

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after_secs:?}s")]
    RateLimit {
        /// Seconds until rate limit resets (if known)
        retry_after_secs: Option<u64>,
    },

    /// Input text too long
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

    /// Empty input provided
    #[error("Empty input provided")]
    EmptyInput,

    /// Dimension mismatch in returned embedding
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensions
        expected: usize,
        /// Actual dimensions received
        actual: usize,
    },
}

impl EmbeddingError {
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

    /// Create a dimension mismatch error.
    #[must_use]
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
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
// Provider Trait
// =============================================================================

/// Trait for embedding providers.
///
/// TigerStyle: Unified interface for simulation and production.
///
/// All providers implement this trait, allowing higher-level components
/// to work with any provider without knowing the concrete type.
///
/// # Example
///
/// ```rust
/// use umi_memory::embedding::{EmbeddingProvider, SimEmbeddingProvider};
///
/// async fn generate_embedding<P: EmbeddingProvider>(provider: &P, text: &str) -> Vec<f32> {
///     provider.embed(text).await.unwrap()
/// }
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text.
    ///
    /// # Arguments
    /// * `text` - The text to embed
    ///
    /// # Returns
    /// Vector of floats representing the embedding (normalized to unit vector)
    ///
    /// # Errors
    /// Returns `EmbeddingError` on failure (rate limit, network error, etc.)
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for multiple texts (batch).
    ///
    /// This is more efficient than calling `embed` multiple times as it can
    /// leverage API batching capabilities.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to embed
    ///
    /// # Returns
    /// Vector of embeddings, one per input text
    ///
    /// # Errors
    /// Returns `EmbeddingError` on failure
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Get the embedding dimensions (e.g., 1536 for text-embedding-3-small).
    fn dimensions(&self) -> usize;

    /// Provider name for logging and debugging.
    fn name(&self) -> &'static str;

    /// Check if this is a simulation provider.
    ///
    /// Returns `true` for `SimEmbeddingProvider`, `false` for real providers.
    fn is_simulation(&self) -> bool;
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Validate that an embedding has the expected dimensions.
///
/// # Arguments
/// * `embedding` - The embedding to validate
/// * `expected` - Expected number of dimensions
///
/// # Errors
/// Returns `EmbeddingError::DimensionMismatch` if dimensions don't match
pub fn validate_dimensions(embedding: &[f32], expected: usize) -> Result<(), EmbeddingError> {
    if embedding.len() != expected {
        return Err(EmbeddingError::dimension_mismatch(
            expected,
            embedding.len(),
        ));
    }
    Ok(())
}

/// Normalize a vector to unit length (L2 norm = 1).
///
/// This ensures cosine similarity can be computed efficiently.
///
/// # Arguments
/// * `vec` - The vector to normalize
///
/// # Panics
/// Panics if the input vector is all zeros (cannot normalize)
#[must_use]
pub fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(norm > 0.0, "Cannot normalize zero vector");

    vec.iter().map(|x| x / norm).collect()
}

/// Check if a vector is normalized (L2 norm ≈ 1.0).
///
/// # Arguments
/// * `vec` - The vector to check
/// * `tolerance` - Acceptable deviation from 1.0 (default: 0.001)
#[must_use]
pub fn is_normalized(vec: &[f32], tolerance: f32) -> bool {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm - 1.0).abs() < tolerance
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EMBEDDING_DIMENSIONS_COUNT;

    #[test]
    fn test_embedding_error_constructors() {
        let err = EmbeddingError::timeout();
        assert!(matches!(err, EmbeddingError::Timeout));

        let err = EmbeddingError::rate_limit(Some(60));
        assert!(matches!(
            err,
            EmbeddingError::RateLimit {
                retry_after_secs: Some(60)
            }
        ));

        let err = EmbeddingError::context_overflow(10000);
        assert!(matches!(
            err,
            EmbeddingError::ContextOverflow { tokens: 10000 }
        ));

        let err = EmbeddingError::invalid_response("bad format");
        assert!(matches!(err, EmbeddingError::InvalidResponse { .. }));

        let err = EmbeddingError::dimension_mismatch(1536, 768);
        assert!(matches!(
            err,
            EmbeddingError::DimensionMismatch {
                expected: 1536,
                actual: 768
            }
        ));
    }

    #[test]
    fn test_embedding_error_is_retryable() {
        assert!(EmbeddingError::timeout().is_retryable());
        assert!(EmbeddingError::rate_limit(Some(60)).is_retryable());
        assert!(EmbeddingError::service_unavailable("down").is_retryable());

        assert!(!EmbeddingError::AuthenticationFailed.is_retryable());
        assert!(!EmbeddingError::EmptyInput.is_retryable());
        assert!(!EmbeddingError::json_error("parse failed").is_retryable());
    }

    #[test]
    fn test_validate_dimensions() {
        let embedding = vec![0.1; EMBEDDING_DIMENSIONS_COUNT];
        assert!(validate_dimensions(&embedding, EMBEDDING_DIMENSIONS_COUNT).is_ok());

        let wrong_size = vec![0.1; 768];
        assert!(validate_dimensions(&wrong_size, EMBEDDING_DIMENSIONS_COUNT).is_err());
    }

    #[test]
    fn test_normalize_vector() {
        let vec = vec![3.0, 4.0]; // Length = 5
        let normalized = normalize_vector(&vec);

        // Should be [0.6, 0.8]
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);

        // Verify unit length
        assert!(is_normalized(&normalized, 0.001));
    }

    #[test]
    fn test_is_normalized() {
        let unit = vec![1.0, 0.0, 0.0];
        assert!(is_normalized(&unit, 0.001));

        let not_unit = vec![2.0, 0.0, 0.0];
        assert!(!is_normalized(&not_unit, 0.001));

        let normalized = vec![0.6, 0.8]; // 3-4-5 triangle
        assert!(is_normalized(&normalized, 0.001));
    }

    #[test]
    #[should_panic(expected = "Cannot normalize zero vector")]
    fn test_normalize_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let _ = normalize_vector(&zero);
    }
}
