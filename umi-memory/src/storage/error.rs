//! Storage Errors
//!
//! `TigerStyle`: Explicit error types with context.

use thiserror::Error;

/// Errors from storage operations.
#[derive(Debug, Clone, Error)]
pub enum StorageError {
    /// Entity not found
    #[error("entity not found: {id}")]
    NotFound {
        /// Entity ID that was not found
        id: String,
    },

    /// Entity already exists (for insert-only operations)
    #[error("entity already exists: {id}")]
    AlreadyExists {
        /// Entity ID that already exists
        id: String,
    },

    /// Validation error
    #[error("validation error: {message}")]
    Validation {
        /// Validation error message
        message: String,
    },

    /// Connection error
    #[error("connection error: {message}")]
    Connection {
        /// Connection error message
        message: String,
    },

    /// Query error
    #[error("query error: {message}")]
    Query {
        /// Query error message
        message: String,
    },

    /// Timeout error
    #[error("timeout after {duration_ms}ms")]
    Timeout {
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Simulated fault (for DST)
    #[error("simulated fault: {fault_type}")]
    SimulatedFault {
        /// Type of simulated fault
        fault_type: String,
    },

    /// Internal error
    #[error("internal error: {message}")]
    Internal {
        /// Error message
        message: String,
    },

    /// Connection failed
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// Write failed
    #[error("write failed: {0}")]
    WriteFailed(String),

    /// Read failed
    #[error("read failed: {0}")]
    ReadFailed(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("deserialization error: {0}")]
    DeserializationError(String),
}

impl StorageError {
    /// Create a not found error.
    #[must_use]
    pub fn not_found(id: impl Into<String>) -> Self {
        Self::NotFound { id: id.into() }
    }

    /// Create an already exists error.
    #[must_use]
    pub fn already_exists(id: impl Into<String>) -> Self {
        Self::AlreadyExists { id: id.into() }
    }

    /// Create a validation error.
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a connection error.
    #[must_use]
    pub fn connection(message: impl Into<String>) -> Self {
        Self::Connection {
            message: message.into(),
        }
    }

    /// Create a query error.
    #[must_use]
    pub fn query(message: impl Into<String>) -> Self {
        Self::Query {
            message: message.into(),
        }
    }

    /// Create a timeout error.
    #[must_use]
    pub fn timeout(duration_ms: u64) -> Self {
        Self::Timeout { duration_ms }
    }

    /// Create a simulated fault error.
    #[must_use]
    pub fn simulated_fault(fault_type: impl Into<String>) -> Self {
        Self::SimulatedFault {
            fault_type: fault_type.into(),
        }
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a read error (wraps query error for reads).
    #[must_use]
    pub fn read(message: impl Into<String>) -> Self {
        Self::Query {
            message: format!("read: {}", message.into()),
        }
    }

    /// Create a write error (wraps query error for writes).
    #[must_use]
    pub fn write(message: impl Into<String>) -> Self {
        Self::Query {
            message: format!("write: {}", message.into()),
        }
    }

    /// Check if this is a transient error (can be retried).
    #[must_use]
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Connection { .. } | Self::Timeout { .. } | Self::SimulatedFault { .. }
        )
    }
}

/// Result type for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_constructors() {
        let err = StorageError::not_found("test-id");
        assert!(matches!(err, StorageError::NotFound { id } if id == "test-id"));

        let err = StorageError::validation("invalid content");
        assert!(
            matches!(err, StorageError::Validation { message } if message == "invalid content")
        );
    }

    #[test]
    fn test_is_transient() {
        assert!(StorageError::connection("failed").is_transient());
        assert!(StorageError::timeout(1000).is_transient());
        assert!(StorageError::simulated_fault("network").is_transient());

        assert!(!StorageError::not_found("id").is_transient());
        assert!(!StorageError::validation("bad").is_transient());
    }
}
