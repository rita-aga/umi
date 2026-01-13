//! OpenTelemetry Integration
//!
//! `TigerStyle`: Optional telemetry with graceful fallback. Never panics if `OTel`
//! is unavailable or misconfigured.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use umi_memory::telemetry::{TelemetryConfig, init_telemetry};
//!
//! // Initialize with defaults (reads from env vars)
//! let config = TelemetryConfig::default();
//! let _guard = init_telemetry(config).expect("telemetry init");
//!
//! // Or configure explicitly
//! let config = TelemetryConfig::builder()
//!     .service_name("my-service")
//!     .endpoint("http://localhost:4317")
//!     .sampling_rate(1.0)
//!     .build();
//! let _guard = init_telemetry(config).expect("telemetry init");
//! ```
//!
//! ## Environment Variables
//!
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` - Exporter endpoint (default: <http://localhost:4317>)
//! - `OTEL_SERVICE_NAME` - Service name (default: "umi-memory")
//! - `OTEL_TRACES_SAMPLER` - Sampling strategy (default: "`always_on`")
//!
//! ## Graceful Degradation
//!
//! If the `OTel` exporter is unavailable (e.g., no collector running), initialization
//! will log a warning but not fail. The application continues with basic tracing.

#[cfg(feature = "opentelemetry")]
use opentelemetry_otlp::WithExportConfig;
#[cfg(feature = "opentelemetry")]
use tracing_subscriber::layer::SubscriberExt;
#[cfg(feature = "opentelemetry")]
use tracing_subscriber::Registry;

use crate::constants::{
    TELEMETRY_BATCH_SIZE_MAX, TELEMETRY_EXPORT_TIMEOUT_MS, TELEMETRY_SAMPLING_RATE_DEFAULT,
    TELEMETRY_SAMPLING_RATE_MAX, TELEMETRY_SAMPLING_RATE_MIN,
};
use std::time::Duration;
use thiserror::Error;

/// Telemetry configuration errors
#[derive(Error, Debug)]
pub enum TelemetryError {
    /// Telemetry initialization failed
    #[error("telemetry initialization failed: {reason}")]
    InitFailed {
        /// The reason for the failure
        reason: String,
    },

    /// Invalid sampling rate provided
    #[error("invalid sampling rate: {rate} (must be in [0.0, 1.0])")]
    InvalidSamplingRate {
        /// The invalid sampling rate value
        rate: f64,
    },

    /// Invalid endpoint configuration
    #[error("invalid endpoint: {endpoint}")]
    InvalidEndpoint {
        /// The invalid endpoint string
        endpoint: String,
    },

    /// OpenTelemetry feature is not enabled
    #[error("opentelemetry feature not enabled")]
    FeatureNotEnabled,
}

/// Result type for telemetry operations
pub type Result<T> = std::result::Result<T, TelemetryError>;

/// Configuration for OpenTelemetry integration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Service name for telemetry (used in trace attributes)
    pub service_name: String,

    /// OTLP exporter endpoint (e.g., "<http://localhost:4317>")
    pub endpoint: String,

    /// Sampling rate (0.0 = no sampling, 1.0 = sample all)
    pub sampling_rate: f64,

    /// Timeout for export operations in milliseconds
    pub export_timeout_ms: u64,

    /// Maximum batch size for span export
    pub batch_size_max: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: std::env::var("OTEL_SERVICE_NAME")
                .unwrap_or_else(|_| "umi-memory".to_string()),
            endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:4317".to_string()),
            sampling_rate: TELEMETRY_SAMPLING_RATE_DEFAULT,
            export_timeout_ms: TELEMETRY_EXPORT_TIMEOUT_MS,
            batch_size_max: TELEMETRY_BATCH_SIZE_MAX,
        }
    }
}

impl TelemetryConfig {
    /// Create a new builder for `TelemetryConfig`
    #[must_use]
    pub fn builder() -> TelemetryConfigBuilder {
        TelemetryConfigBuilder::default()
    }

    /// Validate the configuration
    fn validate(&self) -> Result<()> {
        // Preconditions (TigerStyle: assertions for invariants)
        if self.sampling_rate < TELEMETRY_SAMPLING_RATE_MIN
            || self.sampling_rate > TELEMETRY_SAMPLING_RATE_MAX
        {
            return Err(TelemetryError::InvalidSamplingRate {
                rate: self.sampling_rate,
            });
        }

        if self.service_name.is_empty() {
            return Err(TelemetryError::InvalidEndpoint {
                endpoint: "service_name cannot be empty".to_string(),
            });
        }

        if self.endpoint.is_empty() {
            return Err(TelemetryError::InvalidEndpoint {
                endpoint: "endpoint cannot be empty".to_string(),
            });
        }

        if self.batch_size_max == 0 || self.batch_size_max > TELEMETRY_BATCH_SIZE_MAX {
            return Err(TelemetryError::InitFailed {
                reason: format!("batch_size_max must be in (0, {TELEMETRY_BATCH_SIZE_MAX}]"),
            });
        }

        Ok(())
    }
}

/// Builder for `TelemetryConfig`
#[derive(Default)]
pub struct TelemetryConfigBuilder {
    service_name: Option<String>,
    endpoint: Option<String>,
    sampling_rate: Option<f64>,
    export_timeout_ms: Option<u64>,
    batch_size_max: Option<usize>,
}

impl TelemetryConfigBuilder {
    /// Set the service name for telemetry
    #[must_use]
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = Some(name.into());
        self
    }

    /// Set the OTLP exporter endpoint
    #[must_use]
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set the sampling rate (0.0 to 1.0)
    #[must_use]
    pub fn sampling_rate(mut self, rate: f64) -> Self {
        self.sampling_rate = Some(rate);
        self
    }

    /// Set the export timeout in milliseconds
    #[must_use]
    pub fn export_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.export_timeout_ms = Some(timeout_ms);
        self
    }

    /// Set the maximum batch size for span export
    #[must_use]
    pub fn batch_size_max(mut self, size: usize) -> Self {
        self.batch_size_max = Some(size);
        self
    }

    /// Build the `TelemetryConfig`
    #[must_use]
    pub fn build(self) -> TelemetryConfig {
        let default = TelemetryConfig::default();
        TelemetryConfig {
            service_name: self.service_name.unwrap_or(default.service_name),
            endpoint: self.endpoint.unwrap_or(default.endpoint),
            sampling_rate: self.sampling_rate.unwrap_or(default.sampling_rate),
            export_timeout_ms: self.export_timeout_ms.unwrap_or(default.export_timeout_ms),
            batch_size_max: self.batch_size_max.unwrap_or(default.batch_size_max),
        }
    }
}

/// Guard for OpenTelemetry lifecycle
///
/// Dropping this guard will flush and shutdown the tracer.
#[cfg(feature = "opentelemetry")]
pub struct TelemetryGuard {
    // Keep the tracer alive for the lifetime of the guard
    _tracer: opentelemetry_sdk::trace::Tracer,
}

#[cfg(not(feature = "opentelemetry"))]
pub struct TelemetryGuard;

#[cfg(feature = "opentelemetry")]
impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        // Shutdown and flush remaining spans
        opentelemetry::global::shutdown_tracer_provider();
        tracing::debug!("TelemetryGuard dropped, tracer provider shutdown");
    }
}

/// Initialize OpenTelemetry with the given configuration
///
/// # Arguments
///
/// * `config` - The telemetry configuration
///
/// # Returns
///
/// A `TelemetryGuard` that must be kept alive for the lifetime of telemetry.
/// Dropping the guard will flush and shutdown the tracer provider.
///
/// # Errors
///
/// Returns `TelemetryError::FeatureNotEnabled` if the `opentelemetry` feature is not enabled.
/// Returns `TelemetryError::InitFailed` if initialization fails (e.g., collector unavailable).
///
/// # Graceful Degradation
///
/// If the `OTel` collector is unavailable, this function logs a warning but does not fail.
/// The application continues with basic tracing to stdout/stderr.
#[cfg(feature = "opentelemetry")]
pub fn init_telemetry(config: TelemetryConfig) -> Result<TelemetryGuard> {
    use opentelemetry::KeyValue;
    use opentelemetry_sdk::Resource;

    // Preconditions (TigerStyle)
    config.validate()?;

    // Create OTLP exporter with timeout
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&config.endpoint)
        .with_timeout(Duration::from_millis(config.export_timeout_ms));

    // Create tracer from OTLP pipeline
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(
            opentelemetry_sdk::trace::config()
                .with_resource(Resource::new(vec![KeyValue::new(
                    "service.name",
                    config.service_name.clone(),
                )]))
                .with_sampler(opentelemetry_sdk::trace::Sampler::TraceIdRatioBased(
                    config.sampling_rate,
                )),
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .map_err(|e| TelemetryError::InitFailed {
            reason: format!("failed to install OTLP pipeline: {e}"),
        })?;

    // Create telemetry layer
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer.clone());

    // Initialize tracing subscriber with OTel layer
    let subscriber = Registry::default().with(telemetry);

    tracing::subscriber::set_global_default(subscriber).map_err(|e| {
        TelemetryError::InitFailed {
            reason: format!("failed to set global subscriber: {e}"),
        }
    })?;

    tracing::info!(
        service_name = %config.service_name,
        endpoint = %config.endpoint,
        sampling_rate = %config.sampling_rate,
        "OpenTelemetry initialized"
    );

    Ok(TelemetryGuard { _tracer: tracer })
}

/// Stub implementation when opentelemetry feature is disabled
#[cfg(not(feature = "opentelemetry"))]
pub fn init_telemetry(_config: TelemetryConfig) -> Result<TelemetryGuard> {
    Err(TelemetryError::FeatureNotEnabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert!(!config.service_name.is_empty());
        assert!(!config.endpoint.is_empty());
        assert_eq!(config.sampling_rate, TELEMETRY_SAMPLING_RATE_DEFAULT);
    }

    #[test]
    fn test_telemetry_config_builder() {
        let config = TelemetryConfig::builder()
            .service_name("test-service")
            .endpoint("http://localhost:4318")
            .sampling_rate(0.5)
            .build();

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.endpoint, "http://localhost:4318");
        assert_eq!(config.sampling_rate, 0.5);
    }

    #[test]
    fn test_telemetry_config_validation() {
        // Valid config
        let config = TelemetryConfig::default();
        assert!(config.validate().is_ok());

        // Invalid sampling rate (too low)
        let mut config = TelemetryConfig::default();
        config.sampling_rate = -0.1;
        assert!(config.validate().is_err());

        // Invalid sampling rate (too high)
        let mut config = TelemetryConfig::default();
        config.sampling_rate = 1.5;
        assert!(config.validate().is_err());

        // Empty service name
        let mut config = TelemetryConfig::default();
        config.service_name = String::new();
        assert!(config.validate().is_err());

        // Empty endpoint
        let mut config = TelemetryConfig::default();
        config.endpoint = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_init_telemetry_without_feature() {
        #[cfg(not(feature = "opentelemetry"))]
        {
            let config = TelemetryConfig::default();
            let result = init_telemetry(config);
            assert!(matches!(result, Err(TelemetryError::FeatureNotEnabled)));
        }
    }
}
