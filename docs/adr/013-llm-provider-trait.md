# ADR-013: LLM Provider Trait - Unified Interface for Sim and Production

## Status

Accepted

## Context

With SimLLM implemented in the DST harness (ADR-012), we need a unified interface
that allows higher-level components (EntityExtractor, DualRetriever, EvolutionTracker)
to work with both simulated and real LLM providers.

### Requirements

1. **Simulation-first**: SimLLMProvider wraps DST SimLLM as the primary implementation
2. **Unified interface**: Same trait for sim, Anthropic, and OpenAI providers
3. **Feature-gated**: Real providers are optional (`anthropic`, `openai` features)
4. **Async**: All operations are async for network I/O compatibility
5. **Error handling**: Unified error type across all providers
6. **TigerStyle**: Preconditions, postconditions, explicit limits

### Existing Infrastructure

- `SimLLM` in `dst/llm.rs` - Deterministic LLM simulation
- `LLMError` - Error types for LLM operations
- `FaultInjector` - Fault injection for testing

## Decision

Create an `llm` module with a unified `LLMProvider` trait and multiple implementations.

### Architecture

```
umi-core/src/llm/
├── mod.rs           # Trait + re-exports
├── sim.rs           # SimLLMProvider (wraps DST SimLLM)
├── anthropic.rs     # AnthropicProvider (feature-gated)
└── openai.rs        # OpenAIProvider (feature-gated)
```

### LLMProvider Trait

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Complete a prompt with a text response.
    async fn complete(&self, request: &CompletionRequest) -> Result<String, ProviderError>;

    /// Complete a prompt expecting a JSON response.
    async fn complete_json<T: DeserializeOwned + Send>(
        &self,
        request: &CompletionRequest,
    ) -> Result<T, ProviderError>;

    /// Get the provider name for logging/debugging.
    fn name(&self) -> &'static str;

    /// Check if this is a simulation provider.
    fn is_simulation(&self) -> bool;
}
```

### CompletionRequest

```rust
pub struct CompletionRequest {
    /// The prompt text
    pub prompt: String,
    /// Optional system message
    pub system: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Temperature (0.0-1.0)
    pub temperature: Option<f32>,
}
```

### ProviderError

Unified error type that maps from provider-specific errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("Request timed out")]
    Timeout,

    #[error("Rate limit exceeded, retry after {retry_after_secs:?}s")]
    RateLimit { retry_after_secs: Option<u64> },

    #[error("Context length exceeded: {tokens} tokens")]
    ContextOverflow { tokens: usize },

    #[error("Invalid response: {message}")]
    InvalidResponse { message: String },

    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("JSON parse error: {message}")]
    JsonError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },
}
```

### SimLLMProvider

Wraps the DST SimLLM, providing the primary simulation implementation:

```rust
pub struct SimLLMProvider {
    inner: SimLLM,
}

impl SimLLMProvider {
    /// Create from existing SimLLM (typically from SimEnvironment).
    pub fn from_sim_llm(sim_llm: SimLLM) -> Self;

    /// Create a new standalone SimLLMProvider with given seed.
    pub fn with_seed(seed: u64) -> Self;
}
```

### AnthropicProvider (feature-gated)

```rust
#[cfg(feature = "anthropic")]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self;
    pub fn with_model(self, model: impl Into<String>) -> Self;
}
```

### OpenAIProvider (feature-gated)

```rust
#[cfg(feature = "openai")]
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: Option<String>,  // For OpenAI-compatible APIs
}

impl OpenAIProvider {
    pub fn new(api_key: impl Into<String>) -> Self;
    pub fn with_model(self, model: impl Into<String>) -> Self;
    pub fn with_base_url(self, url: impl Into<String>) -> Self;  // For local/proxy
}
```

### Feature Flags

```toml
[features]
default = []
anthropic = ["dep:reqwest"]
openai = ["dep:reqwest"]
```

### Usage Pattern

```rust
// Simulation (no external dependencies)
let provider = SimLLMProvider::with_seed(42);

// Production with Anthropic
#[cfg(feature = "anthropic")]
let provider = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?);

// Production with OpenAI
#[cfg(feature = "openai")]
let provider = OpenAIProvider::new(std::env::var("OPENAI_API_KEY")?);

// All use the same interface
let response = provider.complete(&request).await?;
```

### Integration with Higher Components

```rust
pub struct EntityExtractor<P: LLMProvider> {
    provider: P,
}

impl<P: LLMProvider> EntityExtractor<P> {
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    pub async fn extract(&self, text: &str) -> Result<ExtractionResult, ExtractionError> {
        let request = CompletionRequest::new(self.build_prompt(text));
        let result: ExtractionResponse = self.provider.complete_json(&request).await?;
        // ...
    }
}

// In simulation tests:
let extractor = EntityExtractor::new(SimLLMProvider::with_seed(42));

// In production:
let extractor = EntityExtractor::new(AnthropicProvider::new(api_key));
```

## Consequences

### Positive

- **Simulation-first**: SimLLMProvider is always available, no external deps
- **Unified interface**: Higher components work with any provider
- **Feature isolation**: Real providers don't add dependencies unless needed
- **Testable**: All components can be tested with deterministic simulation
- **Extensible**: Easy to add new providers (Gemini, local models, etc.)

### Negative

- **Abstraction overhead**: Trait object or generic bounds add complexity
- **Feature matrix**: Must test with different feature combinations

### Mitigations

1. **Generics over trait objects**: Use `impl LLMProvider` for zero-cost abstraction
2. **CI matrix**: Test `--no-default-features`, `--features anthropic`, `--features openai`

## Implementation

### Files to Create

```
umi-core/src/llm/
├── mod.rs           # Trait, CompletionRequest, ProviderError, re-exports
├── sim.rs           # SimLLMProvider
├── anthropic.rs     # AnthropicProvider (feature-gated)
└── openai.rs        # OpenAIProvider (feature-gated)
```

### Cargo.toml Updates

```toml
[dependencies]
reqwest = { version = "0.12", features = ["json"], optional = true }

[features]
default = []
anthropic = ["dep:reqwest"]
openai = ["dep:reqwest"]
```

### Test Coverage

1. **SimLLMProvider**: Determinism, prompt handling, error mapping
2. **AnthropicProvider**: Request formatting, response parsing, error handling
3. **OpenAIProvider**: Request formatting, response parsing, base URL override
4. **Integration**: Higher components work with all providers

## References

- ADR-012: SimLLM (DST implementation)
- Python `umi/providers/` - Original provider implementations
- Anthropic API docs: https://docs.anthropic.com/
- OpenAI API docs: https://platform.openai.com/docs/
