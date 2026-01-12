# 010: OpenTelemetry Integration

**Created:** 2026-01-12 16:34:00
**Status:** Complete
**Owner:** Claude Sonnet 4.5

## Goal

Add OpenTelemetry support to UMI for distributed tracing, metrics, and telemetry export. This will enable production observability while maintaining DST compatibility.

## Context

Currently UMI only has basic `tracing` + `tracing-subscriber` for console logging. No distributed tracing, no metrics export, no OpenTelemetry integration exists.

**Design Principles:**
- Make it an optional feature (don't force OTel dependency)
- Maintain DST compatibility (SimClock, deterministic behavior)
- Follow TigerStyle (constants with units, assertions, no unwrap)
- Support OTLP export (industry standard)

## Plan

### Phase 1: Dependencies & Feature Flag ✅
- [x] Add opentelemetry dependencies to workspace Cargo.toml
- [x] Add tracing-opentelemetry dependency
- [x] Add opentelemetry-otlp for exporters
- [x] Create optional feature flag in umi-memory/Cargo.toml

### Phase 2: Core Integration ✅
- [x] Create `src/telemetry/` module
- [x] Add TelemetryConfig with constants (timeouts, batch sizes)
- [x] Implement initialization function with graceful fallback
- [x] Add OTel layer to tracing-subscriber
- [x] Support OTLP exporters (gRPC + HTTP)
- [x] Create example demonstrating usage

### Phase 3: Instrumentation ✅
- [x] Add #[tracing::instrument] to Memory::remember()
- [x] Add #[tracing::instrument] to Memory::recall()
- [x] Add #[tracing::instrument] to EntityExtractor::extract()
- [x] Add #[tracing::instrument] to DualRetriever::search()
- [x] Add #[tracing::instrument] to SimStorageBackend methods
- [x] Add #[tracing::instrument] to SimLLMProvider::complete()
- [x] Add #[tracing::instrument] to SimEmbeddingProvider::embed_batch()
- [x] Add #[tracing::instrument] to EvolutionTracker::detect()
- [x] Add span events for key operations (extraction, embeddings, storage)

### Phase 4: Testing ✅
- [x] Unit tests for TelemetryConfig (existing tests pass)
- [x] All 524 unit tests pass with instrumentation
- [x] Verified graceful degradation (feature is optional)

### Phase 5: Documentation ✅
- [x] Updated example with actual span names
- [x] Document environment variables (OTEL_EXPORTER_OTLP_ENDPOINT)
- [x] Updated README with feature flag
- [x] Fixed all 16 telemetry documentation warnings
- [x] Added complete API documentation to all public items
- [x] Included verification reports from review agent

### Phase 6: Verification ✅
- [x] Run `cargo test --all-features` (524 tests passed)
- [x] Run `cargo clippy --all-features` (clean)
- [x] Example compiles and demonstrates spans
- [x] Ready to commit and push

## Technical Design

### Feature Flag

```toml
[features]
default = []
opentelemetry = [
    "dep:opentelemetry",
    "dep:opentelemetry-otlp",
    "dep:tracing-opentelemetry"
]
```

### Constants (TigerStyle)

```rust
// src/constants.rs
pub const TELEMETRY_BATCH_SIZE_MAX: usize = 512;
pub const TELEMETRY_EXPORT_TIMEOUT_MS: u64 = 5000;  // 5 seconds
pub const TELEMETRY_SPAN_QUEUE_SIZE_MAX: usize = 2048;
```

### Initialization

```rust
// src/telemetry/mod.rs
pub fn init_telemetry(config: TelemetryConfig) -> Result<()> {
    // With graceful fallback if exporter unavailable
}
```

### Integration Points

Spans will be added at:
- `EntityExtractor::extract()` - entity extraction span
- `DualRetriever::retrieve()` - retrieval span with vector/keyword sub-spans
- `StorageBackend::read()/write()` - storage operation spans
- `LLMProvider::call()` - LLM request spans

### Environment Variables

Standard OTel env vars:
- `OTEL_EXPORTER_OTLP_ENDPOINT` - Exporter endpoint (default: http://localhost:4317)
- `OTEL_SERVICE_NAME` - Service name (default: "umi-memory")
- `OTEL_TRACES_SAMPLER` - Sampling strategy (default: "always_on")

## Success Criteria

- [x] `cargo build --all-features` succeeds
- [x] `cargo test --all-features` passes (524 unit tests passed)
- [x] `cargo clippy --all-features` warnings fixed
- [x] Feature is optional (compiles without it)
- [x] Example created demonstrating usage
- [x] Documentation updated (README, lib.rs, constants.rs)

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| OTel breaks DST determinism | Use SimTelemetry backend in tests |
| Exporter blocks async runtime | Use batch exporter with timeouts |
| Dependency bloat | Make it optional feature |
| Complex initialization | Provide sensible defaults, graceful fallback |

## Instance Log

| Instance | Phase | Status | Notes |
|----------|-------|--------|-------|
| Claude-1 | Phase 1 | In Progress | Adding dependencies |

## Findings

- OpenTelemetry only mentioned in planning docs (Letta rewrite feasibility)
- No existing OTel code in codebase
- Basic tracing infrastructure already present
- Verification by external review agent confirms 100% functional implementation
- Successfully tested with real Jaeger collector - 4 traces with 10 spans captured
- All span attributes present and correct (text_len, query_len, entity_id, etc.)
- Production-ready with graceful degradation

## References

- [OpenTelemetry Rust](https://github.com/open-telemetry/opentelemetry-rust)
- [tracing-opentelemetry](https://docs.rs/tracing-opentelemetry/)
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
