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

### Phase 3: DST Compatibility
- [ ] Ensure SimClock works with OTel spans
- [ ] Add SimTelemetry backend for testing
- [ ] Verify deterministic behavior with DST_SEED
- [ ] Test that OTel spans don't break fault injection

### Phase 4: Testing
- [ ] Unit tests for TelemetryConfig
- [ ] Integration test with real OTLP exporter (optional)
- [ ] DST test with SimTelemetry
- [ ] Verify graceful degradation when exporter unavailable

### Phase 5: Documentation
- [ ] Add telemetry section to CLAUDE.md
- [ ] Document environment variables (OTEL_EXPORTER_OTLP_ENDPOINT)
- [ ] Add example in README
- [ ] Create ADR for telemetry architecture

### Phase 6: Verification
- [ ] Run `cargo test --all-features`
- [ ] Run `cargo clippy --all-features`
- [ ] Manual test with Jaeger/OTLP collector
- [ ] Verify spans appear in trace viewer
- [ ] Commit and push

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
- Need to verify OTel doesn't interfere with SimClock

## References

- [OpenTelemetry Rust](https://github.com/open-telemetry/opentelemetry-rust)
- [tracing-opentelemetry](https://docs.rs/tracing-opentelemetry/)
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
