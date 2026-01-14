# Plan: Make Provider Traits Object-Safe

**Status**: In Progress
**Created**: 2026-01-13
**Goal**: Make provider traits object-safe to enable real provider injection into Memory class

## Problem Statement

Current state:
- `LLMProvider::complete_json<T>` has a generic type parameter
- Generic methods prevent trait objects (`Box<dyn LLMProvider>`)
- Memory class can only use Sim providers
- Python bindings can't inject real providers (Anthropic, OpenAI, Lance, Postgres)

Impact:
- Memory class is only useful for testing, not production
- Python bindings are incomplete
- Contradicts project vision of flexible provider injection

## Solution: Remove Generic from complete_json

Change `complete_json<T>` to return `serde_json::Value` instead of generic `T`:

```rust
// Before (NOT object-safe)
async fn complete_json<T: DeserializeOwned + Send>(&self, request: &CompletionRequest) -> Result<T, ProviderError>;

// After (object-safe)
async fn complete_json(&self, request: &CompletionRequest) -> Result<serde_json::Value, ProviderError>;
```

Callers deserialize manually:
```rust
let value = provider.complete_json(&request).await?;
let typed: MyType = serde_json::from_value(value)?;
```

## Implementation Plan

### Phase 1: Update LLMProvider Trait ✅ COMPLETE
- [x] Change `complete_json` signature in `umi-memory/src/llm/mod.rs`
- [x] Update default implementation
- [x] Verify trait is now object-safe (no compiler errors on `Box<dyn LLMProvider>`)

### Phase 2: Update Test Cases ✅ COMPLETE
- [x] Fix `umi-memory/src/llm/sim.rs` test
- [x] Fix `umi-memory/src/dst/llm.rs` test
- [x] Run `cargo test --all-features -p umi-memory` to verify

### Phase 3: Refactor Components to Use Trait Objects (In Progress)
- [x] Update EntityExtractor to use `Box<dyn LLMProvider>` ✅
  - Added Debug + 'static bounds to LLMProvider trait
  - Updated EntityExtractor struct to use Box<dyn LLMProvider>
  - Fixed all test usages (47 tests pass)
  - Updated Memory and UnifiedMemory to Box providers when creating EntityExtractor
- [ ] Update DualRetriever to use trait objects for all providers
  - Added Debug + 'static bounds to EmbeddingProvider, VectorBackend, StorageBackend traits
  - Need to update DualRetriever struct (next)
- [ ] Update EvolutionTracker to use `Box<dyn LLMProvider>` and `Box<dyn StorageBackend>`
- [ ] Update Memory struct to use trait objects
- [ ] Add Memory::new() constructor that takes real providers
- [ ] Keep Memory::sim() for backward compatibility
- [ ] Update documentation with examples
- [ ] Add integration test with real providers

### Phase 4: Update Python Bindings
- [ ] Update Memory class to accept provider instances
- [ ] Add examples showing real provider injection
- [ ] Update PYTHON.md to remove limitation notice
- [ ] Add Python test with real providers

### Phase 5: Verification
- [ ] Run full test suite: `cargo test --all-features`
- [ ] Run Python tests: `cd umi-py && pytest`
- [ ] Manual verification: Run example with Anthropic provider
- [ ] Update ADRs if needed
- [ ] Update CLAUDE.md with new capability

## Files to Modify

1. `umi-memory/src/llm/mod.rs` - Trait definition
2. `umi-memory/src/llm/sim.rs` - Test case
3. `umi-memory/src/dst/llm.rs` - Test case
4. `umi-memory/src/umi/mod.rs` - Memory class
5. `umi-py/src/lib.rs` - Python bindings
6. `PYTHON.md` - Documentation
7. `docs/adr/017-memory-class.md` - ADR update (if needed)

## Success Criteria

- [ ] All Rust tests pass (~813 tests)
- [ ] Python tests pass
- [ ] Can create `Memory::new(anthropic_provider, lance_storage, ...)`
- [ ] Can inject real providers from Python
- [ ] No object-safety compiler errors
- [ ] Documentation updated
- [ ] Code committed with passing tests

## Risks

- **Breaking change**: Callers of `complete_json` must add manual deserialization
  - Mitigation: Only 2 test usages found, easy to fix
- **Performance**: Extra `serde_json::Value` allocation
  - Mitigation: Negligible for LLM calls (network-bound)

## Timeline

Estimated: 2-3 hours
- Phase 1-2: 30 minutes (trait + tests)
- Phase 3: 1 hour (Memory class changes)
- Phase 4: 30 minutes (Python bindings)
- Phase 5: 30 minutes (verification)

## Instance Log

| Instance | Claimed Phases | Status |
|----------|----------------|--------|
| Claude 1 | Phase 1-2 | In Progress |
|          |                |        |
