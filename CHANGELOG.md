# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-12

### Added
- Comprehensive documentation explaining SimLLM behavior and placeholder data
- Clear examples of using production LLM providers (Anthropic, OpenAI)
- Better error messages for invalid limit parameters

### Changed
- **BREAKING**: `RecallOptions::with_limit()` now returns `Result<Self, MemoryError>` instead of `Self`
- **BREAKING**: `SearchOptions::with_limit()` now returns `Result<Self, RetrievalError>` instead of `Self`
- Invalid limits (0 or >100) now return descriptive errors instead of panicking in debug builds
- `debug_assert!` kept as safety net after validation

### Fixed
- Limit validation now returns errors instead of panicking for better error handling (Issue #1 from user testing)

### Migration Guide

**Before (v0.1.0)**:
```rust
let options = RecallOptions::default().with_limit(10);
```

**After (v0.2.0)**:
```rust
// Option 1: Use ? operator (propagate error)
let options = RecallOptions::default().with_limit(10)?;

// Option 2: Use .unwrap() (panic on error, for tests)
let options = RecallOptions::default().with_limit(10).unwrap();

// Option 3: Handle error explicitly
let options = match RecallOptions::default().with_limit(10) {
    Ok(opts) => opts,
    Err(e) => {
        eprintln!("Invalid limit: {}", e);
        return Err(e.into());
    }
};
```

## [0.1.0] - 2026-01-11

### Added
- Initial release of umi-memory
- Memory orchestrator with remember() and recall() operations
- Entity extraction from text using LLM providers
- Dual retrieval: fast vector search + LLM query rewriting
- Evolution tracking: updates, contradictions, extensions, derivations
- Memory tiers: CoreMemory, WorkingMemory, ArchivalMemory
- Deterministic simulation testing (DST) framework
- SimLLMProvider for reproducible testing
- LanceDB storage backend for production use
- Anthropic and OpenAI LLM provider integrations
- 590+ tests with full coverage
- Criterion benchmarks for performance tracking

[0.2.0]: https://github.com/rita-aga/umi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rita-aga/umi/releases/tag/v0.1.0
