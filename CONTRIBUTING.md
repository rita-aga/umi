# Contributing to Umi

Thank you for your interest in contributing to Umi! This document provides guidelines for contributing to the project.

## Getting Started

1. **Read the documentation**:
   - [README.md](./README.md) - Project overview
   - [VISION.md](./VISION.md) - Project goals and roadmap
   - [CLAUDE.md](./CLAUDE.md) - Detailed development guidelines

2. **Check open issues**: Browse [GitHub Issues](https://github.com/rita-aga/umi/issues) for tasks to work on

3. **Fork and clone**: Fork the repository and clone it locally

## Development Setup

### Rust Development

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/rita-aga/umi.git
cd umi

# Build the project
cargo build --all-features

# Run tests
cargo test --all-features

# Run linters
cargo clippy --all-features -- -D warnings
cargo fmt --check
```

### Python Development

```bash
# Install Python dependencies
cd umi-py
pip install maturin
maturin develop

# Run tests (when available)
pytest -v

# Run linters
ruff check .
ruff format .
mypy umi/
```

## Core Principles

### 1. Simulation-First (Mandatory)

Every component MUST have a simulation implementation for deterministic testing:

```rust
// Good - deterministic simulation
let mut memory = Memory::sim(42);

// Also good - production with real providers
let memory = Memory::builder()
    .with_llm(AnthropicProvider::new())
    .build();
```

### 2. Tests Must Pass

Before every commit:

```bash
# Rust
cargo fmt && cargo clippy --all-features -- -D warnings && cargo test --all-features

# Python (when available)
ruff check . && ruff format . && pytest
```

All tests must pass. Never commit broken code.

### 3. TigerStyle Engineering

Follow TigerBeetle's engineering principles:

- **Explicit constants with units**: `MEMORY_BYTES_MAX`, `TTL_SECONDS_DEFAULT`
- **2+ assertions per function**: Validate preconditions and postconditions
- **No silent truncation**: Explicit conversions only
- **Big-endian naming**: `memory_size_bytes_max` not `max_memory_size_bytes`

See [CLAUDE.md](./CLAUDE.md) for detailed TigerStyle guidelines.

### 4. No Stubs

Complete implementations only. No placeholder code:

```rust
// FORBIDDEN
fn extract_entities(&self, text: &str) -> Vec<Entity> {
    Vec::new()  // TODO: implement
}

// REQUIRED
fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    let prompt = format!("Extract entities from: {}", text);
    let response = self.llm.call(&prompt).await?;
    self.parse_entities(&response)
}
```

### 5. Graceful Degradation

Handle LLM failures elegantly:

```rust
// Good - returns fallback on failure
pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    match self.llm.extract(text).await {
        Ok(entities) => Ok(entities),
        Err(e) => {
            tracing::warn!("LLM extraction failed: {}, returning empty", e);
            Ok(Vec::new())  // Fallback to empty list
        }
    }
}
```

## Pull Request Process

1. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feature/add-postgres-backend
   ```

2. **Make your changes** following the guidelines above

3. **Write tests** for new functionality or bug fixes

4. **Run the full test suite**:
   ```bash
   cargo test --all-features
   ```

5. **Run linters**:
   ```bash
   cargo clippy --all-features -- -D warnings
   cargo fmt
   ```

6. **Commit with a clear message** using conventional commits:
   ```bash
   git commit -m "feat: Add Postgres storage backend"
   ```

   Use these prefixes:
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `refactor:` - Code refactoring
   - `docs:` - Documentation changes
   - `test:` - Test additions/fixes
   - `chore:` - Maintenance tasks

7. **Push and create a PR**:
   ```bash
   git push origin feature/add-postgres-backend
   ```

8. **Describe your changes** in the PR:
   - What problem does this solve?
   - How does it work?
   - Are there any breaking changes?
   - Link to related issues

## What to Work On

### High Priority (P0)

- **PyO3 bindings** - Complete Python type system
- **Real Postgres backend** - Production storage
- **Documentation** - Examples and guides

### Medium Priority (P1)

- **PyPI publishing** - Public Python release
- **Crates.io publishing** - Rust crate distribution
- **Performance improvements** - Optimization work

### Low Priority (P2)

- **Additional LLM providers** - More provider support
- **Advanced features** - New capabilities

See [VISION.md](./VISION.md) for the full roadmap.

## Code Review

All PRs require review. Expect feedback on:

- **Correctness** - Does the code work?
- **Tests** - Are there tests? Do they pass?
- **Style** - Does it follow TigerStyle?
- **Safety** - Are there proper assertions and error handling?
- **Documentation** - Are public APIs documented?

## Architecture Decision Records (ADRs)

For significant architectural changes, create an ADR in `docs/adr/`:

1. Use the next available number: `NNN-short-name.md`
2. Follow the existing format (see `docs/adr/` for examples)
3. Include:
   - Context (why is this needed?)
   - Decision (what are we doing?)
   - Consequences (what are the trade-offs?)
   - Alternatives (what else was considered?)

## Questions?

- Open a [GitHub Issue](https://github.com/rita-aga/umi/issues)
- Read [CLAUDE.md](./CLAUDE.md) for detailed development guidelines
- Check existing ADRs in `docs/adr/`

## Code of Conduct

Be respectful, inclusive, and collaborative. We're all here to build something great together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
