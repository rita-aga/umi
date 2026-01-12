# Learnings from Kelpie for Umi

**Date**: 2026-01-12
**Purpose**: Document improvements to make Umi as clean and straightforward as Kelpie

---

## Executive Summary

Kelpie demonstrates excellent project structure, documentation, and engineering practices. Key learnings:

1. **Documentation visibility**: Top-level VISION.md vs hidden .vision/ directory
2. **Comprehensive README**: Badges, tables, ASCII diagrams, clear examples
3. **Detailed CLAUDE.md**: TigerStyle principles, commit policy, verification requirements
4. **Unified CI**: Single ci.yml with multiple jobs vs separate workflows
5. **Workspace organization**: Clean crates/ directory structure
6. **Professional polish**: Status badges, performance tables, detailed testing docs

---

## Comparison Matrix

| Aspect | Kelpie | Umi | Recommendation |
|--------|---------|-----|----------------|
| **VISION.md** | Top-level, comprehensive (310 lines) | Hidden in .vision/ (104 lines) | Move to top-level, expand content |
| **README.md** | Detailed with badges, tables, diagrams (220 lines) | Good but simpler (169 lines) | Add badges, tables, ASCII architecture |
| **CLAUDE.md** | Extensive with TigerStyle, policies (574 lines) | Functional but basic (208 lines) | Add TigerStyle guide, commit policy, verification |
| **CI Workflows** | Single comprehensive ci.yml (6 jobs) | Split: ci.yml (Python), ci-rust.yml (Rust) | Consolidate or enhance both |
| **Workspace** | crates/ directory for all members | Flat: umi-memory/, umi-py/ | Consider crates/ (optional) |
| **Cargo.toml** | [workspace.package] + [workspace.dependencies] + profiles | Basic workspace config | Add workspace.package, profiles |
| **Status Badges** | CI badge, License badge | None | Add CI and license badges |
| **Testing Docs** | Fault injection types, DST examples | Basic test commands | Add DST documentation |

---

## Detailed Learnings

### 1. Documentation Visibility (VISION.md)

**Kelpie Approach**:
- Top-level `VISION.md` (310 lines)
- Comprehensive sections:
  - What is Kelpie? (clear positioning)
  - Core principles (4 detailed principles)
  - Architecture (ASCII diagram)
  - Memory hierarchy (clear tables)
  - What it IS NOT (clear boundaries)
  - Target use cases (4 specific scenarios)
  - Implementation status (completion tables)
  - Design decisions (why virtual actors, MCP, Firecracker)
  - Performance targets (table with current vs target)
  - Test coverage (table by category)
  - Roadmap (phased, with priorities)
  - Inspiration sources (with explanations)

**Umi Current State**:
- Hidden `.vision/umi-vision.md` (104 lines)
- Good content but:
  - Not visible to casual viewers
  - Less comprehensive
  - Missing performance targets
  - Missing clear test coverage table
  - Roadmap not as detailed

**Recommendation**:
- **Move** `.vision/umi-vision.md` → `VISION.md` (top-level)
- **Expand** with:
  - Architecture ASCII diagram
  - "What Umi Is NOT" section
  - Performance targets table
  - Test coverage by category table
  - Clearer roadmap with priorities (P0/P1/P2)
  - More detailed use cases

---

### 2. README Excellence

**Kelpie Approach** (220 lines):
```markdown
# Kelpie
[![CI](badge)](link)
[![License](badge)](link)

One-line description

## Overview
- Bullet points of key features

## Quick Start
```bash
# Clear, copy-paste examples
```

## Features
### Memory Hierarchy
| Tier | Purpose | Size | Persistence |
|------|---------|------|-------------|
| Core | ... | ... | ... |

### Sandbox Isolation
| Level | Implementation | Boot Time |
|-------|----------------|-----------|

## Architecture
```
ASCII diagram
```

## Crates
| Crate | Description | Status |
|-------|-------------|--------|

## API Compatibility
Detailed API endpoints

## Testing
Fault types, examples

## Configuration
Environment variables table

## Roadmap
Clear next priorities

## Engineering Principles
TigerStyle summary

## Inspiration
Sources with explanations

## License
```

**Umi Current State** (169 lines):
- Good structure but missing:
  - Status badges (CI, license)
  - Feature comparison tables
  - ASCII architecture diagram
  - Configuration table
  - Clear crate status table
  - Detailed testing section

**Recommendation**:
- **Add** CI and license badges
- **Add** ASCII architecture diagram
- **Add** feature comparison table
- **Add** crate status table with completion status
- **Add** configuration environment variables table
- **Enhance** testing section with DST examples
- **Add** clear roadmap section (link to VISION.md)

---

### 3. CLAUDE.md Depth

**Kelpie Approach** (574 lines):
- Quick commands (build, test, benchmark)
- Architecture overview
- **TigerStyle Engineering Principles** (detailed):
  1. Explicit constants with units (examples)
  2. Big-endian naming (examples)
  3. Assertions (2+ per function)
  4. Prefer u64 over usize
  5. No silent truncation
  6. Explicit error handling
  7. Debug assertions for expensive checks
- **DST (Deterministic Simulation Testing)** section:
  - Core principles
  - Running DST tests
  - Writing DST tests (code examples)
  - Fault types table
- **Code Style** guide:
  - Module organization
  - Struct layout
  - Function signatures
- **Testing Guidelines**:
  - Test naming conventions
  - Property-based testing examples
  - DST test coverage checklist
- **Error Handling** patterns
- **Performance Guidelines**
- **Documentation** standards
- **Commit Policy: Only Working Software** (critical):
  - Pre-commit verification
  - Why it matters
  - Commit checklist
  - What to do if tests fail
- **Acceptance Criteria: No Stubs, Verification First**:
  - No stubs policy (with examples)
  - Verification-first development
  - Verification checklist
  - What "done" means
  - Codebase audit commands
- Contributing workflow

**Umi Current State** (208 lines):
- Good basics:
  - What Umi is
  - Development philosophy (simulation-first, TigerStyle, graceful degradation)
  - Build & test commands
  - ADR references
  - Directory structure
- Missing:
  - Detailed TigerStyle examples
  - DST framework documentation
  - Commit policy
  - Acceptance criteria (no stubs, verification first)
  - Code style guide
  - Testing guidelines
  - Performance guidelines
  - Error handling patterns

**Recommendation**:
- **Add** detailed TigerStyle section with examples from kelpie
- **Add** "Commit Policy: Only Working Software" section
- **Add** "Acceptance Criteria" section
- **Add** DST testing documentation
- **Add** code style guide (module org, struct layout, function signatures)
- **Add** testing guidelines with examples
- **Add** error handling patterns
- **Expand** current sections with more detail

---

### 4. CI Workflow Structure

**Kelpie Approach** (single `ci.yml`):
```yaml
jobs:
  check:       # cargo check
  fmt:         # cargo fmt --check
  clippy:      # cargo clippy -D warnings
  test:        # cargo test
  test-dst:    # DST with multiple seeds
  docs:        # cargo doc with -D warnings
  coverage:    # cargo-llvm-cov + codecov
```
- Single comprehensive workflow
- Rust cache optimization
- Documentation validation
- Coverage reporting

**Umi Current State**:
- `ci.yml`: Python (lint, typecheck, test)
- `ci-rust.yml`: Rust (lint, test, build)
- Both are good but:
  - No documentation validation
  - No coverage reporting
  - Could add DST multi-seed testing like kelpie

**Recommendation**:
- **Keep** separate workflows (Python vs Rust makes sense)
- **Enhance** `ci-rust.yml`:
  - Add `docs` job (cargo doc with -D warnings)
  - Add `coverage` job (cargo-llvm-cov)
  - Enhance `test` job to run DST with multiple seeds (like kelpie)
  - Add `check` job (cargo check) before test
- **Add** badges to README pointing to both workflows

---

### 5. Workspace Organization

**Kelpie Approach**:
```
kelpie/
├── crates/
│   ├── kelpie-core/
│   ├── kelpie-runtime/
│   ├── kelpie-registry/
│   └── ... (13 crates)
├── docs/adr/
├── Cargo.toml
├── README.md
├── VISION.md
└── CLAUDE.md
```

**Umi Current State**:
```
umi/
├── umi-memory/
├── umi-py/
├── docs/adr/
├── Cargo.toml
├── README.md
├── CLAUDE.md
└── .vision/umi-vision.md
```

**Recommendation**:
- **Optional**: Move to crates/ structure (not critical)
- **Required**: Move `.vision/umi-vision.md` → `VISION.md`
- **Keep**: Current flat structure is fine for 2 crates

---

### 6. Cargo.toml Completeness

**Kelpie Approach**:
```toml
[workspace]
resolver = "2"
members = [...]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
license = "Apache-2.0"
repository = "..."
authors = [...]

[workspace.dependencies]
# Comprehensive shared dependencies
tokio = { version = "1.34", features = ["full"] }
# ... many more

# Internal crates
kelpie-core = { path = "crates/kelpie-core" }
# ...

[profile.release]
lto = true
codegen-units = 1

[profile.bench]
lto = true
```

**Umi Current State**:
```toml
[workspace]
resolver = "2"
members = [...]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "..."
authors = [...]

[workspace.dependencies]
# Good dependencies
```

**Recommendation**:
- **Add** `rust-version = "1.75"` (or appropriate version)
- **Add** release profile optimizations:
  ```toml
  [profile.release]
  lto = true
  codegen-units = 1
  ```
- **Add** bench profile (if benchmarks exist)
- Current dependency structure is good

---

### 7. README Badges & Polish

**Kelpie**:
```markdown
[![CI](https://github.com/nerdsane/kelpie/actions/workflows/ci.yml/badge.svg)](...)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
```

**Umi**: No badges

**Recommendation**:
- **Add** CI badges for both Python and Rust workflows
- **Add** license badge
- **Consider** adding:
  - Crates.io badge (when published)
  - PyPI badge (when published)
  - Documentation badge (if hosted)

---

## Implementation Plan

### Phase 1: Documentation Restructure (High Impact, Low Effort)

1. **Move VISION.md to top level**
   - `mv .vision/umi-vision.md VISION.md`
   - Update links in CLAUDE.md, README.md

2. **Enhance VISION.md**
   - Add ASCII architecture diagram
   - Add "What Umi Is NOT" section
   - Add performance targets table
   - Add test coverage table
   - Enhance roadmap with P0/P1/P2 priorities

3. **Enhance README.md**
   - Add CI badges (Python + Rust)
   - Add license badge
   - Add ASCII architecture diagram
   - Add feature/crate status table
   - Add configuration table
   - Enhance testing section with DST examples

### Phase 2: Development Guidelines (Medium Impact, Medium Effort)

4. **Enhance CLAUDE.md**
   - Add detailed TigerStyle section with examples
   - Add "Commit Policy: Only Working Software"
   - Add "Acceptance Criteria: No Stubs, Verification First"
   - Add DST testing documentation
   - Add code style guide
   - Add testing guidelines
   - Add error handling patterns

### Phase 3: CI/Build Improvements (Medium Impact, Low Effort)

5. **Enhance Rust CI**
   - Add `docs` job (cargo doc -D warnings)
   - Add `coverage` job
   - Add DST multi-seed testing
   - Add `check` job

6. **Optimize Cargo.toml**
   - Add rust-version
   - Add release profile optimizations
   - Add bench profile

### Phase 4: Optional Structural Changes (Low Priority)

7. **Consider crates/ directory**
   - Only if workspace grows beyond 2-3 crates
   - Not critical for current size

---

## Key Takeaways

1. **Visibility matters**: Top-level docs (VISION.md) are more discoverable
2. **Comprehensive documentation**: Detailed CLAUDE.md helps contributors understand expectations
3. **Professional polish**: Badges, tables, diagrams make project feel mature
4. **Clear standards**: Explicit commit policy and acceptance criteria prevent technical debt
5. **Testing rigor**: DST documentation and multi-seed CI testing ensure reliability

---

## Priority Recommendations

**Must Do** (P0):
1. Move `.vision/umi-vision.md` → `VISION.md`
2. Add CI badges to README
3. Add "Commit Policy" section to CLAUDE.md
4. Add "Acceptance Criteria" section to CLAUDE.md

**Should Do** (P1):
5. Enhance VISION.md with tables and diagrams
6. Add ASCII architecture to README
7. Add detailed TigerStyle examples to CLAUDE.md
8. Enhance Rust CI with docs/coverage jobs

**Nice to Have** (P2):
9. Add performance targets table to VISION.md
10. Add configuration table to README
11. Add release profile optimizations to Cargo.toml
12. Consider crates/ directory if workspace grows

---

## Conclusion

Kelpie demonstrates how good documentation, clear engineering principles, and professional polish create a project that's easy to understand, contribute to, and maintain. Umi has solid fundamentals but can benefit from these structural and documentation improvements without changing its core architecture or code.
