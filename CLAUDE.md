# CLAUDE.md

Instructions for Claude Code when working in the Umi repository.

## What Umi Is

Umi is a memory library for AI agents with two layers:

**Rust core** (`umi-core/`):
- Memory tiers (CoreMemory, WorkingMemory, ArchivalMemory)
- Storage backends with simulation support
- DST (Deterministic Simulation Testing)

**Python layer** (`umi/`):
- Entity extraction from text
- Dual retrieval (fast + LLM query expansion)
- Evolution tracking (detecting updates/contradictions)

## Development Philosophy

### Simulation-First (Mandatory)

Every component MUST have a simulation implementation:

**Rust**: Use `SimConfig::with_seed(N)` for deterministic behavior
```rust
let config = SimConfig::with_seed(42);
let memory = CoreMemory::new(32 * 1024, config);
```

**Python**: Use `seed=N` parameter
```python
memory = Memory(seed=42)  # Deterministic, no LLM calls
```

**Why?** Same seed = same results = reproducible tests and bugs.

### TigerStyle Assertions

**Rust**: Use `debug_assert!` for invariants
```rust
fn store(&mut self, data: &[u8]) -> Result<()> {
    debug_assert!(!data.is_empty(), "data must not be empty");
    debug_assert!(data.len() <= self.capacity, "data exceeds capacity");
    // ...
}
```

**Python**: Use `assert` for pre/postconditions
```python
async def remember(self, text: str) -> list[Entity]:
    assert text, "text must not be empty"
    assert len(text) <= 100_000, "text too large"
    # ...
    assert isinstance(result, list), "must return list"
    return result
```

### Graceful Degradation (Python)

LLM calls can fail. Components should:
- Catch `TimeoutError` and `RuntimeError`
- Return fallback values (empty list, None) instead of crashing

## Build & Test Commands

### Python

```bash
pip install -e ".[dev]"
pytest                      # 145 tests
ruff check .
ruff format --check .
mypy umi/
```

### Rust

```bash
cargo test                  # 232 tests
cargo clippy --all-features
cargo fmt --check
cargo build --release
```

## Architecture Decision Records (ADRs)

Document significant design decisions in `docs/adr/`:
- `009-dual-retrieval.md` - Query rewriting and RRF merging
- `010-entity-extraction.md` - LLM-powered extraction
- `011-evolution-tracking.md` - Memory relationship detection

Create new ADRs for architectural changes. Format: `NNN-short-name.md`

## Directory Structure

```
umi/
├── umi/                    # Python package
│   ├── __init__.py
│   ├── memory.py           # Main Memory class
│   ├── extraction.py       # EntityExtractor
│   ├── retrieval.py        # DualRetriever
│   ├── evolution.py        # EvolutionTracker
│   ├── storage.py          # SimStorage, Entity
│   ├── faults.py           # FaultConfig
│   ├── providers/
│   │   ├── base.py         # LLMProvider protocol
│   │   ├── sim.py          # SimLLMProvider
│   │   ├── anthropic.py
│   │   └── openai.py
│   └── tests/
│
├── umi-core/               # Rust core
│   └── src/
│       ├── lib.rs
│       ├── dst/            # Deterministic simulation
│       ├── memory/         # Memory tiers
│       └── storage/        # Storage backends
│
├── umi-py/                 # PyO3 bindings (not wired up yet)
│
├── Cargo.toml              # Rust workspace
├── pyproject.toml          # Python package
└── docs/adr/               # Architecture decisions
```

## Current Limitations

- **Python storage is in-memory only** - SimStorage doesn't persist
- **PyO3 bindings not wired up** - Python and Rust layers are separate
- **No real database backends** - Postgres/Qdrant planned

## Git Workflow

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`
- Run both Python and Rust tests before pushing
- Update ADRs for architectural changes
- Always commit and push after completing work

## Planning & Progress

**Before starting work**, check these directories:

- `.vision/umi-vision.md` - Core principles and constraints
- `.progress/` - Active plans and roadmaps

Current active plan: `.progress/001_20260111_163000_umi-standalone-roadmap.md`

## Roadmap (Next Steps)

### Phase 1: PyPI Publication
- Package name: `umi-memory` (or similar)
- GitHub Actions workflow for auto-publishing
- Create v0.1.0 release

### Phase 2: Real Storage Backends
- PostgreSQL with asyncpg
- Qdrant for vector search
- Unified Storage protocol

### Phase 3: Wire PyO3 Bindings
- Expose Rust memory tiers to Python
- Optional high-performance backend
- Benchmark Rust vs Python

See `.progress/001_*.md` for full details.

## Origin

Umi was extracted from RikaiOS to be a standalone library. Inspired by:
- **memU** - Dual retrieval (fast + LLM semantic)
- **Mem0** - Entity extraction and evolution tracking
- **Supermemory** - Temporal metadata
- **TigerStyle** - Assertion-based programming
