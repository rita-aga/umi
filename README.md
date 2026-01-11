# Umi

Memory for AI agents - entity extraction, dual retrieval, and evolution tracking.

## What It Does

- **Entity Extraction**: Pulls structured entities (people, orgs, topics) from text using an LLM
- **Dual Retrieval**: Fast substring search + LLM-powered query expansion for better recall
- **Evolution Tracking**: Detects when new information updates, extends, or contradicts existing memories
- **Temporal Metadata**: Tracks both when something was said and when the event occurred

## Architecture

Umi has two layers:

- **Rust core** (`umi-core/`): Memory tiers, storage backends, deterministic simulation testing (DST)
- **Python layer** (`umi/`): LLM integration, entity extraction, smart retrieval

```
┌─────────────────────────────────────────────────────────────┐
│  Python: pip install umi                                     │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │EntityExtract│  │DualRetriever│  │EvolutionTrack│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │                │                  │
│         └───────────────┼────────────────┘                  │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LLMProvider (SimLLMProvider | Anthropic | OpenAI)  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │ PyO3 (future)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Rust: umi-core                                              │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ CoreMemory  │  │WorkingMemory│  │ArchivalMem  │         │
│  │   (32KB)    │  │   (1MB)     │  │ (unlimited) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │                │                  │
│         └───────────────┼────────────────┘                  │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  StorageBackend (SimBackend | Postgres | Vector)    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Python layer (from source, not yet on PyPI)
pip install git+https://github.com/rita-aga/umi.git

# With LLM provider support
pip install "umi[anthropic] @ git+https://github.com/rita-aga/umi.git"
pip install "umi[openai] @ git+https://github.com/rita-aga/umi.git"
```

## Quick Start (Python)

```python
from umi import Memory

# Simulation mode (deterministic, no API calls)
memory = Memory(seed=42)

# Remember information
entities = await memory.remember("I met Alice at Acme Corp")
print(f"Stored {len(entities)} entities")

# Recall memories
results = await memory.recall("Who do I know at Acme?")
for entity in results:
    print(f"  - {entity.name}: {entity.content}")
```

## Production Mode

```python
from umi import Memory

# With Anthropic Claude
memory = Memory(provider="anthropic")

# With OpenAI
memory = Memory(provider="openai")
```

## Rust Core

The Rust layer provides:

- **Memory Tiers**: CoreMemory (32KB), WorkingMemory (1MB TTL), ArchivalMemory (unlimited)
- **DST (Deterministic Simulation Testing)**: Same seed = same results, every time
- **Storage Backends**: SimBackend for testing, Postgres/Vector backends (planned)

```rust
use umi_core::{CoreMemory, SimConfig};

// Create with deterministic seed
let config = SimConfig::with_seed(42);
let memory = CoreMemory::new(32 * 1024, config);

// Store and retrieve
memory.write(b"important context")?;
let data = memory.read_all()?;
```

Run Rust tests:
```bash
cargo test
```

## Current Limitations

- **Python storage is in-memory only** - SimStorage doesn't persist
- **PyO3 bindings not wired up yet** - Python and Rust layers are separate for now
- **No real database backends** - Postgres/Qdrant planned

## Project Structure

```
umi/
├── umi/                    # Python package
│   ├── memory.py           # Main Memory class
│   ├── extraction.py       # EntityExtractor
│   ├── retrieval.py        # DualRetriever
│   ├── evolution.py        # EvolutionTracker
│   ├── providers/          # LLM providers
│   │   ├── sim.py          # SimLLMProvider (testing)
│   │   ├── anthropic.py
│   │   └── openai.py
│   └── tests/              # Python tests (145 passing)
│
├── umi-core/               # Rust core library
│   └── src/
│       ├── dst/            # Deterministic simulation
│       ├── memory/         # Memory tiers
│       └── storage/        # Storage backends
│
├── umi-py/                 # PyO3 bindings (future)
│
├── Cargo.toml              # Rust workspace
├── pyproject.toml          # Python package
└── docs/adr/               # Architecture decisions
```

## Development

```bash
# Python
pip install -e ".[dev]"
pytest                      # 145 tests
ruff check .
mypy umi/

# Rust
cargo test                  # 232 tests
cargo clippy
cargo fmt --check
```

## License

MIT
