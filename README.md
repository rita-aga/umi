# Umi - Memory System That Never Forgets

A simulation-first memory library with provable correctness for AI applications.

## Key Features

- **Simulation-First**: Every component has a deterministic simulation mode for reliable testing
- **TigerStyle**: Preconditions and postconditions throughout for correctness
- **LLM Integration**: Smart entity extraction, dual retrieval, and evolution tracking
- **Fault Injection**: Test edge cases with configurable fault injection

## Installation

```bash
pip install umi
```

With LLM provider support:
```bash
pip install "umi[anthropic]"  # For Anthropic Claude
pip install "umi[openai]"     # For OpenAI
pip install "umi[all]"        # All providers
```

## Quick Start

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

## Core Components

### Memory

Main interface for remember/recall operations.

```python
memory = Memory(seed=42)  # Simulation mode

# Store with metadata
entities = await memory.remember(
    "Alice works at Acme Corp",
    importance=0.8,
    document_time=datetime.now(),
    event_time=datetime(2024, 1, 15),
)

# Search with options
results = await memory.recall(
    "Who works at Acme?",
    limit=10,
    deep_search=True,  # Use LLM for query expansion
)
```

### EntityExtractor

Extract structured entities from text.

```python
from umi import EntityExtractor, SimLLMProvider

llm = SimLLMProvider(seed=42)
extractor = EntityExtractor(llm=llm, seed=42)

result = await extractor.extract("Alice, CEO of Acme, met Bob")
for entity in result.entities:
    print(f"{entity.name} ({entity.entity_type}): {entity.confidence}")
```

### DualRetriever

Smart search with query expansion.

```python
from umi import DualRetriever, SimLLMProvider, SimStorage

retriever = DualRetriever(
    storage=SimStorage(seed=42),
    llm=SimLLMProvider(seed=42),
    seed=42,
)

# Deep search rewrites query into variations
results = await retriever.search("Who works at Acme?", deep_search=True)
```

### EvolutionTracker

Track how memories evolve over time.

```python
from umi import EvolutionTracker, SimLLMProvider, SimStorage

tracker = EvolutionTracker(
    llm=SimLLMProvider(seed=42),
    storage=SimStorage(seed=42),
    seed=42,
)

# Detect if new entity updates/extends/contradicts existing
evolution = await tracker.find_related_and_detect(new_entity)
if evolution:
    print(f"Evolution: {evolution.evolution_type} - {evolution.reason}")
```

## Fault Injection

Test edge cases with configurable faults.

```python
from umi import Memory, FaultConfig

# 50% chance of LLM timeout
faults = FaultConfig(llm_timeout=0.5)
memory = Memory(seed=42, faults=faults)

# Test graceful degradation
entities = await memory.remember("Test")  # May timeout
```

Available faults:
- `llm_timeout`: Simulate LLM timeouts
- `llm_error`: Simulate LLM API errors
- `llm_malformed`: Simulate malformed LLM responses
- `storage_read_error`: Simulate storage read failures
- `storage_write_error`: Simulate storage write failures

## Determinism

Same seed always produces identical results:

```python
# Run 1
memory1 = Memory(seed=42)
entities1 = await memory1.remember("Alice knows Bob")

# Run 2 (same seed)
memory2 = Memory(seed=42)
entities2 = await memory2.remember("Alice knows Bob")

# Identical results
assert entities1[0].name == entities2[0].name
```

## Entity Types

- `person`: People mentioned
- `org`: Organizations, companies
- `project`: Projects, initiatives
- `topic`: Topics, concepts
- `preference`: User preferences
- `task`: Tasks, action items
- `event`: Events, meetings
- `note`: Fallback for unstructured content

## Evolution Types

- `update`: New info replaces old (e.g., job change)
- `extend`: New info adds to old (e.g., more details)
- `derive`: New info concluded from old (e.g., inference)
- `contradict`: New info conflicts with old

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User API: Memory                                            │
│  - remember(text) -> List[Entity]                           │
│  - recall(query) -> List[Entity]                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ EntityExtractor│     │ DualRetriever │     │EvolutionTracker│
│ (extraction)   │     │ (search)      │     │ (relationships)│
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │  LLMProvider Protocol                        │
        │  ┌─────────────┐  ┌─────────────┐           │
        │  │SimLLMProvider│  │ Anthropic/  │           │
        │  │(simulation)  │  │ OpenAI      │           │
        │  └─────────────┘  └─────────────┘           │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  SimStorage (in-memory, deterministic)      │
        └─────────────────────────────────────────────┘
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test file
pytest umi/tests/test_memory.py

# Linting
ruff check .

# Type checking
mypy umi/
```

## License

MIT
