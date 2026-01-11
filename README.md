# Umi

Memory for AI agents - entity extraction, dual retrieval, and evolution tracking.

## What It Does

- **Entity Extraction**: Pulls structured entities (people, orgs, topics) from text using an LLM
- **Dual Retrieval**: Fast substring search + LLM-powered query expansion for better recall
- **Evolution Tracking**: Detects when new information updates, extends, or contradicts existing memories
- **Temporal Metadata**: Tracks both when something was said and when the event occurred

## Installation

```bash
# From source (not yet on PyPI)
pip install git+https://github.com/rita-aga/umi.git

# With LLM provider support
pip install "umi[anthropic] @ git+https://github.com/rita-aga/umi.git"
pip install "umi[openai] @ git+https://github.com/rita-aga/umi.git"
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

## Current Limitations

- **Storage is in-memory only** - no persistence yet (Postgres/vector DB backends planned)
- **SimStorage for testing** - use `seed=42` for deterministic behavior without LLM calls

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
│  Memory API                                                  │
│  - remember(text) -> List[Entity]                           │
│  - recall(query) -> List[Entity]                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│EntityExtractor│     │ DualRetriever │     │EvolutionTracker│
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │  LLMProvider                                 │
        │  - SimLLMProvider (testing, no API calls)   │
        │  - AnthropicProvider                        │
        │  - OpenAIProvider                           │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  SimStorage (in-memory only for now)        │
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
