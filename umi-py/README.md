# Umi Python Bindings

Python bindings for the Umi memory system - a memory library for AI agents with entity extraction, dual retrieval, and evolution tracking.

## Installation

```bash
pip install umi
```

## Quick Start

### Testing with Simulation Providers

```python
import umi
import asyncio

async def main():
    # Create memory with simulation providers (deterministic, for testing)
    memory = umi.Memory.sim(seed=42)

    # Store information
    result = await memory.remember("Alice works at Acme Corp")
    print(f"Stored {result.entity_count()} entities")

    # Retrieve information
    entities = await memory.recall("Alice")
    for entity in entities:
        print(f"- {entity.name}: {entity.content}")

asyncio.run(main())
```

### Production with Real Providers

```python
import umi

# Anthropic LLM + OpenAI embeddings + Lance storage
memory = umi.Memory.with_anthropic(
    anthropic_key="sk-ant-...",
    openai_key="sk-...",
    db_path="./umi_db"
)

# OpenAI LLM + embeddings + Lance storage
memory = umi.Memory.with_openai(
    openai_key="sk-...",
    db_path="./umi_db"
)

# Anthropic LLM + OpenAI embeddings + Postgres storage
memory = umi.Memory.with_postgres(
    anthropic_key="sk-ant-...",
    openai_key="sk-...",
    postgres_url="postgresql://localhost/umi"
)
```

## Features

- **Entity Extraction**: Automatically extract entities from text using LLMs
- **Dual Retrieval**: Hybrid search combining vector similarity and text search
- **Evolution Tracking**: Track how memories evolve and relate over time
- **Multiple Backends**: Support for Lance, Postgres, and in-memory storage
- **Multiple LLM Providers**: Anthropic, OpenAI, or simulation providers

## Core Memory Components

### Core Memory (32KB)
Always-in-context structured blocks:

```python
core = umi.CoreMemory()
core.set_block("system", "You are a helpful assistant.")
core.set_block("persona", "You are friendly and concise.")
print(core.render())  # XML for LLM context
```

### Working Memory (1MB)
Session-scoped KV store with TTL:

```python
working = umi.WorkingMemory()
working.set("session_id", b"abc123", ttl_secs=3600)
value = working.get("session_id")
```

### Archival Memory (Unlimited)
Long-term entity storage with retrieval:

```python
# Remember
result = await memory.remember("Alice works at Acme Corp")

# Recall
entities = await memory.recall("Who works at Acme?")

# Options
options = umi.RememberOptions().without_extraction().with_importance(0.8)
result = await memory.remember("Raw text", options)
```

## License

MIT
