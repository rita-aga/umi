# ADR-010: Entity Extraction - LLM-Powered Entity and Relation Detection

## Status

Accepted

## Context

Phase 4 established DualRetriever for smart search. Now we need structured
entity extraction from text to populate the memory system.

Current `Memory.remember()` has basic extraction via inline JSON parsing.
We need a dedicated `EntityExtractor` class that:

1. Extracts named entities (people, orgs, projects, topics)
2. Detects relationships between entities
3. Assigns confidence scores
4. Works deterministically with SimLLMProvider

### Requirements

1. **Structured output**: Entities with type, name, content, confidence
2. **Relation detection**: Links between extracted entities
3. **Sim-first**: Deterministic with seed, no network in tests
4. **Graceful fallback**: Returns note entity if extraction fails
5. **TigerStyle**: Preconditions, postconditions, explicit limits

## Decision

Implement `EntityExtractor` class with:

1. **Prompt engineering**: Structured extraction prompts
2. **Response parsing**: JSON parsing with validation
3. **Confidence scoring**: LLM-assigned or heuristic-based
4. **Relation extraction**: Source/target/type triples

### Architecture

```
remember("I met Alice at Acme Corp")
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                   EntityExtractor                         │
│                                                           │
│  1. Build extraction prompt                               │
│  2. Call LLM (SimLLMProvider or production)              │
│  3. Parse JSON response                                   │
│  4. Validate and build Entity objects                    │
│  5. Extract relations                                     │
│                                                           │
│  Output:                                                  │
│  - entities: [Entity(name="Alice", type="person"), ...]  │
│  - relations: [Relation(source="Alice", target="Acme")]  │
└──────────────────────────────────────────────────────────┘
```

### Entity Types

```python
ENTITY_TYPES = [
    "person",      # People mentioned
    "org",         # Organizations, companies
    "project",     # Projects, initiatives
    "topic",       # Topics, concepts
    "preference",  # User preferences
    "task",        # Tasks, action items
    "event",       # Events, meetings
    "note",        # Fallback for unstructured
]
```

### Relation Types

```python
RELATION_TYPES = [
    "works_at",    # Person works at Org
    "knows",       # Person knows Person
    "manages",     # Person manages Project
    "relates_to",  # Generic relation
    "prefers",     # User prefers something
    "part_of",     # Entity is part of another
]
```

### API Design

```python
@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    name: str
    entity_type: str
    content: str
    confidence: float  # 0.0-1.0

@dataclass
class ExtractedRelation:
    """Relation between entities."""
    source: str      # Source entity name
    target: str      # Target entity name
    relation_type: str
    confidence: float

@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    raw_text: str

class EntityExtractor:
    """Extract entities and relations from text using LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        seed: int | None = None,
    ):
        """Initialize extractor.

        Args:
            llm: LLM provider for extraction.
            seed: Optional seed for deterministic behavior.
        """

    async def extract(
        self,
        text: str,
        *,
        existing_entities: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relations from text.

        Args:
            text: Text to extract from.
            existing_entities: Known entities for context.

        Returns:
            ExtractionResult with entities and relations.
        """
```

### Extraction Prompt

```
Extract entities and relationships from this text.

Text: {text}

Known entities (for context): {existing_entities}

Return JSON:
{
  "entities": [
    {"name": "...", "type": "person|org|project|topic|preference|task|event", "content": "...", "confidence": 0.0-1.0}
  ],
  "relations": [
    {"source": "name1", "target": "name2", "type": "works_at|knows|manages|relates_to|prefers|part_of", "confidence": 0.0-1.0}
  ]
}

Rules:
- Only extract clear, factual entities
- Confidence 0.9+ for explicit mentions, 0.5-0.8 for inferred
- Skip if uncertain
```

### SimLLMProvider Integration

Already implemented in Phase 2:
- `_sim_entity_extraction()` generates deterministic entities
- Routes prompts containing "extract" and "entit"
- Returns JSON with entities and relations arrays

### TigerStyle Compliance

```python
async def extract(self, text: str) -> ExtractionResult:
    # Preconditions
    assert text, "text must not be empty"
    assert len(text) <= TEXT_BYTES_MAX, f"text exceeds limit"

    # ... extraction logic ...

    # Postconditions
    assert isinstance(result.entities, list), "must return entities list"
    for entity in result.entities:
        assert entity.name, "entity must have name"
        assert 0.0 <= entity.confidence <= 1.0, "invalid confidence"

    return result
```

## Consequences

### Positive

- **Structured memory**: Entities with types enable better search
- **Relation tracking**: Links between entities for graph queries
- **Confidence scores**: Filter low-confidence extractions
- **Deterministic**: Same seed produces same extractions

### Negative

- **LLM dependency**: Extraction quality depends on LLM
- **Latency**: Additional LLM call per remember()

### Mitigations

1. **Graceful fallback**: Return note entity if extraction fails
2. **Confidence filtering**: Skip low-confidence entities
3. **Caching**: Future optimization for repeated text

## Implementation

### Phase 5 Files

```
umi/umi-python/umi/
├── extraction.py         # EntityExtractor class (this ADR)
├── memory.py             # Update remember() to use extractor
└── tests/
    └── test_extraction.py  # EntityExtractor tests
```

### Integration with Memory

```python
class Memory:
    def __post_init__(self):
        # ... existing init ...
        self._extractor = EntityExtractor(llm=self._llm, seed=self.seed)

    async def remember(self, text: str, ...) -> list[Entity]:
        if extract_entities:
            result = await self._extractor.extract(text)
            for extracted in result.entities:
                entity = Entity(
                    name=extracted.name,
                    entity_type=extracted.entity_type,
                    content=extracted.content,
                    importance=importance,
                    # ... temporal metadata ...
                )
                stored = await self._storage.store(entity)
                entities.append(stored)
```

## References

- ADR-007: SimLLMProvider
- ADR-008: Memory Class
- ADR-009: Dual Retrieval
