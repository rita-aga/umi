# ADR-011: Evolution Tracking - Memory Relationship Detection

## Status

Accepted

## Context

Phase 5 established EntityExtractor for extracting entities from text. Now we need
to track how memories evolve over time - detecting when new information updates,
extends, contradicts, or derives from existing memories.

This is a key differentiator from other memory systems which store memories in
isolation without understanding their relationships.

### Requirements

1. **Evolution detection**: Identify relationships between new and existing memories
2. **Evolution types**: Update, extend, derive, contradict (from memU research)
3. **Sim-first**: Deterministic with SimLLMProvider, no network in tests
4. **Confidence scoring**: Track certainty of detected relationships
5. **TigerStyle**: Preconditions, postconditions, explicit limits

## Decision

Implement `EvolutionTracker` class with:

1. **LLM-based detection**: Use LLM to compare new entity with existing ones
2. **Structured output**: EvolutionRelation with type, source, target, reason
3. **Graceful degradation**: Return None if detection fails
4. **Integration**: Optional step in Memory.remember()

### Architecture

```
remember("Alice left Acme and joined StartupX")
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                   EvolutionTracker                        │
│                                                           │
│  1. Get new entity from extraction                        │
│  2. Search for related existing entities                  │
│  3. Build comparison prompt                               │
│  4. Call LLM to detect evolution type                     │
│  5. Return EvolutionRelation if found                     │
│                                                           │
│  Output:                                                  │
│  - EvolutionRelation(                                     │
│      source_id="entity-123",  # old: "Alice at Acme"     │
│      target_id="entity-456",  # new: "Alice at StartupX" │
│      evolution_type="update",                             │
│      reason="Job change supersedes previous employment"  │
│    )                                                      │
└──────────────────────────────────────────────────────────┘
```

### Evolution Types

```python
EVOLUTION_TYPES = frozenset([
    "update",      # New info replaces old (e.g., job change)
    "extend",      # New info adds to old (e.g., additional detail)
    "derive",      # New info concluded from old (e.g., inference)
    "contradict",  # New info conflicts with old (e.g., disagreement)
])
```

### API Design

```python
@dataclass(frozen=True)
class EvolutionRelation:
    """Detected evolution relationship between memories."""
    source_id: str      # Older memory ID
    target_id: str      # Newer memory ID
    evolution_type: str # update|extend|derive|contradict
    reason: str         # Brief explanation
    confidence: float   # Detection confidence (0.0-1.0)

@dataclass
class EvolutionTracker:
    """Track how memories evolve over time."""

    llm: LLMProvider
    storage: SimStorage
    seed: int | None = None

    async def detect(
        self,
        new_entity: Entity,
        existing_entities: list[Entity],
    ) -> EvolutionRelation | None:
        """Detect evolution relationship between new and existing entities.

        Args:
            new_entity: Newly extracted entity.
            existing_entities: Related existing entities.

        Returns:
            EvolutionRelation if detected, None otherwise.
        """
```

### Detection Prompt

```
Compare new information with existing memories and determine the relationship.

New: {new_content}

Existing memories:
{existing_list}

What is the relationship?
- "update": New info replaces/corrects old
- "extend": New info adds to old
- "derive": New info is conclusion from old
- "contradict": New info conflicts with old
- "none": No significant relationship

Return JSON: {"type": "...", "reason": "...", "related_id": "...", "confidence": 0.0-1.0}
```

### SimLLMProvider Integration

Already implemented in Phase 2:
- `_sim_evolution_detection()` generates deterministic evolution types
- Routes prompts containing "detect" and "evolution"
- Returns JSON with type, reason, related_id, confidence

### TigerStyle Compliance

```python
async def detect(
    self,
    new_entity: Entity,
    existing_entities: list[Entity],
) -> EvolutionRelation | None:
    # Preconditions
    assert new_entity, "new_entity must not be None"
    assert new_entity.id, "new_entity must have id"
    assert isinstance(existing_entities, list), "existing_entities must be list"

    # ... detection logic ...

    # Postcondition
    if result is not None:
        assert result.evolution_type in EVOLUTION_TYPES, "invalid evolution type"
        assert 0.0 <= result.confidence <= 1.0, "invalid confidence"

    return result
```

## Consequences

### Positive

- **Memory coherence**: Understand how memories relate over time
- **Contradiction detection**: Alert when new info conflicts
- **Knowledge evolution**: Track how understanding develops
- **Deterministic**: Same seed produces same detections

### Negative

- **LLM dependency**: Detection quality depends on LLM
- **Latency**: Additional LLM call when related entities exist

### Mitigations

1. **Graceful degradation**: Return None if detection fails
2. **Confidence threshold**: Skip low-confidence detections
3. **Optional integration**: Can disable in Memory.remember()

## Implementation

### Phase 6 Files

```
umi/umi-python/umi/
├── evolution.py           # EvolutionTracker class (this ADR)
├── memory.py              # Update remember() to optionally track evolution
└── tests/
    └── test_evolution.py  # EvolutionTracker tests
```

### Integration with Memory

```python
class Memory:
    def __post_init__(self):
        # ... existing init ...
        self._evolution = EvolutionTracker(
            llm=self._llm,
            storage=self._storage,
            seed=self.seed,
        )

    async def remember(
        self,
        text: str,
        ...,
        track_evolution: bool = True,
    ) -> list[Entity]:
        # Extract entities
        result = await self._extractor.extract(text)

        stored = []
        for extracted in result.entities:
            entity = Entity(...)

            if track_evolution:
                # Find related entities
                existing = await self._storage.search(entity.name, limit=5)
                if existing:
                    evolution = await self._evolution.detect(entity, existing)
                    if evolution:
                        # Store the evolution relation
                        # (design decision: inline metadata vs separate table)
                        entity.metadata["evolution"] = {
                            "type": evolution.evolution_type,
                            "related_id": evolution.source_id,
                            "reason": evolution.reason,
                        }

            stored_entity = await self._storage.store(entity)
            stored.append(stored_entity)
```

## References

- ADR-007: SimLLMProvider
- ADR-008: Memory Class
- ADR-010: Entity Extraction
- memU paper: Memory evolution concepts
