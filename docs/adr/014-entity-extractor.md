# ADR-014: EntityExtractor - Rust Port

## Status

Accepted

## Context

Porting the Python `EntityExtractor` to Rust as part of the full Rust migration.
The extractor uses LLM to identify named entities and relations from unstructured text.

### Requirements

1. **Simulation-first**: Works with `SimLLMProvider` for deterministic testing
2. **Generic over provider**: `EntityExtractor<P: LLMProvider>`
3. **Graceful degradation**: Returns fallback on LLM failure
4. **TigerStyle**: Preconditions, postconditions, explicit limits
5. **Parity with Python**: Same entity types, relation types, prompt format

### Python Reference

The Python implementation provides:
- `ExtractedEntity` with name, type, content, confidence
- `ExtractedRelation` with source, target, type, confidence
- `ExtractionResult` containing entities, relations, raw_text
- Prompt template for LLM extraction
- Fallback to "note" entity on parse failure

## Decision

Create an `extraction` module with types and extractor following Rust idioms.

### Architecture

```
umi-core/src/extraction/
├── mod.rs       # EntityExtractor + re-exports
├── types.rs     # EntityType, RelationType, Extracted*, ExtractionResult
├── prompts.rs   # Prompt template
└── error.rs     # ExtractionError
```

### Type Design

Use Rust enums for type safety:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Project,
    Topic,
    Preference,
    Task,
    Event,
    Note,  // Fallback
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    WorksAt,
    Knows,
    Manages,
    RelatesTo,
    Prefers,
    PartOf,
}
```

### EntityExtractor

Generic over provider for sim/production flexibility:

```rust
pub struct EntityExtractor<P: LLMProvider> {
    provider: P,
}

impl<P: LLMProvider> EntityExtractor<P> {
    pub fn new(provider: P) -> Self;

    pub async fn extract(
        &self,
        text: &str,
        options: ExtractionOptions,
    ) -> Result<ExtractionResult, ExtractionError>;

    pub async fn extract_entities_only(
        &self,
        text: &str,
    ) -> Result<Vec<ExtractedEntity>, ExtractionError>;
}
```

### ExtractionOptions

Builder pattern for optional parameters:

```rust
pub struct ExtractionOptions {
    pub existing_entities: Vec<String>,
    pub min_confidence: f64,
}

impl Default for ExtractionOptions {
    fn default() -> Self {
        Self {
            existing_entities: Vec::new(),
            min_confidence: 0.0,
        }
    }
}
```

### Graceful Degradation

On LLM failure or parse error:
1. Catch `ProviderError` from LLM call
2. Create fallback "note" entity from input text
3. Return success with fallback (not error)

This matches Python behavior and ensures memory operations don't fail
just because LLM is unavailable.

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("Text is empty")]
    EmptyText,

    #[error("Text too long: {len} bytes (max {max})")]
    TextTooLong { len: usize, max: usize },

    #[error("Invalid confidence: {value} (must be 0.0-1.0)")]
    InvalidConfidence { value: f64 },
}
```

Note: LLM errors are NOT propagated - we degrade gracefully instead.

### Prompt Template

Same structure as Python for compatibility:

```
Extract entities and relationships from this text.

Text: {text}

{context_section}

Return JSON with this exact structure:
{
  "entities": [...],
  "relations": [...]
}
...
```

### Constants

Add to `constants.rs`:

```rust
pub const EXTRACTION_TEXT_BYTES_MAX: usize = 100_000;
pub const EXTRACTION_ENTITIES_MAX: usize = 50;
pub const EXTRACTION_RELATIONS_MAX: usize = 100;
pub const EXTRACTION_CONFIDENCE_MIN: f64 = 0.0;
pub const EXTRACTION_CONFIDENCE_MAX: f64 = 1.0;
pub const EXTRACTION_CONFIDENCE_DEFAULT: f64 = 0.5;
```

## Consequences

### Positive

- **Type-safe**: Rust enums prevent invalid entity/relation types
- **Testable**: Works with SimLLMProvider for deterministic tests
- **Resilient**: Graceful degradation on LLM failure
- **Consistent**: Same behavior as Python implementation

### Negative

- **JSON parsing**: Need robust handling of malformed LLM output
- **String matching**: Entity type inference from strings is imprecise

### Mitigations

1. Use `serde` with default values for missing fields
2. Fall back to "note" type for unknown entity types
3. Comprehensive tests with various malformed inputs

## References

- Python `umi/extraction.py`
- ADR-010: Entity Extraction (Python design)
- ADR-013: LLM Provider Trait
