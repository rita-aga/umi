# ADR-016: EvolutionTracker - Rust Implementation

## Status

Accepted

## Context

ADR-011 established the Python EvolutionTracker. As part of Phase R1 (porting Python layer to Rust), we need a Rust implementation that:

1. Uses existing types from `storage/evolution.rs` (`EvolutionType`, `EvolutionRelation`)
2. Is generic over `LLMProvider` and `StorageBackend` traits
3. Follows simulation-first design (SimLLMProvider always available)
4. Implements graceful degradation on LLM failures

### Existing Rust Types

The storage module already defines:

```rust
pub enum EvolutionType { Update, Extend, Derive, Contradict }

pub struct EvolutionRelation {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub evolution_type: EvolutionType,
    pub reason: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}
```

### Requirements

1. **Generic design**: `EvolutionTracker<L: LLMProvider, S: StorageBackend>`
2. **Sim-first**: All tests use `SimLLMProvider`, real providers feature-gated
3. **Graceful degradation**: Return `None` on LLM failure, not crash
4. **TigerStyle**: Preconditions, postconditions, explicit limits
5. **Confidence threshold**: Filter low-confidence detections

## Decision

Create `umi-core/src/evolution/` module with:

### Module Structure

```
umi-core/src/evolution/
├── mod.rs          # EvolutionTracker, DetectionOptions, DetectionResult, EvolutionError
└── prompts.rs      # Detection prompt templates
```

### Core Types

```rust
/// Options for evolution detection.
pub struct DetectionOptions {
    /// Minimum confidence threshold (default: 0.3)
    pub min_confidence: f32,
    /// Maximum existing entities to compare (default: 10)
    pub max_comparisons: usize,
}

/// Result of evolution detection.
pub struct DetectionResult {
    /// The detected evolution relation
    pub relation: EvolutionRelation,
    /// Whether LLM was used (vs fallback)
    pub llm_used: bool,
}

/// Evolution detection errors.
pub enum EvolutionError {
    /// Invalid detection options
    InvalidOptions(String),
}
```

### EvolutionTracker Design

```rust
pub struct EvolutionTracker<L: LLMProvider, S: StorageBackend> {
    llm: L,
    storage: S,
}

impl<L: LLMProvider, S: StorageBackend> EvolutionTracker<L, S> {
    /// Detect evolution relationship between new and existing entities.
    pub async fn detect(
        &self,
        new_entity: &Entity,
        existing_entities: &[Entity],
        options: DetectionOptions,
    ) -> Result<Option<DetectionResult>, EvolutionError>;

    /// Find related entities and detect evolution in one call.
    pub async fn find_and_detect(
        &self,
        new_entity: &Entity,
        options: DetectionOptions,
    ) -> Result<Option<DetectionResult>, EvolutionError>;

    /// Parse LLM response into EvolutionRelation.
    fn parse_response(
        &self,
        response: &str,
        new_entity_id: &str,
    ) -> Option<EvolutionRelation>;
}
```

### Detection Prompt

```rust
pub const EVOLUTION_DETECTION_PROMPT: &str = r#"Compare new information with existing memories and determine the relationship.

New information:
{new_content}

Existing memories:
{existing_list}

What is the relationship between the new information and existing memories?
- "update": New info replaces/corrects old (e.g., changed job, moved address)
- "extend": New info adds to old (e.g., more details, clarification)
- "derive": New info is conclusion from old (e.g., inference, deduction)
- "contradict": New info conflicts with old (e.g., disagreement, correction)
- "none": No significant relationship

Return JSON with this exact structure:
{"type": "update|extend|derive|contradict|none", "reason": "brief explanation", "related_id": "id of most related memory or null", "confidence": 0.0-1.0}

Only return the JSON, nothing else."#;
```

### SimLLM Routing

The `SimLLMProvider` already routes evolution prompts via `_sim_evolution_detection()`.
Prompts containing both "evolution" and "existing memories" trigger this route.

### Graceful Degradation

```rust
pub async fn detect(&self, ...) -> Result<Option<DetectionResult>, EvolutionError> {
    // Preconditions
    debug_assert!(!new_entity.id.is_empty(), "new_entity must have id");

    if existing_entities.is_empty() {
        return Ok(None);
    }

    // Build and send prompt
    let prompt = build_detection_prompt(new_entity, existing_entities);

    // Graceful degradation: return None on LLM failure
    let response = match self.llm.complete(CompletionRequest::new(&prompt)).await {
        Ok(resp) => resp,
        Err(_) => return Ok(None),  // LLM failure → None, not error
    };

    // Parse response
    let relation = match self.parse_response(&response.content, &new_entity.id) {
        Some(r) => r,
        None => return Ok(None),  // Parse failure → None
    };

    // Apply confidence threshold
    if relation.confidence < options.min_confidence {
        return Ok(None);
    }

    Ok(Some(DetectionResult {
        relation,
        llm_used: true,
    }))
}
```

### TigerStyle Compliance

1. **Preconditions**: Validate entity IDs, confidence ranges
2. **Postconditions**: Ensure returned type is valid, confidence in range
3. **Explicit limits**: All constants in `constants.rs`
4. **Debug assertions**: Catch bugs in tests without production overhead

## Constants

Add to `constants.rs`:

```rust
/// Maximum existing entities to compare against
pub const EVOLUTION_EXISTING_ENTITIES_COUNT_MAX: usize = 10;

/// Default confidence threshold for evolution detection
pub const EVOLUTION_CONFIDENCE_THRESHOLD_DEFAULT: f64 = 0.3;
```

## Testing Strategy

All tests use `SimLLMProvider`:

```rust
#[tokio::test]
async fn test_detect_update() {
    let llm = SimLLMProvider::new(SimConfig::with_seed(42));
    let storage = SimStorageBackend::new(SimConfig::with_seed(42));
    let tracker = EvolutionTracker::new(llm, storage);

    let old_entity = Entity::new(EntityType::Person, "Alice", "Works at Acme");
    let new_entity = Entity::new(EntityType::Person, "Alice", "Left Acme, now at StartupX");

    let result = tracker.detect(&new_entity, &[old_entity], DetectionOptions::default()).await?;

    assert!(result.is_some());
    let detection = result.unwrap();
    assert_eq!(detection.relation.evolution_type, EvolutionType::Update);
}
```

## Consequences

### Positive

- Consistent with EntityExtractor and DualRetriever patterns
- Uses existing EvolutionType and EvolutionRelation from storage
- Full DST support via SimLLMProvider
- Type-safe, no runtime type errors

### Negative

- Additional module to maintain
- LLM call adds latency when detecting evolution

### Mitigations

- Graceful degradation means failures don't crash
- Confidence threshold reduces noise
- Optional: skip evolution detection in performance-critical paths

## References

- ADR-011: Evolution Tracking (Python)
- ADR-013: LLM Provider Trait
- ADR-014: EntityExtractor
- ADR-015: DualRetriever
- `storage/evolution.rs`: Existing EvolutionType and EvolutionRelation
