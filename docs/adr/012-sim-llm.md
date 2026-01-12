# ADR-012: SimLLM - Deterministic LLM Simulation for DST

## Status

Accepted

## Context

Umi is being ported to full Rust with simulation-first architecture. The Python layer
has `SimLLMProvider` for deterministic LLM simulation during testing. The Rust DST
harness needs equivalent functionality.

### Requirements

1. **Deterministic**: Same seed produces identical responses
2. **Prompt routing**: Different response generators for entity extraction, query rewriting, evolution detection
3. **Fault injection**: Integrate with existing `FaultInjector` (LLM fault types already exist)
4. **TigerStyle**: Preconditions, postconditions, explicit limits
5. **Shareable**: Can be wrapped in `Arc` like other DST components

### Existing Infrastructure

The DST harness already provides:
- `DeterministicRng` - Seeded random number generator
- `FaultInjector` - Fault injection with LLM fault types (`LlmTimeout`, `LlmRateLimit`, etc.)
- `SimClock` - Simulated time
- `SimEnvironment` - Container for simulation resources

## Decision

Add `SimLLM` to the DST module following existing patterns:

### Architecture

```
SimEnvironment
├── config: SimConfig
├── clock: SimClock
├── rng: DeterministicRng
├── faults: Arc<FaultInjector>
├── storage: SimStorage
├── network: SimNetwork
└── llm: SimLLM  ← NEW
```

### SimLLM Design

```rust
pub struct SimLLM {
    rng: RefCell<DeterministicRng>,  // Interior mutability for &self methods
    fault_injector: Arc<FaultInjector>,
    clock: SimClock,
}

impl SimLLM {
    /// Complete a prompt with deterministic response.
    pub async fn complete(&self, prompt: &str) -> Result<String, LLMError>;

    /// Complete a prompt expecting JSON response.
    pub async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T, LLMError>;
}
```

### Prompt Routing

Routes prompts to domain-specific generators based on content:

| Pattern | Generator | Output |
|---------|-----------|--------|
| "extract" + "entit" | `sim_entity_extraction` | JSON with entities/relations |
| "rewrite" + "query" | `sim_query_rewrite` | JSON array of query variations |
| "detect" + "evolution" | `sim_evolution_detection` | JSON with type/reason/confidence |
| default | `sim_generic` | Acknowledgment string |

### Response Generation Strategy

1. **Hash-based variation**: Use SHA256 hash of prompt for deterministic variation
2. **Pattern matching**: Extract names/orgs from prompt text for realistic responses
3. **Weighted selection**: Use RNG with weights for evolution types

Example entity extraction:
```rust
fn sim_entity_extraction(&self, prompt: &str) -> String {
    let mut entities = Vec::new();

    // Detect common names in prompt
    for name in COMMON_NAMES {
        if prompt.to_uppercase().contains(&name.to_uppercase()) {
            entities.push(json!({
                "name": name,
                "type": "person",
                "content": format!("Extracted from: {}...", &prompt[..50.min(prompt.len())]),
                "confidence": 0.7 + self.rng.borrow_mut().next_float() * 0.3,
            }));
        }
    }

    // Fallback to note entity if nothing found
    if entities.is_empty() {
        let hash = self.prompt_hash(prompt);
        entities.push(json!({
            "name": format!("Note_{}", hash),
            "type": "note",
            "content": &prompt[..100.min(prompt.len())],
            "confidence": 0.5,
        }));
    }

    serde_json::to_string(&json!({
        "entities": entities,
        "relations": [],
    })).unwrap()
}
```

### Fault Injection Integration

Uses existing fault types from `fault.rs`:
- `FaultType::LlmTimeout` → `LLMError::Timeout`
- `FaultType::LlmRateLimit` → `LLMError::RateLimit`
- `FaultType::LlmContextOverflow` → `LLMError::ContextOverflow`
- `FaultType::LlmInvalidResponse` → `LLMError::InvalidResponse`
- `FaultType::LlmServiceUnavailable` → `LLMError::ServiceUnavailable`

### Error Types

```rust
#[derive(Debug, Clone, thiserror::Error)]
pub enum LLMError {
    #[error("LLM request timed out")]
    Timeout,

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Context length exceeded: {0} tokens")]
    ContextOverflow(usize),

    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    #[error("Service unavailable")]
    ServiceUnavailable,

    #[error("JSON parse error: {0}")]
    JsonError(String),
}
```

### TigerStyle Compliance

```rust
pub async fn complete(&self, prompt: &str) -> Result<String, LLMError> {
    // Preconditions
    debug_assert!(!prompt.is_empty(), "prompt must not be empty");
    debug_assert!(prompt.len() <= LLM_PROMPT_BYTES_MAX, "prompt exceeds limit");

    // Check for faults
    if let Some(fault) = self.fault_injector.should_inject("llm_complete") {
        return Err(self.fault_to_error(fault));
    }

    // Simulate latency
    self.simulate_latency().await;

    // Route and generate
    let response = self.route_prompt(prompt);

    // Postcondition
    debug_assert!(!response.is_empty(), "response must not be empty");
    Ok(response)
}
```

### Constants to Add

```rust
// LLM Simulation Limits
pub const LLM_PROMPT_BYTES_MAX: usize = 100_000;      // 100KB
pub const LLM_RESPONSE_BYTES_MAX: usize = 50_000;     // 50KB
pub const LLM_LATENCY_MS_MIN: u64 = 50;               // Minimum simulated latency
pub const LLM_LATENCY_MS_MAX: u64 = 2000;             // Maximum simulated latency
pub const LLM_LATENCY_MS_DEFAULT: u64 = 100;          // Default simulated latency
```

## Consequences

### Positive

- **Full DST coverage**: LLM calls are now deterministic and testable
- **Fault injection**: Can test LLM failure handling with existing infrastructure
- **Consistent patterns**: Follows established DST module conventions
- **No external deps**: SimLLM has no network dependencies

### Negative

- **Simulation fidelity**: Responses don't match real LLM quality
- **Maintenance**: Must keep response generators updated with real usage patterns

### Mitigations

1. **Golden tests**: Compare SimLLM outputs with real LLM outputs for key scenarios
2. **Pattern coverage**: Ensure all prompt patterns used in production have sim handlers

## Implementation

### Files to Create/Modify

```
umi-core/src/
├── constants.rs          # Add LLM_* constants
├── dst/
│   ├── mod.rs            # Export SimLLM, LLMError
│   ├── llm.rs            # NEW: SimLLM implementation
│   └── simulation.rs     # Add SimLLM to SimEnvironment
└── lib.rs                # Re-export LLMError
```

### Test Coverage

1. **Determinism**: Same seed + same prompt = same response
2. **Prompt routing**: Each pattern routes to correct generator
3. **Fault injection**: All LLM fault types produce correct errors
4. **JSON parsing**: `complete_json` correctly deserializes responses

## References

- Python `SimLLMProvider`: `umi/providers/sim.py`
- ADR-007: SimLLMProvider (Python design)
- ADR-009: Dual Retrieval (uses query rewrite)
- ADR-010: Entity Extraction (uses extraction prompts)
- ADR-011: Evolution Tracking (uses evolution detection)
