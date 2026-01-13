# Plan: memU Parity Features for Umi

**Created:** 2026-01-13
**Status:** Planning
**Estimated Phases:** 4

## Overview

Based on analysis of memU's actual source code vs Umi's implementation, this plan adds features to achieve parity and address gaps identified in session handoff, rendering, and query intelligence.

## Features

1. **Markdown Rendering** - Core Memory can render as XML or Markdown
2. **Pronoun Resolution** - Query rewriting resolves pronouns from conversation context
3. **Semantic Category Routing** - LLM decides initial block type (like memU categories)
4. **Working Memory Persistence** - Session state survives restarts and supports handoff

---

## Phase 1: Markdown Rendering for Core Memory ✅ COMPLETE

**Goal:** Add `render_markdown()` method to CoreMemory alongside existing `render()` (XML).

**Status:** Complete (2026-01-13)
**ADR:** docs/adr/020-markdown-rendering.md

### Files to Modify

- `umi-memory/src/memory/core.rs`
- `umi-memory/src/memory/block.rs`

### Implementation

```rust
// core.rs - Add new method
impl CoreMemory {
    /// Render core memory as Markdown for human display.
    ///
    /// # Example Output
    /// ```markdown
    /// # Core Memory
    ///
    /// ## System (importance: 0.95)
    /// You are a helpful assistant.
    ///
    /// ## Human (importance: 0.75)
    /// User: Alice, software engineer
    /// ```
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut output = String::with_capacity(self.current_bytes + 256);
        output.push_str("# Core Memory\n\n");

        for block in self.blocks_ordered() {
            output.push_str(&block.render_markdown());
            output.push_str("\n\n");
        }

        output
    }
}

// block.rs - Add new method
impl MemoryBlock {
    /// Render block as Markdown.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let type_name = self.block_type.as_str();
        let header = match &self.label {
            Some(label) => format!("## {} - {} (importance: {:.2})",
                capitalize(type_name), label, self.importance),
            None => format!("## {} (importance: {:.2})",
                capitalize(type_name), self.importance),
        };
        format!("{}\n{}", header, self.content)
    }
}
```

### Tests (DST-First)

```rust
#[test]
fn test_render_markdown_empty() {
    let core = CoreMemory::new();
    let md = core.render_markdown();
    assert_eq!(md, "# Core Memory\n\n");
}

#[test]
fn test_render_markdown_with_blocks() {
    let mut core = CoreMemory::new();
    core.set_block(MemoryBlockType::System, "Be helpful.").unwrap();
    core.set_block(MemoryBlockType::Human, "User: Alice").unwrap();

    let md = core.render_markdown();
    assert!(md.contains("# Core Memory"));
    assert!(md.contains("## System"));
    assert!(md.contains("Be helpful."));
    assert!(md.contains("## Human"));
    assert!(md.contains("User: Alice"));
}

#[test]
fn test_render_markdown_order_matches_xml() {
    let mut core = CoreMemory::new();
    core.set_block(MemoryBlockType::Scratch, "5").unwrap();
    core.set_block(MemoryBlockType::System, "1").unwrap();

    let md = core.render_markdown();
    let xml = core.render();

    // Both should have System before Scratch
    let md_sys = md.find("System").unwrap();
    let md_scratch = md.find("Scratch").unwrap();
    assert!(md_sys < md_scratch);
}
```

### Acceptance Criteria

- [x] `render_markdown()` method exists on CoreMemory
- [x] `render_markdown()` method exists on MemoryBlock
- [x] Output follows Markdown format with headers
- [x] Block order matches XML render order
- [x] Importance is displayed
- [x] Labels are displayed when present
- [x] All tests pass (11 tests: 8 unit + 3 DST)

---

## Phase 2: Pronoun Resolution in Query Rewriting

**Goal:** Resolve pronouns ("he", "she", "they", "it") using conversation context before generating query variations.

### Files to Modify

- `umi-memory/src/retrieval/mod.rs`
- `umi-memory/src/retrieval/prompts.rs`
- `umi-memory/src/retrieval/types.rs`

### Design

```rust
// types.rs - Add conversation context to SearchOptions
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub limit: usize,
    pub deep_search: bool,
    pub time_range: Option<(u64, u64)>,
    pub conversation_context: Option<Vec<ConversationTurn>>,  // NEW
}

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub role: String,      // "user" or "assistant"
    pub content: String,
}

// prompts.rs - New prompt for pronoun resolution
pub const PRONOUN_RESOLUTION_PROMPT: &str = r#"Given the conversation context below, resolve any pronouns in the query to their referents.

Conversation:
{conversation}

Query: {query}

If the query contains pronouns (he, she, they, it, this, that, etc.) that refer to entities mentioned in the conversation, rewrite the query with the pronouns replaced by the actual entity names.

If no pronouns need resolution, return the original query unchanged.

Return ONLY the resolved query, nothing else."#;
```

### Implementation Flow

```
┌─────────────────────────────────────────────────────────┐
│  recall("Where does she work?", options_with_context)   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  1. Pronoun Resolution   │
              │  LLM: "she" → "Alice"   │
              │  Result: "Where does    │
              │  Alice work?"           │
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  2. Query Rewriting      │
              │  Generate variations:    │
              │  - "Alice workplace"     │
              │  - "Alice employer"      │
              │  - "Alice company"       │
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  3. Vector Search        │
              │  Search all variations   │
              │  Merge with RRF          │
              └─────────────────────────┘
```

### Tests (DST-First)

```rust
#[tokio::test]
async fn test_pronoun_resolution_basic() {
    let retriever = create_test_retriever(42);

    let context = vec![
        ConversationTurn { role: "user".into(), content: "Tell me about Alice".into() },
        ConversationTurn { role: "assistant".into(), content: "Alice is a software engineer at Acme Corp".into() },
    ];

    let resolved = retriever.resolve_pronouns("Where does she work?", &context).await;

    // Should resolve "she" to "Alice"
    assert!(resolved.contains("Alice"));
    assert!(!resolved.contains("she"));
}

#[tokio::test]
async fn test_pronoun_resolution_no_pronouns() {
    let retriever = create_test_retriever(42);

    let context = vec![];
    let query = "What is the capital of France?";

    let resolved = retriever.resolve_pronouns(query, &context).await;

    // Should return unchanged
    assert_eq!(resolved, query);
}

#[tokio::test]
async fn test_search_with_context() {
    let retriever = create_test_retriever(42);

    let options = SearchOptions::default()
        .with_conversation_context(vec![
            ConversationTurn { role: "user".into(), content: "Tell me about Bob".into() },
        ]);

    let result = retriever.search("What does he do?", options).await.unwrap();

    // Should have searched for Bob-related queries
    assert!(!result.entities.is_empty());
}
```

### Acceptance Criteria

- [ ] `SearchOptions` accepts conversation context
- [ ] `resolve_pronouns()` method on DualRetriever
- [ ] Pronoun resolution runs before query rewriting
- [ ] Graceful degradation if resolution fails
- [ ] SimLLMProvider handles pronoun resolution prompts
- [ ] All tests pass

---

## Phase 3: Semantic Category Routing

**Goal:** LLM decides initial block type for entities (like memU's category system).

### Files to Modify

- `umi-memory/src/extraction/mod.rs`
- `umi-memory/src/extraction/prompts.rs`
- `umi-memory/src/orchestration/unified.rs`

### Design

Currently, entity type → block type mapping is hardcoded:

```rust
// Current: Hardcoded mapping
fn entity_type_to_block_type(entity_type: &EntityType) -> MemoryBlockType {
    match entity_type {
        EntityType::Self_ => MemoryBlockType::Persona,
        EntityType::Person => MemoryBlockType::Human,
        EntityType::Project => MemoryBlockType::Facts,
        // ...
    }
}
```

**New: LLM-assisted routing with context:**

```rust
// New: LLM decides based on content and context
pub struct RoutingDecision {
    pub entity_id: String,
    pub suggested_block: MemoryBlockType,
    pub confidence: f64,
    pub reason: String,
    pub should_promote_immediately: bool,  // e.g., Self_ info always in Core
}

impl UnifiedMemory {
    /// Route entities to appropriate blocks using LLM.
    async fn route_entities(&self, entities: &[Entity], context: &str) -> Vec<RoutingDecision> {
        let prompt = build_routing_prompt(entities, context);
        let response = self.llm.complete(&prompt).await?;
        parse_routing_decisions(&response)
    }
}
```

### Routing Prompt

```rust
pub const ROUTING_PROMPT: &str = r#"You are categorizing memory items into storage blocks.

AVAILABLE BLOCKS:
- Persona: Information about the AI assistant itself (identity, capabilities)
- Human: Information about users/people the AI interacts with
- Facts: General knowledge, project details, technical information
- Goals: Tasks, objectives, things to accomplish
- Scratch: Temporary notes, working information

ENTITIES TO ROUTE:
{entities}

CONVERSATION CONTEXT:
{context}

For each entity, decide:
1. Which block it belongs in
2. Whether it should be immediately promoted to Core Memory (always in context)
3. Confidence (0.0-1.0)

Rules:
- Self/identity information → Persona, always promote
- User preferences/profile → Human, promote if frequently accessed
- Project/technical details → Facts
- Tasks/todos → Goals
- Temporary/session-specific → Scratch, don't promote

Return JSON:
[
  {"entity_id": "...", "block": "Human", "promote": true, "confidence": 0.95, "reason": "User profile info"},
  ...
]"#;
```

### Implementation

```rust
impl UnifiedMemory {
    pub async fn remember(&mut self, text: &str) -> UnifiedMemoryResult<UnifiedRememberResult> {
        // 1. Extract entities
        let extracted = self.extractor.extract(text, options).await?;

        // 2. NEW: Route entities using LLM
        let routing = if self.config.semantic_routing {
            self.route_entities(&extracted, text).await?
        } else {
            // Fallback to hardcoded mapping
            extracted.iter().map(|e| RoutingDecision::from_entity_type(e)).collect()
        };

        // 3. Store in archival
        for entity in extracted {
            self.storage.store_entity(&entity).await?;
        }

        // 4. Promote based on routing decisions
        for decision in routing.iter().filter(|d| d.should_promote_immediately) {
            self.promote_entity_to_core(&decision.entity_id, decision.suggested_block).await?;
        }

        // 5. Access-based promotion for the rest
        // ... existing logic
    }
}
```

### Tests (DST-First)

```rust
#[tokio::test]
async fn test_routing_self_info_to_persona() {
    let mut memory = create_unified_memory(42);
    memory.config_mut().semantic_routing = true;

    let result = memory.remember("My name is Umi and I help with memory management").await.unwrap();

    // Self info should route to Persona and be immediately promoted
    assert!(memory.core().get_block(MemoryBlockType::Persona).is_some());
}

#[tokio::test]
async fn test_routing_user_info_to_human() {
    let mut memory = create_unified_memory(42);
    memory.config_mut().semantic_routing = true;

    let result = memory.remember("Alice is a software engineer who prefers dark mode").await.unwrap();

    // User preference should route to Human
    let routing = memory.last_routing_decisions();
    assert!(routing.iter().any(|r| r.suggested_block == MemoryBlockType::Human));
}

#[tokio::test]
async fn test_routing_disabled_uses_hardcoded() {
    let mut memory = create_unified_memory(42);
    memory.config_mut().semantic_routing = false;  // Disabled

    let result = memory.remember("Alice works at Acme").await.unwrap();

    // Should use hardcoded Person → Human mapping
    // (deterministic, DST-friendly)
}
```

### Acceptance Criteria

- [ ] `UnifiedMemoryConfig` has `semantic_routing: bool` option
- [ ] `route_entities()` method uses LLM for routing decisions
- [ ] Routing prompt considers entity content and context
- [ ] Immediate promotion for high-priority entities (Self_, user preferences)
- [ ] Fallback to hardcoded mapping when disabled
- [ ] SimLLMProvider handles routing prompts
- [ ] All tests pass

---

## Phase 4: Working Memory Persistence

**Goal:** Working Memory can be saved/loaded for session continuity.

### Files to Modify

- `umi-memory/src/memory/working.rs`
- `umi-memory/src/orchestration/unified.rs`

### Design

```rust
// working.rs - Add serialization
impl WorkingMemory {
    /// Serialize working memory to bytes for persistence.
    pub fn serialize(&self) -> WorkingMemoryResult<Vec<u8>> {
        let snapshot = WorkingMemorySnapshot {
            entries: self.entries.iter()
                .filter(|(_, e)| e.expires_at_ms > self.clock_ms)  // Only non-expired
                .map(|(k, e)| (k.clone(), SerializedEntry::from(e)))
                .collect(),
            clock_ms: self.clock_ms,
            version: 1,
        };

        serde_json::to_vec(&snapshot)
            .map_err(|e| WorkingMemoryError::SerializationFailed { reason: e.to_string() })
    }

    /// Deserialize working memory from bytes.
    pub fn deserialize(&mut self, data: &[u8]) -> WorkingMemoryResult<()> {
        let snapshot: WorkingMemorySnapshot = serde_json::from_slice(data)
            .map_err(|e| WorkingMemoryError::DeserializationFailed { reason: e.to_string() })?;

        // Clear existing and load
        self.entries.clear();
        self.current_bytes = 0;

        for (key, entry) in snapshot.entries {
            // Adjust TTL based on time elapsed
            let adjusted_expires = entry.expires_at_ms;
            if adjusted_expires > self.clock_ms {
                self.set(&key, &entry.value, Some(adjusted_expires - self.clock_ms))?;
            }
        }

        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct WorkingMemorySnapshot {
    entries: HashMap<String, SerializedEntry>,
    clock_ms: u64,
    version: u32,
}

#[derive(Serialize, Deserialize)]
struct SerializedEntry {
    value: Vec<u8>,
    expires_at_ms: u64,
}
```

### Session Management

```rust
// orchestration/unified.rs - Add session support
impl<L, E, S, V> UnifiedMemory<L, E, S, V> {
    /// Save current session state.
    pub async fn save_session(&self, session_id: &str) -> UnifiedMemoryResult<()> {
        let working_data = self.working.serialize()?;

        // Save to storage backend with special session key
        let session_key = format!("__session__{}", session_id);
        self.storage.set_raw(&session_key, &working_data).await?;

        Ok(())
    }

    /// Load a previous session.
    pub async fn load_session(&mut self, session_id: &str) -> UnifiedMemoryResult<()> {
        let session_key = format!("__session__{}", session_id);

        if let Some(data) = self.storage.get_raw(&session_key).await? {
            self.working.deserialize(&data)?;
        }

        Ok(())
    }

    /// Create a new session and return its ID.
    pub fn create_session(&mut self) -> String {
        let session_id = generate_session_id(self.clock.now_ms());

        // Auto-populate session metadata
        self.working.set("__session_id", session_id.as_bytes(), None).ok();
        self.working.set("__session_start_ms",
            &self.clock.now_ms().to_le_bytes(), None).ok();

        session_id
    }
}
```

### Auto-Population

```rust
impl<L, E, S, V> UnifiedMemory<L, E, S, V> {
    pub async fn remember(&mut self, text: &str) -> UnifiedMemoryResult<UnifiedRememberResult> {
        // ... existing logic ...

        // NEW: Auto-populate working memory with session context
        if self.config.auto_populate_working {
            // Track last remembered text (truncated)
            let truncated = if text.len() > 200 { &text[..200] } else { text };
            self.working.set("__last_input", truncated.as_bytes(), None).ok();

            // Track interaction count
            self.working.incr("__interaction_count", 1).ok();

            // Track last entities (for context)
            let entity_names: Vec<&str> = result.entities.iter()
                .map(|e| e.name.as_str())
                .collect();
            let names_json = serde_json::to_vec(&entity_names).unwrap_or_default();
            self.working.set("__last_entities", &names_json, None).ok();
        }

        Ok(result)
    }

    pub async fn recall(&mut self, query: &str, options: SearchOptions) -> ... {
        // ... existing logic ...

        // NEW: Track last query for pronoun resolution
        if self.config.auto_populate_working {
            self.working.set("__last_query", query.as_bytes(), None).ok();
        }

        Ok(result)
    }
}
```

### Tests (DST-First)

```rust
#[tokio::test]
async fn test_working_memory_serialize_roundtrip() {
    let mut working = WorkingMemory::new();
    working.set_clock_ms(1000);
    working.set("key1", b"value1", Some(5000)).unwrap();
    working.set("key2", b"value2", Some(10000)).unwrap();

    let serialized = working.serialize().unwrap();

    let mut working2 = WorkingMemory::new();
    working2.set_clock_ms(1000);
    working2.deserialize(&serialized).unwrap();

    assert_eq!(working2.get("key1"), Some(b"value1".as_slice()));
    assert_eq!(working2.get("key2"), Some(b"value2".as_slice()));
}

#[tokio::test]
async fn test_session_save_load() {
    let mut memory = create_unified_memory(42);

    // Create session and add data
    let session_id = memory.create_session();
    memory.working_mut().set("topic", b"project deadlines", None).unwrap();
    memory.working_mut().set("user_mood", b"focused", None).unwrap();

    // Save session
    memory.save_session(&session_id).await.unwrap();

    // Create new memory instance (simulating restart)
    let mut memory2 = create_unified_memory(42);

    // Load session
    memory2.load_session(&session_id).await.unwrap();

    assert_eq!(memory2.working().get("topic"), Some(b"project deadlines".as_slice()));
    assert_eq!(memory2.working().get("user_mood"), Some(b"focused".as_slice()));
}

#[tokio::test]
async fn test_auto_populate_working() {
    let mut memory = create_unified_memory(42);
    memory.config_mut().auto_populate_working = true;

    memory.remember("Alice is working on the report").await.unwrap();
    memory.remember("Bob joined the project").await.unwrap();

    // Should have auto-populated metadata
    assert!(memory.working().get("__last_input").is_some());

    let count_bytes = memory.working().get("__interaction_count").unwrap();
    let count = i64::from_le_bytes(count_bytes.try_into().unwrap());
    assert_eq!(count, 2);
}

#[tokio::test]
async fn test_session_handoff() {
    // Agent 1 creates session
    let mut agent1 = create_unified_memory(42);
    let session_id = agent1.create_session();
    agent1.remember("User prefers dark mode").await.unwrap();
    agent1.working_mut().set("current_task", b"UI review", None).unwrap();
    agent1.save_session(&session_id).await.unwrap();

    // Agent 2 picks up session
    let mut agent2 = create_unified_memory(43);  // Different seed
    agent2.load_session(&session_id).await.unwrap();

    // Agent 2 has full context
    assert_eq!(agent2.working().get("current_task"), Some(b"UI review".as_slice()));
}
```

### Acceptance Criteria

- [ ] `serialize()` / `deserialize()` methods on WorkingMemory
- [ ] `save_session()` / `load_session()` on UnifiedMemory
- [ ] `create_session()` generates session ID and populates metadata
- [ ] `auto_populate_working` config option
- [ ] TTL is preserved across serialization (adjusted for elapsed time)
- [ ] Session data stored in storage backend with special prefix
- [ ] All tests pass

---

## Implementation Order

1. **Phase 1: Markdown Rendering** (~2 hours)
   - Smallest scope, no dependencies
   - Good warmup

2. **Phase 2: Pronoun Resolution** (~4 hours)
   - Builds on existing query rewriting
   - Needs SimLLMProvider update

3. **Phase 3: Semantic Category Routing** (~6 hours)
   - Larger LLM integration
   - Needs config option for backward compatibility

4. **Phase 4: Working Memory Persistence** (~6 hours)
   - Most impactful for real-world use
   - Needs storage backend integration

---

## Configuration Summary

New config options to add to `UnifiedMemoryConfig`:

```rust
pub struct UnifiedMemoryConfig {
    // Existing...
    pub auto_promote: bool,
    pub auto_evict: bool,

    // NEW
    pub semantic_routing: bool,        // Phase 3: LLM-based routing
    pub auto_populate_working: bool,   // Phase 4: Auto-populate session context
}
```

---

## DST Considerations

All features must be DST-compatible:

| Feature | DST Strategy |
|---------|--------------|
| Markdown rendering | Deterministic string output |
| Pronoun resolution | SimLLMProvider handles prompt |
| Semantic routing | SimLLMProvider returns consistent decisions |
| Working Memory | SimClock for TTL, deterministic serialization |

---

## Verification Checklist

Before marking complete:

- [ ] All tests pass: `cargo test --all-features`
- [ ] No clippy warnings: `cargo clippy --all-features`
- [ ] Manual verification of each feature
- [ ] DST tests reproduce with same seed
- [ ] Documentation updated
- [ ] CLAUDE.md reflects new features

---

## Notes

- Semantic routing can be disabled for pure DST testing
- Working Memory persistence uses JSON for debuggability
- Pronoun resolution is best-effort (graceful degradation)
- All features maintain TigerStyle principles
