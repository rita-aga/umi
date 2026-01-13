# Replacing Kelpie's Memory with UMI: Architecture Analysis

## The Surprising Discovery

**Kelpie's server doesn't actually use the `kelpie-memory` crate!**

```rust
// What exists in kelpie-memory/
pub struct CoreMemory { ... }
pub struct WorkingMemory { ... }

// What kelpie-server actually uses:
pub struct AppState {
    agents: RwLock<HashMap<String, AgentState>>,  // Just a HashMap!
    messages: RwLock<HashMap<String, Vec<Message>>>,  // Just a Vec!
    archival: RwLock<HashMap<String, Vec<ArchivalEntry>>>,  // Just a Vec!
}

pub struct AgentState {
    pub blocks: Vec<Block>,  // No CoreMemory object, just Vec<Block>!
}
```

The `kelpie-memory` crate exists mainly for **DST testing** and **type definitions**, but the actual server implementation uses raw data structures.

---

## Three Replacement Strategies

### Strategy A: Full Replacement (Most Radical)

Replace Kelpie's entire memory system with UMI as the single source of truth.

#### Architecture:
```
┌─────────────────────────────────────────────────┐
│            Kelpie Agent API                     │
│  (endpoints: /blocks, /messages, /archival)     │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│                UMI Memory                        │
│  ┌───────────────────────────────────────────┐  │
│  │  Core Memory (always in context)          │  │
│  │  - render() returns XML for LLM           │  │
│  │  - Entities with high importance          │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Working Memory (session KV store)        │  │
│  │  - incr/append/touch operations           │  │
│  │  - TTL-based expiration                   │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Archival Memory (unlimited storage)      │  │
│  │  - Entity extraction from conversations   │  │
│  │  - Dual retrieval (vector + keyword)      │  │
│  │  - Evolution tracking                     │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### What Changes:
```rust
// OLD: kelpie-server/src/state.rs
pub struct AppState {
    agents: RwLock<HashMap<String, AgentState>>,
    messages: RwLock<HashMap<String, Vec<Message>>>,
    archival: RwLock<HashMap<String, Vec<ArchivalEntry>>>,
}

// NEW: kelpie-server/src/state.rs
pub struct AppState {
    agents: RwLock<HashMap<String, AgentConfig>>,  // Metadata only
    umi: RwLock<HashMap<String, umi_memory::Memory>>,  // One UMI instance per agent
}
```

#### API Handler Changes:

**Block Operations (Core Memory):**
```rust
// OLD: Direct Vec<Block> manipulation
pub async fn list_blocks(agent_id: &str) -> Result<Vec<Block>> {
    let agents = state.agents.read().await;
    let agent = agents.get(agent_id)?;
    Ok(agent.blocks.clone())
}

// NEW: Query UMI's core memory
pub async fn list_blocks(agent_id: &str) -> Result<Vec<Block>> {
    let umi_instances = state.umi.read().await;
    let umi = umi_instances.get(agent_id)?;

    // Get core memory and convert to Block format
    let core = umi.core_memory();
    let blocks = core.blocks()
        .map(|b| Block {
            id: b.id().to_string(),
            label: b.label().unwrap_or(b.block_type().as_str()).to_string(),
            value: b.content().to_string(),
            limit: None,
        })
        .collect();
    Ok(blocks)
}

// Block updates store as entities
pub async fn update_block(agent_id: &str, block_id: &str, value: String) -> Result<Block> {
    let umi_instances = state.umi.write().await;
    let umi = umi_instances.get_mut(agent_id)?;

    // Update core memory block
    let block_type = find_block_type(block_id)?;
    umi.core_memory_mut().set_block(block_type, &value)?;

    // Also store as entity in archival for history
    let entity = Entity::new(EntityType::Note, "block_update", value);
    umi.remember_entity(entity).await?;

    // Return updated block
    Ok(Block { /* ... */ })
}
```

**Message Operations (Working Memory):**
```rust
// OLD: Direct Vec<Message> storage
pub async fn add_message(agent_id: &str, message: Message) -> Result<Message> {
    let mut messages = state.messages.write().await;
    messages.entry(agent_id).or_default().push(message.clone());
    Ok(message)
}

// NEW: Store in UMI archival + track in working memory
pub async fn add_message(agent_id: &str, message: Message) -> Result<Message> {
    let umi_instances = state.umi.write().await;
    let umi = umi_instances.get_mut(agent_id)?;

    // Store message as entity in archival (for semantic search)
    let entity = Entity::builder(
        EntityType::Note,
        format!("message_{}", message.id),
        message.content.clone()
    )
    .with_metadata("role", message.role.to_string())
    .with_metadata("timestamp", message.created_at.to_string())
    .build();

    umi.remember_entity(entity).await?;

    // Track message count in working memory
    umi.working_memory_mut().incr("message_count", 1)?;

    Ok(message)
}

// Retrieve messages with semantic search
pub async fn list_messages(
    agent_id: &str,
    limit: usize,
    query: Option<&str>
) -> Result<Vec<Message>> {
    let umi_instances = state.umi.read().await;
    let umi = umi_instances.get(agent_id)?;

    // Use UMI's dual retrieval instead of Vec lookup
    let options = RecallOptions::new()
        .with_limit(limit)
        .with_entity_type(EntityType::Note);

    let entities = if let Some(q) = query {
        umi.recall(q, options).await?
    } else {
        umi.recent_entities(EntityType::Note, limit).await?
    };

    // Convert entities back to messages
    let messages = entities.into_iter()
        .map(|e| Message {
            id: e.id,
            content: e.content,
            role: e.get_metadata("role").unwrap_or("user"),
            // ...
        })
        .collect();

    Ok(messages)
}
```

**Archival Operations:**
```rust
// OLD: Direct Vec<ArchivalEntry> storage
pub async fn add_archival(agent_id: &str, content: String) -> Result<ArchivalEntry> {
    let mut archival = state.archival.write().await;
    let entry = ArchivalEntry::new(content);
    archival.entry(agent_id).or_default().push(entry.clone());
    Ok(entry)
}

// NEW: Use UMI's remember() - gets entity extraction for free!
pub async fn add_archival(agent_id: &str, content: String) -> Result<ArchivalEntry> {
    let umi_instances = state.umi.write().await;
    let umi = umi_instances.get_mut(agent_id)?;

    // UMI automatically extracts entities, creates embeddings, etc.
    let result = umi.remember(&content).await?;

    // Return first entity as archival entry
    let entry = ArchivalEntry {
        id: result.entities[0].id.clone(),
        content,
        created_at: result.entities[0].created_at,
    };

    Ok(entry)
}

// Search uses UMI's dual retrieval (vector + keyword)
pub async fn search_archival(
    agent_id: &str,
    query: &str,
    limit: usize
) -> Result<Vec<ArchivalEntry>> {
    let umi_instances = state.umi.read().await;
    let umi = umi_instances.get(agent_id)?;

    // UMI's dual retrieval is way better than Vec iteration!
    let options = RecallOptions::new().with_limit(limit);
    let entities = umi.recall(query, options).await?;

    let entries = entities.into_iter()
        .map(|e| ArchivalEntry {
            id: e.id,
            content: e.content,
            created_at: e.created_at,
        })
        .collect();

    Ok(entries)
}
```

**System Prompt Building:**
```rust
// OLD: Manual XML building from blocks
fn build_system_prompt(system: &Option<String>, blocks: &[Block]) -> String {
    let mut parts = Vec::new();
    if let Some(sys) = system {
        parts.push(sys.clone());
    }
    if !blocks.is_empty() {
        parts.push("\n\n<memory>".to_string());
        for block in blocks {
            parts.push(format!("<{}>\n{}\n</{}>", block.label, block.value, block.label));
        }
        parts.push("</memory>".to_string());
    }
    parts.join("\n")
}

// NEW: Use UMI's render()
fn build_system_prompt(system: &Option<String>, umi: &umi_memory::Memory) -> String {
    let mut parts = Vec::new();
    if let Some(sys) = system {
        parts.push(sys.clone());
    }

    // UMI renders core memory with importance, sorted by priority
    let core_xml = umi.core_memory().render();
    parts.push(core_xml);

    parts.join("\n\n")
}
```

#### What We Gain:

1. **Entity Extraction** - Automatic extraction of people, topics, projects from conversations
2. **Semantic Search** - Dual retrieval (vector + keyword) instead of linear Vec scan
3. **Evolution Tracking** - Automatic relationship detection between entities
4. **Better Memory Management** - Importance-based promotion to core memory
5. **DST Testing** - Deterministic simulation for all memory operations
6. **Unified Storage** - Single system instead of separate blocks/messages/archival

#### What We Lose:

1. **Block-based API** - Need compatibility layer to preserve API contracts
2. **Simple Vec Storage** - More complex backend (but much more capable)
3. **Direct Access** - Must go through UMI's API instead of direct HashMap lookups

#### Migration Path:

1. **Phase 1**: Add UMI alongside existing storage (dual-write)
2. **Phase 2**: Migrate read operations to UMI one endpoint at a time
3. **Phase 3**: Stop writing to old storage, verify everything works
4. **Phase 4**: Remove old storage code
5. **Phase 5**: Expose new UMI-only features (entity queries, evolution tracking)

---

### Strategy B: Hybrid Approach (Conservative)

Keep Kelpie's API and blocks structure, but use UMI as the **backend storage**.

#### Architecture:
```
┌─────────────────────────────────────────────────┐
│         Kelpie API (unchanged)                   │
│    /blocks, /messages, /archival                 │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│          Kelpie Memory Adapter                   │
│  (translates blocks ↔ entities)                  │
│  - Block CRUD → Entity operations                │
│  - Messages → Entity storage                     │
│  - Archival → UMI archival                       │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│              UMI Memory (backend)                │
│  - Entity extraction                             │
│  - Dual retrieval                                │
│  - Evolution tracking                            │
│  - Smart core memory management                  │
└─────────────────────────────────────────────────┘
```

#### Implementation:
```rust
// kelpie-server/src/memory_adapter.rs
pub struct UmiMemoryAdapter {
    umi: umi_memory::Memory,
    // Cache layer to maintain Block IDs and labels
    block_metadata: HashMap<String, BlockMetadata>,
}

impl UmiMemoryAdapter {
    // Translate Block → Entity
    pub async fn update_block(&mut self, block_id: &str, value: String) -> Result<Block> {
        let metadata = self.block_metadata.get(block_id)?;

        // Store as entity
        let entity = Entity::new(
            metadata.entity_type,
            metadata.label.clone(),
            value
        );
        self.umi.remember_entity(entity).await?;

        // Update block metadata
        let block = Block {
            id: block_id.to_string(),
            label: metadata.label.clone(),
            value,
            // ...
        };
        Ok(block)
    }

    // Translate Entity → Block
    pub fn get_blocks(&self) -> Vec<Block> {
        self.umi.core_memory()
            .blocks()
            .map(|b| Block {
                id: self.get_block_id(b),
                label: b.label().unwrap_or(b.block_type().as_str()).to_string(),
                value: b.content().to_string(),
                // ...
            })
            .collect()
    }
}

// kelpie-server/src/state.rs
pub struct AppState {
    // Replace HashMap with UMI adapters
    agents: RwLock<HashMap<String, AgentConfig>>,
    memory: RwLock<HashMap<String, UmiMemoryAdapter>>,
}
```

#### What We Gain:
- **API Compatibility** - Existing clients work unchanged
- **UMI Features** - Get entity extraction, search, evolution tracking
- **Gradual Migration** - Can switch backend without changing API

#### What We Lose:
- **Extra Complexity** - Need adapter layer to translate
- **Some Overhead** - Translation between Block and Entity representations

---

### Strategy C: Delegation Pattern (Minimal Changes)

Keep Kelpie's CoreMemory and WorkingMemory classes, but delegate to UMI internally.

#### Architecture:
```
┌─────────────────────────────────────────────────┐
│         Kelpie CoreMemory (wrapper)              │
│  - add_block() → umi.remember_entity()           │
│  - get_block() → umi.core_memory().get()         │
│  - render() → umi.core_memory().render()         │
└─────────────────┬───────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────┐
│              UMI Memory (delegate)               │
└─────────────────────────────────────────────────┘
```

#### Implementation:
```rust
// kelpie-memory/src/core.rs - MODIFIED
pub struct CoreMemory {
    // OLD: blocks: Vec<MemoryBlock>
    // NEW: delegate to UMI
    umi: umi_memory::Memory,

    // Keep for compatibility
    max_bytes: usize,
}

impl CoreMemory {
    pub fn add_block(&mut self, block: MemoryBlock) -> Result<()> {
        // Convert block to entity and delegate
        let entity = Entity::new(
            self.map_block_type_to_entity(block.block_type()),
            block.label().unwrap_or(""),
            block.content()
        );

        self.umi.remember_entity(entity).await?;
        Ok(())
    }

    pub fn render(&self) -> String {
        // Delegate to UMI's render
        self.umi.core_memory().render()
    }

    pub fn get_block(&self, id: &MemoryBlockId) -> Option<&MemoryBlock> {
        // Query UMI and convert back to MemoryBlock
        // (requires caching layer)
    }
}
```

#### What We Gain:
- **Minimal API Changes** - Same interface, different backend
- **Gradual Adoption** - Can keep old code paths during migration

#### What We Lose:
- **Less Flexibility** - Constrained by existing CoreMemory interface
- **Translation Overhead** - Still need Block ↔ Entity conversion

---

## Impact on Mappings

### Current Situation (With Mapping):
```
UMI Archival                   Kelpie Core Memory
EntityType::Person  ──────┐
EntityType::Topic   ──────┼─→  MemoryBlockType::Facts
                          │
EntityType::Project ──────┐
EntityType::Task    ──────┼─→  MemoryBlockType::Goals
```

### After Full Replacement (No Mapping Needed):
```
UMI Archival → UMI Core Memory (importance-based promotion)

Entity { type: Person, importance: 0.9 }  ──┐
Entity { type: Topic, importance: 0.85 }  ──┼─→ Core Memory XML
Entity { type: Project, importance: 0.95 } ─┘    (rendered directly)

<core_memory>
  <entity type="person" importance="0.90">Bob works at Acme</entity>
  <entity type="topic" importance="0.85">Rust programming</entity>
  <entity type="project" importance="0.95">Build Umi</entity>
</core_memory>
```

**The mapping layer becomes unnecessary** because UMI directly manages what goes into core memory based on importance scores, not predefined block types.

### Hybrid Approach (Mapping Still Needed):
The adapter layer would still use the mapping to maintain API compatibility:

```rust
// In UmiMemoryAdapter
fn entity_to_block(&self, entity: &Entity) -> Block {
    let kelpie_type = KelpieBlockType::from(entity.entity_type);

    Block {
        label: kelpie_type.as_str().to_string(),
        value: entity.content.clone(),
        // ...
    }
}
```

---

## Recommendation

**Strategy B (Hybrid Approach)** is the best path forward:

### Why Hybrid?

1. **Backward Compatible** - Existing Kelpie API clients continue working
2. **Progressive Enhancement** - Add UMI features gradually
3. **Clear Migration Path** - Can eventually move to Strategy A
4. **Less Risky** - Adapter layer provides safety net
5. **Best of Both** - Keep Kelpie's simple API, get UMI's advanced features

### Implementation Timeline

**Phase 1: Foundation (Week 1-2)**
- Create `UmiMemoryAdapter` in kelpie-server
- Implement basic Block ↔ Entity translation
- Add unit tests for adapter

**Phase 2: Dual Write (Week 3)**
- Write to both old storage AND UMI
- Read from old storage (for safety)
- Monitor for consistency

**Phase 3: Dual Read (Week 4)**
- Read from UMI, fallback to old storage
- Monitor error rates
- Gain confidence in UMI backend

**Phase 4: UMI Primary (Week 5)**
- Stop writing to old storage
- UMI is source of truth
- Keep old storage for emergency rollback

**Phase 5: Cleanup (Week 6)**
- Remove old storage code
- Remove compatibility shims
- Expose new UMI-only features

**Phase 6: Full Replacement (Future)**
- Migrate API to entity-based model
- Remove mapping layer
- Pure UMI architecture

---

## What Happens to Our Recent Work?

The integration features we just built are **still valuable** but their purpose shifts:

### Current Use (With Mapping):
```rust
// When promoting UMI entities to Kelpie blocks
let kelpie_type = KelpieBlockType::from(entity.entity_type);
core_memory.add_block(kelpie_type, entity.content, entity.importance);
```

### After Replacement:
```rust
// Direct entity management, no mapping
umi.promote_to_core_memory(&entity);  // Uses importance score directly
```

The **atomic operations** (incr/append/touch) and **XML rendering with importance** are still used - they become internal UMI features instead of integration features.

The **mapping layer** becomes a **compatibility shim** during migration, then gets removed once we fully commit to UMI as the memory system.
