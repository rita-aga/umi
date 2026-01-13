# memU vs UMI: Memory Management Comparison

## The Question

Does memU do automatic promotion from archival → core, use all three tiers, and have built-in importance-based eviction?

## The Answer: **YES**, memU does ALL of these things

memU has a **complete memory orchestrator** that automatically manages memory across multiple layers with promotion and forgetting mechanisms.

---

## memU Architecture (Complete Orchestration)

### Three-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: MEMORY CATEGORY (What enters LLM context)        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                              │
│  Aggregated summaries as markdown files:                    │
│  - preferences.md                                            │
│  - relationships.md                                          │
│  - skills.md                                                 │
│                                                              │
│  "Only category-level files enter agent's context window"   │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ AUTOMATIC PROMOTION
                       │ (Based on access patterns & importance)
                       │
┌──────────────────────┴───────────────────────────────────────┐
│  LAYER 2: MEMORY ITEM (Extracted discrete units)            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                              │
│  Discrete meaningful units:                                  │
│  - User prefers technical explanations                       │
│  - User works with Bob at Acme                               │
│  - User is building Umi project                              │
│                                                              │
│  "Smallest meaningful unit that can be understood on its own"│
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ EXTRACTION
                       │ (LLM-powered)
                       │
┌──────────────────────┴───────────────────────────────────────┐
│  LAYER 1: RESOURCE (Raw storage, never deleted)             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                              │
│  Raw multimodal data:                                        │
│  - Conversations                                             │
│  - Documents                                                 │
│  - Images, video, audio                                      │
│  - Code logs                                                 │
│                                                              │
│  "Resources are never pruned or discarded"                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Automatic Promotion

**How it works:**
1. User input → stored in Resource layer
2. memU agent extracts → Memory Item layer
3. Frequently accessed items → promoted to Category layer
4. Category files → included in LLM context

**From the docs:**
> "Information flows upward through deliberate abstraction. As memories progress through layers, meaning becomes clearer and structure becomes more stable."

### Automatic Forgetting (Eviction)

**How it works:**
- Memories not accessed for a long time → removed from Category layer
- System falls back to Item layer
- If still not found → retrieves from Resource layer
- Resources are NEVER deleted (only demoted)

**From the docs:**
> "When a memory has not been referenced for a long time, it may no longer appear at the category layer and is effectively 'forgotten' at that level. In this case, the system falls back to retrieve from deeper layers."

### Self-Evolution

**How it works:**
- System tracks access patterns
- Category structure adapts based on usage
- "Continuous feedback from real usage allows the memory structure to gradually align with how the Agent actually works"

---

## UMI Architecture (Component Library)

### Current State: Single Tier with Optional Components

```
┌─────────────────────────────────────────────────────────────┐
│  Memory Orchestrator (Current Implementation)                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                              │
│  Uses ONLY archival storage:                                 │
│  - EntityExtractor                                           │
│  - ArchivalMemory (StorageBackend + VectorBackend)          │
│  - DualRetriever                                             │
│  - EvolutionTracker                                          │
│                                                              │
│  Does NOT use:                                               │
│  ❌ CoreMemory                                               │
│  ❌ WorkingMemory                                            │
│  ❌ Automatic promotion                                      │
│  ❌ Importance-based eviction                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Available Components (Not Wired Together)

```
┌─────────────────────────────────────────────────────────────┐
│  CoreMemory (Separate component, not auto-managed)          │
│  - Can hold ~32KB                                            │
│  - Renders as XML with importance                            │
│  - Manual management only                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WorkingMemory (Separate component, not auto-managed)        │
│  - Session KV store with TTL                                 │
│  - Atomic operations (incr, append, touch)                   │
│  - Manual management only                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ArchivalMemory (Currently used by Memory orchestrator)      │
│  - Unlimited storage                                         │
│  - Entity extraction                                         │
│  - Dual retrieval                                            │
│  - Evolution tracking                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  KelpieBlockType Mapping (Utility, not auto-used)           │
│  - Translates EntityType → KelpieBlockType                   │
│  - Only used in tests                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Direct Comparison

| Feature | memU | UMI |
|---------|------|-----|
| **Multiple memory tiers** | ✅ Yes (3 layers) | ⚠️ Components exist, not integrated |
| **Automatic promotion** | ✅ Yes (Resource → Item → Category) | ❌ No |
| **Importance-based eviction** | ✅ Yes (access-pattern based forgetting) | ❌ No |
| **What enters LLM context** | ✅ Category layer (auto-managed) | ⚠️ User decides (manual) |
| **Memory orchestrator** | ✅ Complete (manages all layers) | ⚠️ Partial (only archival) |
| **Self-evolution** | ✅ Yes (structure adapts to usage) | ❌ No |
| **Graceful degradation** | ✅ Yes (fallback to deeper layers) | ✅ Yes (LLM/storage failures) |
| **Never delete raw data** | ✅ Resources never pruned | ✅ Entities persist |
| **Philosophy** | Framework (opinionated) | Library (flexible) |

---

## What memU Does That UMI Doesn't

### 1. Automatic Tier Management

**memU:**
```python
# User just calls remember()
await memu.remember("Alice works at Acme")

# Behind the scenes:
# 1. Store in Resource layer ✅
# 2. Extract to Item layer ✅
# 3. Aggregate to Category layer ✅
# 4. Category enters LLM context ✅
```

**UMI:**
```rust
// User calls remember()
memory.remember("Alice works at Acme").await?;

// Behind the scenes:
// 1. Store in Archival ✅
// 2. No promotion to core ❌
// 3. User must manually decide what goes in LLM context ❌
```

### 2. Automatic Forgetting Based on Access

**memU:**
- Tracks memory access patterns
- Unused memories demoted from Category layer
- Still accessible via fallback to Item/Resource layers

**UMI:**
- No access tracking
- No automatic demotion
- All entities stay in archival indefinitely

### 3. Self-Evolving Structure

**memU:**
- Category files adapt based on usage patterns
- Memory organization evolves with agent behavior
- "Structure gradually aligns with how Agent works"

**UMI:**
- EntityTypes are fixed (Self, Person, Project, Topic, Note, Task)
- No automatic reorganization
- Structure defined at design time

---

## Why the Difference?

### memU's Philosophy: **Opinionated Framework**

memU makes architectural decisions for you:
- ✅ Decides what enters LLM context (Category layer)
- ✅ Manages promotion automatically
- ✅ Handles forgetting automatically
- ✅ Evolves structure based on usage

**Trade-off:** Less control, but complete out-of-the-box solution

### UMI's Philosophy: **Flexible Library**

UMI provides building blocks, you decide architecture:
- ⚠️ You decide what enters LLM context
- ⚠️ You implement promotion if needed
- ⚠️ You implement eviction if needed
- ⚠️ You define memory structure

**Trade-off:** More control, but more implementation work

---

## Can UMI Match memU's Features?

**Yes, but it requires building a new orchestrator:**

```rust
// Hypothetical UnifiedMemory orchestrator
pub struct UnifiedMemory {
    resource: StorageBackend,    // Layer 1: Raw data
    items: Vec<Entity>,          // Layer 2: Extracted entities
    categories: CoreMemory,      // Layer 3: Aggregated for LLM

    access_tracker: AccessTracker,
    promotion_policy: PromotionPolicy,
}

impl UnifiedMemory {
    async fn remember(&mut self, text: &str) {
        // 1. Store in resource layer
        let resource_id = self.resource.store_raw(text).await?;

        // 2. Extract entities (item layer)
        let entities = self.extractor.extract(text).await?;
        self.items.extend(entities.clone());

        // 3. Promote high-importance to categories (using mapping)
        for entity in entities {
            if entity.importance > 0.8 {
                let block_type = KelpieBlockType::from(entity.entity_type);
                self.categories.append_to_block(block_type, &entity.content)?;
            }
        }

        // 4. Track access
        self.access_tracker.record_access(&entities);

        // 5. Evict unused memories from categories
        self.evict_forgotten().await?;
    }

    async fn recall(&self, query: &str) -> Vec<Entity> {
        // Try category layer first (fast, in-context)
        if let Some(results) = self.categories.search(query) {
            return results;
        }

        // Fallback to item layer
        if let Some(results) = self.search_items(query) {
            return results;
        }

        // Fallback to resource layer
        self.resource.search(query).await?
    }

    async fn evict_forgotten(&mut self) {
        // Remove unused memories from categories
        for entity in self.categories.all_entities() {
            if self.access_tracker.last_access(&entity) > 30_days {
                self.categories.remove(&entity);
            }
        }
    }
}
```

This would give UMI the same capabilities as memU.

---

## Summary

### Does memU do these things?

1. **Automatic promotion from archival → core?**
   - ✅ **YES** - Resource → Item → Category with automatic flow

2. **Memory orchestrator that uses all three tiers?**
   - ✅ **YES** - Single orchestrator manages Resource/Item/Category layers

3. **Built-in importance-based eviction from core?**
   - ✅ **YES** - Access-pattern based forgetting from Category layer

### Does UMI do these things?

1. **Automatic promotion from archival → core?**
   - ❌ **NO** - Only stores in archival, no automatic promotion

2. **Memory orchestrator that uses all three tiers?**
   - ⚠️ **PARTIALLY** - Has components, but Memory class only uses archival

3. **Built-in importance-based eviction from core?**
   - ❌ **NO** - No eviction logic implemented

### The Bottom Line

**memU is a complete memory management framework** that automatically handles promotion, eviction, and tier management.

**UMI is a component library** that gives you the tools (CoreMemory, WorkingMemory, mapping) but doesn't wire them together into an automatic system.

To match memU's capabilities, UMI would need a **new UnifiedMemory orchestrator** that uses all three tiers with automatic promotion/eviction logic.

---

## References

- [memU GitHub Repository](https://github.com/NevaMind-AI/memU)
- [memU 1.0.0: Memory-Driven Agent Evolution](https://dev.to/memu_ai/memu-100-memory-driven-agent-evolution-ane)
- [memU Official Website](https://memu.pro)
- [memU PyPI Package](https://pypi.org/project/memu-py/)
