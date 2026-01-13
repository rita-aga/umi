# UMI Architecture Clarified: What Are Those 6 Types For?

**Your Question**: "UMI has six definitions too, and you say there is no blocks in UMI. So, what is UMI defining then?"

**The Answer**: UMI DOES have structure - the 6 EntityTypes. But they serve a **different memory tier** than Kelpie's 6 MemoryBlockTypes.

## The Key Insight: Different Memory Tiers

```
┌─────────────────────────────────────────────────────────┐
│                    UMI ARCHITECTURE                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ARCHIVAL MEMORY (Unlimited Storage)                    │
│  ↓                                                       │
│  Uses 6 EntityTypes for organization:                   │
│  - Self_   : User's self-representation                 │
│  - Person  : Other people                               │
│  - Project : Projects/initiatives                       │
│  - Topic   : Topics/concepts                            │
│  - Note    : General notes                              │
│  - Task    : Tasks/todos                                │
│                                                          │
│  Purpose: Semantic categorization for search            │
│  Storage: Millions of entities, searchable by embedding │
│  Location: storage/entity.rs                            │
│                                                          │
│           ┌──────────────────┐                          │
│           │   PROMOTION      │                          │
│           │ (importance +    │                          │
│           │  recency based)  │                          │
│           └──────────────────┘                          │
│                    ↓                                     │
│                                                          │
│  CORE MEMORY (~32KB, Always in LLM Context)             │
│  ↓                                                       │
│  Can be structured as:                                  │
│  Option A: Importance-based (no fixed blocks)           │
│  Option B: Kelpie-compatible blocks (via mapping)       │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  KELPIE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  CORE MEMORY (~32KB, Always in LLM Context)             │
│  ↓                                                       │
│  Uses 6 MemoryBlockTypes for organization:              │
│  - System   : System instructions                       │
│  - Persona  : AI personality                            │
│  - Human    : User information                          │
│  - Facts    : Key facts and knowledge                   │
│  - Goals    : Current objectives                        │
│  - Scratch  : Temporary working space                   │
│                                                          │
│  Purpose: Semantic grouping for LLM prompt              │
│  Storage: ~32KB total, fixed structure                  │
│  Location: memory/block.rs (MemoryBlockType enum)       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## What My Confusing Statement Meant

When I said "UMI has no blocks", I meant:

**❌ WRONG INTERPRETATION**: "UMI has no structure at all"

**✅ CORRECT INTERPRETATION**: "UMI's core memory doesn't force content into predefined semantic categories - it can promote entities based purely on importance/recency without requiring them to fit into specific block types"

## The Truth: Both Have Structure, Different Purposes

| System | Structure | Purpose | Memory Tier | Location in Code |
|--------|-----------|---------|-------------|------------------|
| **UMI EntityTypes** | 6 types (Self_, Person, Project, Topic, Note, Task) | Categorize entities in unlimited archival storage for semantic search | Archival Memory | `storage/entity.rs` |
| **Kelpie MemoryBlockTypes** | 6 types (System, Persona, Human, Facts, Goals, Scratch) | Group content in ~32KB core memory for LLM context | Core Memory | `memory/block.rs` |

## Why UMI Has 6 EntityTypes

**Purpose**: Semantic categorization for archival memory

```rust
// storage/entity.rs
pub enum EntityType {
    Self_,    // "I prefer technical explanations"
    Person,   // "Bob works at Acme"
    Project,  // "Building Umi memory system"
    Topic,    // "Rust programming language"
    Note,     // "Meeting notes from Tuesday"
    Task,     // "Implement Kelpie integration"
}
```

**Usage Example**:
```rust
// User says: "I'm working with Bob from Acme on the Umi project using Rust"

// UMI extracts and categorizes:
entities = [
    Entity { type: Self_,   content: "User is working on Umi" },
    Entity { type: Person,  content: "Bob works at Acme" },
    Entity { type: Project, content: "Umi memory system" },
    Entity { type: Topic,   content: "Rust programming language" },
]

// All stored in archival memory with embeddings for semantic search
```

**Why These 6?**
- Provides semantic categorization for search ("find all people", "find all projects")
- Enables evolution tracking (relationships between entities of different types)
- Supports filtering in recall ("only show me tasks")
- Allows type-specific processing (e.g., tasks have completion status)

## Why Kelpie Has 6 MemoryBlockTypes

**Purpose**: Semantic grouping for core memory

```rust
// memory/block.rs (via MemoryBlockType enum)
pub enum KelpieBlockType {
    System,   // System instructions (highest priority)
    Persona,  // AI personality and behavior
    Human,    // User information
    Facts,    // Key facts and knowledge
    Goals,    // Current objectives and tasks
    Scratch,  // Temporary working space
}
```

**Usage Example**:
```rust
// Kelpie's core memory (what goes into every LLM prompt):

<core_memory>
  <block type="persona" importance="0.95">
    I prefer concise technical explanations. I work in Rust.
  </block>

  <block type="facts" importance="0.75">
    Bob works at Acme Corp. Rust is a systems programming language.
  </block>

  <block type="goals" importance="0.85">
    Building Umi memory system. Task: Implement Kelpie integration.
  </block>
</core_memory>

// This entire structure is ~32KB and included in every LLM call
```

**Why These 6?**
- Maps to cognitive architecture (declarative memory, working memory, etc.)
- Provides predictable structure for LLM reasoning
- Enables priority-based rendering (System first, Scratch last)
- Allows importance weighting by category

## The Mapping Layer: Bridging Two Organizational Schemes

**The Problem**: UMI stores entities using EntityTypes (archival), but Kelpie expects MemoryBlockTypes (core). We need to translate between them.

**The Solution**: `kelpie_mapping.rs` provides bidirectional conversion

```rust
// kelpie_mapping.rs

// UMI EntityType → Kelpie MemoryBlockType
impl From<EntityType> for KelpieBlockType {
    fn from(entity_type: EntityType) -> Self {
        match entity_type {
            EntityType::Self_   => KelpieBlockType::Persona,  // 1:1
            EntityType::Person  => KelpieBlockType::Facts,    // N:1
            EntityType::Project => KelpieBlockType::Goals,    // 1:1
            EntityType::Topic   => KelpieBlockType::Facts,    // N:1
            EntityType::Note    => KelpieBlockType::Scratch,  // 1:1
            EntityType::Task    => KelpieBlockType::Goals,    // N:1
        }
    }
}
```

**Flow**:
```
1. UMI stores: Entity { type: Person, content: "Bob works at Acme" }
                     ↓ (in archival memory)
2. Promotion time: Entity has high importance, promote to core
                     ↓
3. Mapping layer: EntityType::Person → KelpieBlockType::Facts
                     ↓
4. Kelpie core: <block type="facts">Bob works at Acme</block>
```

## Why Multiple EntityTypes Map to One MemoryBlockType (N:1)

Notice:
- `Person` → `Facts`
- `Topic` → `Facts`

Both map to the same Kelpie block because:

1. **Core memory is space-constrained** (~32KB total)
2. **LLM efficiency**: Fewer blocks = simpler structure = better reasoning
3. **Semantic grouping**: Related content should be together (all factual knowledge in one block)
4. **Priority optimization**: Rendering order is predictable (Facts always 3rd)

**Example**:
```rust
// Archival: 2 separate entities with different types
Entity { type: Person, name: "Bob",  content: "Bob works at Acme" }
Entity { type: Topic,  name: "Rust", content: "Systems programming language" }

// Core memory: aggregated into one Facts block
<block type="facts" importance="0.75">
  Bob works at Acme. Rust is a systems programming language.
</block>
```

This is **intentional compression** - we lose the distinction between "person facts" and "topic facts" in core memory, but gain space efficiency and simpler structure for the LLM.

## Comparison Table

| Aspect | UMI EntityTypes | Kelpie MemoryBlockTypes |
|--------|-----------------|-------------------------|
| **Memory Tier** | Archival (unlimited) | Core (~32KB) |
| **Purpose** | Semantic categorization for search | Semantic grouping for LLM context |
| **Count** | 6 types | 6 types (coincidence!) |
| **Flexibility** | Can add more types | Fixed 6 types |
| **Storage** | Millions of entities | ~5-10 blocks total |
| **Organization** | Each entity has one type | Multiple entities → one block |
| **Code Location** | `storage/entity.rs` | `memory/block.rs` |
| **Used By** | EntityExtractor, DualRetriever | CoreMemory, LLM providers |

## The Confusion Resolved

**What you thought I meant**: "UMI has no structure, just chaos"

**What I actually meant**: "UMI's core memory CAN BE flexible (importance-based promotion without fixed blocks), but UMI DOES have structure in its archival layer (6 EntityTypes)"

**The reality**:
- UMI has 6 EntityTypes for **archival memory organization**
- Kelpie has 6 MemoryBlockTypes for **core memory organization**
- The mapping layer translates between these two when promoting entities to core

## Analogy: Library vs Reading Room

Think of it like this:

**UMI's EntityTypes = Library Organization System**
- Books categorized by type: Fiction, Non-Fiction, Biography, Science, History, Reference
- Millions of books in the library
- Use categories to find what you need via search
- Books stay in the library indefinitely

**Kelpie's MemoryBlockTypes = Reading Room Organization**
- Limited reading room with 6 shelves: Background, Character, Reader, Knowledge, Objectives, Notes
- Only ~20 most important books allowed in reading room at once
- These books are what the reader (LLM) sees for every task
- Books can be moved between reading room shelves based on which shelf fits their content best

**The Mapping**:
- When a "Biography" book is promoted to the reading room, it goes on the "Knowledge" shelf
- When a "Science" book is promoted, it also goes on the "Knowledge" shelf
- Both types of books (Biography, Science) map to the same reading room shelf (Knowledge)

## Why This Matters for Integration

When integrating UMI with Kelpie:

1. **Archival Layer**: UMI's EntityTypes provide rich semantic categorization
2. **Promotion Layer**: Mapping translates EntityTypes → MemoryBlockTypes
3. **Core Layer**: Kelpie's MemoryBlockTypes structure the LLM context

**Without the mapping**: UMI entities wouldn't know which Kelpie block to populate
**Without EntityTypes**: UMI couldn't do semantic search or evolution tracking
**Without MemoryBlockTypes**: Kelpie's core memory would be unstructured

## Summary

**UMI's 6 EntityTypes are for organizing unlimited archival storage.**
**Kelpie's 6 MemoryBlockTypes are for organizing limited core memory.**
**The mapping layer translates between them during promotion.**

Both systems have structure. Both use 6 categories (coincidentally). But they serve different memory tiers with different constraints and different purposes.

The statement "UMI has no blocks" should have been: "UMI's core memory doesn't require fixed block structure - it can promote based on importance/recency. But UMI's archival memory DOES use structured EntityTypes for semantic organization."

---

Does this clear up the confusion? The key insight is: **different memory tiers = different organizational needs**.
