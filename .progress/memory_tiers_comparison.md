# Memory Tiers Comparison: UMI vs Kelpie

## Side-by-Side Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UMI: THREE-TIER ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: ARCHIVAL MEMORY (Unlimited)                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                              │
│  Organized by: 6 EntityTypes                                                │
│  ┌────────────┬────────────┬────────────┬────────────┬────────┬────────┐  │
│  │   Self_    │   Person   │  Project   │   Topic    │  Note  │  Task  │  │
│  ├────────────┼────────────┼────────────┼────────────┼────────┼────────┤  │
│  │ "I prefer  │ "Bob works │ "Building  │ "Rust is a │ "Notes │ "Impl- │  │
│  │  concise   │  at Acme   │  Umi mem-  │  systems   │  from  │  ement │  │
│  │  explana-  │  Corp as   │  ory sys-  │  program-  │  meet- │  Kelp- │  │
│  │  tions"    │  CTO"      │  tem"      │  ming lang"│  ing"  │  ie"   │  │
│  │            │            │            │            │        │        │  │
│  │ (20 more)  │ (50 more)  │ (10 more)  │ (100 more) │ (30)   │ (15)   │  │
│  └────────────┴────────────┴────────────┴────────────┴────────┴────────┘  │
│                                                                              │
│  Storage: LanceDB with embeddings (millions of entities)                    │
│  Search: Semantic search by content + type filtering                        │
│  Evolution: Track relationships between entities                            │
│                                                                              │
│                           ╱╲  PROMOTION  ╱╲                                 │
│                          ╱  ╲ (importance╱  ╲                               │
│                         ╱    ╲ + recency╱    ╲                              │
│                        ╱      ╲       ╱      ╲                              │
│                       ╱        ╲     ╱        ╲                             │
│                      ╱          ╲   ╱          ╲                            │
│                     ╱            ╲ ╱            ╲                           │
│                    ╱              V              ╲                          │
└───────────────────────────────────────────────────────────────────────────┬─┘
                                                                              │
┌─────────────────────────────────────────────────────────────────────────┬─┘
│  TIER 2: CORE MEMORY (~32KB)                                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                          │
│  UMI's Approach: Flexible Structure                                     │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Option A: Importance-Based (No Fixed Blocks)                  │   │
│  │  ─────────────────────────────────────────────────────────     │   │
│  │  <entity type="self" importance="1.0">I prefer concise...</e>  │   │
│  │  <entity type="project" importance="0.95">Building Umi...</e>  │   │
│  │  <entity type="person" importance="0.9">Bob works...</e>       │   │
│  │  <entity type="topic" importance="0.85">Rust is...</e>         │   │
│  │                                                                 │   │
│  │  Option B: Kelpie-Compatible Blocks (Via Mapping)              │   │
│  │  ──────────────────────────────────────────────────            │   │
│  │  <block type="persona" importance="0.95">                      │   │
│  │    I prefer concise explanations                               │   │
│  │  </block>                                                       │   │
│  │  <block type="facts" importance="0.75">                        │   │
│  │    Bob works at Acme. Rust is a systems language.              │   │
│  │  </block>                                                       │   │
│  │  <block type="goals" importance="0.85">                        │   │
│  │    Building Umi. Task: Implement Kelpie integration            │   │
│  │  </block>                                                       │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Always in LLM context: YES (every prompt includes this)                │
│  Flexibility: Can be structured or importance-based                     │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 3: WORKING MEMORY (~1MB)                                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                          │
│  Session-scoped KV store with TTL                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Key-Value Pairs:                                               │   │
│  │  ─────────────────                                              │   │
│  │  "conversation_count" → "5"                (TTL: 1 hour)        │   │
│  │  "last_topic"        → "memory systems"    (TTL: 1 hour)        │   │
│  │  "temp_notes"        → "Check performance" (TTL: 10 minutes)    │   │
│  │  "retry_count"       → "2"                 (TTL: 5 minutes)     │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Operations: get, set, incr, append, touch, delete                      │
│  Expiration: Automatic TTL-based cleanup                                │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        KELPIE: SINGLE-TIER ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  CORE MEMORY (~32KB)                                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                              │
│  Organized by: 6 MemoryBlockTypes (FIXED)                                   │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐                  │
│  │System  │Persona │ Human  │ Facts  │ Goals  │Scratch │                  │
│  │(pri 0) │(pri 1) │(pri 2) │(pri 3) │(pri 4) │(pri 5) │                  │
│  ├────────┼────────┼────────┼────────┼────────┼────────┤                  │
│  │"You    │"I am   │"User   │"Bob    │"Build  │"Notes  │                  │
│  │ are a  │ help-  │ pre-   │ works  │ Umi    │ from   │                  │
│  │ help-  │ ful    │ fers   │ at     │ memory │ meet-  │                  │
│  │ ful    │ and    │ tech-  │ Acme.  │ system"│ ing"   │                  │
│  │ assis- │ kind"  │ nical  │ Rust   │        │        │                  │
│  │ tant"  │        │ style" │ is a   │ "Task: │        │                  │
│  │        │        │        │ sys-   │ Impl-  │        │                  │
│  │        │        │        │ tems   │ ement  │        │                  │
│  │        │        │        │ lang"  │ Kelp"  │        │                  │
│  └────────┴────────┴────────┴────────┴────────┴────────┘                  │
│                                                                              │
│  XML Format (what LLM sees):                                                │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ <core_memory>                                                       │   │
│  │   <block type="system" importance="1.0">                            │   │
│  │     You are a helpful assistant                                     │   │
│  │   </block>                                                           │   │
│  │   <block type="persona" importance="0.95">                          │   │
│  │     I am helpful and kind                                           │   │
│  │   </block>                                                           │   │
│  │   <block type="human" importance="0.90">                            │   │
│  │     User prefers technical style                                    │   │
│  │   </block>                                                           │   │
│  │   <block type="facts" importance="0.75">                            │   │
│  │     Bob works at Acme. Rust is a systems language.                  │   │
│  │   </block>                                                           │   │
│  │   <block type="goals" importance="0.85">                            │   │
│  │     Build Umi memory system. Task: Implement Kelpie                 │   │
│  │   </block>                                                           │   │
│  │   <block type="scratch" importance="0.50">                          │   │
│  │     Notes from meeting                                              │   │
│  │   </block>                                                           │   │
│  │ </core_memory>                                                       │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Always in LLM context: YES (every prompt includes this)                    │
│  Flexibility: Fixed 6-block structure                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Key Differences

| Feature | UMI | Kelpie |
|---------|-----|--------|
| **Memory Tiers** | 3 tiers (Core, Working, Archival) | 1 tier (Core only) |
| **Archival Memory** | Yes (unlimited entities with EntityTypes) | No (no long-term storage) |
| **Core Memory Structure** | Flexible (can be blocks or importance-based) | Fixed (6 MemoryBlockTypes) |
| **Working Memory** | Yes (session KV store with TTL) | No (stateless across sessions) |
| **Entity Types** | 6 types for archival categorization | N/A (no entities) |
| **Block Types** | 6 types for Kelpie compatibility | 6 types (mandatory) |
| **Semantic Search** | Yes (in archival memory) | No (only core memory) |
| **Evolution Tracking** | Yes (entity relationships) | No |
| **LLM Context** | Core memory only (~32KB) | Core memory only (~32KB) |

## The Integration Strategy

When UMI integrates with Kelpie:

```
┌─────────────────────────────────────────────────────┐
│  User Input: "I'm working with Bob on Umi"          │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────┐
│  UMI LAYER                                          │
│  ──────────                                         │
│  1. EntityExtractor extracts:                       │
│     - Entity { type: Self_, content: "working..." } │
│     - Entity { type: Person, content: "Bob" }       │
│     - Entity { type: Project, content: "Umi" }      │
│                                                     │
│  2. Store in ARCHIVAL with embeddings               │
│                                                     │
│  3. Promote high-importance to CORE                 │
│     - Check importance scores                       │
│     - Select top N entities                         │
│                                                     │
│  4. MAPPING LAYER translates:                       │
│     - EntityType::Self_   → KelpieBlockType::Persona│
│     - EntityType::Person  → KelpieBlockType::Facts  │
│     - EntityType::Project → KelpieBlockType::Goals  │
│                                                     │
│  5. Render as Kelpie XML:                           │
│     <block type="persona" importance="0.95">        │
│       User is working on Umi                        │
│     </block>                                        │
│     <block type="facts" importance="0.75">          │
│       Bob is a person                               │
│     </block>                                        │
│     <block type="goals" importance="0.85">          │
│       Umi project                                   │
│     </block>                                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────┐
│  KELPIE LAYER                                       │
│  ───────────                                        │
│  Receives structured XML with 6 block types         │
│  Passes to LLM in every prompt                      │
│  LLM reasons using core memory context              │
└─────────────────────────────────────────────────────┘
```

## Why The Mapping Exists

**Without the mapping**:
- UMI would store entities with EntityTypes (Self_, Person, etc.)
- Kelpie expects MemoryBlockTypes (Persona, Facts, etc.)
- No way to translate between the two → integration fails

**With the mapping**:
- UMI entities promoted from archival → mapped to Kelpie blocks
- Kelpie receives familiar 6-block structure
- LLM sees predictable, structured core memory

## Example: Full Workflow

**User message**: "I'm Alice, I work at Acme with Bob. We're building Umi in Rust. My task is to implement vector search."

### Step 1: UMI Extraction & Storage (Archival)
```rust
// EntityExtractor identifies 5 entities
entities = [
    Entity { type: Self_,   name: "Alice",        importance: 1.0 },
    Entity { type: Person,  name: "Bob",          importance: 0.7 },
    Entity { type: Project, name: "Umi",          importance: 0.9 },
    Entity { type: Topic,   name: "Rust",         importance: 0.6 },
    Entity { type: Task,    name: "vector search",importance: 0.8 },
]

// All stored in archival memory with embeddings
```

### Step 2: Promotion to Core (Importance-Based)
```rust
// Top 3 by importance get promoted
promoted = [
    Entity { type: Self_,   importance: 1.0 },
    Entity { type: Project, importance: 0.9 },
    Entity { type: Task,    importance: 0.8 },
]
```

### Step 3: Mapping to Kelpie Blocks
```rust
// Apply EntityType → KelpieBlockType mapping
Self_   → Persona  // "Alice" info goes to Persona block
Project → Goals    // "Umi" goes to Goals block
Task    → Goals    // "vector search" goes to Goals block
```

### Step 4: Render as Kelpie XML
```xml
<core_memory>
  <block type="persona" importance="1.0">
    I am Alice, working at Acme
  </block>

  <block type="goals" importance="0.85">
    Building Umi project. Task: Implement vector search.
  </block>
</core_memory>
```

### Step 5: Kelpie Uses It
```
LLM receives this core memory in every prompt:
- Knows the user is Alice
- Knows the current project is Umi
- Knows the current task is vector search
- Can reference this context in responses
```

## Summary: What Each System's "6 Types" Are For

### UMI's 6 EntityTypes (Archival Memory)
- **Purpose**: Categorize unlimited entities for semantic search
- **Storage**: LanceDB with embeddings (millions of entities)
- **Flexibility**: Can add more types if needed
- **Usage**: "Find all Person entities", "Get tasks related to Rust"

### Kelpie's 6 MemoryBlockTypes (Core Memory)
- **Purpose**: Structure limited core memory for LLM context
- **Storage**: In-memory, ~32KB total (5-10 blocks)
- **Flexibility**: Fixed 6 types, cannot add more
- **Usage**: LLM sees these blocks in every prompt

### The Mapping (kelpie_mapping.rs)
- **Purpose**: Translate between archival categories and core categories
- **Direction**: Bidirectional (but lossy in reverse)
- **N:1 Nature**: Multiple EntityTypes → One MemoryBlockType
- **Usage**: During promotion from archival to core

---

**The confusion was**: I said "UMI has no blocks" when I meant "UMI's core memory can be flexible", but UMI DOES have structure (6 EntityTypes) for its archival layer.

**The truth is**: Both have 6 categories, but for different memory tiers with different purposes.
