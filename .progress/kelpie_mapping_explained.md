# UMI → Kelpie Block Type Mapping Explained

## The Problem We're Solving

When UMI entities get promoted from archival storage to Kelpie's core memory (the ~32KB that stays in every LLM prompt), they need to be grouped into semantic categories. This mapping defines those groupings.

## Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHIVAL MEMORY (UMI)                         │
│                  Unlimited, Searchable, Embedded                 │
│                                                                   │
│  EntityType::Person → [Bob works at Acme]                        │
│  EntityType::Person → [Alice is a designer]                      │
│  EntityType::Topic  → [Rust programming language]                │
│  EntityType::Topic  → [TigerStyle engineering principles]        │
│  EntityType::Project→ [Building Umi memory system]               │
│  EntityType::Task   → [Implement Kelpie integration]             │
│  EntityType::Self_  → [I prefer concise technical explanations]  │
│  EntityType::Note   → [Meeting notes from Tuesday]               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Promotion to Core Memory
                                │ (based on importance/recency)
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CORE MEMORY (Kelpie)                          │
│                   ~32KB, Always in LLM Context                   │
│                                                                   │
│  <block type="facts" importance="0.80">                          │
│    Bob works at Acme. Alice is a designer.                       │
│    Rust programming language. TigerStyle engineering principles. │
│  </block>                                                         │
│                                                                   │
│  <block type="goals" importance="0.85">                          │
│    Building Umi memory system. Implement Kelpie integration.     │
│  </block>                                                         │
│                                                                   │
│  <block type="persona" importance="0.95">                        │
│    I prefer concise technical explanations.                      │
│  </block>                                                         │
│                                                                   │
│  <block type="scratch" importance="0.50">                        │
│    Meeting notes from Tuesday                                    │
│  </block>                                                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Mapping Rationale (Entity → Block)

### 1. Self → Persona ✅
**UMI:** `EntityType::Self_` - "User's self-representation"
**Kelpie:** `MemoryBlockType::Persona` - "AI personality and behavior guidelines"

**Example:**
```rust
// UMI archival
Entity {
    type: Self_,
    content: "I prefer concise technical explanations. I work in Rust."
}

// Promoted to Kelpie core memory
<block type="persona" importance="0.95">
I prefer concise technical explanations. I work in Rust.
</block>
```

**Why:** The AI's understanding of the user's preferences/identity maps directly to persona.

---

### 2. Person → Facts ❓
**UMI:** `EntityType::Person` - "Other people"
**Kelpie:** `MemoryBlockType::Facts` - "Key facts and knowledge to remember"

**Example:**
```rust
// UMI archival (multiple entities)
Entity { type: Person, name: "Bob", content: "Bob works at Acme Corp as CTO" }
Entity { type: Person, name: "Alice", content: "Alice is a product designer" }

// Aggregated in Kelpie core memory
<block type="facts" importance="0.75">
Bob works at Acme Corp as CTO. Alice is a product designer.
</block>
```

**Why:** Information about people is factual knowledge. Kelpie doesn't distinguish between "facts about people" vs "facts about other things" - it's all just facts.

---

### 3. Topic → Facts ❓
**UMI:** `EntityType::Topic` - "Topics/concepts"
**Kelpie:** `MemoryBlockType::Facts` - "Key facts and knowledge to remember"

**Example:**
```rust
// UMI archival
Entity { type: Topic, name: "Rust", content: "Systems programming language" }
Entity { type: Topic, name: "TigerStyle", content: "Engineering philosophy from TigerBeetle" }

// Aggregated in Kelpie core memory
<block type="facts" importance="0.80">
Rust is a systems programming language. TigerStyle is an engineering philosophy from TigerBeetle.
</block>
```

**Why:** Topics/concepts are factual knowledge, just like people information.

---

### 4. Project → Goals ✅
**UMI:** `EntityType::Project` - "Projects/initiatives"
**Kelpie:** `MemoryBlockType::Goals` - "Current objectives and tasks"

**Example:**
```rust
// UMI archival
Entity {
    type: Project,
    name: "Umi",
    content: "Building a memory system for AI agents"
}

// Promoted to Kelpie core memory
<block type="goals" importance="0.85">
Building a memory system for AI agents (Umi project)
</block>
```

**Why:** Projects are objectives/goals the AI is working towards.

---

### 5. Task → Goals ❓
**UMI:** `EntityType::Task` - "Tasks/todos"
**Kelpie:** `MemoryBlockType::Goals` - "Current objectives and tasks"

**Example:**
```rust
// UMI archival (multiple tasks)
Entity { type: Task, content: "Implement Kelpie integration" }
Entity { type: Task, content: "Write documentation" }
Entity { type: Task, content: "Run benchmarks" }

// Aggregated in Kelpie core memory
<block type="goals" importance="0.90">
Current tasks: Implement Kelpie integration, write documentation, run benchmarks
</block>
```

**Why:** Tasks are literally goals/objectives. Both Projects and Tasks map to the same Goals block because they're both things the AI is trying to accomplish.

---

### 6. Note → Scratch ✅
**UMI:** `EntityType::Note` - "General notes"
**Kelpie:** `MemoryBlockType::Scratch` - "Temporary working space for reasoning"

**Example:**
```rust
// UMI archival
Entity {
    type: Note,
    content: "Meeting notes: Discussed feature priorities, decided on MVP scope"
}

// In Kelpie core memory
<block type="scratch" importance="0.50">
Meeting notes: Discussed feature priorities, decided on MVP scope
</block>
```

**Why:** Notes are typically temporary/scratch information, not permanent facts or goals.

---

## The N:1 Problem (Many → One)

Notice that multiple UMI types map to the same Kelpie type:

```
Person  ──┐
Topic   ──┼──→  Facts
          │
Project ──┐
Task    ──┼──→  Goals
          │
```

This is **intentional** because:
1. **Kelpie has fewer, broader categories** (6 types vs UMI's 6+)
2. **Core memory is space-constrained** (~32KB total)
3. **LLM efficiency**: Fewer blocks = simpler XML structure
4. **Semantic grouping**: Related content should be together

## Reverse Mapping is Lossy

When converting back from Kelpie → UMI, we lose information:

```rust
// Forward: Specific → General (lossless)
EntityType::Person → KelpieBlockType::Facts  ✅
EntityType::Topic  → KelpieBlockType::Facts  ✅

// Reverse: General → Specific (lossy, defaults)
KelpieBlockType::Facts → EntityType::Topic  ❓ (could have been Person!)
```

This is fine because the reverse mapping is rarely needed. The typical flow is:

```
UMI archival (granular entities)
    ↓
Promote to Kelpie core (aggregated blocks)
    ↓
LLM sees simplified structure
    ↓
LLM extracts new info
    ↓
Back to UMI archival (granular entities)
```

## Real-World Example

User conversation:
> "I'm working with Bob from Acme on the Umi project. We're using Rust and TigerStyle principles. My first task is to implement Kelpie integration."

**UMI Archival Storage (6 entities):**
```rust
entities = [
    Entity { type: Person, name: "Bob", content: "Bob works at Acme Corp" },
    Entity { type: Project, name: "Umi", content: "Memory system for AI agents" },
    Entity { type: Topic, name: "Rust", content: "Programming language" },
    Entity { type: Topic, name: "TigerStyle", content: "Engineering principles" },
    Entity { type: Task, content: "Implement Kelpie integration" },
    Entity { type: Self_, content: "User is collaborating with Bob on Umi" },
]
```

**Kelpie Core Memory (3 blocks):**
```xml
<core_memory>
  <block type="facts" importance="0.75">
    Bob works at Acme Corp. Rust is the programming language.
    TigerStyle are the engineering principles being followed.
  </block>

  <block type="goals" importance="0.85">
    Working on Umi (memory system for AI agents).
    Task: Implement Kelpie integration.
  </block>

  <block type="persona" importance="0.90">
    User is collaborating with Bob on the Umi project.
  </block>
</core_memory>
```

Notice how:
- 3 separate entities (Bob, Rust, TigerStyle) → 1 Facts block
- 2 separate entities (Project, Task) → 1 Goals block
- 1 Self entity → 1 Persona block

This is **more efficient** for the LLM (328 chars vs potentially 6 separate blocks).

## When Mapping Happens

```rust
// In Kelpie integration layer
let entities_from_archival: Vec<Entity> = umi.recall("recent important info")?;

for entity in entities_from_archival {
    // Convert UMI type to Kelpie type
    let kelpie_type = KelpieBlockType::from(entity.entity_type);

    // Find or create the appropriate block
    let block = core_memory.get_or_create_block(kelpie_type);

    // Append entity content to block
    block.append_content(&format!("\n{}", entity.content));

    // Update importance based on entity importance
    if entity.importance > block.importance() {
        block.set_importance(entity.importance);
    }
}
```

## Configuration Option

If the default mapping doesn't work for your use case, you can override it:

```rust
// Custom mapping strategy
struct CustomMapping;

impl EntityToBlockMapping for CustomMapping {
    fn map(&self, entity_type: EntityType) -> KelpieBlockType {
        match entity_type {
            EntityType::Person => KelpieBlockType::Human,  // Put people in Human block
            EntityType::Topic => KelpieBlockType::Facts,
            EntityType::Task => KelpieBlockType::Scratch,   // Tasks in scratch instead
            // ... custom logic
        }
    }
}
```

## Summary

The mapping exists because:
1. **Different memory tiers** have different purposes (archival vs core)
2. **Different granularity** (6 specific types vs 6 broad categories)
3. **Different constraints** (unlimited vs 32KB)
4. **Integration efficiency** (seamless UMI→Kelpie promotion)

The "confusing" mappings (Person→Facts, Topic→Facts, Task→Goals) make sense when you realize Kelpie is optimizing for **semantic grouping** and **LLM context efficiency**, not entity type specificity.
