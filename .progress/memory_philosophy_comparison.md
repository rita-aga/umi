# Memory Philosophy: Letta vs Kelpie vs UMI

## The Three Approaches to Agent Memory

### Letta/MemGPT: Minimal + Flexible

**Philosophy**: "Give agents self-editing memory with minimal constraints"

**Core Memory Blocks**:
- `persona` - Agent's self-concept (read-write by agent)
- `human` - User information (read-write by agent)
- `system` - System instructions (read-only)
- **Custom blocks** - User-defined for any purpose

**Key Insight**: Letta doesn't prescribe categories. You create whatever blocks you need:
- `organization` for company context
- `policies` for rules
- `emotional_state` for tracking mood
- `scratchpad` for working notes

**From Letta Docs**:
> "Memory blocks are entirely customizable. Use for any purpose: knowledge, guidelines, state tracking, scratchpad space."

**Architecture**:
```
┌────────────────────────────────────────┐
│   Letta Core Memory (Flexible)         │
│                                         │
│   <persona>I am helpful and kind</persona>
│   <human>User likes Python</human>
│   <organization>Works at Acme Corp</organization>
│   <goals>Help user learn Rust</goals>
│   <scratchpad>Meeting at 3pm</scratchpad>
│                                         │
│   (You define the blocks!)              │
└────────────────────────────────────────┘
```

**Agent Capabilities**:
- **Self-editing**: Agents can modify their own memory blocks
- **Learning**: As agent learns about user, updates `human` block
- **Adaptation**: Can update `persona` to change behavior over time

---

### Kelpie: Structured + Semantic

**Philosophy**: "Organize memory by semantic role with predefined categories"

**Core Memory Blocks** (6 fixed types):
1. `System` - System instructions (highest priority, immutable)
2. `Persona` - AI personality (stable, high priority)
3. `Human` - User information (stable, high priority)
4. `Facts` - Knowledge to remember (accumulated over time)
5. `Goals` - Objectives and tasks (updated frequently)
6. `Scratch` - Temporary workspace (ephemeral, low priority)

**Key Insight**: Kelpie prescribes structure. Each block type has a specific semantic purpose and typical importance level.

**Architecture**:
```
┌────────────────────────────────────────┐
│   Kelpie Core Memory (Structured)      │
│                                         │
│   Priority 0: System (importance 1.0)  │
│   Priority 1: Persona (importance 0.95)│
│   Priority 2: Human (importance 0.90)  │
│   Priority 3: Facts (importance 0.80)  │
│   Priority 4: Goals (importance 0.85)  │
│   Priority 5: Scratch (importance 0.50)│
│                                         │
│   (Fixed categories, sorted rendering) │
└────────────────────────────────────────┘
```

**Advantages**:
- **Consistency**: All Kelpie agents use same structure
- **Optimization**: Rendering order is predictable
- **Reasoning**: LLM understands semantic categories
- **Interoperability**: Agents can share memory format

**Disadvantages**:
- **Rigidity**: Can't create custom categories
- **Mismatch**: What if your use case doesn't fit these 6?
- **Overhead**: Might not need all 6 types for simple agents

**Why These 6?**

Looking at the categories, they map to different aspects of agent cognition:
- **System/Persona** = Agent identity ("Who am I?")
- **Human** = Context about user ("Who am I talking to?")
- **Facts** = Static knowledge ("What do I know?")
- **Goals** = Dynamic objectives ("What am I trying to do?")
- **Scratch** = Working memory ("What am I thinking about right now?")

This is similar to human cognitive architecture:
- Long-term memory = Facts
- Working memory = Scratch
- Self-concept = Persona
- Social context = Human
- Executive function = Goals

---

### UMI: Entity-Centric + Importance-Based

**Philosophy**: "Memory is a collection of entities, promoted based on importance and recency"

**Entity Types** (6 semantic categories for archival):
- `Self` - User's self-representation
- `Person` - Other people
- `Project` - Projects/initiatives
- `Topic` - Topics/concepts
- `Note` - General notes
- `Task` - Tasks/todos

**Key Insight**: UMI doesn't force blocks. Entities live in archival memory, and the **most important/recent ones get promoted to core memory automatically**.

**Architecture**:
```
┌─────────────────────────────────────────────────┐
│           UMI Archival Memory                    │
│              (Unlimited storage)                 │
│                                                  │
│  Entity { type: Person, importance: 0.9 }        │
│  Entity { type: Topic, importance: 0.85 }        │
│  Entity { type: Project, importance: 0.95 }      │
│  Entity { type: Task, importance: 0.7 }          │
│  Entity { type: Self, importance: 1.0 }          │
│  ... (thousands more) ...                        │
└──────────────────┬──────────────────────────────┘
                   │
                   │ Automatic promotion
                   │ (importance + recency)
                   ↓
┌─────────────────────────────────────────────────┐
│         UMI Core Memory (~32KB)                  │
│         (Only high-importance entities)          │
│                                                  │
│  <entity type="self" importance="1.0">          │
│    I prefer technical explanations               │
│  </entity>                                       │
│                                                  │
│  <entity type="project" importance="0.95">      │
│    Building Umi memory system                    │
│  </entity>                                       │
│                                                  │
│  <entity type="person" importance="0.9">        │
│    Bob works at Acme                             │
│  </entity>                                       │
│                                                  │
│  (Dynamic, importance-driven)                    │
└─────────────────────────────────────────────────┘
```

**Advantages**:
- **Automatic curation**: High-importance content surfaces naturally
- **Semantic search**: Find entities by meaning, not just keywords
- **Evolution tracking**: Detect relationships between entities
- **Flexible**: Entity types are for organization, not constraints

**Disadvantages**:
- **Less predictable**: Core memory contents vary by importance
- **No fixed structure**: Can't assume "Facts" block exists
- **More complex**: Promotion algorithm needs tuning

---

## Direct Comparison

| Aspect | Letta | Kelpie | UMI |
|--------|-------|--------|-----|
| **Categories** | 2 fixed + custom | 6 fixed | 6 semantic (for archival only) |
| **Core Memory** | User-defined blocks | Fixed block types | Importance-based promotion |
| **Flexibility** | Very high | Low | Very high |
| **Predictability** | Low | Very high | Medium |
| **Agent editing** | Yes (self-editing) | No | No (but learning via extraction) |
| **Structure** | Minimal | Structured | Entity-centric |
| **Use case** | General agents | Structured agents | Knowledge-intensive agents |

---

## Philosophy Summary

### Letta: "Agents should define their own memory"
```python
# Letta: You decide what blocks to create
agent = create_agent(blocks=[
    {"label": "persona", "value": "..."},
    {"label": "human", "value": "..."},
    {"label": "whatever_i_want", "value": "..."},  # Anything!
])
```

### Kelpie: "Memory should follow cognitive architecture"
```python
# Kelpie: Blocks are predefined
agent = create_agent()
agent.core_memory.set_block(MemoryBlockType.System, "...")  # Fixed types
agent.core_memory.set_block(MemoryBlockType.Facts, "...")
agent.core_memory.set_block(MemoryBlockType.Goals, "...")
```

### UMI: "Memory is entities, organized by importance"
```python
# UMI: Entities in archival, automatic promotion
umi = Memory()
await umi.remember("Bob works at Acme")  # Extracts entities
entities = await umi.recall("who works at acme")  # Semantic search

# Core memory filled automatically by importance
core_xml = umi.core_memory().render()  # Top N important entities
```

---

## Answering Your Questions

### Q1: Why did Kelpie have those categories specifically?

**A**: Kelpie expanded on Letta's 2-block model (Persona + Human) by adding 4 more categories to match cognitive architecture:
- **Facts** = Long-term memory (what do I know?)
- **Goals** = Executive function (what am I trying to do?)
- **Scratch** = Working memory (what am I thinking about?)
- **System** = Core directives (how should I behave?)

This creates a more structured cognitive model than Letta's flexible approach.

### Q2: Is it trying to replicate Letta AI?

**A**: Partially. Kelpie borrows Letta's core idea (Persona + Human blocks) but **extends it significantly**:
- Letta: 2 fixed + unlimited custom blocks
- Kelpie: 6 fixed semantic blocks

Kelpie trades Letta's flexibility for predictability and structure.

### Q3: Do we need to keep those categories?

**A**: It depends on your goals:

**Keep them if:**
- You want Kelpie API compatibility
- You value predictable structure
- Your use cases fit the 6 categories
- You want cognitive architecture alignment

**Replace them if:**
- You want flexibility (like Letta)
- You want automatic importance-based promotion (like UMI)
- The 6 categories don't match your domain
- You want entity-centric memory instead of block-centric

### Q4: How do we reconcile the approaches?

**Three options:**

**Option A: Kelpie-compatible mode**
```rust
// Map UMI entities to Kelpie blocks
let kelpie_type = KelpieBlockType::from(entity.entity_type);
core.add_block(kelpie_type, entity.content, entity.importance);
```
Keep the 6-block structure, use mapping layer.

**Option B: Pure UMI mode**
```rust
// No blocks, just entities
umi.promote_to_core_memory(&entity);  // Importance-based
```
Abandon the 6-block structure, use importance-based promotion.

**Option C: Hybrid mode**
```rust
// Configurable block types
let custom_blocks = vec![
    BlockType::custom("persona", 0.95),
    BlockType::custom("context", 0.90),
    BlockType::custom("knowledge", 0.80),
];
umi.configure_core_memory(custom_blocks);
```
Let users define their own block categories (like Letta).

---

## Recommendation

**Option C (Hybrid)** gives you the best of all worlds:

1. **Default to Kelpie's 6 blocks** for backward compatibility
2. **Allow custom block types** for flexibility (like Letta)
3. **Use UMI's importance-based promotion** under the hood
4. **Provide entity-based API** for advanced users

```rust
// config.rs
pub struct CoreMemoryConfig {
    pub block_types: Vec<BlockTypeDefinition>,
    pub promotion_strategy: PromotionStrategy,
}

pub enum PromotionStrategy {
    Fixed,  // Always use predefined blocks (Kelpie mode)
    ImportanceBased { threshold: f64 },  // UMI mode
    Custom(Box<dyn Fn(&Entity) -> BlockType>),  // Letta mode
}
```

This lets users choose:
- **Kelpie users**: Get the familiar 6-block structure
- **Letta users**: Define custom blocks
- **UMI users**: Use pure entity-based memory with auto-promotion

No lock-in, maximum flexibility!
