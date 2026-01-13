# The Simple Answer to "What Are UMI's 6 Types For?"

## Your Question

> "But UMI has six definitions now too, and you say there is no blocks in UMI. So, what is UMI defining then?"

## The Simple Answer

**UMI's 6 EntityTypes organize the WAREHOUSE.**
**Kelpie's 6 MemoryBlockTypes organize the SHOP WINDOW.**

## The Analogy

Imagine a retail store:

### The Warehouse (UMI's Archival Memory)
- Unlimited storage space in the back
- Organized by 6 categories: Self, Person, Project, Topic, Note, Task
- Millions of items with searchable labels
- Items rarely leave the warehouse

**This is what UMI's 6 EntityTypes are for.**

### The Shop Window (Core Memory)
- Limited display space (~32KB)
- Only the most important items on display
- What customers (the LLM) see and can access
- Items can be arranged in Kelpie's 6 categories: System, Persona, Human, Facts, Goals, Scratch

**This is what Kelpie's 6 MemoryBlockTypes are for.**

### The Mapping (kelpie_mapping.rs)
When moving items from warehouse → shop window:
- "Person" items get displayed on the "Facts" shelf
- "Project" items get displayed on the "Goals" shelf
- "Self" items get displayed on the "Persona" shelf
- etc.

**This is what the mapping layer does.**

## Why I Said "UMI Has No Blocks"

I meant: **UMI's shop window (core memory) doesn't REQUIRE fixed shelves - you could just put the most important items on display without organizing them into specific categories.**

But UMI's warehouse (archival memory) DOES have organization - the 6 EntityTypes.

## The Key Insight

**Both systems have 6 categories. Both have structure. But they organize DIFFERENT things:**

- **UMI's 6 types** organize unlimited archival storage (the warehouse)
- **Kelpie's 6 types** organize limited core memory (the shop window)

When UMI integrates with Kelpie:
1. UMI stores everything in the warehouse using EntityTypes
2. UMI promotes important items to the shop window
3. The mapping layer decides which Kelpie shelf each item goes on
4. Kelpie sees a familiar 6-shelf display

## Code Location Summary

```
UMI's 6 EntityTypes:
  umi-memory/src/storage/entity.rs
  Used for: Archival memory (unlimited)

Kelpie's 6 MemoryBlockTypes:
  umi-memory/src/memory/block.rs
  Used for: Core memory (~32KB)

The Mapping:
  umi-memory/src/memory/kelpie_mapping.rs
  Translates: EntityType → KelpieBlockType
```

## One Sentence Summary

**UMI has 6 EntityTypes for organizing archival storage, Kelpie has 6 MemoryBlockTypes for organizing core memory, and the mapping translates between them during promotion.**

That's it. Different memory tiers, different purposes, same number of categories (coincidentally).
