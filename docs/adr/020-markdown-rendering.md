# ADR 020: Markdown Rendering for Core Memory

## Status

Accepted

## Context

Umi's CoreMemory is always included in LLM context, rendered as XML for machine consumption. However, when developers debug, inspect memory state, or build UIs that display memory to users, XML is harder to read than Markdown.

memU (https://github.com/NevaMind-AI/memU) provides human-friendly Markdown rendering alongside XML for LLMs. We need equivalent functionality for Umi to support:

1. Debugging - Developers inspecting memory state during development
2. User Interfaces - Apps displaying memory contents to end users
3. Documentation - Examples and tutorials showing memory state
4. Session Handoff - Readable memory snapshots for agent transitions

## Decision

We will add `render_markdown()` methods to both `CoreMemory` and `MemoryBlock` that produce human-friendly Markdown output while maintaining the same block ordering as XML rendering.

### Design Principles

1. **Dual Format Support**: Both XML (for LLMs) and Markdown (for humans) available
2. **Consistent Ordering**: Markdown and XML render blocks in identical order
3. **TigerStyle Compliance**: Deterministic output, explicit formatting
4. **DST-First**: Full test coverage with deterministic simulation tests

### Implementation

#### MemoryBlock Markdown Rendering

```rust
impl MemoryBlock {
    /// Render block as Markdown for human display.
    ///
    /// Example output:
    /// ```markdown
    /// ## System (importance: 0.95)
    /// You are a helpful assistant.
    /// ```
    pub fn render_markdown(&self) -> String {
        let type_name = self.block_type.as_str();
        let capitalized_type = capitalize(type_name);

        let header = match &self.label {
            Some(label) => format!(
                "## {} - {} (importance: {:.2})",
                capitalized_type, label, self.importance
            ),
            None => format!("## {} (importance: {:.2})", capitalized_type, self.importance),
        };

        format!("{}\n{}", header, self.content)
    }
}
```

#### CoreMemory Markdown Rendering

```rust
impl CoreMemory {
    /// Render core memory as Markdown for human display.
    ///
    /// Example output:
    /// ```markdown
    /// # Core Memory
    ///
    /// ## System (importance: 0.95)
    /// You are a helpful assistant.
    ///
    /// ## Human (importance: 0.75)
    /// User: Alice, software engineer
    /// ```
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
```

#### Helper Function

```rust
/// Capitalize the first letter of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    }
}
```

### Output Format

**XML (for LLMs)**:
```xml
<core_memory>
<block type="system" importance="0.95">
You are a helpful assistant.
</block>
<block type="human" label="profile" importance="0.75">
User: Alice, software engineer
</block>
</core_memory>
```

**Markdown (for humans)**:
```markdown
# Core Memory

## System (importance: 0.95)
You are a helpful assistant.

## Human - profile (importance: 0.75)
User: Alice, software engineer
```

### Test Coverage

11 tests covering:
- Empty core memory rendering
- Single and multiple blocks
- Blocks with labels
- Different importance levels
- Order consistency between XML and Markdown
- DST determinism
- Metadata preservation

All tests pass with deterministic simulation testing.

## Consequences

### Positive

1. **Better Developer Experience**: Human-readable memory inspection
2. **UI-Friendly**: Easy integration into user-facing applications
3. **Parity with memU**: Same dual-format capability
4. **Backward Compatible**: Existing XML rendering unchanged
5. **Deterministic**: Same output for same input (TigerStyle)
6. **Well-Tested**: 11 tests including DST coverage

### Negative

1. **Code Duplication**: Two rendering paths to maintain
2. **Minor Performance Cost**: Additional formatting logic
3. **API Surface Increase**: More methods to document

### Mitigation

- Keep rendering logic simple and parallel to XML
- Use `#[inline]` for hot paths if profiling shows issues
- Document both methods clearly with examples

## Usage Examples

### Debugging

```rust
let mut core = CoreMemory::new();
core.set_block(MemoryBlockType::System, "Be helpful.")?;
core.set_block(MemoryBlockType::Human, "User: Alice")?;

// For LLM context
let xml = core.render();

// For debugging/inspection
let md = core.render_markdown();
println!("{}", md);
```

### UI Integration

```rust
// Display memory to user in web interface
#[get("/memory")]
async fn get_memory(memory: &State<CoreMemory>) -> String {
    memory.render_markdown()
}
```

### Session Handoff

```rust
// Save readable snapshot for agent handoff
let snapshot = memory.render_markdown();
fs::write("session_snapshot.md", snapshot)?;
```

## Related Decisions

- ADR 001: Core Memory Design
- ADR 013: LLM Provider Trait
- ADR 017: Memory Class (UnifiedMemory)

## References

- memU Markdown Rendering: https://github.com/NevaMind-AI/memU
- TigerStyle: https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md
- Markdown Specification: https://commonmark.org/
