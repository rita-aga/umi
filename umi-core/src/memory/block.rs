//! Memory Block - Individual blocks within Core Memory
//!
//! TigerStyle: Each block has explicit type, label, and size tracking.

use std::fmt;
use uuid::Uuid;

use crate::constants::{CORE_MEMORY_BLOCK_LABEL_BYTES_MAX, CORE_MEMORY_BLOCK_SIZE_BYTES_MAX};

/// Unique identifier for a memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryBlockId(Uuid);

impl MemoryBlockId {
    /// Create a new random block ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create a block ID from a UUID.
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for MemoryBlockId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Types of memory blocks in core memory.
///
/// TigerStyle: Fixed set of block types with clear purposes.
///
/// # Block Types
///
/// - `System` - System instructions and prompts (highest priority)
/// - `Persona` - AI personality and behavior guidelines
/// - `Human` - Information about the human user
/// - `Facts` - Key facts and knowledge to remember
/// - `Goals` - Current objectives and tasks
/// - `Scratch` - Temporary working space for reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryBlockType {
    /// System instructions and prompts
    System,
    /// AI personality and behavior guidelines
    Persona,
    /// Information about the human user
    Human,
    /// Key facts and knowledge to remember
    Facts,
    /// Current objectives and tasks
    Goals,
    /// Temporary working space for reasoning
    Scratch,
}

impl MemoryBlockType {
    /// Get the string representation for XML rendering.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Persona => "persona",
            Self::Human => "human",
            Self::Facts => "facts",
            Self::Goals => "goals",
            Self::Scratch => "scratch",
        }
    }

    /// Get render priority (lower = rendered first).
    ///
    /// TigerStyle: Explicit ordering for deterministic rendering.
    #[must_use]
    pub fn priority(&self) -> u8 {
        match self {
            Self::System => 0,
            Self::Persona => 1,
            Self::Human => 2,
            Self::Facts => 3,
            Self::Goals => 4,
            Self::Scratch => 5,
        }
    }

    /// Get all block types in render order.
    #[must_use]
    pub fn all_ordered() -> &'static [MemoryBlockType] {
        &[
            Self::System,
            Self::Persona,
            Self::Human,
            Self::Facts,
            Self::Goals,
            Self::Scratch,
        ]
    }
}

impl fmt::Display for MemoryBlockType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single memory block.
///
/// TigerStyle: Immutable content with explicit size tracking.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Unique identifier
    id: MemoryBlockId,
    /// Block type
    block_type: MemoryBlockType,
    /// Optional human-readable label
    label: Option<String>,
    /// Block content
    content: String,
    /// Cached content size in bytes
    size_bytes: usize,
    /// Creation timestamp (milliseconds since epoch)
    created_at_ms: u64,
    /// Last modification timestamp (milliseconds since epoch)
    modified_at_ms: u64,
}

impl MemoryBlock {
    /// Create a new memory block.
    ///
    /// # Panics
    /// Panics if content exceeds `CORE_MEMORY_BLOCK_SIZE_BYTES_MAX`.
    /// Panics if label exceeds `CORE_MEMORY_BLOCK_LABEL_BYTES_MAX`.
    #[must_use]
    pub fn new(block_type: MemoryBlockType, content: impl Into<String>, now_ms: u64) -> Self {
        let content = content.into();

        // Preconditions
        assert!(
            content.len() <= CORE_MEMORY_BLOCK_SIZE_BYTES_MAX,
            "block content {} bytes exceeds max {}",
            content.len(),
            CORE_MEMORY_BLOCK_SIZE_BYTES_MAX
        );

        let size_bytes = content.len();
        let id = MemoryBlockId::new();

        // Postconditions
        let result = Self {
            id,
            block_type,
            label: None,
            content,
            size_bytes,
            created_at_ms: now_ms,
            modified_at_ms: now_ms,
        };

        assert_eq!(
            result.size_bytes,
            result.content.len(),
            "size must match content"
        );

        result
    }

    /// Create a new memory block with a label.
    ///
    /// # Panics
    /// Panics if content exceeds `CORE_MEMORY_BLOCK_SIZE_BYTES_MAX`.
    /// Panics if label exceeds `CORE_MEMORY_BLOCK_LABEL_BYTES_MAX`.
    #[must_use]
    pub fn with_label(
        block_type: MemoryBlockType,
        label: impl Into<String>,
        content: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        let label = label.into();
        let content = content.into();

        // Preconditions
        assert!(
            label.len() <= CORE_MEMORY_BLOCK_LABEL_BYTES_MAX,
            "block label {} bytes exceeds max {}",
            label.len(),
            CORE_MEMORY_BLOCK_LABEL_BYTES_MAX
        );
        assert!(
            content.len() <= CORE_MEMORY_BLOCK_SIZE_BYTES_MAX,
            "block content {} bytes exceeds max {}",
            content.len(),
            CORE_MEMORY_BLOCK_SIZE_BYTES_MAX
        );

        let size_bytes = content.len();
        let id = MemoryBlockId::new();

        Self {
            id,
            block_type,
            label: Some(label),
            content,
            size_bytes,
            created_at_ms: now_ms,
            modified_at_ms: now_ms,
        }
    }

    /// Get the block ID.
    #[must_use]
    pub fn id(&self) -> MemoryBlockId {
        self.id
    }

    /// Get the block type.
    #[must_use]
    pub fn block_type(&self) -> MemoryBlockType {
        self.block_type
    }

    /// Get the optional label.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the block content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get the size in bytes (cached).
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get creation timestamp.
    #[must_use]
    pub fn created_at_ms(&self) -> u64 {
        self.created_at_ms
    }

    /// Get modification timestamp.
    #[must_use]
    pub fn modified_at_ms(&self) -> u64 {
        self.modified_at_ms
    }

    /// Update the content.
    ///
    /// # Panics
    /// Panics if new content exceeds `CORE_MEMORY_BLOCK_SIZE_BYTES_MAX`.
    pub fn set_content(&mut self, content: impl Into<String>, now_ms: u64) {
        let content = content.into();

        // Precondition
        assert!(
            content.len() <= CORE_MEMORY_BLOCK_SIZE_BYTES_MAX,
            "block content {} bytes exceeds max {}",
            content.len(),
            CORE_MEMORY_BLOCK_SIZE_BYTES_MAX
        );

        self.size_bytes = content.len();
        self.content = content;
        self.modified_at_ms = now_ms;

        // Postcondition
        assert_eq!(
            self.size_bytes,
            self.content.len(),
            "size must match content"
        );
    }

    /// Render the block as XML for LLM context.
    ///
    /// TigerStyle: Deterministic, predictable output format.
    #[must_use]
    pub fn render(&self) -> String {
        let type_attr = self.block_type.as_str();
        match &self.label {
            Some(label) => {
                format!(
                    "<block type=\"{}\" label=\"{}\">\n{}\n</block>",
                    type_attr, label, self.content
                )
            }
            None => {
                format!("<block type=\"{}\">\n{}\n</block>", type_attr, self.content)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_type_as_str() {
        assert_eq!(MemoryBlockType::System.as_str(), "system");
        assert_eq!(MemoryBlockType::Persona.as_str(), "persona");
        assert_eq!(MemoryBlockType::Human.as_str(), "human");
        assert_eq!(MemoryBlockType::Facts.as_str(), "facts");
        assert_eq!(MemoryBlockType::Goals.as_str(), "goals");
        assert_eq!(MemoryBlockType::Scratch.as_str(), "scratch");
    }

    #[test]
    fn test_block_type_priority() {
        assert_eq!(MemoryBlockType::System.priority(), 0);
        assert_eq!(MemoryBlockType::Scratch.priority(), 5);
    }

    #[test]
    fn test_block_type_all_ordered() {
        let types = MemoryBlockType::all_ordered();
        assert_eq!(types.len(), 6);
        assert_eq!(types[0], MemoryBlockType::System);
        assert_eq!(types[5], MemoryBlockType::Scratch);
    }

    #[test]
    fn test_memory_block_new() {
        let block = MemoryBlock::new(MemoryBlockType::System, "Hello world", 1000);

        assert_eq!(block.block_type(), MemoryBlockType::System);
        assert_eq!(block.content(), "Hello world");
        assert_eq!(block.size_bytes(), 11);
        assert!(block.label().is_none());
        assert_eq!(block.created_at_ms(), 1000);
        assert_eq!(block.modified_at_ms(), 1000);
    }

    #[test]
    fn test_memory_block_with_label() {
        let block = MemoryBlock::with_label(
            MemoryBlockType::Facts,
            "user_preferences",
            "Likes cats",
            2000,
        );

        assert_eq!(block.block_type(), MemoryBlockType::Facts);
        assert_eq!(block.label(), Some("user_preferences"));
        assert_eq!(block.content(), "Likes cats");
        assert_eq!(block.size_bytes(), 10);
    }

    #[test]
    fn test_memory_block_set_content() {
        let mut block = MemoryBlock::new(MemoryBlockType::Scratch, "initial", 1000);
        assert_eq!(block.size_bytes(), 7);

        block.set_content("updated content here", 2000);

        assert_eq!(block.content(), "updated content here");
        assert_eq!(block.size_bytes(), 20);
        assert_eq!(block.created_at_ms(), 1000);
        assert_eq!(block.modified_at_ms(), 2000);
    }

    #[test]
    fn test_memory_block_render() {
        let block = MemoryBlock::new(MemoryBlockType::System, "You are helpful.", 1000);
        let rendered = block.render();

        assert!(rendered.contains("<block type=\"system\">"));
        assert!(rendered.contains("You are helpful."));
        assert!(rendered.contains("</block>"));
    }

    #[test]
    fn test_memory_block_render_with_label() {
        let block = MemoryBlock::with_label(MemoryBlockType::Human, "profile", "Name: Alice", 1000);
        let rendered = block.render();

        assert!(rendered.contains("type=\"human\""));
        assert!(rendered.contains("label=\"profile\""));
        assert!(rendered.contains("Name: Alice"));
    }

    #[test]
    fn test_memory_block_id_unique() {
        let id1 = MemoryBlockId::new();
        let id2 = MemoryBlockId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    #[should_panic(expected = "block content")]
    fn test_memory_block_content_too_large() {
        let large_content = "x".repeat(CORE_MEMORY_BLOCK_SIZE_BYTES_MAX + 1);
        let _ = MemoryBlock::new(MemoryBlockType::System, large_content, 1000);
    }

    #[test]
    #[should_panic(expected = "block label")]
    fn test_memory_block_label_too_large() {
        let large_label = "x".repeat(CORE_MEMORY_BLOCK_LABEL_BYTES_MAX + 1);
        let _ = MemoryBlock::with_label(MemoryBlockType::System, large_label, "content", 1000);
    }
}
