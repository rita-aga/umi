//! Memory - Core and Working Memory for Umi
//!
//! TigerStyle: Three-tier memory architecture with explicit limits.
//!
//! # Memory Tiers
//!
//! 1. **Core Memory** (~32KB) - Always in LLM context
//! 2. **Working Memory** (~1MB) - Session state, KV store with TTL
//! 3. **Archival Memory** (unlimited) - Long-term storage (separate module)
//!
//! # Example
//!
//! ```rust
//! use umi_core::memory::{CoreMemory, MemoryBlockType};
//!
//! let mut core = CoreMemory::new();
//! core.set_block(MemoryBlockType::System, "You are a helpful assistant.").unwrap();
//! core.set_block(MemoryBlockType::Human, "User prefers concise responses.").unwrap();
//!
//! let context = core.render();
//! // <core_memory>
//! // <block type="system">You are a helpful assistant.</block>
//! // <block type="human">User prefers concise responses.</block>
//! // </core_memory>
//! ```

mod archival;
mod block;
mod core;
mod working;

pub use archival::{ArchivalMemory, ArchivalMemoryConfig};
pub use block::{MemoryBlock, MemoryBlockId, MemoryBlockType};
pub use core::{CoreMemory, CoreMemoryConfig, CoreMemoryError};
pub use working::{WorkingMemory, WorkingMemoryConfig, WorkingMemoryError};
