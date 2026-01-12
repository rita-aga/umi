//! # Umi Memory
//!
//! A production-ready memory library for AI agents with deterministic simulation testing.
//!
//! ## Features
//!
//! - **🧠 Smart Memory Management**: Core, working, and archival memory tiers with automatic eviction
//! - **🔍 Dual Retrieval**: Fast vector search + LLM-powered semantic query expansion
//! - **🔄 Evolution Tracking**: Automatically detect updates, contradictions, and derived insights
//! - **✅ Graceful Degradation**: System continues operating even when LLM/storage components fail
//! - **🎯 Deterministic Testing**: Full DST (Deterministic Simulation Testing) for reproducible fault injection
//! - **🚀 Production Backends**: LanceDB for embedded vectors, Postgres for persistence
//!
//! ## Quick Start
//!
//! ```rust
//! use umi_memory::umi::{Memory, RememberOptions, RecallOptions};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create memory with simulation providers (deterministic, seed 42)
//! let mut memory = Memory::sim(42);
//!
//! // Remember information
//! memory.remember(
//!     "Alice is a software engineer at Acme Corp",
//!     RememberOptions::default()
//! ).await?;
//!
//! // Recall information
//! let results = memory.recall("Who works at Acme?", RecallOptions::default()).await?;
//!
//! for entity in results {
//!     println!("Found: {} - {}", entity.name, entity.content);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! **Note on SimLLM**: [`Memory::sim()`](umi::Memory::sim) uses simulation providers that return
//! deterministic placeholder data (entity names like "Alice", "Bob", generic content). This is
//! by design for reproducible testing. For real content extraction, use production LLM providers
//! like [`AnthropicProvider`](llm::AnthropicProvider) or [`OpenAIProvider`](llm::OpenAIProvider).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Memory Orchestrator                   │
//! ├─────────────────────────────────────────────────────────┤
//! │  EntityExtractor  │ DualRetriever  │ EvolutionTracker   │
//! ├─────────────────────────────────────────────────────────┤
//! │  Core Memory (32KB)      │ Always loaded, persistent   │
//! │  Working Memory (1MB)    │ TTL-based eviction, cache   │
//! │  Archival Memory         │ Vector search + storage     │
//! ├─────────────────────────────────────────────────────────┤
//! │  DST Framework           │ Fault injection + simulation│
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Core Components
//!
//! - [`Memory`](umi::Memory) - Main orchestrator, coordinates all components
//! - [`EntityExtractor`](extraction::EntityExtractor) - Extracts structured entities from text
//! - [`DualRetriever`](retrieval::DualRetriever) - Fast + semantic search with RRF merging
//! - [`EvolutionTracker`](evolution::EvolutionTracker) - Detects memory evolution patterns
//!
//! ## Simulation-First Philosophy
//!
//! > "If you're not testing with fault injection, you're not testing."
//!
//! Every component has a deterministic simulation implementation:
//!
//! ```rust
//! use umi_memory::dst::{Simulation, SimConfig, FaultConfig, FaultType};
//!
//! # #[tokio::test]
//! # async fn test_example() {
//! let sim = Simulation::new(SimConfig::with_seed(42))
//!     .with_fault(FaultConfig::new(FaultType::LlmTimeout, 0.1));
//!
//! sim.run(|env| async move {
//!     // Test code with deterministic fault injection
//!     // Same seed = same faults = reproducible bugs
//!     Ok::<_, anyhow::Error>(())
//! }).await.unwrap();
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! - `lance` - LanceDB storage backend
//! - `postgres` - PostgreSQL storage backend
//! - `anthropic` - Anthropic LLM provider (Claude)
//! - `openai` - OpenAI LLM provider (GPT, embeddings)
//! - `llm-providers` - All LLM providers
//! - `embedding-providers` - All embedding providers
//!
//! ## Examples
//!
//! See the [examples directory](https://github.com/rita-aga/umi/tree/main/umi-memory/examples) for:
//!
//! - `quick_start.rs` - Basic remember/recall workflow
//! - `production_setup.rs` - Production configuration
//! - `configuration.rs` - Custom memory settings
//!
//! ## Documentation
//!
//! - [GitHub Repository](https://github.com/rita-aga/umi)
//! - [Architecture Decision Records](https://github.com/rita-aga/umi/tree/main/docs/adr)
//! - [Development Guide](https://github.com/rita-aga/umi/blob/main/CLAUDE.md)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod constants;
pub mod dst;
pub mod embedding;
pub mod evolution;
pub mod extraction;
pub mod llm;
pub mod memory;
pub mod retrieval;
pub mod storage;
pub mod umi;

// Re-export common types
pub use constants::*;
pub use dst::{
    create_simulation,
    run_property_tests,
    test_seeds,
    DeterministicRng,
    FaultConfig,
    FaultInjector,
    FaultType,
    NetworkError,
    NetworkMessage,
    // Property-based testing
    PropertyTest,
    PropertyTestFailure,
    PropertyTestResult,
    PropertyTestable,
    SimClock,
    SimConfig,
    SimEnvironment,
    SimNetwork,
    SimStorage,
    Simulation,
    StorageError,
    TimeAdvanceConfig,
};
pub use memory::{
    ArchivalMemory, ArchivalMemoryConfig, CoreMemory, CoreMemoryConfig, CoreMemoryError,
    MemoryBlock, MemoryBlockId, MemoryBlockType, WorkingMemory, WorkingMemoryConfig,
    WorkingMemoryError,
};
pub use storage::{Entity, EntityBuilder, EntityType, SimStorageBackend, StorageBackend};
// Note: storage::StorageError not re-exported to avoid conflict with dst::StorageError
// Use `umi_memory::storage::StorageError` explicitly if needed

#[cfg(feature = "lance")]
pub use storage::LanceStorageBackend;

// LLM Provider exports
pub use llm::{CompletionRequest, LLMProvider, ProviderError, SimLLMProvider};

#[cfg(feature = "anthropic")]
pub use llm::AnthropicProvider;

#[cfg(feature = "openai")]
pub use llm::OpenAIProvider;

// Embedding Provider exports
pub use embedding::{EmbeddingError, EmbeddingProvider, SimEmbeddingProvider};

#[cfg(feature = "embedding-openai")]
pub use embedding::OpenAIEmbeddingProvider;

// Extraction exports
pub use extraction::{
    EntityExtractor, ExtractedEntity, ExtractedRelation, ExtractionError, ExtractionOptions,
    ExtractionResult,
};
// Note: extraction::EntityType and extraction::RelationType not re-exported
// to avoid conflict with storage::EntityType. Use explicit paths if needed.

// Retrieval exports
pub use retrieval::{DualRetriever, RetrievalError, SearchOptions, SearchResult};

// Evolution exports
pub use evolution::{DetectionOptions, DetectionResult, EvolutionError, EvolutionTracker};

// Umi Memory exports (main API)
pub use umi::{Memory, MemoryError, RecallOptions, RememberOptions, RememberResult};
