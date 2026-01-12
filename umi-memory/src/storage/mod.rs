//! Storage - Backend Trait and Implementations
//!
//! TigerStyle: Abstract storage with simulation-first testing.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StorageBackend Trait                      │
//! └─────────────────────────────────────────────────────────────┘
//!          ↑                    ↑                    ↑
//!          │                    │                    │
//! ┌────────┴────────┐  ┌────────┴────────┐  ┌───────┴────────┐
//! │SimStorageBackend│  │LanceStorageBack │  │ PostgresBackend│
//! │   (testing)     │  │   (embedded)    │  │   (server)     │
//! └─────────────────┘  └─────────────────┘  └────────────────┘
//! ```
//!
//! # Simulation-First
//!
//! Tests are written BEFORE implementation. SimStorageBackend enables
//! deterministic testing with fault injection.

mod backend;
mod entity;
mod error;
mod evolution;
mod sim;
mod vector;

#[cfg(feature = "postgres")]
mod postgres;

#[cfg(feature = "lance")]
mod lance;

pub use backend::StorageBackend;
pub use entity::{Entity, EntityBuilder, EntityType, SourceRef};
pub use error::{StorageError, StorageResult};
pub use evolution::{EvolutionRelation, EvolutionRelationBuilder, EvolutionType};
pub use sim::SimStorageBackend;
pub use vector::{SimVectorBackend, VectorBackend, VectorSearchResult};

#[cfg(feature = "postgres")]
pub use postgres::PostgresBackend;

#[cfg(feature = "lance")]
pub use lance::LanceStorageBackend;
