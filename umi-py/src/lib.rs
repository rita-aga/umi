//! Umi Python Bindings
//!
//! PyO3 bindings for the Umi memory system.
//!
//! # Usage from Python
//!
//! ```python
//! import umi
//!
//! # Core Memory (32KB, always in context)
//! core = umi.CoreMemory()
//! core.set_block("system", "You are a helpful assistant.")
//! core.set_block("human", "User prefers concise responses.")
//! context = core.render()
//!
//! # Working Memory (1MB KV store with TTL)
//! working = umi.WorkingMemory()
//! working.set("session_id", b"abc123")
//! value = working.get("session_id")
//!
//! # Entities for storage
//! entity = umi.Entity("person", "Alice", "My friend Alice")
//! ```

use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{create_exception, PyErr};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use umi_memory::memory::{
    CoreMemory as RustCoreMemory, CoreMemoryConfig, CoreMemoryError, MemoryBlockType,
    WorkingMemory as RustWorkingMemory, WorkingMemoryError,
};
use umi_memory::storage::{
    Entity as RustEntity, EntityType as RustEntityType, EvolutionRelation as RustEvolutionRelation,
    EvolutionType as RustEvolutionType,
};

// =============================================================================
// MemoryBlockType Helpers
// =============================================================================

/// Parse a block type string to MemoryBlockType.
fn parse_block_type(s: &str) -> PyResult<MemoryBlockType> {
    match s.to_lowercase().as_str() {
        "system" => Ok(MemoryBlockType::System),
        "persona" => Ok(MemoryBlockType::Persona),
        "human" => Ok(MemoryBlockType::Human),
        "facts" => Ok(MemoryBlockType::Facts),
        "goals" => Ok(MemoryBlockType::Goals),
        "scratch" => Ok(MemoryBlockType::Scratch),
        _ => Err(PyValueError::new_err(format!(
            "Invalid block type: '{}'. Valid types: system, persona, human, facts, goals, scratch",
            s
        ))),
    }
}

/// Convert MemoryBlockType to string.
#[allow(dead_code)]
fn block_type_to_str(bt: MemoryBlockType) -> &'static str {
    match bt {
        MemoryBlockType::System => "system",
        MemoryBlockType::Persona => "persona",
        MemoryBlockType::Human => "human",
        MemoryBlockType::Facts => "facts",
        MemoryBlockType::Goals => "goals",
        MemoryBlockType::Scratch => "scratch",
    }
}

// =============================================================================
// EntityType Helpers
// =============================================================================

/// Parse an entity type string to RustEntityType.
fn parse_entity_type(s: &str) -> PyResult<RustEntityType> {
    match s.to_lowercase().as_str() {
        "self" => Ok(RustEntityType::Self_),
        "person" => Ok(RustEntityType::Person),
        "project" => Ok(RustEntityType::Project),
        "topic" => Ok(RustEntityType::Topic),
        "note" => Ok(RustEntityType::Note),
        "task" => Ok(RustEntityType::Task),
        _ => Err(PyValueError::new_err(format!(
            "Invalid entity type: '{}'. Valid types: self, person, project, topic, note, task",
            s
        ))),
    }
}

/// Convert RustEntityType to string.
fn entity_type_to_str(et: RustEntityType) -> &'static str {
    match et {
        RustEntityType::Self_ => "self",
        RustEntityType::Person => "person",
        RustEntityType::Organization => "organization",
        RustEntityType::Project => "project",
        RustEntityType::Topic => "topic",
        RustEntityType::Location => "location",
        RustEntityType::Event => "event",
        RustEntityType::Note => "note",
        RustEntityType::Task => "task",
    }
}

// =============================================================================
// EvolutionType Helpers (ADR-006)
// =============================================================================

/// Parse an evolution type string to RustEvolutionType.
fn parse_evolution_type(s: &str) -> PyResult<RustEvolutionType> {
    match s.to_lowercase().as_str() {
        "update" => Ok(RustEvolutionType::Update),
        "extend" => Ok(RustEvolutionType::Extend),
        "derive" => Ok(RustEvolutionType::Derive),
        "contradict" => Ok(RustEvolutionType::Contradict),
        _ => Err(PyValueError::new_err(format!(
            "Invalid evolution type: '{}'. Valid types: update, extend, derive, contradict",
            s
        ))),
    }
}

/// Convert RustEvolutionType to string.
fn evolution_type_to_str(et: RustEvolutionType) -> &'static str {
    match et {
        RustEvolutionType::Update => "update",
        RustEvolutionType::Extend => "extend",
        RustEvolutionType::Derive => "derive",
        RustEvolutionType::Contradict => "contradict",
    }
}

// =============================================================================
// CoreMemory
// =============================================================================

/// Core Memory - Always in LLM context (~32KB).
///
/// Stores structured blocks of information that are always included
/// in the LLM context window.
///
/// Block types (as strings):
///   - "system": System instructions
///   - "persona": Agent persona
///   - "human": Human/user preferences
///   - "facts": Known facts
///   - "goals": Current goals
///   - "scratch": Scratch/temporary notes
#[pyclass(name = "CoreMemory")]
pub struct PyCoreMemory {
    inner: RustCoreMemory,
}

#[pymethods]
impl PyCoreMemory {
    /// Create a new CoreMemory with default configuration.
    ///
    /// Args:
    ///     max_size_bytes: Optional maximum size in bytes (default: 32KB)
    #[new]
    #[pyo3(signature = (max_size_bytes=None))]
    fn new(max_size_bytes: Option<usize>) -> PyResult<Self> {
        let inner = match max_size_bytes {
            Some(size) => RustCoreMemory::with_config(CoreMemoryConfig::new(size)),
            None => RustCoreMemory::new(),
        };
        Ok(Self { inner })
    }

    /// Set a block's content.
    ///
    /// Args:
    ///     block_type: Block type string (system, persona, human, facts, goals, scratch)
    ///     content: The content string
    #[pyo3(signature = (block_type, content))]
    fn set_block(&mut self, block_type: &str, content: &str) -> PyResult<()> {
        let bt = parse_block_type(block_type)?;

        self.inner.set_block(bt, content).map_err(|e| match e {
            CoreMemoryError::BlockTooLarge {
                size_bytes,
                max_bytes,
            } => {
                PyValueError::new_err(format!("Block too large: {size_bytes} > {max_bytes} bytes"))
            }
            CoreMemoryError::Full {
                current_bytes,
                max_bytes,
                requested_bytes,
            } => PyValueError::new_err(format!(
                "Memory full: {current_bytes}/{max_bytes} bytes, need {requested_bytes}"
            )),
            _ => PyValueError::new_err(format!("Core memory error: {e}")),
        })?;

        Ok(())
    }

    /// Set a block's content with a label.
    ///
    /// Args:
    ///     block_type: Block type string
    ///     label: Label for the block
    ///     content: The content string
    fn set_block_with_label(
        &mut self,
        block_type: &str,
        label: &str,
        content: &str,
    ) -> PyResult<()> {
        let bt = parse_block_type(block_type)?;

        self.inner
            .set_block_with_label(bt, label, content)
            .map_err(|e| PyValueError::new_err(format!("Core memory error: {e}")))?;

        Ok(())
    }

    /// Get a block's content.
    ///
    /// Args:
    ///     block_type: Block type string
    ///
    /// Returns:
    ///     The content string or None if block doesn't exist
    fn get_block(&self, block_type: &str) -> PyResult<Option<String>> {
        let bt = parse_block_type(block_type)?;
        Ok(self.inner.get_block(bt).map(|b| b.content().to_string()))
    }

    /// Remove a block.
    ///
    /// Returns True if block existed and was removed.
    fn remove_block(&mut self, block_type: &str) -> PyResult<bool> {
        let bt = parse_block_type(block_type)?;
        Ok(self.inner.remove_block(bt).is_ok())
    }

    /// Check if a block exists.
    fn has_block(&self, block_type: &str) -> PyResult<bool> {
        let bt = parse_block_type(block_type)?;
        Ok(self.inner.has_block(bt))
    }

    /// Render all blocks to XML string for LLM context.
    fn render(&self) -> String {
        self.inner.render()
    }

    /// Get current used bytes.
    #[getter]
    fn used_bytes(&self) -> usize {
        self.inner.used_bytes()
    }

    /// Get maximum capacity in bytes.
    #[getter]
    fn max_bytes(&self) -> usize {
        self.inner.max_bytes()
    }

    /// Get utilization as a fraction (0.0 to 1.0).
    #[getter]
    fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    /// Clear all blocks.
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "CoreMemory(used={}, max={}, utilization={:.1}%)",
            self.inner.used_bytes(),
            self.inner.max_bytes(),
            self.inner.utilization() * 100.0
        )
    }
}

// =============================================================================
// WorkingMemory
// =============================================================================

/// Working Memory - Session-scoped KV store (~1MB).
///
/// Key-value store with TTL-based expiration for session data.
#[pyclass(name = "WorkingMemory")]
pub struct PyWorkingMemory {
    inner: RustWorkingMemory,
}

#[pymethods]
impl PyWorkingMemory {
    /// Create a new WorkingMemory with default configuration.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustWorkingMemory::new(),
        }
    }

    /// Set a key-value pair with optional TTL.
    ///
    /// Args:
    ///     key: The key string
    ///     value: The value as bytes
    ///     ttl_secs: Optional TTL in seconds (uses default if not specified)
    #[pyo3(signature = (key, value, ttl_secs=None))]
    fn set(&mut self, key: &str, value: &[u8], ttl_secs: Option<u64>) -> PyResult<()> {
        let ttl_ms = ttl_secs.map(|s| s * 1000);
        self.inner.set(key, value, ttl_ms).map_err(|e| match e {
            WorkingMemoryError::EntryTooLarge {
                size_bytes,
                max_bytes,
            } => {
                PyValueError::new_err(format!("Entry too large: {size_bytes} > {max_bytes} bytes"))
            }
            WorkingMemoryError::MemoryFull {
                current_bytes,
                max_bytes,
            } => PyValueError::new_err(format!("Memory full: {current_bytes}/{max_bytes} bytes")),
            WorkingMemoryError::TooManyEntries { count, max_count } => {
                PyValueError::new_err(format!("Too many entries: {count} >= {max_count}"))
            }
            _ => PyValueError::new_err(format!("Working memory error: {e}")),
        })
    }

    /// Get a value by key.
    ///
    /// Returns None if key doesn't exist or has expired.
    fn get(&self, py: Python<'_>, key: &str) -> Option<PyObject> {
        self.inner.get(key).map(|v| PyBytes::new(py, v).into_py(py))
    }

    /// Delete a key.
    ///
    /// Returns True if key existed and was deleted.
    fn delete(&mut self, key: &str) -> bool {
        self.inner.delete(key)
    }

    /// Check if a key exists (and is not expired).
    fn exists(&self, key: &str) -> bool {
        self.inner.exists(key)
    }

    /// Remove all expired entries.
    ///
    /// Returns the number of entries removed.
    fn cleanup_expired(&mut self) -> usize {
        self.inner.cleanup_expired()
    }

    /// Get current used bytes.
    #[getter]
    fn used_bytes(&self) -> usize {
        self.inner.used_bytes()
    }

    /// Get number of entries.
    #[getter]
    fn entry_count(&self) -> usize {
        self.inner.entry_count()
    }

    /// Check if empty.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all entries.
    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkingMemory(entries={}, used={} bytes)",
            self.inner.entry_count(),
            self.inner.used_bytes()
        )
    }
}

// =============================================================================
// Entity
// =============================================================================

/// An entity in archival memory.
///
/// Entity types (as strings):
///   - "self": User's self-representation
///   - "person": Other people
///   - "project": Projects/initiatives
///   - "topic": Topics/concepts
///   - "note": General notes
///   - "task": Tasks/todos
#[pyclass(name = "Entity")]
#[derive(Clone)]
pub struct PyEntity {
    inner: RustEntity,
}

#[pymethods]
impl PyEntity {
    /// Create a new entity.
    ///
    /// Args:
    ///     entity_type: Entity type string (self, person, project, topic, note, task)
    ///     name: Display name
    ///     content: Main content
    #[new]
    fn new(entity_type: &str, name: String, content: String) -> PyResult<Self> {
        let et = parse_entity_type(entity_type)?;
        Ok(Self {
            inner: RustEntity::new(et, name, content),
        })
    }

    /// Get the entity ID.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Get the entity type as string.
    #[getter]
    fn entity_type(&self) -> &'static str {
        entity_type_to_str(self.inner.entity_type)
    }

    /// Get the entity name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get the entity content.
    #[getter]
    fn content(&self) -> &str {
        &self.inner.content
    }

    /// Set the entity content.
    #[setter]
    fn set_content(&mut self, content: String) {
        self.inner.update_content(content);
    }

    /// Get metadata as a dictionary.
    #[getter]
    fn metadata(&self) -> HashMap<String, String> {
        self.inner.metadata.clone()
    }

    /// Get a metadata value by key.
    fn get_meta(&self, key: &str) -> Option<String> {
        self.inner.get_metadata(key).map(|s| s.to_string())
    }

    /// Set a metadata value.
    fn set_meta(&mut self, key: String, value: String) {
        self.inner.metadata.insert(key, value);
    }

    /// Check if entity has an embedding.
    #[getter]
    fn has_embedding(&self) -> bool {
        self.inner.has_embedding()
    }

    /// Get created_at timestamp as ISO string.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.to_rfc3339()
    }

    /// Get updated_at timestamp as ISO string.
    #[getter]
    fn updated_at(&self) -> String {
        self.inner.updated_at.to_rfc3339()
    }

    // =========================================================================
    // Temporal Fields (ADR-006)
    // =========================================================================

    /// Get document_time timestamp as ISO string (when source was created).
    #[getter]
    fn document_time(&self) -> Option<String> {
        self.inner.document_time.map(|t| t.to_rfc3339())
    }

    /// Set document_time from ISO string.
    #[setter]
    fn set_document_time(&mut self, value: Option<String>) -> PyResult<()> {
        self.inner.document_time = match value {
            Some(s) => Some(
                chrono::DateTime::parse_from_rfc3339(&s)
                    .map_err(|e| PyValueError::new_err(format!("Invalid datetime: {e}")))?
                    .with_timezone(&chrono::Utc),
            ),
            None => None,
        };
        Ok(())
    }

    /// Get event_time timestamp as ISO string (when event occurred).
    #[getter]
    fn event_time(&self) -> Option<String> {
        self.inner.event_time.map(|t| t.to_rfc3339())
    }

    /// Set event_time from ISO string.
    #[setter]
    fn set_event_time(&mut self, value: Option<String>) -> PyResult<()> {
        self.inner.event_time = match value {
            Some(s) => Some(
                chrono::DateTime::parse_from_rfc3339(&s)
                    .map_err(|e| PyValueError::new_err(format!("Invalid datetime: {e}")))?
                    .with_timezone(&chrono::Utc),
            ),
            None => None,
        };
        Ok(())
    }

    /// Check if entity has temporal metadata.
    fn has_temporal_metadata(&self) -> bool {
        self.inner.has_temporal_metadata()
    }

    fn __repr__(&self) -> String {
        format!(
            "Entity(id='{}', type='{}', name='{}')",
            self.inner.id,
            entity_type_to_str(self.inner.entity_type),
            self.inner.name
        )
    }
}

impl PyEntity {
    /// Convert from Rust entity.
    pub fn from_rust(entity: RustEntity) -> Self {
        Self { inner: entity }
    }

    /// Get the inner Rust entity.
    pub fn to_rust(&self) -> RustEntity {
        self.inner.clone()
    }
}

// =============================================================================
// EvolutionRelation (ADR-006)
// =============================================================================

/// An evolution relationship between two memories.
///
/// Evolution types (as strings):
///   - "update": New info replaces/corrects old
///   - "extend": New info adds to old
///   - "derive": New info concluded from old
///   - "contradict": New info conflicts with old
#[pyclass(name = "EvolutionRelation")]
#[derive(Clone)]
pub struct PyEvolutionRelation {
    inner: RustEvolutionRelation,
}

#[pymethods]
impl PyEvolutionRelation {
    /// Create a new evolution relation.
    ///
    /// Args:
    ///     source_id: ID of the older/source memory
    ///     target_id: ID of the newer/target memory
    ///     evolution_type: Type string (update, extend, derive, contradict)
    ///     reason: Human-readable reason for the relationship
    ///     confidence: Confidence score (0.0 to 1.0)
    #[new]
    fn new(
        source_id: String,
        target_id: String,
        evolution_type: &str,
        reason: String,
        confidence: f32,
    ) -> PyResult<Self> {
        let et = parse_evolution_type(evolution_type)?;
        Ok(Self {
            inner: RustEvolutionRelation::new(source_id, target_id, et, reason, confidence),
        })
    }

    /// Get the relation ID.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Get the source memory ID.
    #[getter]
    fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    /// Get the target memory ID.
    #[getter]
    fn target_id(&self) -> &str {
        &self.inner.target_id
    }

    /// Get the evolution type as string.
    #[getter]
    fn evolution_type(&self) -> &'static str {
        evolution_type_to_str(self.inner.evolution_type)
    }

    /// Get the reason.
    #[getter]
    fn reason(&self) -> &str {
        &self.inner.reason
    }

    /// Get the confidence score.
    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    /// Get created_at timestamp as ISO string.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.to_rfc3339()
    }

    /// Check if this is a high-confidence relation.
    fn is_high_confidence(&self) -> bool {
        self.inner.is_high_confidence()
    }

    /// Check if this is a conflict that needs resolution.
    fn needs_resolution(&self) -> bool {
        self.inner.needs_resolution()
    }

    /// Check if evolution type is a conflict.
    fn is_conflict(&self) -> bool {
        self.inner.evolution_type.is_conflict()
    }

    /// Check if evolution type is additive.
    fn is_additive(&self) -> bool {
        self.inner.evolution_type.is_additive()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvolutionRelation(source='{}', target='{}', type='{}', confidence={:.2})",
            self.inner.source_id,
            self.inner.target_id,
            evolution_type_to_str(self.inner.evolution_type),
            self.inner.confidence
        )
    }
}

impl PyEvolutionRelation {
    /// Convert from Rust evolution relation.
    pub fn from_rust(relation: RustEvolutionRelation) -> Self {
        Self { inner: relation }
    }

    /// Get the inner Rust evolution relation.
    pub fn to_rust(&self) -> RustEvolutionRelation {
        self.inner.clone()
    }
}

// =============================================================================
// Provider Classes - LLM
// =============================================================================

/// Anthropic LLM Provider (Claude).
///
/// Uses the Anthropic API for text generation and entity extraction.
///
/// Requires `ANTHROPIC_API_KEY` environment variable or pass as constructor argument.
///
/// Example:
///     provider = umi.AnthropicProvider(api_key="sk-ant-...")
#[pyclass(name = "AnthropicProvider")]
pub struct PyAnthropicProvider {
    #[allow(dead_code)]
    inner: umi_memory::llm::AnthropicProvider,
}

#[pymethods]
impl PyAnthropicProvider {
    /// Create a new Anthropic provider.
    ///
    /// Args:
    ///     api_key: Anthropic API key (sk-ant-...)
    #[new]
    fn new(api_key: String) -> PyResult<Self> {
        let inner = umi_memory::llm::AnthropicProvider::new(api_key);
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "AnthropicProvider(model='claude-sonnet-4')".to_string()
    }
}

/// OpenAI LLM Provider (GPT).
///
/// Uses the OpenAI API for text generation and entity extraction.
///
/// Requires `OPENAI_API_KEY` environment variable or pass as constructor argument.
///
/// Example:
///     provider = umi.OpenAIProvider(api_key="sk-...")
#[pyclass(name = "OpenAIProvider")]
pub struct PyOpenAIProvider {
    #[allow(dead_code)]
    inner: umi_memory::llm::OpenAIProvider,
}

#[pymethods]
impl PyOpenAIProvider {
    /// Create a new OpenAI provider.
    ///
    /// Args:
    ///     api_key: OpenAI API key (sk-...)
    #[new]
    fn new(api_key: String) -> PyResult<Self> {
        let inner = umi_memory::llm::OpenAIProvider::new(api_key);
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "OpenAIProvider(model='gpt-4o')".to_string()
    }
}

/// Simulation LLM Provider (for testing).
///
/// Deterministic LLM provider for testing. Returns predictable responses based on seed.
///
/// Example:
///     provider = umi.SimLLMProvider(seed=42)
#[pyclass(name = "SimLLMProvider")]
pub struct PySimLLMProvider {
    #[allow(dead_code)]
    inner: umi_memory::llm::SimLLMProvider,
}

#[pymethods]
impl PySimLLMProvider {
    /// Create a new simulation LLM provider.
    ///
    /// Args:
    ///     seed: Random seed for deterministic behavior
    #[new]
    fn new(seed: u64) -> Self {
        let inner = umi_memory::llm::SimLLMProvider::with_seed(seed);
        Self { inner }
    }

    fn __repr__(&self) -> String {
        "SimLLMProvider(deterministic)".to_string()
    }
}

// =============================================================================
// Provider Classes - Embedding
// =============================================================================

/// OpenAI Embedding Provider.
///
/// Uses OpenAI's text-embedding-3-small model (1536 dimensions).
///
/// Example:
///     provider = umi.OpenAIEmbeddingProvider(api_key="sk-...")
#[pyclass(name = "OpenAIEmbeddingProvider")]
pub struct PyOpenAIEmbeddingProvider {
    #[allow(dead_code)]
    inner: umi_memory::embedding::OpenAIEmbeddingProvider,
}

#[pymethods]
impl PyOpenAIEmbeddingProvider {
    /// Create a new OpenAI embedding provider.
    ///
    /// Args:
    ///     api_key: OpenAI API key (sk-...)
    #[new]
    fn new(api_key: String) -> PyResult<Self> {
        let inner = umi_memory::embedding::OpenAIEmbeddingProvider::new(api_key);
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "OpenAIEmbeddingProvider(model='text-embedding-3-small')".to_string()
    }
}

/// Simulation Embedding Provider (for testing).
///
/// Deterministic embedding provider for testing. Returns predictable embeddings based on seed.
///
/// Example:
///     provider = umi.SimEmbeddingProvider(seed=42)
#[pyclass(name = "SimEmbeddingProvider")]
pub struct PySimEmbeddingProvider {
    #[allow(dead_code)]
    inner: umi_memory::embedding::SimEmbeddingProvider,
}

#[pymethods]
impl PySimEmbeddingProvider {
    /// Create a new simulation embedding provider.
    ///
    /// Args:
    ///     seed: Random seed for deterministic behavior
    #[new]
    fn new(seed: u64) -> Self {
        let inner = umi_memory::embedding::SimEmbeddingProvider::with_seed(seed);
        Self { inner }
    }

    fn __repr__(&self) -> String {
        "SimEmbeddingProvider(deterministic)".to_string()
    }
}

// =============================================================================
// Provider Classes - Storage Backend
// =============================================================================

/// LanceDB Storage Backend.
///
/// Persistent storage using LanceDB (embedded vector database).
///
/// Example:
///     storage = await umi.LanceStorageBackend.connect(path="./umi_db")
#[pyclass(name = "LanceStorageBackend")]
pub struct PyLanceStorageBackend {
    #[allow(dead_code)]
    inner: umi_memory::storage::LanceStorageBackend,
}

#[pymethods]
impl PyLanceStorageBackend {
    /// Connect to LanceDB storage (async constructor).
    ///
    /// Args:
    ///     path: Path to LanceDB directory
    ///
    /// Returns:
    ///     Connected storage backend
    #[staticmethod]
    #[pyo3(name = "connect")]
    fn connect_sync(_py: Python<'_>, path: String) -> PyResult<Self> {
        // Block on async connect
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = runtime
            .block_on(umi_memory::storage::LanceStorageBackend::connect(&path))
            .map_err(|e| PyValueError::new_err(format!("Failed to connect to LanceDB: {}", e)))?;

        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "LanceStorageBackend(connected)".to_string()
    }
}

/// Postgres Storage Backend.
///
/// Persistent storage using PostgreSQL database.
///
/// Example:
///     storage = await umi.PostgresStorageBackend.connect(url="postgresql://localhost/umi")
#[pyclass(name = "PostgresStorageBackend")]
pub struct PyPostgresStorageBackend {
    #[allow(dead_code)]
    inner: umi_memory::storage::PostgresBackend,
}

#[pymethods]
impl PyPostgresStorageBackend {
    /// Connect to Postgres storage (async constructor).
    ///
    /// Args:
    ///     url: Postgres connection URL (postgresql://...)
    ///
    /// Returns:
    ///     Connected storage backend
    #[staticmethod]
    #[pyo3(name = "connect")]
    fn connect_sync(_py: Python<'_>, url: String) -> PyResult<Self> {
        // Block on async connect
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = runtime
            .block_on(umi_memory::storage::PostgresBackend::new(&url))
            .map_err(|e| PyValueError::new_err(format!("Failed to connect to Postgres: {}", e)))?;

        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "PostgresStorageBackend(connected)".to_string()
    }
}

/// Simulation Storage Backend (for testing).
///
/// In-memory storage backend for testing. Data is not persisted.
///
/// Example:
///     storage = umi.SimStorageBackend(seed=42)
#[pyclass(name = "SimStorageBackend")]
pub struct PySimStorageBackend {
    #[allow(dead_code)]
    inner: umi_memory::storage::SimStorageBackend,
}

#[pymethods]
impl PySimStorageBackend {
    /// Create a new simulation storage backend.
    ///
    /// Args:
    ///     seed: Random seed for deterministic behavior
    #[new]
    fn new(seed: u64) -> Self {
        let config = umi_memory::dst::SimConfig::with_seed(seed);
        let inner = umi_memory::storage::SimStorageBackend::new(config);
        Self { inner }
    }

    fn __repr__(&self) -> String {
        "SimStorageBackend(in-memory)".to_string()
    }
}

// =============================================================================
// Provider Classes - Vector Backend
// =============================================================================

/// LanceDB Vector Backend.
///
/// Vector similarity search using LanceDB.
///
/// Example:
///     vector = await umi.LanceVectorBackend.connect(path="./umi_db")
#[pyclass(name = "LanceVectorBackend")]
pub struct PyLanceVectorBackend {
    #[allow(dead_code)]
    inner: umi_memory::storage::LanceVectorBackend,
}

#[pymethods]
impl PyLanceVectorBackend {
    /// Connect to LanceDB vector backend (async constructor).
    ///
    /// Args:
    ///     path: Path to LanceDB directory
    ///
    /// Returns:
    ///     Connected vector backend
    #[staticmethod]
    #[pyo3(name = "connect")]
    fn connect_sync(_py: Python<'_>, path: String) -> PyResult<Self> {
        // Block on async connect
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = runtime
            .block_on(umi_memory::storage::LanceVectorBackend::connect(&path))
            .map_err(|e| PyValueError::new_err(format!("Failed to connect to LanceDB: {}", e)))?;

        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "LanceVectorBackend(connected)".to_string()
    }
}

/// Postgres Vector Backend.
///
/// Vector similarity search using PostgreSQL with pgvector extension.
///
/// Example:
///     vector = await umi.PostgresVectorBackend.connect(url="postgresql://localhost/umi")
#[pyclass(name = "PostgresVectorBackend")]
pub struct PyPostgresVectorBackend {
    #[allow(dead_code)]
    inner: umi_memory::storage::PostgresVectorBackend,
}

#[pymethods]
impl PyPostgresVectorBackend {
    /// Connect to Postgres vector backend (async constructor).
    ///
    /// Args:
    ///     url: Postgres connection URL (postgresql://...)
    ///
    /// Returns:
    ///     Connected vector backend
    #[staticmethod]
    #[pyo3(name = "connect")]
    fn connect_sync(_py: Python<'_>, url: String) -> PyResult<Self> {
        // Block on async connect
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let inner = runtime
            .block_on(umi_memory::storage::PostgresVectorBackend::connect(&url))
            .map_err(|e| PyValueError::new_err(format!("Failed to connect to Postgres: {}", e)))?;

        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        "PostgresVectorBackend(connected)".to_string()
    }
}

/// Simulation Vector Backend (for testing).
///
/// In-memory vector backend for testing. Uses simple cosine similarity.
///
/// Example:
///     vector = umi.SimVectorBackend(seed=42)
#[pyclass(name = "SimVectorBackend")]
pub struct PySimVectorBackend {
    #[allow(dead_code)]
    inner: umi_memory::storage::SimVectorBackend,
}

#[pymethods]
impl PySimVectorBackend {
    /// Create a new simulation vector backend.
    ///
    /// Args:
    ///     seed: Random seed for deterministic behavior
    #[new]
    fn new(seed: u64) -> Self {
        let inner = umi_memory::storage::SimVectorBackend::new(seed);
        Self { inner }
    }

    fn __repr__(&self) -> String {
        "SimVectorBackend(in-memory)".to_string()
    }
}

// =============================================================================
// Options Types
// =============================================================================

/// Options for remember operations.
///
/// Controls how memories are stored, including entity extraction,
/// evolution tracking, and embedding generation.
///
/// Example:
///     options = umi.RememberOptions()
///     options = options.without_extraction().with_importance(0.8)
#[pyclass(name = "RememberOptions")]
#[derive(Clone)]
pub struct PyRememberOptions {
    inner: umi_memory::umi::RememberOptions,
}

#[pymethods]
impl PyRememberOptions {
    /// Create new options with defaults.
    #[new]
    fn new() -> Self {
        Self {
            inner: umi_memory::umi::RememberOptions::default(),
        }
    }

    /// Disable entity extraction (store as raw text).
    fn without_extraction(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().without_extraction();
        slf
    }

    /// Disable evolution tracking.
    fn without_evolution(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().without_evolution();
        slf
    }

    /// Set importance score (0.0-1.0).
    ///
    /// Args:
    ///     importance: Importance score (0.0 = low, 1.0 = high)
    fn with_importance(mut slf: PyRefMut<'_, Self>, importance: f32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_importance(importance);
        slf
    }

    /// Disable embedding generation.
    fn without_embeddings(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().without_embeddings();
        slf
    }

    /// Enable embedding generation (default).
    fn with_embeddings(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_embeddings();
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "RememberOptions(extract={}, evolve={}, embeddings={}, importance={:.2})",
            self.inner.extract_entities,
            self.inner.track_evolution,
            self.inner.generate_embeddings,
            self.inner.importance
        )
    }
}

/// Options for recall operations.
///
/// Controls how memories are retrieved, including result limits,
/// deep search, and time range filtering.
///
/// Example:
///     options = umi.RecallOptions().with_limit(20).with_deep_search()
#[pyclass(name = "RecallOptions")]
#[derive(Clone)]
pub struct PyRecallOptions {
    inner: umi_memory::umi::RecallOptions,
}

#[pymethods]
impl PyRecallOptions {
    /// Create new options with defaults.
    #[new]
    fn new() -> Self {
        Self {
            inner: umi_memory::umi::RecallOptions::default(),
        }
    }

    /// Set maximum number of results (1-100).
    ///
    /// Args:
    ///     limit: Maximum results
    fn with_limit(mut slf: PyRefMut<'_, Self>, limit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner = slf
            .inner
            .clone()
            .with_limit(limit)
            .map_err(|e| PyValueError::new_err(format!("Invalid limit: {}", e)))?;
        Ok(slf)
    }

    /// Enable deep search (LLM rewrites query).
    fn with_deep_search(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_deep_search();
        slf
    }

    /// Disable deep search (fast text-only search).
    fn fast_only(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().fast_only();
        slf
    }

    /// Set time range filter (Unix timestamps in milliseconds).
    ///
    /// Args:
    ///     start_ms: Start time (Unix timestamp in milliseconds)
    ///     end_ms: End time (Unix timestamp in milliseconds)
    fn with_time_range(
        mut slf: PyRefMut<'_, Self>,
        start_ms: u64,
        end_ms: u64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_time_range(start_ms, end_ms);
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "RecallOptions(limit={}, deep_search={:?})",
            self.inner.limit, self.inner.deep_search
        )
    }
}

/// Memory configuration.
///
/// Controls memory behavior like default recall limits.
///
/// Example:
///     config = umi.MemoryConfig().with_recall_limit(50)
#[pyclass(name = "MemoryConfig")]
#[derive(Clone)]
pub struct PyMemoryConfig {
    inner: umi_memory::umi::MemoryConfig,
}

#[pymethods]
impl PyMemoryConfig {
    /// Create new config with defaults.
    #[new]
    fn new() -> Self {
        Self {
            inner: umi_memory::umi::MemoryConfig::default(),
        }
    }

    /// Set default recall limit.
    ///
    /// Args:
    ///     limit: Default limit for recall operations
    fn with_recall_limit(mut slf: PyRefMut<'_, Self>, limit: usize) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_recall_limit(limit);
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryConfig(recall_limit={})",
            self.inner.default_recall_limit
        )
    }
}

// =============================================================================
// Result Types
// =============================================================================

/// Result of a remember operation.
///
/// Contains the stored entities and any detected evolution relationships.
///
/// Example:
///     result = await memory.remember("text", options)
///     print(f"Stored {result.entity_count()} entities")
#[pyclass(name = "RememberResult")]
pub struct PyRememberResult {
    inner: umi_memory::umi::RememberResult,
}

#[pymethods]
impl PyRememberResult {
    /// Get the list of stored entities.
    #[getter]
    fn entities(&self) -> Vec<PyEntity> {
        self.inner
            .entities
            .iter()
            .map(|e| PyEntity::from_rust(e.clone()))
            .collect()
    }

    /// Get the list of detected evolution relationships.
    #[getter]
    fn evolutions(&self) -> Vec<PyEvolutionRelation> {
        self.inner
            .evolutions
            .iter()
            .map(|e| PyEvolutionRelation::from_rust(e.clone()))
            .collect()
    }

    /// Get the number of stored entities.
    fn entity_count(&self) -> usize {
        self.inner.entity_count()
    }

    /// Check if any evolution relationships were detected.
    fn has_evolutions(&self) -> bool {
        self.inner.has_evolutions()
    }

    fn __repr__(&self) -> String {
        format!(
            "RememberResult(entities={}, evolutions={})",
            self.inner.entity_count(),
            self.inner.evolutions.len()
        )
    }
}

impl PyRememberResult {
    /// Convert from Rust RememberResult.
    pub fn from_rust(result: umi_memory::umi::RememberResult) -> Self {
        Self { inner: result }
    }
}

// =============================================================================
// Memory Class
// =============================================================================

/// Main memory interface for Umi.
///
/// Orchestrates all components for remember/recall operations.
///
/// Example (Sim providers):
///     memory = umi.Memory.sim(seed=42)
///     result = await memory.remember("text", options)
///
/// Example (Real providers):
///     llm = umi.AnthropicProvider(api_key="sk-ant-...")
///     embedder = umi.OpenAIEmbeddingProvider(api_key="sk-...")
///     vector = umi.LanceVectorBackend.connect("./db")
///     storage = umi.LanceStorageBackend.connect("./db")
///     memory = umi.Memory.new(llm, embedder, vector, storage)
#[pyclass(name = "Memory")]
pub struct PyMemory {
    // Wrapped in Arc<Mutex<>> for sharing across async boundaries
    // Memory no longer has generic type parameters (uses trait objects internally)
    inner: Arc<Mutex<umi_memory::umi::Memory>>,
}

#[pymethods]
impl PyMemory {
    /// Create a Memory with Sim providers (for testing).
    ///
    /// Args:
    ///     seed: Random seed for deterministic behavior
    ///
    /// Example:
    ///     memory = umi.Memory.sim(seed=42)
    #[staticmethod]
    fn sim(seed: u64) -> Self {
        let inner = umi_memory::umi::Memory::sim(seed);
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Create a Memory with Anthropic LLM + OpenAI embeddings + Lance storage.
    ///
    /// Args:
    ///     anthropic_key: Anthropic API key (sk-ant-...)
    ///     openai_key: OpenAI API key for embeddings (sk-...)
    ///     db_path: Path to LanceDB directory
    ///
    /// Example:
    ///     memory = umi.Memory.with_anthropic(
    ///         anthropic_key="sk-ant-...",
    ///         openai_key="sk-...",
    ///         db_path="./umi_db"
    ///     )
    #[staticmethod]
    fn with_anthropic(
        anthropic_key: String,
        openai_key: String,
        db_path: String,
    ) -> PyResult<Self> {
        // Create providers
        let llm = umi_memory::llm::AnthropicProvider::new(anthropic_key);
        let embedder = umi_memory::embedding::OpenAIEmbeddingProvider::new(openai_key);

        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        // Connect to LanceDB
        let vector = runtime
            .block_on(umi_memory::storage::LanceVectorBackend::connect(&db_path))
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to connect to Lance vector: {}", e))
            })?;

        let storage = runtime
            .block_on(umi_memory::storage::LanceStorageBackend::connect(&db_path))
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to connect to Lance storage: {}", e))
            })?;

        // Create Memory with providers
        let inner = umi_memory::umi::Memory::new(llm, embedder, vector, storage);

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Create a Memory with OpenAI LLM + embeddings + Lance storage.
    ///
    /// Args:
    ///     openai_key: OpenAI API key (sk-...)
    ///     db_path: Path to LanceDB directory
    ///
    /// Example:
    ///     memory = umi.Memory.with_openai(
    ///         openai_key="sk-...",
    ///         db_path="./umi_db"
    ///     )
    #[staticmethod]
    fn with_openai(openai_key: String, db_path: String) -> PyResult<Self> {
        // Create providers
        let llm = umi_memory::llm::OpenAIProvider::new(openai_key.clone());
        let embedder = umi_memory::embedding::OpenAIEmbeddingProvider::new(openai_key);

        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        // Connect to LanceDB
        let vector = runtime
            .block_on(umi_memory::storage::LanceVectorBackend::connect(&db_path))
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to connect to Lance vector: {}", e))
            })?;

        let storage = runtime
            .block_on(umi_memory::storage::LanceStorageBackend::connect(&db_path))
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to connect to Lance storage: {}", e))
            })?;

        // Create Memory with providers
        let inner = umi_memory::umi::Memory::new(llm, embedder, vector, storage);

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Create a Memory with Anthropic LLM + OpenAI embeddings + Postgres storage.
    ///
    /// Args:
    ///     anthropic_key: Anthropic API key (sk-ant-...)
    ///     openai_key: OpenAI API key for embeddings (sk-...)
    ///     postgres_url: PostgreSQL connection URL (postgresql://...)
    ///
    /// Example:
    ///     memory = umi.Memory.with_postgres(
    ///         anthropic_key="sk-ant-...",
    ///         openai_key="sk-...",
    ///         postgres_url="postgresql://localhost/umi"
    ///     )
    #[staticmethod]
    fn with_postgres(
        anthropic_key: String,
        openai_key: String,
        postgres_url: String,
    ) -> PyResult<Self> {
        // Create providers
        let llm = umi_memory::llm::AnthropicProvider::new(anthropic_key);
        let embedder = umi_memory::embedding::OpenAIEmbeddingProvider::new(openai_key);

        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        // Connect to Postgres
        let vector = runtime
            .block_on(umi_memory::storage::PostgresVectorBackend::connect(
                &postgres_url,
            ))
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to connect to Postgres vector: {}", e))
            })?;

        let storage = runtime
            .block_on(umi_memory::storage::PostgresBackend::new(&postgres_url))
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to connect to Postgres storage: {}", e))
            })?;

        // Create Memory with providers
        let inner = umi_memory::umi::Memory::new(llm, embedder, vector, storage);

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    // =========================================================================
    // Async API (native Python async/await)
    // =========================================================================

    /// Store information in memory (async).
    ///
    /// Args:
    ///     text: Text to remember
    ///     options: Remember options
    ///
    /// Returns:
    ///     RememberResult with stored entities and evolutions
    fn remember<'p>(
        &'p mut self,
        py: Python<'p>,
        text: String,
        options: PyRememberOptions,
    ) -> PyResult<&'p PyAny> {
        let inner = Arc::clone(&self.inner);
        let opts = options.inner.clone();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let result = guard
                .remember(&text, opts)
                .await
                .map_err(|e| PyValueError::new_err(format!("Remember failed: {}", e)))?;

            Ok(Python::with_gil(|_py| PyRememberResult::from_rust(result)))
        })
    }

    /// Retrieve memories matching query (async).
    ///
    /// Args:
    ///     query: Search query
    ///     options: Recall options
    ///
    /// Returns:
    ///     List of matching entities
    fn recall<'p>(
        &'p self,
        py: Python<'p>,
        query: String,
        options: PyRecallOptions,
    ) -> PyResult<&'p PyAny> {
        let inner = Arc::clone(&self.inner);
        let opts = options.inner.clone();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let entities = guard
                .recall(&query, opts)
                .await
                .map_err(|e| PyValueError::new_err(format!("Recall failed: {}", e)))?;

            Ok(Python::with_gil(|_py| {
                entities
                    .into_iter()
                    .map(PyEntity::from_rust)
                    .collect::<Vec<_>>()
            }))
        })
    }

    /// Delete entity by ID (async).
    ///
    /// Args:
    ///     entity_id: ID of entity to delete
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn forget<'p>(&'p mut self, py: Python<'p>, entity_id: String) -> PyResult<&'p PyAny> {
        let inner = Arc::clone(&self.inner);

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let deleted = guard
                .forget(&entity_id)
                .await
                .map_err(|e| PyValueError::new_err(format!("Forget failed: {}", e)))?;

            Ok(deleted)
        })
    }

    /// Get entity by ID (async).
    ///
    /// Args:
    ///     entity_id: Entity ID
    ///
    /// Returns:
    ///     Entity if found, None otherwise
    fn get<'p>(&'p self, py: Python<'p>, entity_id: String) -> PyResult<&'p PyAny> {
        let inner = Arc::clone(&self.inner);

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let entity = guard
                .get(&entity_id)
                .await
                .map_err(|e| PyValueError::new_err(format!("Get failed: {}", e)))?;

            Ok(Python::with_gil(|_py| entity.map(PyEntity::from_rust)))
        })
    }

    /// Count total entities in storage (async).
    ///
    /// Returns:
    ///     Total number of entities
    fn count<'p>(&'p self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let inner = Arc::clone(&self.inner);

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let count = guard
                .count()
                .await
                .map_err(|e| PyValueError::new_err(format!("Count failed: {}", e)))?;

            Ok(count)
        })
    }

    // =========================================================================
    // Sync API (blocking)
    // =========================================================================

    /// Store information in memory (blocking).
    ///
    /// Args:
    ///     text: Text to remember
    ///     options: Remember options
    ///
    /// Returns:
    ///     RememberResult with stored entities and evolutions
    fn remember_sync(
        &mut self,
        text: String,
        options: PyRememberOptions,
    ) -> PyResult<PyRememberResult> {
        // Create a tokio runtime and block on the async method
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let mut guard = self.inner.blocking_lock();
        let result = runtime
            .block_on(guard.remember(&text, options.inner.clone()))
            .map_err(|e| PyValueError::new_err(format!("Remember failed: {}", e)))?;

        Ok(PyRememberResult::from_rust(result))
    }

    /// Retrieve memories matching query (blocking).
    ///
    /// Args:
    ///     query: Search query
    ///     options: Recall options
    ///
    /// Returns:
    ///     List of matching entities
    fn recall_sync(&self, query: String, options: PyRecallOptions) -> PyResult<Vec<PyEntity>> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let guard = self.inner.blocking_lock();
        let entities = runtime
            .block_on(guard.recall(&query, options.inner.clone()))
            .map_err(|e| PyValueError::new_err(format!("Recall failed: {}", e)))?;

        Ok(entities.into_iter().map(PyEntity::from_rust).collect())
    }

    /// Delete entity by ID (blocking).
    ///
    /// Args:
    ///     entity_id: ID of entity to delete
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn forget_sync(&mut self, entity_id: String) -> PyResult<bool> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let mut guard = self.inner.blocking_lock();
        let deleted = runtime
            .block_on(guard.forget(&entity_id))
            .map_err(|e| PyValueError::new_err(format!("Forget failed: {}", e)))?;

        Ok(deleted)
    }

    /// Get entity by ID (blocking).
    ///
    /// Args:
    ///     entity_id: Entity ID
    ///
    /// Returns:
    ///     Entity if found, None otherwise
    fn get_sync(&self, entity_id: String) -> PyResult<Option<PyEntity>> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let guard = self.inner.blocking_lock();
        let entity = runtime
            .block_on(guard.get(&entity_id))
            .map_err(|e| PyValueError::new_err(format!("Get failed: {}", e)))?;

        Ok(entity.map(PyEntity::from_rust))
    }

    /// Count total entities in storage (blocking).
    ///
    /// Returns:
    ///     Total number of entities
    fn count_sync(&self) -> PyResult<usize> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;

        let guard = self.inner.blocking_lock();
        let count = runtime
            .block_on(guard.count())
            .map_err(|e| PyValueError::new_err(format!("Count failed: {}", e)))?;

        Ok(count)
    }

    fn __repr__(&self) -> String {
        "Memory(providers=configured)".to_string()
    }
}

// =============================================================================
// Python Exceptions
// =============================================================================

// Base exception for all Umi errors
create_exception!(umi, UmiError, PyException);

// Specific exception types
create_exception!(umi, EmptyTextError, UmiError);
create_exception!(umi, TextTooLongError, UmiError);
create_exception!(umi, EmptyQueryError, UmiError);
create_exception!(umi, InvalidLimitError, UmiError);
create_exception!(umi, StorageError, UmiError);
create_exception!(umi, EmbeddingError, UmiError);
create_exception!(umi, ProviderError, UmiError);

// =============================================================================
// Module
// =============================================================================

/// Umi Python module.
#[pymodule]
fn umi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Memory types
    m.add_class::<PyCoreMemory>()?;
    m.add_class::<PyWorkingMemory>()?;

    // Entity and Evolution (ADR-006)
    m.add_class::<PyEntity>()?;
    m.add_class::<PyEvolutionRelation>()?;

    // LLM Providers
    m.add_class::<PyAnthropicProvider>()?;
    m.add_class::<PyOpenAIProvider>()?;
    m.add_class::<PySimLLMProvider>()?;

    // Embedding Providers
    m.add_class::<PyOpenAIEmbeddingProvider>()?;
    m.add_class::<PySimEmbeddingProvider>()?;

    // Storage Backends
    m.add_class::<PyLanceStorageBackend>()?;
    m.add_class::<PyPostgresStorageBackend>()?;
    m.add_class::<PySimStorageBackend>()?;

    // Vector Backends
    m.add_class::<PyLanceVectorBackend>()?;
    m.add_class::<PyPostgresVectorBackend>()?;
    m.add_class::<PySimVectorBackend>()?;

    // Options and Config
    m.add_class::<PyRememberOptions>()?;
    m.add_class::<PyRecallOptions>()?;
    m.add_class::<PyMemoryConfig>()?;

    // Result Types
    m.add_class::<PyRememberResult>()?;

    // Memory
    m.add_class::<PyMemory>()?;

    // Exceptions
    m.add("UmiError", m.py().get_type::<UmiError>())?;
    m.add("EmptyTextError", m.py().get_type::<EmptyTextError>())?;
    m.add("TextTooLongError", m.py().get_type::<TextTooLongError>())?;
    m.add("EmptyQueryError", m.py().get_type::<EmptyQueryError>())?;
    m.add("InvalidLimitError", m.py().get_type::<InvalidLimitError>())?;
    m.add("StorageError", m.py().get_type::<StorageError>())?;
    m.add("EmbeddingError", m.py().get_type::<EmbeddingError>())?;
    m.add("ProviderError", m.py().get_type::<ProviderError>())?;

    Ok(())
}
