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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

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
        RustEntityType::Project => "project",
        RustEntityType::Topic => "topic",
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

    // Convenience aliases (for cleaner API)
    m.add("CoreMemory", m.getattr("PyCoreMemory")?)?;
    m.add("WorkingMemory", m.getattr("PyWorkingMemory")?)?;
    m.add("Entity", m.getattr("PyEntity")?)?;
    m.add("EvolutionRelation", m.getattr("PyEvolutionRelation")?)?;

    Ok(())
}
