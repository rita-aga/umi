//! Entity - Structured data for Archival Memory
//!
//! `TigerStyle`: Explicit types, validation, builder pattern.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::constants::{ENTITY_CONTENT_BYTES_MAX, ENTITY_NAME_BYTES_MAX};

// =============================================================================
// Entity Type
// =============================================================================

/// Types of entities in archival memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// User's self-representation
    #[serde(rename = "self")]
    Self_,
    /// Other people
    Person,
    /// Organizations or companies
    Organization,
    /// Projects/initiatives
    Project,
    /// Topics/concepts
    Topic,
    /// Locations or places
    Location,
    /// Events or meetings
    Event,
    /// General notes
    Note,
    /// Tasks/todos
    Task,
}

impl EntityType {
    /// Get string representation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Self_ => "self",
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Project => "project",
            Self::Topic => "topic",
            Self::Location => "location",
            Self::Event => "event",
            Self::Note => "note",
            Self::Task => "task",
        }
    }

    /// Parse from string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "self" => Some(Self::Self_),
            "person" => Some(Self::Person),
            "organization" | "org" => Some(Self::Organization),
            "project" => Some(Self::Project),
            "topic" => Some(Self::Topic),
            "location" | "place" => Some(Self::Location),
            "event" | "meeting" => Some(Self::Event),
            "note" => Some(Self::Note),
            "task" => Some(Self::Task),
            _ => None,
        }
    }

    /// Get all entity types in order.
    #[must_use]
    pub fn all() -> &'static [EntityType] {
        &[
            Self::Self_,
            Self::Person,
            Self::Organization,
            Self::Project,
            Self::Topic,
            Self::Location,
            Self::Event,
            Self::Note,
            Self::Task,
        ]
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Source Reference (for multimedia content)
// =============================================================================

/// Reference to source content (URL, file path, S3 URI, etc.)
///
/// Used when an entity was extracted from multimedia content (images, audio,
/// video, PDFs, web pages). The entity stores the extracted text/summary,
/// while `SourceRef` points to the original content.
///
/// # Example
///
/// ```
/// use umi_memory::storage::SourceRef;
///
/// // Image that was analyzed
/// let image_ref = SourceRef::new("file:///photos/meeting.jpg".to_string())
///     .with_mime_type("image/jpeg".to_string())
///     .with_size_bytes(1024 * 500);
///
/// // PDF that was extracted
/// let pdf_ref = SourceRef::new("s3://docs/report.pdf".to_string())
///     .with_mime_type("application/pdf".to_string())
///     .with_checksum("sha256:abc123...".to_string());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    /// URI to the source (file://, https://, s3://, etc.)
    pub uri: String,
    /// MIME type of the source (image/png, audio/mp3, application/pdf, etc.)
    pub mime_type: Option<String>,
    /// Size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Checksum for integrity verification (e.g., "sha256:abc123...")
    pub checksum: Option<String>,
}

impl SourceRef {
    /// Create a new source reference with just the URI.
    #[must_use]
    pub fn new(uri: String) -> Self {
        Self {
            uri,
            mime_type: None,
            size_bytes: None,
            checksum: None,
        }
    }

    /// Set the MIME type.
    #[must_use]
    pub fn with_mime_type(mut self, mime_type: String) -> Self {
        self.mime_type = Some(mime_type);
        self
    }

    /// Set the size in bytes.
    #[must_use]
    pub fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    /// Set the checksum.
    #[must_use]
    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = Some(checksum);
        self
    }

    /// Check if this is a local file reference.
    #[must_use]
    pub fn is_local(&self) -> bool {
        self.uri.starts_with("file://")
    }

    /// Check if this is a remote URL.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        self.uri.starts_with("http://") || self.uri.starts_with("https://")
    }

    /// Check if this is an S3 reference.
    #[must_use]
    pub fn is_s3(&self) -> bool {
        self.uri.starts_with("s3://")
    }

    /// Get the file extension from the URI (if any).
    #[must_use]
    pub fn extension(&self) -> Option<&str> {
        self.uri.rsplit('.').next().filter(|ext| !ext.contains('/'))
    }
}

impl std::fmt::Display for SourceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.uri)
    }
}

// =============================================================================
// Entity
// =============================================================================

/// An entity in archival memory.
///
/// `TigerStyle`: Explicit fields, no Option where not needed.
///
/// # Temporal Metadata (ADR-006)
///
/// Entities support bi-temporal tracking:
/// - `document_time`: When the source document was created (e.g., email sent date)
/// - `event_time`: When the event actually occurred (e.g., "I met Alice last Tuesday")
///
/// This enables queries like "What happened last week?" to find events by when they
/// occurred, not just when they were recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier (UUID v4)
    pub id: String,
    /// Type of entity
    pub entity_type: EntityType,
    /// Display name
    pub name: String,
    /// Main content
    pub content: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Embedding vector (for semantic search)
    pub embedding: Option<Vec<f32>>,
    /// Creation timestamp (when stored in Umi)
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// When source document was created (e.g., email sent date)
    pub document_time: Option<DateTime<Utc>>,
    /// When event actually occurred (e.g., "last Tuesday")
    pub event_time: Option<DateTime<Utc>>,
    /// Reference to source content (for multimedia workflows)
    pub source_ref: Option<SourceRef>,
}

impl Entity {
    /// Generate a deterministic ID for an entity based on its name and type.
    ///
    /// This ensures the same entity (name + type) always gets the same ID,
    /// enabling storage layer deduplication.
    ///
    /// Uses UUID v5 (SHA-1 hash) with a fixed namespace for determinism.
    #[must_use]
    fn generate_deterministic_id(entity_type: EntityType, name: &str) -> String {
        // Use a fixed namespace UUID for Umi entities
        // This UUID was randomly generated once and is now fixed
        const UMI_NAMESPACE: uuid::Uuid = uuid::Uuid::from_bytes([
            0x6b, 0xa4, 0x28, 0x1c, 0x4d, 0x9f, 0x4f, 0x3a, 0x8c, 0x7e, 0x9d, 0x2b, 0x5f, 0x1e,
            0x6a, 0x3c,
        ]);

        // Combine entity type and name for uniqueness
        let unique_key = format!("{}:{}", entity_type.as_str(), name);

        // Generate deterministic UUID v5
        uuid::Uuid::new_v5(&UMI_NAMESPACE, unique_key.as_bytes()).to_string()
    }

    /// Create a new entity with required fields.
    ///
    /// **Deduplication**: Entity IDs are deterministic based on name + type.
    /// This means storing the same entity twice will use the same ID,
    /// allowing storage backends to deduplicate properly.
    ///
    /// # Panics
    /// Panics if name or content exceed limits.
    #[must_use]
    pub fn new(entity_type: EntityType, name: String, content: String) -> Self {
        // Preconditions
        assert!(
            name.len() <= ENTITY_NAME_BYTES_MAX,
            "name {} bytes exceeds max {}",
            name.len(),
            ENTITY_NAME_BYTES_MAX
        );
        assert!(
            content.len() <= ENTITY_CONTENT_BYTES_MAX,
            "content {} bytes exceeds max {}",
            content.len(),
            ENTITY_CONTENT_BYTES_MAX
        );

        let now = Utc::now();
        Self {
            id: Self::generate_deterministic_id(entity_type, &name),
            entity_type,
            name,
            content,
            metadata: HashMap::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
            document_time: None,
            event_time: None,
            source_ref: None,
        }
    }

    /// Create a builder for more complex entity construction.
    #[must_use]
    pub fn builder(entity_type: EntityType, name: String, content: String) -> EntityBuilder {
        EntityBuilder::new(entity_type, name, content)
    }

    /// Check if entity has an embedding.
    #[must_use]
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Get metadata value.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Update content and timestamp.
    pub fn update_content(&mut self, content: String) {
        assert!(
            content.len() <= ENTITY_CONTENT_BYTES_MAX,
            "content {} bytes exceeds max {}",
            content.len(),
            ENTITY_CONTENT_BYTES_MAX
        );
        self.content = content;
        self.updated_at = Utc::now();
    }

    /// Set embedding.
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
        self.updated_at = Utc::now();
    }

    /// Set document time (when source was created).
    pub fn set_document_time(&mut self, time: DateTime<Utc>) {
        self.document_time = Some(time);
        self.updated_at = Utc::now();
    }

    /// Set event time (when event actually occurred).
    pub fn set_event_time(&mut self, time: DateTime<Utc>) {
        self.event_time = Some(time);
        self.updated_at = Utc::now();
    }

    /// Get document time.
    #[must_use]
    pub fn document_time(&self) -> Option<DateTime<Utc>> {
        self.document_time
    }

    /// Get event time.
    #[must_use]
    pub fn event_time(&self) -> Option<DateTime<Utc>> {
        self.event_time
    }

    /// Check if entity has temporal metadata.
    #[must_use]
    pub fn has_temporal_metadata(&self) -> bool {
        self.document_time.is_some() || self.event_time.is_some()
    }

    /// Set source reference (for multimedia content).
    pub fn set_source_ref(&mut self, source_ref: SourceRef) {
        self.source_ref = Some(source_ref);
        self.updated_at = Utc::now();
    }

    /// Get source reference.
    #[must_use]
    pub fn source_ref(&self) -> Option<&SourceRef> {
        self.source_ref.as_ref()
    }

    /// Check if entity has a source reference.
    #[must_use]
    pub fn has_source_ref(&self) -> bool {
        self.source_ref.is_some()
    }
}

// =============================================================================
// Entity Builder
// =============================================================================

/// Builder for Entity with fluent API.
#[derive(Debug)]
pub struct EntityBuilder {
    entity_type: EntityType,
    name: String,
    content: String,
    id: Option<String>,
    metadata: HashMap<String, String>,
    embedding: Option<Vec<f32>>,
    created_at: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
    document_time: Option<DateTime<Utc>>,
    event_time: Option<DateTime<Utc>>,
    source_ref: Option<SourceRef>,
}

impl EntityBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(entity_type: EntityType, name: String, content: String) -> Self {
        Self {
            entity_type,
            name,
            content,
            id: None,
            metadata: HashMap::new(),
            embedding: None,
            created_at: None,
            updated_at: None,
            document_time: None,
            event_time: None,
            source_ref: None,
        }
    }

    /// Set custom ID.
    #[must_use]
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// Add metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set embedding.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set creation timestamp (for DST).
    #[must_use]
    pub fn with_created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = Some(created_at);
        self
    }

    /// Set update timestamp (for DST).
    #[must_use]
    pub fn with_updated_at(mut self, updated_at: DateTime<Utc>) -> Self {
        self.updated_at = Some(updated_at);
        self
    }

    /// Set document time (when source was created).
    #[must_use]
    pub fn with_document_time(mut self, document_time: DateTime<Utc>) -> Self {
        self.document_time = Some(document_time);
        self
    }

    /// Set event time (when event actually occurred).
    #[must_use]
    pub fn with_event_time(mut self, event_time: DateTime<Utc>) -> Self {
        self.event_time = Some(event_time);
        self
    }

    /// Set source reference (for multimedia content).
    #[must_use]
    pub fn with_source_ref(mut self, source_ref: SourceRef) -> Self {
        self.source_ref = Some(source_ref);
        self
    }

    /// Build the entity.
    ///
    /// # Panics
    /// Panics if name or content exceed limits.
    #[must_use]
    pub fn build(self) -> Entity {
        // Preconditions
        assert!(
            self.name.len() <= ENTITY_NAME_BYTES_MAX,
            "name {} bytes exceeds max {}",
            self.name.len(),
            ENTITY_NAME_BYTES_MAX
        );
        assert!(
            self.content.len() <= ENTITY_CONTENT_BYTES_MAX,
            "content {} bytes exceeds max {}",
            self.content.len(),
            ENTITY_CONTENT_BYTES_MAX
        );

        let now = Utc::now();
        Entity {
            id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            entity_type: self.entity_type,
            name: self.name,
            content: self.content,
            metadata: self.metadata,
            embedding: self.embedding,
            created_at: self.created_at.unwrap_or(now),
            updated_at: self.updated_at.unwrap_or(now),
            document_time: self.document_time,
            event_time: self.event_time,
            source_ref: self.source_ref,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_as_str() {
        assert_eq!(EntityType::Self_.as_str(), "self");
        assert_eq!(EntityType::Person.as_str(), "person");
        assert_eq!(EntityType::Project.as_str(), "project");
        assert_eq!(EntityType::Topic.as_str(), "topic");
        assert_eq!(EntityType::Note.as_str(), "note");
        assert_eq!(EntityType::Task.as_str(), "task");
    }

    #[test]
    fn test_entity_type_from_str() {
        assert_eq!(EntityType::from_str("self"), Some(EntityType::Self_));
        assert_eq!(EntityType::from_str("PERSON"), Some(EntityType::Person));
        assert_eq!(EntityType::from_str("Project"), Some(EntityType::Project));
        assert_eq!(EntityType::from_str("unknown"), None);
    }

    #[test]
    fn test_entity_new() {
        let entity = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "My friend Alice".to_string(),
        );

        assert!(!entity.id.is_empty());
        assert_eq!(entity.entity_type, EntityType::Person);
        assert_eq!(entity.name, "Alice");
        assert_eq!(entity.content, "My friend Alice");
        assert!(entity.metadata.is_empty());
        assert!(entity.embedding.is_none());
    }

    #[test]
    fn test_entity_builder() {
        let entity = Entity::builder(
            EntityType::Project,
            "Umi".to_string(),
            "Memory system".to_string(),
        )
        .with_id("custom-id".to_string())
        .with_metadata("status".to_string(), "active".to_string())
        .with_embedding(vec![0.1, 0.2, 0.3])
        .build();

        assert_eq!(entity.id, "custom-id");
        assert_eq!(entity.entity_type, EntityType::Project);
        assert_eq!(entity.get_metadata("status"), Some("active"));
        assert!(entity.has_embedding());
    }

    #[test]
    fn test_entity_update_content() {
        let mut entity = Entity::new(
            EntityType::Note,
            "Test".to_string(),
            "Original content".to_string(),
        );
        let original_updated = entity.updated_at;

        // Small delay to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(10));

        entity.update_content("New content".to_string());

        assert_eq!(entity.content, "New content");
        assert!(entity.updated_at >= original_updated);
    }

    #[test]
    #[should_panic(expected = "name")]
    fn test_entity_name_too_long() {
        let long_name = "x".repeat(ENTITY_NAME_BYTES_MAX + 1);
        let _ = Entity::new(EntityType::Note, long_name, "content".to_string());
    }

    #[test]
    #[should_panic(expected = "content")]
    fn test_entity_content_too_long() {
        let long_content = "x".repeat(ENTITY_CONTENT_BYTES_MAX + 1);
        let _ = Entity::new(EntityType::Note, "name".to_string(), long_content);
    }

    // =========================================================================
    // Temporal Metadata Tests (ADR-006)
    // =========================================================================

    #[test]
    fn test_entity_new_has_no_temporal_metadata() {
        let entity = Entity::new(EntityType::Note, "Test".to_string(), "Content".to_string());

        // Precondition: new entities have no temporal metadata
        assert!(entity.document_time.is_none());
        assert!(entity.event_time.is_none());
        assert!(!entity.has_temporal_metadata());
    }

    #[test]
    fn test_entity_set_document_time() {
        let mut entity = Entity::new(
            EntityType::Note,
            "Email".to_string(),
            "Content from email".to_string(),
        );

        let email_sent_time = Utc::now() - chrono::Duration::days(7);
        entity.set_document_time(email_sent_time);

        // Postconditions
        assert_eq!(entity.document_time(), Some(email_sent_time));
        assert!(entity.has_temporal_metadata());
        assert!(entity.event_time().is_none()); // Unchanged
    }

    #[test]
    fn test_entity_set_event_time() {
        let mut entity = Entity::new(
            EntityType::Person,
            "Alice".to_string(),
            "Met at conference".to_string(),
        );

        // "I met Alice last Tuesday"
        let event_occurred = Utc::now() - chrono::Duration::days(5);
        entity.set_event_time(event_occurred);

        // Postconditions
        assert_eq!(entity.event_time(), Some(event_occurred));
        assert!(entity.has_temporal_metadata());
        assert!(entity.document_time().is_none()); // Unchanged
    }

    #[test]
    fn test_entity_builder_with_temporal_metadata() {
        let doc_time = Utc::now() - chrono::Duration::days(10);
        let event_time = Utc::now() - chrono::Duration::days(14);

        let entity = Entity::builder(
            EntityType::Note,
            "Meeting Notes".to_string(),
            "Discussed project timeline".to_string(),
        )
        .with_document_time(doc_time)
        .with_event_time(event_time)
        .build();

        // Postconditions
        assert_eq!(entity.document_time(), Some(doc_time));
        assert_eq!(entity.event_time(), Some(event_time));
        assert!(entity.has_temporal_metadata());
    }

    #[test]
    fn test_temporal_metadata_bi_temporal_scenario() {
        // Scenario: User says "I met Bob at the conference last month"
        // - document_time: When user said this (now)
        // - event_time: When user actually met Bob (last month)

        let now = Utc::now();
        let last_month = now - chrono::Duration::days(30);

        let entity = Entity::builder(
            EntityType::Person,
            "Bob".to_string(),
            "Met at conference".to_string(),
        )
        .with_document_time(now)
        .with_event_time(last_month)
        .build();

        // Document time should be more recent than event time
        assert!(entity.document_time().unwrap() > entity.event_time().unwrap());

        // Both times should be in the past relative to created_at
        assert!(entity.document_time().unwrap() <= entity.created_at);
    }

    #[test]
    fn test_temporal_metadata_updates_timestamp() {
        let mut entity = Entity::new(EntityType::Note, "Test".to_string(), "Content".to_string());
        let original_updated = entity.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(10));

        entity.set_event_time(Utc::now());

        // Setting temporal metadata should update the timestamp
        assert!(entity.updated_at > original_updated);
    }

    // =========================================================================
    // Source Reference Tests (Multimedia Support)
    // =========================================================================

    #[test]
    fn test_source_ref_new() {
        let source_ref = SourceRef::new("file:///photos/meeting.jpg".to_string());

        assert_eq!(source_ref.uri, "file:///photos/meeting.jpg");
        assert!(source_ref.mime_type.is_none());
        assert!(source_ref.size_bytes.is_none());
        assert!(source_ref.checksum.is_none());
    }

    #[test]
    fn test_source_ref_builder_pattern() {
        let source_ref = SourceRef::new("s3://bucket/report.pdf".to_string())
            .with_mime_type("application/pdf".to_string())
            .with_size_bytes(1024 * 1024)
            .with_checksum("sha256:abc123".to_string());

        assert_eq!(source_ref.uri, "s3://bucket/report.pdf");
        assert_eq!(source_ref.mime_type, Some("application/pdf".to_string()));
        assert_eq!(source_ref.size_bytes, Some(1024 * 1024));
        assert_eq!(source_ref.checksum, Some("sha256:abc123".to_string()));
    }

    #[test]
    fn test_source_ref_is_local() {
        let local = SourceRef::new("file:///home/user/doc.pdf".to_string());
        let remote = SourceRef::new("https://example.com/doc.pdf".to_string());
        let s3 = SourceRef::new("s3://bucket/doc.pdf".to_string());

        assert!(local.is_local());
        assert!(!remote.is_local());
        assert!(!s3.is_local());
    }

    #[test]
    fn test_source_ref_is_remote() {
        let http = SourceRef::new("http://example.com/doc.pdf".to_string());
        let https = SourceRef::new("https://example.com/doc.pdf".to_string());
        let local = SourceRef::new("file:///home/user/doc.pdf".to_string());

        assert!(http.is_remote());
        assert!(https.is_remote());
        assert!(!local.is_remote());
    }

    #[test]
    fn test_source_ref_is_s3() {
        let s3 = SourceRef::new("s3://my-bucket/path/to/file.pdf".to_string());
        let local = SourceRef::new("file:///home/user/doc.pdf".to_string());

        assert!(s3.is_s3());
        assert!(!local.is_s3());
    }

    #[test]
    fn test_source_ref_extension() {
        let pdf = SourceRef::new("file:///docs/report.pdf".to_string());
        let jpg = SourceRef::new("https://example.com/image.jpg".to_string());
        let no_ext = SourceRef::new("s3://bucket/file".to_string());

        assert_eq!(pdf.extension(), Some("pdf"));
        assert_eq!(jpg.extension(), Some("jpg"));
        assert_eq!(no_ext.extension(), None);
    }

    #[test]
    fn test_entity_new_has_no_source_ref() {
        let entity = Entity::new(EntityType::Note, "Test".to_string(), "Content".to_string());

        assert!(entity.source_ref.is_none());
        assert!(!entity.has_source_ref());
    }

    #[test]
    fn test_entity_set_source_ref() {
        let mut entity = Entity::new(
            EntityType::Note,
            "Image Analysis".to_string(),
            "A photo of a whiteboard with meeting notes".to_string(),
        );

        let source_ref = SourceRef::new("file:///photos/whiteboard.jpg".to_string())
            .with_mime_type("image/jpeg".to_string());

        entity.set_source_ref(source_ref);

        assert!(entity.has_source_ref());
        assert_eq!(
            entity.source_ref().unwrap().uri,
            "file:///photos/whiteboard.jpg"
        );
        assert_eq!(
            entity.source_ref().unwrap().mime_type,
            Some("image/jpeg".to_string())
        );
    }

    #[test]
    fn test_entity_builder_with_source_ref() {
        let source_ref = SourceRef::new("https://storage.example.com/audio/memo.mp3".to_string())
            .with_mime_type("audio/mpeg".to_string())
            .with_size_bytes(5 * 1024 * 1024);

        let entity = Entity::builder(
            EntityType::Note,
            "Voice Memo".to_string(),
            "Discussed Q4 planning with the team".to_string(),
        )
        .with_source_ref(source_ref)
        .build();

        assert!(entity.has_source_ref());
        let ref_data = entity.source_ref().unwrap();
        assert_eq!(ref_data.uri, "https://storage.example.com/audio/memo.mp3");
        assert_eq!(ref_data.mime_type, Some("audio/mpeg".to_string()));
        assert_eq!(ref_data.size_bytes, Some(5 * 1024 * 1024));
    }

    #[test]
    fn test_source_ref_updates_timestamp() {
        let mut entity = Entity::new(EntityType::Note, "Test".to_string(), "Content".to_string());
        let original_updated = entity.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(10));

        let source_ref = SourceRef::new("file:///test.pdf".to_string());
        entity.set_source_ref(source_ref);

        // Setting source ref should update the timestamp
        assert!(entity.updated_at > original_updated);
    }

    #[test]
    fn test_source_ref_display() {
        let source_ref = SourceRef::new("file:///photos/image.png".to_string());
        assert_eq!(format!("{source_ref}"), "file:///photos/image.png");
    }
}
