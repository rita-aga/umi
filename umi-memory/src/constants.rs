//! `TigerStyle` Constants
//!
//! All limits use big-endian naming: `CATEGORY_SPECIFICS_UNIT_LIMIT`
//! Example: `CORE_MEMORY_SIZE_BYTES_MAX` (not `MAX_CORE_MEMORY_SIZE`)
//!
//! Every constant includes units in the name:
//! - _`BYTES_MAX/MIN` for size limits
//! - _`SECS_DEFAULT` for time durations
//! - _`COUNT_MAX` for quantity limits
//! - _MS for milliseconds

// =============================================================================
// Core Memory Limits
// =============================================================================

/// Maximum size of core memory (always in LLM context)
pub const CORE_MEMORY_SIZE_BYTES_MAX: usize = 32 * 1024; // 32KB

/// Minimum size of core memory
pub const CORE_MEMORY_SIZE_BYTES_MIN: usize = 4 * 1024; // 4KB

/// Maximum size of a single memory block
pub const CORE_MEMORY_BLOCK_SIZE_BYTES_MAX: usize = 16 * 1024; // 16KB

/// Number of core memory block types
pub const CORE_MEMORY_BLOCK_TYPES_COUNT: usize = 6;

/// Maximum length of a block label
pub const CORE_MEMORY_BLOCK_LABEL_BYTES_MAX: usize = 64;

/// Maximum number of blocks in core memory
pub const CORE_MEMORY_BLOCKS_COUNT_MAX: usize = 32;

/// Default importance for memory blocks
pub const CORE_MEMORY_BLOCK_IMPORTANCE_DEFAULT: f64 = 0.5;

/// Minimum importance for memory blocks
pub const CORE_MEMORY_BLOCK_IMPORTANCE_MIN: f64 = 0.0;

/// Maximum importance for memory blocks
pub const CORE_MEMORY_BLOCK_IMPORTANCE_MAX: f64 = 1.0;

// =============================================================================
// Working Memory Limits
// =============================================================================

/// Maximum total size of working memory
pub const WORKING_MEMORY_SIZE_BYTES_MAX: usize = 1024 * 1024; // 1MB

/// Maximum size of a single working memory entry
pub const WORKING_MEMORY_ENTRY_SIZE_BYTES_MAX: usize = 64 * 1024; // 64KB

/// Default TTL for working memory entries
pub const WORKING_MEMORY_TTL_SECS_DEFAULT: u64 = 3600; // 1 hour

/// Maximum TTL for working memory entries
pub const WORKING_MEMORY_TTL_SECS_MAX: u64 = 86400 * 7; // 7 days

/// Maximum number of working memory entries
pub const WORKING_MEMORY_ENTRIES_COUNT_MAX: usize = 10_000;

// =============================================================================
// Entity Limits
// =============================================================================

/// Maximum size of entity content
pub const ENTITY_CONTENT_BYTES_MAX: usize = 1_000_000; // 1MB

/// Maximum length of entity name
pub const ENTITY_NAME_BYTES_MAX: usize = 256;

/// Maximum length of entity ID
pub const ENTITY_ID_BYTES_MAX: usize = 256;

/// Maximum number of tags per entity
pub const ENTITY_TAGS_COUNT_MAX: usize = 100;

/// Maximum length of a single tag
pub const ENTITY_TAG_BYTES_MAX: usize = 256;

// =============================================================================
// Evolution Limits (ADR-006, ADR-016)
// =============================================================================

/// Maximum length of evolution relation reason
pub const EVOLUTION_REASON_BYTES_MAX: usize = 1024;

/// Maximum number of evolution relations per entity
pub const EVOLUTION_RELATIONS_PER_ENTITY_COUNT_MAX: usize = 100;

/// Maximum existing entities to compare against in detection
pub const EVOLUTION_EXISTING_ENTITIES_COUNT_MAX: usize = 10;

/// Default confidence threshold for evolution detection
pub const EVOLUTION_CONFIDENCE_THRESHOLD_DEFAULT: f64 = 0.3;

/// Minimum confidence for evolution detection
pub const EVOLUTION_CONFIDENCE_MIN: f64 = 0.0;

/// Maximum confidence for evolution detection
pub const EVOLUTION_CONFIDENCE_MAX: f64 = 1.0;

// =============================================================================
// Search Limits
// =============================================================================

/// Maximum number of search results
pub const SEARCH_RESULTS_COUNT_MAX: usize = 100;

/// Default number of search results
pub const SEARCH_RESULTS_COUNT_DEFAULT: usize = 10;

/// Maximum length of search query
pub const SEARCH_QUERY_BYTES_MAX: usize = 10_000;

// =============================================================================
// Embedding Limits
// =============================================================================

/// Number of dimensions in embeddings
pub const EMBEDDING_DIMENSIONS_COUNT: usize = 1536;

/// Maximum batch size for embedding requests
pub const EMBEDDING_BATCH_SIZE_MAX: usize = 100;

// =============================================================================
// Storage Limits
// =============================================================================

/// Maximum number of retry attempts for storage operations
pub const STORAGE_RETRY_COUNT_MAX: u32 = 3;

/// Base delay between retries in milliseconds
pub const STORAGE_RETRY_DELAY_MS_BASE: u64 = 100;

/// Maximum delay between retries in milliseconds
pub const STORAGE_RETRY_DELAY_MS_MAX: u64 = 5000;

// =============================================================================
// DST (Deterministic Simulation Testing) Limits
// =============================================================================

/// Maximum number of simulation steps
pub const DST_SIMULATION_STEPS_MAX: u64 = 1_000_000;

/// Maximum probability for fault injection (1.0 = 100%)
pub const DST_FAULT_PROBABILITY_MAX: f64 = 1.0;

/// Maximum time advance per step in milliseconds
pub const DST_TIME_ADVANCE_MS_MAX: u64 = 86_400_000; // 24 hours

/// Maximum latency for simulated operations in milliseconds
pub const DST_LATENCY_MS_MAX: u64 = 10_000; // 10 seconds

// =============================================================================
// LLM Simulation Limits
// =============================================================================

/// Maximum size of LLM prompt
pub const LLM_PROMPT_BYTES_MAX: usize = 100_000; // 100KB

/// Maximum size of LLM response
pub const LLM_RESPONSE_BYTES_MAX: usize = 50_000; // 50KB

/// Minimum simulated latency for LLM calls
pub const LLM_LATENCY_MS_MIN: u64 = 50;

/// Maximum simulated latency for LLM calls
pub const LLM_LATENCY_MS_MAX: u64 = 2000;

/// Default simulated latency for LLM calls
pub const LLM_LATENCY_MS_DEFAULT: u64 = 100;

/// Maximum number of entities in extraction response
pub const LLM_ENTITIES_COUNT_MAX: usize = 50;

/// Maximum number of query rewrites
pub const LLM_QUERY_REWRITES_COUNT_MAX: usize = 5;

// =============================================================================
// Network Simulation Limits
// =============================================================================

/// Maximum network latency in milliseconds
pub const NETWORK_LATENCY_MS_MAX: u64 = 30_000; // 30 seconds

/// Default network base latency in milliseconds
pub const NETWORK_LATENCY_MS_DEFAULT: u64 = 1;

/// Default network latency jitter in milliseconds
pub const NETWORK_JITTER_MS_DEFAULT: u64 = 5;

/// Maximum pending messages per node
pub const NETWORK_PENDING_MESSAGES_COUNT_MAX: usize = 10_000;

// =============================================================================
// Entity Extraction Limits
// =============================================================================

/// Maximum size of text to extract from
pub const EXTRACTION_TEXT_BYTES_MAX: usize = 100_000; // 100KB

/// Maximum number of entities per extraction
pub const EXTRACTION_ENTITIES_COUNT_MAX: usize = 50;

/// Maximum number of relations per extraction
pub const EXTRACTION_RELATIONS_COUNT_MAX: usize = 100;

/// Minimum confidence for extracted entities/relations
pub const EXTRACTION_CONFIDENCE_MIN: f64 = 0.0;

/// Maximum confidence for extracted entities/relations
pub const EXTRACTION_CONFIDENCE_MAX: f64 = 1.0;

/// Default confidence when not specified
pub const EXTRACTION_CONFIDENCE_DEFAULT: f64 = 0.5;

/// Maximum length of entity name
pub const EXTRACTION_ENTITY_NAME_BYTES_MAX: usize = 256;

/// Maximum length of entity content
pub const EXTRACTION_ENTITY_CONTENT_BYTES_MAX: usize = 1000;

// =============================================================================
// Retrieval Limits (ADR-015)
// =============================================================================

/// Maximum number of search results
pub const RETRIEVAL_RESULTS_COUNT_MAX: usize = 100;

/// Default number of search results
pub const RETRIEVAL_RESULTS_COUNT_DEFAULT: usize = 10;

/// Maximum length of search query
pub const RETRIEVAL_QUERY_BYTES_MAX: usize = 10_000;

/// Maximum number of query rewrites from LLM
pub const RETRIEVAL_QUERY_REWRITE_COUNT_MAX: usize = 3;

/// RRF constant (standard value from literature)
pub const RETRIEVAL_RRF_K: usize = 60;

// =============================================================================
// Memory Class Limits (ADR-017)
// =============================================================================

/// Maximum text size for remember operations
pub const MEMORY_TEXT_BYTES_MAX: usize = 100_000;

/// Maximum results for recall operations
pub const MEMORY_RECALL_LIMIT_MAX: usize = 100;

/// Default results for recall operations
pub const MEMORY_RECALL_LIMIT_DEFAULT: usize = 10;

/// Default importance for entities
pub const MEMORY_IMPORTANCE_DEFAULT: f32 = 0.5;

/// Minimum importance value
pub const MEMORY_IMPORTANCE_MIN: f32 = 0.0;

/// Maximum importance value
pub const MEMORY_IMPORTANCE_MAX: f32 = 1.0;

// =============================================================================
// Time Constants
// =============================================================================

/// Milliseconds per second
pub const TIME_MS_PER_SEC: u64 = 1000;

/// Milliseconds per minute
pub const TIME_MS_PER_MIN: u64 = 60 * TIME_MS_PER_SEC;

/// Milliseconds per hour
pub const TIME_MS_PER_HOUR: u64 = 60 * TIME_MS_PER_MIN;

/// Milliseconds per day
pub const TIME_MS_PER_DAY: u64 = 24 * TIME_MS_PER_HOUR;

// =============================================================================
// OpenTelemetry Limits
// =============================================================================

/// Maximum batch size for telemetry spans
pub const TELEMETRY_BATCH_SIZE_MAX: usize = 512;

/// Timeout for telemetry export in milliseconds
pub const TELEMETRY_EXPORT_TIMEOUT_MS: u64 = 5000; // 5 seconds

/// Maximum size of telemetry span queue
pub const TELEMETRY_SPAN_QUEUE_SIZE_MAX: usize = 2048;

/// Default sampling rate (1.0 = 100%, all spans sampled)
pub const TELEMETRY_SAMPLING_RATE_DEFAULT: f64 = 1.0;

/// Minimum sampling rate
pub const TELEMETRY_SAMPLING_RATE_MIN: f64 = 0.0;

/// Maximum sampling rate
pub const TELEMETRY_SAMPLING_RATE_MAX: f64 = 1.0;

/// Default OTLP endpoint port
pub const TELEMETRY_OTLP_PORT_DEFAULT: u16 = 4317;

/// Maximum scheduled delay for batch export in milliseconds
pub const TELEMETRY_BATCH_DELAY_MS_MAX: u64 = 30_000; // 30 seconds

// =============================================================================
// Access Tracking (Phase 1: Orchestration)
// =============================================================================

/// Halflife for access recency decay (7 days)
pub const ACCESS_TRACKER_DECAY_HALFLIFE_MS: u64 = 7 * 24 * 60 * 60 * 1000;

/// Minimum importance score
pub const ACCESS_TRACKER_MIN_IMPORTANCE: f64 = 0.0;

/// Maximum importance score
pub const ACCESS_TRACKER_MAX_IMPORTANCE: f64 = 1.0;

/// Threshold for pruning old access records (90 days)
pub const ACCESS_TRACKER_PRUNE_THRESHOLD_MS: u64 = 90 * 24 * 60 * 60 * 1000;

/// Maximum batch size for recording accesses
pub const ACCESS_TRACKER_BATCH_SIZE_MAX: usize = 1000;

// =============================================================================
// Promotion Policy (Phase 2: Orchestration)
// =============================================================================

/// Default importance threshold for promotion
pub const PROMOTION_IMPORTANCE_THRESHOLD_DEFAULT: f64 = 0.7;

/// Default score threshold for hybrid promotion
pub const PROMOTION_SCORE_THRESHOLD_DEFAULT: f64 = 0.75;

/// Maximum entities in core memory
pub const PROMOTION_CORE_MEMORY_ENTITIES_MAX: usize = 50;

/// Hybrid policy weight for base importance
pub const PROMOTION_WEIGHT_IMPORTANCE: f64 = 0.4;

/// Hybrid policy weight for recency score
pub const PROMOTION_WEIGHT_RECENCY: f64 = 0.3;

/// Hybrid policy weight for frequency score
pub const PROMOTION_WEIGHT_FREQUENCY: f64 = 0.2;

/// Hybrid policy weight for entity type priority
pub const PROMOTION_WEIGHT_TYPE_PRIORITY: f64 = 0.1;

/// Entity type priority for Self_ (highest)
pub const ENTITY_TYPE_PRIORITY_SELF: f64 = 1.0;

/// Entity type priority for Project
pub const ENTITY_TYPE_PRIORITY_PROJECT: f64 = 0.9;

/// Entity type priority for Task
pub const ENTITY_TYPE_PRIORITY_TASK: f64 = 0.85;

/// Entity type priority for Person
pub const ENTITY_TYPE_PRIORITY_PERSON: f64 = 0.7;

/// Entity type priority for Topic
pub const ENTITY_TYPE_PRIORITY_TOPIC: f64 = 0.6;

/// Entity type priority for Note (lowest)
pub const ENTITY_TYPE_PRIORITY_NOTE: f64 = 0.4;

// =============================================================================
// Eviction Policy (Phase 3: Orchestration)
// =============================================================================

/// Maximum size of core memory before eviction triggers
pub const EVICTION_CORE_MEMORY_SIZE_BYTES_MAX: usize = 32 * 1024; // 32KB

/// Maximum entities in core memory before eviction triggers
pub const EVICTION_CORE_MEMORY_ENTITIES_MAX: usize = 50;

/// Number of entities to evict in a single batch
pub const EVICTION_BATCH_SIZE: usize = 10;

/// Minimum importance threshold (never evict above this)
pub const EVICTION_IMPORTANCE_THRESHOLD_MIN: f64 = 0.5;

/// Time threshold for LRU eviction (30 days without access)
pub const EVICTION_LAST_ACCESS_THRESHOLD_MS: u64 = 30 * 24 * 60 * 60 * 1000;

/// Hybrid eviction weight for importance score
pub const EVICTION_WEIGHT_IMPORTANCE: f64 = 0.6;

/// Hybrid eviction weight for recency score
pub const EVICTION_WEIGHT_RECENCY: f64 = 0.4;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_memory_limits_valid() {
        assert!(CORE_MEMORY_SIZE_BYTES_MIN < CORE_MEMORY_SIZE_BYTES_MAX);
        assert!(CORE_MEMORY_SIZE_BYTES_MIN > 0);
    }

    #[test]
    fn test_working_memory_limits_valid() {
        assert!(WORKING_MEMORY_ENTRY_SIZE_BYTES_MAX < WORKING_MEMORY_SIZE_BYTES_MAX);
        assert!(WORKING_MEMORY_TTL_SECS_DEFAULT < WORKING_MEMORY_TTL_SECS_MAX);
    }

    #[test]
    fn test_time_constants_consistent() {
        assert_eq!(TIME_MS_PER_MIN, 60_000);
        assert_eq!(TIME_MS_PER_HOUR, 3_600_000);
        assert_eq!(TIME_MS_PER_DAY, 86_400_000);
    }
}
