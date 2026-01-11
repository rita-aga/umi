//! TigerStyle Constants
//!
//! All limits use big-endian naming: CATEGORY_SPECIFICS_UNIT_LIMIT
//! Example: CORE_MEMORY_SIZE_BYTES_MAX (not MAX_CORE_MEMORY_SIZE)
//!
//! Every constant includes units in the name:
//! - _BYTES_MAX/MIN for size limits
//! - _SECS_DEFAULT for time durations
//! - _COUNT_MAX for quantity limits
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
// Evolution Limits (ADR-006)
// =============================================================================

/// Maximum length of evolution relation reason
pub const EVOLUTION_REASON_BYTES_MAX: usize = 1024;

/// Maximum number of evolution relations per entity
pub const EVOLUTION_RELATIONS_PER_ENTITY_COUNT_MAX: usize = 100;

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
