//! Memory orchestration for unified tier management.
//!
//! `TigerStyle`: Automatic promotion, eviction, and self-evolution.
//!
//! This module provides the orchestration layer for managing memory across
//! three tiers (Core, Working, Archival) with automatic promotion and eviction
//! based on access patterns and importance scores.

pub mod access_tracker;
pub mod category_evolution;
pub mod eviction;
pub mod promotion;
pub mod unified;

#[cfg(test)]
mod tests;

pub use access_tracker::{AccessPattern, AccessTracker};
pub use category_evolution::{
    CategoryEvolver, EvolutionAnalysis, EvolutionSuggestion, EVOLUTION_ANALYSIS_INTERVAL_MS,
    EVOLUTION_BLOCK_USAGE_THRESHOLD_MAX, EVOLUTION_BLOCK_USAGE_THRESHOLD_MIN,
    EVOLUTION_CO_OCCURRENCE_THRESHOLD, EVOLUTION_MIN_SAMPLES_COUNT,
};
pub use eviction::{
    EvictionPolicy, HybridEvictionPolicy, ImportanceEvictionPolicy, LRUEvictionPolicy,
};
pub use promotion::{HybridPolicy, ImportanceBasedPolicy, PromotionPolicy};
pub use unified::{UnifiedMemory, UnifiedMemoryConfig, UnifiedMemoryError, UnifiedRememberResult};
