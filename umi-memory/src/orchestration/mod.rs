//! Memory orchestration for unified tier management.
//!
//! `TigerStyle`: Automatic promotion, eviction, and self-evolution.
//!
//! This module provides the orchestration layer for managing memory across
//! three tiers (Core, Working, Archival) with automatic promotion and eviction
//! based on access patterns and importance scores.

pub mod access_tracker;
pub mod eviction;
pub mod promotion;

pub use access_tracker::{AccessPattern, AccessTracker};
pub use eviction::{
    EvictionPolicy, HybridEvictionPolicy, ImportanceEvictionPolicy, LRUEvictionPolicy,
};
pub use promotion::{HybridPolicy, ImportanceBasedPolicy, PromotionPolicy};
