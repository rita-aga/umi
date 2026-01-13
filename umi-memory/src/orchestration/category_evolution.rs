//! Category Evolution - Self-adapting memory structure based on usage patterns.
//!
//! `TigerStyle`: DST-first, threshold-based suggestions, no automatic changes.
//!
//! # Design
//!
//! CategoryEvolver tracks:
//! - Co-occurrence of entity types (accessed together)
//! - Block type usage statistics
//! - Entity type distribution per block
//!
//! Based on patterns, it suggests:
//! - Creating new blocks for frequently co-occurring entity types
//! - Merging rarely-used blocks
//! - Splitting over-used blocks
//!
//! # Important
//!
//! **No automatic structure changes.** All suggestions require explicit approval.

use std::collections::HashMap;

use crate::dst::SimClock;
use crate::memory::MemoryBlockType;
use crate::storage::EntityType;

// =============================================================================
// Constants (TigerStyle: units in name)
// =============================================================================

/// Minimum samples before generating suggestions.
pub const EVOLUTION_MIN_SAMPLES_COUNT: usize = 100;

/// Co-occurrence threshold (0.0-1.0). Entities accessed together >= this often
/// are candidates for a shared block.
pub const EVOLUTION_CO_OCCURRENCE_THRESHOLD: f64 = 0.7;

/// Block usage threshold. Blocks used less than this fraction of total
/// are candidates for merging.
pub const EVOLUTION_BLOCK_USAGE_THRESHOLD_MIN: f64 = 0.1;

/// Block usage threshold. Blocks used more than this fraction of total
/// are candidates for splitting.
pub const EVOLUTION_BLOCK_USAGE_THRESHOLD_MAX: f64 = 0.5;

/// Analysis interval in milliseconds (7 days).
pub const EVOLUTION_ANALYSIS_INTERVAL_MS: u64 = 7 * 24 * 60 * 60 * 1000;

// =============================================================================
// Types
// =============================================================================

/// A suggestion for evolving the category structure.
#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionSuggestion {
    /// Create a new block for frequently co-occurring entity types.
    CreateBlock {
        /// Suggested name for new block
        name: String,
        /// Entity types that should go in this block
        entity_types: Vec<EntityType>,
        /// Human-readable reason
        reason: String,
        /// Co-occurrence score that triggered this
        co_occurrence_score: f64,
    },

    /// Merge two rarely-used blocks.
    MergeBlocks {
        /// First block to merge
        block1: MemoryBlockType,
        /// Second block to merge
        block2: MemoryBlockType,
        /// Name for merged block
        into_name: String,
        /// Human-readable reason
        reason: String,
        /// Usage scores that triggered this
        usage_scores: (f64, f64),
    },

    /// Split an over-used block.
    SplitBlock {
        /// Block to split
        block: MemoryBlockType,
        /// Suggested split names
        into_names: Vec<String>,
        /// Human-readable reason
        reason: String,
        /// Usage score that triggered this
        usage_score: f64,
    },
}

/// Result of pattern analysis.
#[derive(Debug, Clone, Default)]
pub struct EvolutionAnalysis {
    /// Generated suggestions
    pub suggestions: Vec<EvolutionSuggestion>,
    /// Total samples analyzed
    pub total_samples: usize,
    /// Time of analysis (ms since epoch)
    pub analysis_time_ms: u64,
}

/// Access event for tracking.
#[derive(Debug, Clone)]
pub struct AccessEvent {
    /// Entity type accessed
    pub entity_type: EntityType,
    /// Block type it was stored in
    pub block_type: MemoryBlockType,
    /// Timestamp of access
    pub timestamp_ms: u64,
}

// =============================================================================
// CategoryEvolver
// =============================================================================

/// Tracks usage patterns and suggests structure changes.
///
/// `TigerStyle`:
/// - DST-first with SimClock
/// - Threshold-based suggestions
/// - No automatic changes
pub struct CategoryEvolver {
    /// Clock for time-based operations
    clock: SimClock,

    /// Access events (recent history)
    access_events: Vec<AccessEvent>,

    /// Co-occurrence counts: (EntityType, EntityType) -> count
    /// Only stores (a, b) where a < b (lexicographically)
    co_occurrence_counts: HashMap<(EntityType, EntityType), u64>,

    /// Block usage counts: BlockType -> count
    block_usage_counts: HashMap<MemoryBlockType, u64>,

    /// Entity type counts: EntityType -> count
    entity_type_counts: HashMap<EntityType, u64>,

    /// Total access count
    total_accesses: u64,

    /// Last analysis timestamp
    last_analysis_ms: u64,

    /// Configuration
    min_samples: usize,
    co_occurrence_threshold: f64,
    block_usage_min: f64,
    block_usage_max: f64,
}

impl CategoryEvolver {
    /// Create a new CategoryEvolver.
    #[must_use]
    pub fn new(clock: SimClock) -> Self {
        Self {
            clock,
            access_events: Vec::new(),
            co_occurrence_counts: HashMap::new(),
            block_usage_counts: HashMap::new(),
            entity_type_counts: HashMap::new(),
            total_accesses: 0,
            last_analysis_ms: 0,
            min_samples: EVOLUTION_MIN_SAMPLES_COUNT,
            co_occurrence_threshold: EVOLUTION_CO_OCCURRENCE_THRESHOLD,
            block_usage_min: EVOLUTION_BLOCK_USAGE_THRESHOLD_MIN,
            block_usage_max: EVOLUTION_BLOCK_USAGE_THRESHOLD_MAX,
        }
    }

    /// Create with custom thresholds (for testing).
    #[must_use]
    pub fn with_thresholds(
        clock: SimClock,
        min_samples: usize,
        co_occurrence_threshold: f64,
        block_usage_min: f64,
        block_usage_max: f64,
    ) -> Self {
        // Preconditions
        assert!(min_samples > 0, "min_samples must be positive");
        assert!(
            (0.0..=1.0).contains(&co_occurrence_threshold),
            "co_occurrence_threshold must be 0.0-1.0"
        );
        assert!(
            (0.0..=1.0).contains(&block_usage_min),
            "block_usage_min must be 0.0-1.0"
        );
        assert!(
            (0.0..=1.0).contains(&block_usage_max),
            "block_usage_max must be 0.0-1.0"
        );
        assert!(
            block_usage_min < block_usage_max,
            "block_usage_min must be < block_usage_max"
        );

        Self {
            clock,
            access_events: Vec::new(),
            co_occurrence_counts: HashMap::new(),
            block_usage_counts: HashMap::new(),
            entity_type_counts: HashMap::new(),
            total_accesses: 0,
            last_analysis_ms: 0,
            min_samples,
            co_occurrence_threshold,
            block_usage_min,
            block_usage_max,
        }
    }

    /// Track an access event.
    pub fn track_access(&mut self, entity_type: EntityType, block_type: MemoryBlockType) {
        let timestamp_ms = self.clock.now_ms();

        // Record the event
        self.access_events.push(AccessEvent {
            entity_type: entity_type.clone(),
            block_type,
            timestamp_ms,
        });

        // Update counts
        *self.block_usage_counts.entry(block_type).or_insert(0) += 1;
        *self
            .entity_type_counts
            .entry(entity_type.clone())
            .or_insert(0) += 1;
        self.total_accesses += 1;

        // Update co-occurrence with recent events (within time window)
        self.update_co_occurrence(&entity_type, timestamp_ms);
    }

    /// Update co-occurrence counts for recent accesses.
    fn update_co_occurrence(&mut self, new_entity_type: &EntityType, timestamp_ms: u64) {
        // Time window for co-occurrence (1 hour)
        const CO_OCCURRENCE_WINDOW_MS: u64 = 60 * 60 * 1000;

        for event in &self.access_events {
            // Skip if outside time window
            if timestamp_ms.saturating_sub(event.timestamp_ms) > CO_OCCURRENCE_WINDOW_MS {
                continue;
            }

            // Skip if same entity type
            if &event.entity_type == new_entity_type {
                continue;
            }

            // Create ordered pair (smaller first for consistency)
            let pair = if format!("{:?}", event.entity_type) < format!("{:?}", new_entity_type) {
                (event.entity_type.clone(), new_entity_type.clone())
            } else {
                (new_entity_type.clone(), event.entity_type.clone())
            };

            *self.co_occurrence_counts.entry(pair).or_insert(0) += 1;
        }
    }

    /// Analyze patterns and generate suggestions.
    ///
    /// Returns `None` if not enough samples or too soon since last analysis.
    #[must_use]
    pub fn analyze(&mut self) -> Option<EvolutionAnalysis> {
        let current_time = self.clock.now_ms();

        // Check if enough samples
        if self.total_accesses < self.min_samples as u64 {
            return None;
        }

        // Check if enough time since last analysis
        if current_time.saturating_sub(self.last_analysis_ms) < EVOLUTION_ANALYSIS_INTERVAL_MS {
            return None;
        }

        // Update last analysis time
        self.last_analysis_ms = current_time;

        let mut suggestions = Vec::new();

        // 1. Check for high co-occurrence (suggest CreateBlock)
        self.check_co_occurrence_suggestions(&mut suggestions);

        // 2. Check for low block usage (suggest MergeBlocks)
        self.check_merge_suggestions(&mut suggestions);

        // 3. Check for high block usage (suggest SplitBlock)
        self.check_split_suggestions(&mut suggestions);

        Some(EvolutionAnalysis {
            suggestions,
            total_samples: self.total_accesses as usize,
            analysis_time_ms: current_time,
        })
    }

    /// Check co-occurrence and suggest new blocks.
    fn check_co_occurrence_suggestions(&self, suggestions: &mut Vec<EvolutionSuggestion>) {
        // Find entity type pairs with high co-occurrence
        for ((type1, type2), _count) in &self.co_occurrence_counts {
            let score = self.co_occurrence_score(type1, type2);

            if score >= self.co_occurrence_threshold {
                // Suggest creating a new block for these entity types
                let name = format!("{:?}_{:?}_Block", type1, type2);
                suggestions.push(EvolutionSuggestion::CreateBlock {
                    name,
                    entity_types: vec![type1.clone(), type2.clone()],
                    reason: format!(
                        "{:?} and {:?} are accessed together {:.0}% of the time",
                        type1,
                        type2,
                        score * 100.0
                    ),
                    co_occurrence_score: score,
                });
            }
        }
    }

    /// Check for low-usage blocks to merge.
    fn check_merge_suggestions(&self, suggestions: &mut Vec<EvolutionSuggestion>) {
        // Collect blocks below minimum usage threshold
        let low_usage_blocks: Vec<_> = self
            .block_usage_counts
            .iter()
            .filter_map(|(block, _count)| {
                let score = self.block_usage_score(*block);
                if score < self.block_usage_min {
                    Some((*block, score))
                } else {
                    None
                }
            })
            .collect();

        // If we have 2+ low-usage blocks, suggest merging
        if low_usage_blocks.len() >= 2 {
            let (block1, score1) = low_usage_blocks[0];
            let (block2, score2) = low_usage_blocks[1];

            suggestions.push(EvolutionSuggestion::MergeBlocks {
                block1,
                block2,
                into_name: format!("{:?}_{:?}_Merged", block1, block2),
                reason: format!(
                    "{:?} ({:.1}%) and {:?} ({:.1}%) are both below {:.0}% usage threshold",
                    block1,
                    score1 * 100.0,
                    block2,
                    score2 * 100.0,
                    self.block_usage_min * 100.0
                ),
                usage_scores: (score1, score2),
            });
        }
    }

    /// Check for high-usage blocks to split.
    fn check_split_suggestions(&self, suggestions: &mut Vec<EvolutionSuggestion>) {
        // Find blocks above maximum usage threshold
        for (block, _count) in &self.block_usage_counts {
            let score = self.block_usage_score(*block);

            if score > self.block_usage_max {
                // Suggest splitting this block
                suggestions.push(EvolutionSuggestion::SplitBlock {
                    block: *block,
                    into_names: vec![
                        format!("{:?}_Part1", block),
                        format!("{:?}_Part2", block),
                    ],
                    reason: format!(
                        "{:?} has {:.0}% of all accesses (threshold: {:.0}%)",
                        block,
                        score * 100.0,
                        self.block_usage_max * 100.0
                    ),
                    usage_score: score,
                });
            }
        }
    }

    /// Get total access count.
    #[must_use]
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// Get co-occurrence score between two entity types.
    /// Returns 0.0-1.0 based on how often they're accessed together.
    ///
    /// Score = co_occurrence_count / min(count_type1, count_type2)
    /// This measures: "of the times we accessed the less frequent type,
    /// how often was the other type accessed together?"
    #[must_use]
    pub fn co_occurrence_score(&self, type1: &EntityType, type2: &EntityType) -> f64 {
        // Get individual counts
        let count1 = self.entity_type_counts.get(type1).copied().unwrap_or(0);
        let count2 = self.entity_type_counts.get(type2).copied().unwrap_or(0);

        if count1 == 0 || count2 == 0 {
            return 0.0;
        }

        // Create ordered pair (smaller first for consistency)
        let pair = if format!("{:?}", type1) < format!("{:?}", type2) {
            (type1.clone(), type2.clone())
        } else {
            (type2.clone(), type1.clone())
        };

        let co_occurrence_count = self.co_occurrence_counts.get(&pair).copied().unwrap_or(0);

        // Score = co_occurrence / min(count1, count2)
        // This tells us: of the rarer entity's accesses, what fraction co-occurred?
        let min_count = count1.min(count2);
        co_occurrence_count as f64 / min_count as f64
    }

    /// Get block usage score (fraction of total accesses).
    #[must_use]
    pub fn block_usage_score(&self, block_type: MemoryBlockType) -> f64 {
        if self.total_accesses == 0 {
            return 0.0;
        }

        let count = self.block_usage_counts.get(&block_type).copied().unwrap_or(0);
        count as f64 / self.total_accesses as f64
    }

    /// Get clock reference (for testing).
    #[must_use]
    pub fn clock(&self) -> &SimClock {
        &self.clock
    }

    /// Force analysis (ignore time check, for testing).
    #[must_use]
    pub fn analyze_force(&mut self) -> Option<EvolutionAnalysis> {
        // Check if enough samples
        if self.total_accesses < self.min_samples as u64 {
            return None;
        }

        let current_time = self.clock.now_ms();
        self.last_analysis_ms = current_time;

        let mut suggestions = Vec::new();
        self.check_co_occurrence_suggestions(&mut suggestions);
        self.check_merge_suggestions(&mut suggestions);
        self.check_split_suggestions(&mut suggestions);

        Some(EvolutionAnalysis {
            suggestions,
            total_samples: self.total_accesses as usize,
            analysis_time_ms: current_time,
        })
    }
}

// =============================================================================
// Tests - Written FIRST, Should FAIL with stub implementation
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn create_evolver(seed: u64) -> CategoryEvolver {
        let clock = SimClock::at_ms(1_000_000_000);
        CategoryEvolver::new(clock)
    }

    fn create_evolver_low_threshold(seed: u64) -> CategoryEvolver {
        let clock = SimClock::at_ms(1_000_000_000);
        // Low min_samples so we can test without 100 events
        CategoryEvolver::with_thresholds(
            clock, 10,   // min_samples
            0.5,  // co_occurrence_threshold (50%)
            0.1,  // block_usage_min
            0.5,  // block_usage_max
        )
    }

    // =========================================================================
    // Basic Tests (should pass even with stubs)
    // =========================================================================

    #[test]
    fn test_evolver_new() {
        let evolver = create_evolver(42);
        assert_eq!(evolver.total_accesses(), 0);
    }

    #[test]
    fn test_track_access_increments_count() {
        let mut evolver = create_evolver(42);

        evolver.track_access(EntityType::Person, MemoryBlockType::Human);

        assert_eq!(evolver.total_accesses(), 1);
    }

    #[test]
    fn test_track_multiple_accesses() {
        let mut evolver = create_evolver(42);

        evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        evolver.track_access(EntityType::Project, MemoryBlockType::Facts);
        evolver.track_access(EntityType::Task, MemoryBlockType::Goals);

        assert_eq!(evolver.total_accesses(), 3);
    }

    #[test]
    fn test_analyze_returns_none_without_enough_samples() {
        let mut evolver = create_evolver(42);

        // Only 5 accesses, need 100 by default
        for _ in 0..5 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        let result = evolver.analyze();
        assert!(result.is_none(), "should return None without enough samples");
    }

    // =========================================================================
    // Block Usage Tests (should pass - no stub needed)
    // =========================================================================

    #[test]
    fn test_block_usage_score_calculation() {
        let mut evolver = create_evolver(42);

        // 3 Person, 1 Project = 75% Human, 25% Facts
        evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        evolver.track_access(EntityType::Project, MemoryBlockType::Facts);

        let human_score = evolver.block_usage_score(MemoryBlockType::Human);
        let facts_score = evolver.block_usage_score(MemoryBlockType::Facts);

        // These should be actual values, not stubs
        assert!(
            (human_score - 0.75).abs() < 0.01,
            "Human block should be 75% usage, got {}",
            human_score
        );
        assert!(
            (facts_score - 0.25).abs() < 0.01,
            "Facts block should be 25% usage, got {}",
            facts_score
        );
    }

    #[test]
    fn test_block_usage_score_empty() {
        let evolver = create_evolver(42);

        let score = evolver.block_usage_score(MemoryBlockType::Human);
        assert_eq!(score, 0.0, "empty evolver should return 0.0");
    }

    // =========================================================================
    // Co-occurrence Tests - SHOULD FAIL with stub
    // =========================================================================

    #[test]
    fn test_co_occurrence_score_calculation() {
        let mut evolver = create_evolver_low_threshold(42);

        // Access Person and Project together multiple times
        for _ in 0..10 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver.track_access(EntityType::Project, MemoryBlockType::Facts);
        }

        let score = evolver.co_occurrence_score(&EntityType::Person, &EntityType::Project);

        // This test should FAIL with stub implementation (returns 0.0)
        // Real implementation should return > 0.5 since they're always together
        assert!(
            score > 0.5,
            "Person and Project accessed together should have high co-occurrence, got {}",
            score
        );
    }

    #[test]
    fn test_co_occurrence_score_low_when_separate() {
        let mut evolver = create_evolver_low_threshold(42);

        // Access Person alone, then wait, then Project alone
        for _ in 0..5 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        // Advance time past co-occurrence window (1 hour)
        let _ = evolver.clock.advance_ms(2 * 60 * 60 * 1000);

        for _ in 0..5 {
            evolver.track_access(EntityType::Project, MemoryBlockType::Facts);
        }

        let score = evolver.co_occurrence_score(&EntityType::Person, &EntityType::Project);

        // Should be low since they weren't accessed together
        assert!(
            score < 0.3,
            "Person and Project accessed separately should have low co-occurrence, got {}",
            score
        );
    }

    // =========================================================================
    // Suggestion Generation Tests - SHOULD FAIL with stub
    // =========================================================================

    #[test]
    fn test_suggest_create_block_for_high_co_occurrence() {
        let mut evolver = create_evolver_low_threshold(42);

        // Access Person and Task together (high co-occurrence)
        for _ in 0..15 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver.track_access(EntityType::Task, MemoryBlockType::Goals);
        }

        let analysis = evolver.analyze_force();
        assert!(analysis.is_some(), "should have enough samples");

        let analysis = analysis.unwrap();

        // This test should FAIL with stub implementation (no suggestions)
        let create_suggestions: Vec<_> = analysis
            .suggestions
            .iter()
            .filter(|s| matches!(s, EvolutionSuggestion::CreateBlock { .. }))
            .collect();

        assert!(
            !create_suggestions.is_empty(),
            "should suggest creating block for high co-occurrence entities"
        );
    }

    #[test]
    fn test_suggest_merge_for_low_usage_blocks() {
        let mut evolver = create_evolver_low_threshold(42);

        // Create scenario where two blocks are both below 10% threshold
        // Total: 20 accesses
        // Human: 18 (90%)
        // Facts: 1 (5%)
        // Goals: 1 (5%)
        for _ in 0..18 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }
        evolver.track_access(EntityType::Project, MemoryBlockType::Facts);
        evolver.track_access(EntityType::Task, MemoryBlockType::Goals);

        let analysis = evolver.analyze_force();
        assert!(analysis.is_some(), "should have enough samples");

        let analysis = analysis.unwrap();

        // Verify block usage scores
        let human_score = evolver.block_usage_score(MemoryBlockType::Human);
        let facts_score = evolver.block_usage_score(MemoryBlockType::Facts);
        let goals_score = evolver.block_usage_score(MemoryBlockType::Goals);

        assert!(human_score > 0.5, "Human should be > 50%, got {}", human_score);
        assert!(
            facts_score < 0.1,
            "Facts should be < 10%, got {}",
            facts_score
        );
        assert!(
            goals_score < 0.1,
            "Goals should be < 10%, got {}",
            goals_score
        );

        // Should suggest merging the two low-usage blocks
        let merge_suggestions: Vec<_> = analysis
            .suggestions
            .iter()
            .filter(|s| matches!(s, EvolutionSuggestion::MergeBlocks { .. }))
            .collect();

        assert!(
            !merge_suggestions.is_empty(),
            "should suggest merging low-usage blocks (Facts: {:.1}%, Goals: {:.1}%)",
            facts_score * 100.0,
            goals_score * 100.0
        );
    }

    #[test]
    fn test_suggest_split_for_high_usage_block() {
        let mut evolver = create_evolver_low_threshold(42);

        // All accesses go to Human block (100%)
        for _ in 0..20 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        let analysis = evolver.analyze_force();
        assert!(analysis.is_some());

        let analysis = analysis.unwrap();

        // This test should FAIL with stub implementation
        let split_suggestions: Vec<_> = analysis
            .suggestions
            .iter()
            .filter(|s| matches!(s, EvolutionSuggestion::SplitBlock { .. }))
            .collect();

        assert!(
            !split_suggestions.is_empty(),
            "should suggest splitting block with >50% usage"
        );
    }

    // =========================================================================
    // Time-based Analysis Tests
    // =========================================================================

    #[test]
    fn test_analyze_respects_time_interval() {
        let mut evolver = create_evolver_low_threshold(42);

        // Generate enough samples
        for _ in 0..15 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        // First analysis should work
        let result1 = evolver.analyze();
        assert!(result1.is_some(), "first analysis should work");

        // Second analysis immediately should fail (too soon)
        let result2 = evolver.analyze();
        assert!(
            result2.is_none(),
            "second analysis should be blocked by time interval"
        );

        // Advance past interval (SimClock max is 1 day, so advance 8 days)
        // EVOLUTION_ANALYSIS_INTERVAL_MS = 7 days
        const ONE_DAY_MS: u64 = 24 * 60 * 60 * 1000;
        for _ in 0..8 {
            let _ = evolver.clock.advance_ms(ONE_DAY_MS);
        }

        // Add more samples to meet threshold again
        for _ in 0..5 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        // Third analysis should work
        let result3 = evolver.analyze();
        assert!(
            result3.is_some(),
            "analysis should work after time interval"
        );
    }

    // =========================================================================
    // Determinism Tests
    // =========================================================================

    #[test]
    fn test_evolver_deterministic() {
        let mut evolver1 = create_evolver_low_threshold(42);
        let mut evolver2 = create_evolver_low_threshold(42);

        // Same operations
        for _ in 0..15 {
            evolver1.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver1.track_access(EntityType::Project, MemoryBlockType::Facts);

            evolver2.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver2.track_access(EntityType::Project, MemoryBlockType::Facts);
        }

        let analysis1 = evolver1.analyze_force();
        let analysis2 = evolver2.analyze_force();

        assert_eq!(
            analysis1.as_ref().map(|a| a.suggestions.len()),
            analysis2.as_ref().map(|a| a.suggestions.len()),
            "same inputs should produce same number of suggestions"
        );
    }

    // =========================================================================
    // Threshold Configuration Tests
    // =========================================================================

    #[test]
    #[should_panic(expected = "min_samples must be positive")]
    fn test_invalid_min_samples() {
        let clock = SimClock::at_ms(1_000_000_000);
        CategoryEvolver::with_thresholds(clock, 0, 0.5, 0.1, 0.5);
    }

    #[test]
    #[should_panic(expected = "co_occurrence_threshold must be 0.0-1.0")]
    fn test_invalid_co_occurrence_threshold() {
        let clock = SimClock::at_ms(1_000_000_000);
        CategoryEvolver::with_thresholds(clock, 10, 1.5, 0.1, 0.5);
    }

    #[test]
    #[should_panic(expected = "block_usage_min must be < block_usage_max")]
    fn test_invalid_block_usage_range() {
        let clock = SimClock::at_ms(1_000_000_000);
        CategoryEvolver::with_thresholds(clock, 10, 0.5, 0.6, 0.5);
    }

    // =========================================================================
    // DST Simulation Tests
    // =========================================================================

    #[test]
    fn test_dst_deterministic_co_occurrence_score() {
        // DST: Same seed, same clock, same operations = same results
        for seed in [42, 123, 999] {
            let clock1 = SimClock::at_ms(1_000_000_000);
            let clock2 = SimClock::at_ms(1_000_000_000);

            let mut evolver1 = CategoryEvolver::with_thresholds(clock1, 10, 0.5, 0.1, 0.5);
            let mut evolver2 = CategoryEvolver::with_thresholds(clock2, 10, 0.5, 0.1, 0.5);

            // Same sequence of operations
            for _ in 0..15 {
                evolver1.track_access(EntityType::Person, MemoryBlockType::Human);
                evolver1.track_access(EntityType::Project, MemoryBlockType::Facts);

                evolver2.track_access(EntityType::Person, MemoryBlockType::Human);
                evolver2.track_access(EntityType::Project, MemoryBlockType::Facts);
            }

            let score1 = evolver1.co_occurrence_score(&EntityType::Person, &EntityType::Project);
            let score2 = evolver2.co_occurrence_score(&EntityType::Person, &EntityType::Project);

            assert!(
                (score1 - score2).abs() < 0.001,
                "DST: seed {} should produce identical co-occurrence scores",
                seed
            );
        }
    }

    #[test]
    fn test_dst_time_based_analysis_deterministic() {
        // DST: Time advancement should produce identical results
        let clock1 = SimClock::at_ms(1_000_000_000);
        let clock2 = SimClock::at_ms(1_000_000_000);

        let mut evolver1 = CategoryEvolver::with_thresholds(clock1, 10, 0.5, 0.1, 0.5);
        let mut evolver2 = CategoryEvolver::with_thresholds(clock2, 10, 0.5, 0.1, 0.5);

        // Generate samples
        for _ in 0..15 {
            evolver1.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver2.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        // First analysis
        let result1a = evolver1.analyze();
        let result2a = evolver2.analyze();

        assert_eq!(
            result1a.is_some(),
            result2a.is_some(),
            "DST: first analysis should have same availability"
        );

        // Advance time identically
        const ONE_DAY_MS: u64 = 24 * 60 * 60 * 1000;
        for _ in 0..8 {
            let _ = evolver1.clock.advance_ms(ONE_DAY_MS);
            let _ = evolver2.clock.advance_ms(ONE_DAY_MS);
        }

        // More samples
        for _ in 0..5 {
            evolver1.track_access(EntityType::Task, MemoryBlockType::Goals);
            evolver2.track_access(EntityType::Task, MemoryBlockType::Goals);
        }

        // Second analysis
        let result1b = evolver1.analyze();
        let result2b = evolver2.analyze();

        assert_eq!(
            result1b.is_some(),
            result2b.is_some(),
            "DST: second analysis should have same availability"
        );
    }

    #[test]
    fn test_edge_case_single_entity_type() {
        let mut evolver = create_evolver_low_threshold(42);

        // Only track one entity type
        for _ in 0..15 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        let analysis = evolver.analyze_force();
        assert!(analysis.is_some());

        let analysis = analysis.unwrap();

        // Should suggest split (100% usage of one block)
        let split_count = analysis
            .suggestions
            .iter()
            .filter(|s| matches!(s, EvolutionSuggestion::SplitBlock { .. }))
            .count();

        assert!(
            split_count > 0,
            "single entity type should trigger split suggestion"
        );

        // No co-occurrence suggestions (need 2+ types)
        let create_count = analysis
            .suggestions
            .iter()
            .filter(|s| matches!(s, EvolutionSuggestion::CreateBlock { .. }))
            .count();

        assert_eq!(
            create_count, 0,
            "single entity type cannot have co-occurrence suggestions"
        );
    }

    #[test]
    fn test_edge_case_all_types_equal_usage() {
        let mut evolver = create_evolver_low_threshold(42);

        // Equal distribution across 4 entity types in different blocks
        // Each block gets 25% - below 50% max threshold, above 10% min
        for _ in 0..3 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver.track_access(EntityType::Project, MemoryBlockType::Facts);
            evolver.track_access(EntityType::Task, MemoryBlockType::Goals);
            evolver.track_access(EntityType::Topic, MemoryBlockType::System);
        }

        let analysis = evolver.analyze_force();
        assert!(analysis.is_some());

        let analysis = analysis.unwrap();

        // No split suggestions (no block > 50%)
        let split_count = analysis
            .suggestions
            .iter()
            .filter(|s| matches!(s, EvolutionSuggestion::SplitBlock { .. }))
            .count();

        assert_eq!(
            split_count, 0,
            "equal distribution (25% each) should not trigger split suggestions"
        );
    }

    #[test]
    fn test_co_occurrence_score_symmetric() {
        let mut evolver = create_evolver_low_threshold(42);

        for _ in 0..15 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
            evolver.track_access(EntityType::Project, MemoryBlockType::Facts);
        }

        // Score should be symmetric
        let score_ab = evolver.co_occurrence_score(&EntityType::Person, &EntityType::Project);
        let score_ba = evolver.co_occurrence_score(&EntityType::Project, &EntityType::Person);

        assert!(
            (score_ab - score_ba).abs() < 0.001,
            "co-occurrence score should be symmetric: A-B={} vs B-A={}",
            score_ab,
            score_ba
        );
    }

    #[test]
    fn test_no_suggestions_below_min_samples() {
        let mut evolver = create_evolver_low_threshold(42); // min_samples = 10

        // Only 5 samples
        for _ in 0..5 {
            evolver.track_access(EntityType::Person, MemoryBlockType::Human);
        }

        let analysis = evolver.analyze_force();
        assert!(
            analysis.is_none(),
            "should not analyze with fewer than min_samples"
        );
    }
}
