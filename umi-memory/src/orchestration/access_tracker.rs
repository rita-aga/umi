//! Access pattern tracking for importance-based promotion.
//!
//! `TigerStyle`: Track access patterns with deterministic scoring using SimClock.
//!
//! This module tracks when entities are accessed and calculates importance
//! scores based on frequency, recency, and base importance. Used for
//! automatic promotion and eviction decisions.

use crate::constants::{
    ACCESS_TRACKER_BATCH_SIZE_MAX, ACCESS_TRACKER_DECAY_HALFLIFE_MS, ACCESS_TRACKER_MAX_IMPORTANCE,
    ACCESS_TRACKER_MIN_IMPORTANCE,
};
use crate::dst::SimClock;
use std::collections::HashMap;

/// Access pattern metadata for a single entity.
#[derive(Debug, Clone, PartialEq)]
pub struct AccessPattern {
    /// When entity was first accessed (milliseconds)
    pub first_access_ms: u64,
    /// When entity was most recently accessed (milliseconds)
    pub last_access_ms: u64,
    /// Total number of accesses
    pub access_count: u64,
    /// Recency score (0.0-1.0, exponential decay)
    pub recency_score: f64,
    /// Frequency score (0.0-1.0, normalized)
    pub frequency_score: f64,
    /// Combined importance score (0.0-1.0)
    pub combined_importance: f64,
}

/// Tracks entity access patterns for importance scoring.
#[derive(Debug, Clone)]
pub struct AccessTracker {
    /// Access records by entity ID
    access_records: HashMap<String, AccessRecord>,
    /// Simulated clock for deterministic time
    clock: SimClock,
}

#[derive(Debug, Clone)]
struct AccessRecord {
    first_access_ms: u64,
    last_access_ms: u64,
    access_count: u64,
    base_importance: f64,
}

impl AccessTracker {
    /// Create a new access tracker with a simulated clock.
    ///
    /// `TigerStyle`: Uses SimClock for deterministic testing.
    #[must_use]
    pub fn new(clock: SimClock) -> Self {
        // TigerStyle: clock.now_ms() is u64, always non-negative
        Self {
            access_records: HashMap::new(),
            clock,
        }
    }

    /// Record a single entity access at current clock time.
    pub fn record_access(&mut self, entity_id: &str, base_importance: f64) {
        // Preconditions
        assert!(!entity_id.is_empty(), "entity_id cannot be empty");
        assert!(
            base_importance >= ACCESS_TRACKER_MIN_IMPORTANCE
                && base_importance <= ACCESS_TRACKER_MAX_IMPORTANCE,
            "base_importance {} outside valid range [{}, {}]",
            base_importance,
            ACCESS_TRACKER_MIN_IMPORTANCE,
            ACCESS_TRACKER_MAX_IMPORTANCE
        );

        let current_time = self.clock.now_ms();

        match self.access_records.get_mut(entity_id) {
            Some(record) => {
                // Update existing record
                record.last_access_ms = current_time;
                record.access_count += 1;
                record.base_importance = base_importance;
            }
            None => {
                // Create new record
                self.access_records.insert(
                    entity_id.to_string(),
                    AccessRecord {
                        first_access_ms: current_time,
                        last_access_ms: current_time,
                        access_count: 1,
                        base_importance,
                    },
                );
            }
        }

        // Postconditions
        assert!(
            self.access_records.contains_key(entity_id),
            "entity_id must be in records after recording"
        );
    }

    /// Record multiple entity accesses (batch operation).
    pub fn record_batch_access(&mut self, entity_ids: &[(&str, f64)]) {
        // Preconditions
        assert!(
            entity_ids.len() <= ACCESS_TRACKER_BATCH_SIZE_MAX,
            "batch size {} exceeds maximum {}",
            entity_ids.len(),
            ACCESS_TRACKER_BATCH_SIZE_MAX
        );

        for (entity_id, base_importance) in entity_ids {
            self.record_access(entity_id, *base_importance);
        }
    }

    /// Get access pattern for an entity.
    pub fn get_access_pattern(&self, entity_id: &str) -> Option<AccessPattern> {
        // Preconditions
        assert!(!entity_id.is_empty(), "entity_id cannot be empty");

        let record = self.access_records.get(entity_id)?;
        let current_time = self.clock.now_ms();

        // Calculate recency score (exponential decay)
        let time_since_last_access = current_time.saturating_sub(record.last_access_ms);
        let recency_score = self.calculate_recency_score(time_since_last_access);

        // Calculate frequency score (normalized by max possible accesses)
        let time_since_first_access = current_time.saturating_sub(record.first_access_ms);
        let frequency_score =
            self.calculate_frequency_score(record.access_count, time_since_first_access);

        // Calculate combined importance
        let combined_importance = self.calculate_combined_importance(
            record.base_importance,
            recency_score,
            frequency_score,
        );

        // Postconditions
        assert!(
            recency_score >= ACCESS_TRACKER_MIN_IMPORTANCE
                && recency_score <= ACCESS_TRACKER_MAX_IMPORTANCE,
            "recency_score {} outside valid range",
            recency_score
        );
        assert!(
            frequency_score >= ACCESS_TRACKER_MIN_IMPORTANCE
                && frequency_score <= ACCESS_TRACKER_MAX_IMPORTANCE,
            "frequency_score {} outside valid range",
            frequency_score
        );
        assert!(
            combined_importance >= ACCESS_TRACKER_MIN_IMPORTANCE
                && combined_importance <= ACCESS_TRACKER_MAX_IMPORTANCE,
            "combined_importance {} outside valid range",
            combined_importance
        );

        Some(AccessPattern {
            first_access_ms: record.first_access_ms,
            last_access_ms: record.last_access_ms,
            access_count: record.access_count,
            recency_score,
            frequency_score,
            combined_importance,
        })
    }

    /// Prune old access records.
    pub fn prune_old_records(&mut self, before_ms: u64) -> usize {
        // Preconditions
        assert!(before_ms > 0, "before_ms must be positive: {}", before_ms);

        let initial_count = self.access_records.len();

        self.access_records
            .retain(|_id, record| record.last_access_ms >= before_ms);

        let removed_count = initial_count - self.access_records.len();

        // Postconditions
        assert!(
            removed_count <= initial_count,
            "cannot remove more than initial count"
        );

        removed_count
    }

    /// Calculate recency score using exponential decay.
    fn calculate_recency_score(&self, time_since_access_ms: u64) -> f64 {
        // Exponential decay: score = 0.5^(t / halflife)
        let decay_factor = time_since_access_ms as f64 / ACCESS_TRACKER_DECAY_HALFLIFE_MS as f64;
        let score = 0.5_f64.powf(decay_factor);

        // Clamp to [0.0, 1.0]
        score
            .max(ACCESS_TRACKER_MIN_IMPORTANCE)
            .min(ACCESS_TRACKER_MAX_IMPORTANCE)
    }

    /// Calculate frequency score.
    fn calculate_frequency_score(&self, access_count: u64, time_since_first_access_ms: u64) -> f64 {
        if time_since_first_access_ms == 0 {
            // Just accessed for first time
            return 0.5;
        }

        // Accesses per day
        let days = (time_since_first_access_ms as f64) / (24.0 * 60.0 * 60.0 * 1000.0);
        let accesses_per_day = (access_count as f64) / days.max(1.0);

        // Normalize: 0 accesses/day = 0.0, 10+ accesses/day = 1.0
        let score = (accesses_per_day / 10.0).min(1.0);

        score
            .max(ACCESS_TRACKER_MIN_IMPORTANCE)
            .min(ACCESS_TRACKER_MAX_IMPORTANCE)
    }

    /// Calculate combined importance score.
    fn calculate_combined_importance(
        &self,
        base_importance: f64,
        recency_score: f64,
        frequency_score: f64,
    ) -> f64 {
        // Weighted combination: 50% base + 30% recency + 20% frequency
        let combined = 0.5 * base_importance + 0.3 * recency_score + 0.2 * frequency_score;

        combined
            .max(ACCESS_TRACKER_MIN_IMPORTANCE)
            .min(ACCESS_TRACKER_MAX_IMPORTANCE)
    }

    /// Get access record count (for testing).
    #[cfg(test)]
    pub fn record_count(&self) -> usize {
        self.access_records.len()
    }

    /// Get clock reference.
    ///
    /// Used by eviction policies to get current time for LRU calculations.
    #[must_use]
    pub fn clock(&self) -> &SimClock {
        &self.clock
    }
}

// =============================================================================
// DST Tests (Deterministic Simulation Testing)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::DST_TIME_ADVANCE_MS_MAX;

    // Test constants
    const BASE_TIME_MS: u64 = 1_000_000_000;
    const ONE_DAY_MS: u64 = 24 * 60 * 60 * 1000;
    const ONE_WEEK_MS: u64 = 7 * ONE_DAY_MS;

    /// Helper: Advance clock by large amounts in safe increments
    fn advance_clock_by(clock: &SimClock, total_ms: u64) {
        let mut remaining = total_ms;
        while remaining > 0 {
            let advance = remaining.min(DST_TIME_ADVANCE_MS_MAX);
            clock.advance_ms(advance);
            remaining -= advance;
        }
    }

    #[test]
    fn test_access_tracker_new() {
        // TigerStyle: Test constructor with SimClock
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let tracker = AccessTracker::new(clock.clone());

        // Postconditions
        assert_eq!(tracker.clock().now_ms(), BASE_TIME_MS);
        assert_eq!(tracker.record_count(), 0);
    }

    #[test]
    fn test_record_single_access() {
        // DST: Test basic access recording with SimClock
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // Record access
        tracker.record_access("entity1", 0.8);

        // Postconditions
        assert_eq!(tracker.record_count(), 1);

        // Get pattern
        let pattern = tracker.get_access_pattern("entity1").unwrap();
        assert_eq!(pattern.first_access_ms, BASE_TIME_MS);
        assert_eq!(pattern.last_access_ms, BASE_TIME_MS);
        assert_eq!(pattern.access_count, 1);
        assert!(pattern.combined_importance > 0.0);
    }

    #[test]
    fn test_record_multiple_accesses_same_entity() {
        // DST: Test access count increments with SimClock time advancement
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // Record multiple accesses with time advancement
        tracker.record_access("entity1", 0.8);
        clock.advance_ms(1000);
        tracker.record_access("entity1", 0.8);
        clock.advance_ms(1000);
        tracker.record_access("entity1", 0.8);

        // Postconditions
        assert_eq!(tracker.record_count(), 1); // Still one entity

        let pattern = tracker.get_access_pattern("entity1").unwrap();
        assert_eq!(pattern.first_access_ms, BASE_TIME_MS);
        assert_eq!(pattern.last_access_ms, BASE_TIME_MS + 2000);
        assert_eq!(pattern.access_count, 3);
    }

    #[test]
    fn test_recency_decay() {
        // DST: Test exponential decay over time (deterministic with SimClock)
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // Record access at t=0
        tracker.record_access("entity1", 0.5);
        let pattern_t0 = tracker.get_access_pattern("entity1").unwrap();
        let recency_t0 = pattern_t0.recency_score;

        // Advance time by 1 week (1 halflife) - use helper for large advances
        advance_clock_by(&clock, ONE_WEEK_MS);
        let pattern_t1 = tracker.get_access_pattern("entity1").unwrap();
        let recency_t1 = pattern_t1.recency_score;

        // Advance time by another week (2 halflifes total)
        advance_clock_by(&clock, ONE_WEEK_MS);
        let pattern_t2 = tracker.get_access_pattern("entity1").unwrap();
        let recency_t2 = pattern_t2.recency_score;

        // Postconditions: Exponential decay (each halflife → 50% of previous)
        assert!(recency_t0 > recency_t1, "recency must decay over time");
        assert!(recency_t1 > recency_t2, "recency must continue decaying");
        assert!(
            (recency_t1 / recency_t0 - 0.5).abs() < 0.01,
            "recency should halve after one halflife: t0={}, t1={}, ratio={}",
            recency_t0,
            recency_t1,
            recency_t1 / recency_t0
        );
    }

    #[test]
    fn test_frequency_score() {
        // DST: Test frequency calculation (deterministic with SimClock)
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // Record 10 accesses over 1 day (10 accesses/day = max score)
        for i in 0..10 {
            clock.advance_ms(ONE_DAY_MS / 10);
            tracker.record_access("entity1", 0.5);
        }

        let pattern = tracker.get_access_pattern("entity1").unwrap();

        // Postconditions: High frequency → high score
        assert_eq!(pattern.access_count, 10);
        assert!(
            pattern.frequency_score > 0.8,
            "frequency score should be high for 10 accesses/day: {}",
            pattern.frequency_score
        );
    }

    #[test]
    fn test_combined_importance() {
        // DST: Test importance combines base + recency + frequency
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // High base importance, recent access, low frequency
        tracker.record_access("entity_high_base", 0.9);
        let pattern1 = tracker.get_access_pattern("entity_high_base").unwrap();

        // Low base importance, old access, high frequency
        tracker.record_access("entity_low_base", 0.3);
        for _i in 1..10 {
            clock.advance_ms(1000);
            tracker.record_access("entity_low_base", 0.3);
        }
        advance_clock_by(&clock, ONE_WEEK_MS); // Age it
        let pattern2 = tracker.get_access_pattern("entity_low_base").unwrap();

        // Postconditions
        assert!(
            pattern1.combined_importance > 0.7,
            "high base → high combined"
        );
        assert!(
            pattern2.combined_importance < pattern1.combined_importance,
            "low base + old access → lower combined"
        );
    }

    #[test]
    fn test_batch_access_recording() {
        // TigerStyle: Test batch operations
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock);

        let batch = vec![("entity1", 0.8), ("entity2", 0.7), ("entity3", 0.9)];

        tracker.record_batch_access(&batch);

        // Postconditions
        assert_eq!(tracker.record_count(), 3);
        assert!(tracker.get_access_pattern("entity1").is_some());
        assert!(tracker.get_access_pattern("entity2").is_some());
        assert!(tracker.get_access_pattern("entity3").is_some());
    }

    #[test]
    #[should_panic(expected = "batch size")]
    fn test_batch_access_exceeds_max() {
        // TigerStyle: Test batch size limit
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock);

        // Create oversized batch (more than ACCESS_TRACKER_BATCH_SIZE_MAX)
        let mut oversized_batch = Vec::new();
        for _ in 0..=ACCESS_TRACKER_BATCH_SIZE_MAX {
            oversized_batch.push(("entity", 0.5));
        }

        tracker.record_batch_access(&oversized_batch);
    }

    #[test]
    fn test_prune_old_records() {
        // DST: Test pruning with deterministic time
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // Record old access
        tracker.record_access("old_entity", 0.5);

        // Record new access (advance 100 days)
        advance_clock_by(&clock, 100 * ONE_DAY_MS);
        tracker.record_access("new_entity", 0.5);

        // Prune records older than 90 days
        let cutoff = BASE_TIME_MS + 90 * ONE_DAY_MS;
        let removed = tracker.prune_old_records(cutoff);

        // Postconditions
        assert_eq!(removed, 1, "should remove 1 old record");
        assert_eq!(tracker.record_count(), 1, "should keep 1 new record");
        assert!(tracker.get_access_pattern("old_entity").is_none());
        assert!(tracker.get_access_pattern("new_entity").is_some());
    }

    #[test]
    fn test_get_nonexistent_entity() {
        // TigerStyle: Test missing entity
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let tracker = AccessTracker::new(clock);

        let pattern = tracker.get_access_pattern("nonexistent");

        // Postcondition
        assert!(pattern.is_none(), "nonexistent entity should return None");
    }

    #[test]
    #[should_panic(expected = "entity_id cannot be empty")]
    fn test_record_access_empty_id() {
        // TigerStyle: Test precondition violation
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock);
        tracker.record_access("", 0.5);
    }

    #[test]
    #[should_panic(expected = "outside valid range")]
    fn test_record_access_invalid_importance() {
        // TigerStyle: Test precondition violation
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock);
        tracker.record_access("entity1", 1.5); // > 1.0
    }

    #[test]
    fn test_determinism_same_seed() {
        // DST: Test deterministic behavior with same clock start time
        let clock1 = SimClock::at_ms(BASE_TIME_MS);
        let tracker1 = AccessTracker::new(clock1.clone());

        let clock2 = SimClock::at_ms(BASE_TIME_MS);
        let tracker2 = AccessTracker::new(clock2.clone());

        // Same initial state
        assert_eq!(tracker1.clock().now_ms(), tracker2.clock().now_ms());
        assert_eq!(tracker1.record_count(), tracker2.record_count());
    }

    #[test]
    fn test_importance_bounds() {
        // TigerStyle: Test all scores stay in bounds
        let clock = SimClock::at_ms(BASE_TIME_MS);
        let mut tracker = AccessTracker::new(clock.clone());

        // Test various scenarios
        tracker.record_access("entity1", ACCESS_TRACKER_MIN_IMPORTANCE);
        tracker.record_access("entity2", ACCESS_TRACKER_MAX_IMPORTANCE);
        tracker.record_access("entity3", 0.5);

        // Advance time significantly (365 days)
        advance_clock_by(&clock, 365 * ONE_DAY_MS);

        // Check all patterns
        for entity_id in ["entity1", "entity2", "entity3"] {
            if let Some(pattern) = tracker.get_access_pattern(entity_id) {
                assert!(
                    pattern.recency_score >= ACCESS_TRACKER_MIN_IMPORTANCE
                        && pattern.recency_score <= ACCESS_TRACKER_MAX_IMPORTANCE
                );
                assert!(
                    pattern.frequency_score >= ACCESS_TRACKER_MIN_IMPORTANCE
                        && pattern.frequency_score <= ACCESS_TRACKER_MAX_IMPORTANCE
                );
                assert!(
                    pattern.combined_importance >= ACCESS_TRACKER_MIN_IMPORTANCE
                        && pattern.combined_importance <= ACCESS_TRACKER_MAX_IMPORTANCE
                );
            }
        }
    }
}
