//! SimClock - Simulated Time
//!
//! TigerStyle: Deterministic, controllable time for simulation.
//! Supports async sleep/notify for coordinating time-dependent tasks.

use crate::constants::{DST_TIME_ADVANCE_MS_MAX, TIME_MS_PER_SEC};
use chrono::{DateTime, Duration, Utc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

/// A simulated clock for deterministic testing.
///
/// TigerStyle:
/// - Time only moves forward
/// - All time operations are explicit
/// - No reliance on system time
/// - Supports async sleep with notify for coordination
///
/// Thread-safe via Arc<AtomicU64> for current time.
#[derive(Debug, Clone)]
pub struct SimClock {
    /// Current time in milliseconds since epoch (thread-safe)
    current_ms: Arc<AtomicU64>,
    /// Notify waiters when time advances
    notify: Arc<Notify>,
}

impl SimClock {
    /// Create a new clock starting at time zero.
    ///
    /// # Example
    /// ```
    /// use umi_core::dst::SimClock;
    /// let clock = SimClock::new();
    /// assert_eq!(clock.now_ms(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_ms: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Create a clock starting at the given millisecond timestamp.
    #[must_use]
    pub fn at_ms(start_ms: u64) -> Self {
        Self {
            current_ms: Arc::new(AtomicU64::new(start_ms)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Create a clock starting at the given DateTime.
    #[must_use]
    pub fn at_datetime(dt: DateTime<Utc>) -> Self {
        let ms = dt.timestamp_millis() as u64;
        Self::at_ms(ms)
    }

    /// Create a clock starting at Unix epoch (1970-01-01).
    #[must_use]
    pub fn from_epoch() -> Self {
        Self::new()
    }

    /// Get current time in milliseconds.
    #[must_use]
    pub fn now_ms(&self) -> u64 {
        self.current_ms.load(Ordering::SeqCst)
    }

    /// Get current time in seconds (truncated).
    #[must_use]
    pub fn now_secs(&self) -> u64 {
        self.now_ms() / TIME_MS_PER_SEC
    }

    /// Get current time as DateTime<Utc>.
    #[must_use]
    pub fn now(&self) -> DateTime<Utc> {
        let ms = self.now_ms() as i64;
        DateTime::from_timestamp_millis(ms).unwrap_or_else(|| {
            // Fallback for invalid timestamps
            DateTime::from_timestamp(0, 0).unwrap()
        })
    }

    /// Advance time by the given milliseconds.
    ///
    /// # Panics
    /// Panics if ms exceeds DST_TIME_ADVANCE_MS_MAX.
    ///
    /// # Returns
    /// The new current time.
    pub fn advance_ms(&self, ms: u64) -> u64 {
        // Preconditions
        assert!(
            ms <= DST_TIME_ADVANCE_MS_MAX,
            "advance_ms({}) exceeds max ({})",
            ms,
            DST_TIME_ADVANCE_MS_MAX
        );

        let old_time = self.current_ms.fetch_add(ms, Ordering::SeqCst);
        let new_time = old_time.saturating_add(ms);

        // Notify all waiters that time has advanced
        self.notify.notify_waiters();

        // Postcondition
        assert!(new_time >= old_time, "time must not go backwards");

        new_time
    }

    /// Advance time by the given seconds.
    ///
    /// # Panics
    /// Panics if resulting ms exceeds DST_TIME_ADVANCE_MS_MAX.
    pub fn advance_secs(&self, secs: f64) -> u64 {
        // Precondition
        assert!(secs >= 0.0, "secs must be non-negative, got {}", secs);

        let ms = (secs * 1000.0) as u64;
        self.advance_ms(ms)
    }

    /// Advance time by a chrono Duration.
    pub fn advance(&self, duration: Duration) {
        debug_assert!(duration >= Duration::zero(), "cannot go back in time");

        let delta_ms = duration.num_milliseconds() as u64;
        self.advance_ms(delta_ms);
    }

    /// Set time to absolute value.
    ///
    /// # Panics
    /// Panics if new time is less than current time.
    pub fn set_ms(&self, ms: u64) {
        let current = self.now_ms();
        // Precondition
        assert!(
            ms >= current,
            "cannot set time backwards: {} < {}",
            ms,
            current
        );

        self.current_ms.store(ms, Ordering::SeqCst);
        self.notify.notify_waiters();

        // Postcondition
        assert_eq!(self.now_ms(), ms, "time must be set correctly");
    }

    /// Set time to a DateTime.
    pub fn set(&self, time: DateTime<Utc>) {
        let ms = time.timestamp_millis() as u64;
        self.set_ms(ms);
    }

    /// Get elapsed time since a given timestamp.
    ///
    /// # Panics
    /// Panics if since is in the future.
    #[must_use]
    pub fn elapsed_since(&self, since: u64) -> u64 {
        let current = self.now_ms();
        // Precondition
        assert!(
            since <= current,
            "elapsed_since({}) is in the future (now={})",
            since,
            current
        );

        current - since
    }

    /// Check if a given duration has elapsed since a timestamp.
    #[must_use]
    pub fn has_elapsed(&self, since: u64, duration_ms: u64) -> bool {
        self.elapsed_since(since) >= duration_ms
    }

    /// Check if a deadline (in ms) has passed.
    #[must_use]
    pub fn is_past_ms(&self, deadline_ms: u64) -> bool {
        self.now_ms() >= deadline_ms
    }

    /// Check if a DateTime deadline has passed.
    #[must_use]
    pub fn is_past(&self, deadline: DateTime<Utc>) -> bool {
        self.now() >= deadline
    }

    /// Get a timestamp that represents "now" for storing.
    #[must_use]
    pub fn timestamp(&self) -> u64 {
        self.now_ms()
    }

    /// Sleep until the specified duration has passed.
    ///
    /// In simulation mode, this yields and waits for time to be advanced.
    /// Returns when current_time >= start_time + duration_ms.
    pub async fn sleep_ms(&self, duration_ms: u64) {
        let target_ms = self.now_ms() + duration_ms;

        while self.now_ms() < target_ms {
            self.notify.notified().await;
        }
    }

    /// Sleep for a chrono Duration.
    pub async fn sleep(&self, duration: Duration) {
        let ms = duration.num_milliseconds() as u64;
        self.sleep_ms(ms).await;
    }

    /// Sleep until a specific deadline.
    pub async fn sleep_until_ms(&self, deadline_ms: u64) {
        while self.now_ms() < deadline_ms {
            self.notify.notified().await;
        }
    }
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_time() {
        let clock = SimClock::new();
        assert_eq!(clock.now_ms(), 0);
        assert_eq!(clock.now_secs(), 0);
    }

    #[test]
    fn test_at_ms() {
        let clock = SimClock::at_ms(5000);
        assert_eq!(clock.now_ms(), 5000);
        assert_eq!(clock.now_secs(), 5);
    }

    #[test]
    fn test_at_datetime() {
        let dt = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .to_utc();
        let clock = SimClock::at_datetime(dt);
        assert_eq!(clock.now(), dt);
    }

    #[test]
    fn test_advance_ms() {
        let clock = SimClock::new();

        let new_time = clock.advance_ms(1000);

        assert_eq!(new_time, 1000);
        assert_eq!(clock.now_ms(), 1000);
    }

    #[test]
    fn test_advance_secs() {
        let clock = SimClock::new();

        let new_time = clock.advance_secs(1.5);

        assert_eq!(new_time, 1500);
        assert_eq!(clock.now_ms(), 1500);
    }

    #[test]
    fn test_advance_duration() {
        let clock = SimClock::new();

        clock.advance(Duration::seconds(10));

        assert_eq!(clock.now_ms(), 10_000);
    }

    #[test]
    fn test_multiple_advances() {
        let clock = SimClock::new();

        clock.advance_ms(100);
        clock.advance_ms(200);
        clock.advance_ms(300);

        assert_eq!(clock.now_ms(), 600);
    }

    #[test]
    #[should_panic(expected = "advance_ms")]
    fn test_advance_exceeds_max() {
        let clock = SimClock::new();
        clock.advance_ms(DST_TIME_ADVANCE_MS_MAX + 1);
    }

    #[test]
    fn test_set_ms() {
        let clock = SimClock::new();

        clock.set_ms(5000);

        assert_eq!(clock.now_ms(), 5000);
    }

    #[test]
    #[should_panic(expected = "cannot set time backwards")]
    fn test_set_ms_backwards() {
        let clock = SimClock::new();
        clock.advance_ms(1000);
        clock.set_ms(500);
    }

    #[test]
    fn test_elapsed_since() {
        let clock = SimClock::new();
        let start = clock.now_ms();
        clock.advance_ms(500);

        let elapsed = clock.elapsed_since(start);

        assert_eq!(elapsed, 500);
    }

    #[test]
    fn test_has_elapsed() {
        let clock = SimClock::new();
        let start = clock.now_ms();

        assert!(!clock.has_elapsed(start, 1000));

        clock.advance_ms(500);
        assert!(!clock.has_elapsed(start, 1000));

        clock.advance_ms(500);
        assert!(clock.has_elapsed(start, 1000));

        clock.advance_ms(100);
        assert!(clock.has_elapsed(start, 1000));
    }

    #[test]
    #[should_panic(expected = "is in the future")]
    fn test_elapsed_since_future() {
        let clock = SimClock::new();
        let _ = clock.elapsed_since(1000);
    }

    #[test]
    fn test_timestamp() {
        let clock = SimClock::new();
        clock.advance_ms(12345);
        assert_eq!(clock.timestamp(), 12345);
    }

    #[test]
    fn test_is_past_ms() {
        let clock = SimClock::at_ms(1000);

        assert!(clock.is_past_ms(500));
        assert!(clock.is_past_ms(1000));
        assert!(!clock.is_past_ms(1500));
    }

    #[test]
    fn test_now_datetime() {
        let clock = SimClock::at_ms(0);
        let epoch = DateTime::from_timestamp(0, 0).unwrap();
        assert_eq!(clock.now(), epoch);
    }

    #[test]
    fn test_clone_shares_time() {
        let clock1 = SimClock::new();
        let clock2 = clock1.clone();

        clock1.advance_ms(1000);

        // Both clocks should see the same time (shared state)
        assert_eq!(clock1.now_ms(), 1000);
        assert_eq!(clock2.now_ms(), 1000);
    }

    #[tokio::test]
    async fn test_sleep_ms() {
        let clock = SimClock::new();
        let clock_clone = clock.clone();

        // Spawn a task that sleeps
        let handle = tokio::spawn(async move {
            clock_clone.sleep_ms(100).await;
            clock_clone.now_ms()
        });

        // Advance time in increments
        tokio::task::yield_now().await;
        clock.advance_ms(50);
        tokio::task::yield_now().await;
        clock.advance_ms(50);
        tokio::task::yield_now().await;

        let result = handle.await.unwrap();
        assert!(result >= 100);
    }

    #[tokio::test]
    async fn test_sleep_duration() {
        let clock = SimClock::new();
        let clock_clone = clock.clone();

        let handle = tokio::spawn(async move {
            clock_clone.sleep(Duration::milliseconds(200)).await;
            clock_clone.now_ms()
        });

        tokio::task::yield_now().await;
        clock.advance_ms(200);
        tokio::task::yield_now().await;

        let result = handle.await.unwrap();
        assert!(result >= 200);
    }
}
