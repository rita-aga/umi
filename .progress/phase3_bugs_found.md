# Phase 3: Bugs Found Through DST-First Development

## Summary

By following DST-first principles (write tests FIRST, run iteratively), we found **2 bugs** during Phase 3 implementation.

## The Process

### What I Did Right ✅
1. **Wrote tests FIRST** before implementing eviction policies
2. **Used SimClock** from DST harness (lesson from Phase 1)
3. **Ran tests iteratively** to find bugs
4. **Fixed bugs incrementally** and re-ran tests
5. **All 17 tests passing** with no regressions

### Bugs Found

#### Bug #1: LRU Current Time Calculation Error (Runtime Failure)
**What Happened**:
- Test `test_lru_eviction_with_threshold` failed
- Expected 1 entity to evict but got 0

**Error Message**:
```
thread 'orchestration::eviction::tests::test_lru_eviction_with_threshold'
panicked at umi-memory/src/orchestration/eviction.rs:582:9:
assertion `left == right` failed
  left: 0
 right: 1
```

**Root Cause**:
The LRU policy was trying to get "current time" to calculate time differences:
```rust
// WRONG: Using an entity's last_access_ms as "current time"
if let Some(first_entity) = core_entities.first() {
    if let Some(pattern) = access_tracker.get_access_pattern(&first_entity.id) {
        let current_time = pattern.last_access_ms; // Bug!
        candidates.retain(|(_, last_access)| {
            current_time.saturating_sub(*last_access) >= threshold
        });
    }
}
```

**Why This is Wrong**:
- Used the FIRST entity's `last_access_ms` as "current time"
- If the first entity was the oldest, its time difference from "current" would be 0
- This meant old entities wouldn't be evicted

**In the Test**:
- note1 accessed at time 0, then advanced 5 days
- note2 accessed at time 5 days, then advanced 0 days
- If note1 was first in the list, current_time = 0
- Time difference: 0 - 0 = 0 (not >= 3 day threshold)
- So note1 wouldn't be evicted even though it should be

**Fix**:
Get the actual current time from the clock via AccessTracker:
```rust
// CORRECT: Get current time from the clock
let current_time = access_tracker.clock().now_ms();
candidates.retain(|(_, last_access)| {
    current_time.saturating_sub(*last_access) >= threshold
});
```

**Files Fixed**: `eviction.rs` lines 121-127

**Lesson**: Don't approximate "current time" from arbitrary data. Use the actual clock.

---

#### Bug #2: Combined Importance Misunderstanding (Runtime Failure)
**What Happened**:
- Test `test_importance_eviction_with_threshold` failed
- Expected 1 entity to evict but got 0

**Error Message**:
```
thread 'orchestration::eviction::tests::test_importance_eviction_with_threshold'
panicked at umi-memory/src/orchestration/eviction.rs:648:9:
assertion `left == right` failed
  left: 0
 right: 1
```

**Root Cause**:
Same issue as Phase 2 Bug #5 - misunderstanding how `combined_importance` works.

**In the Test**:
```rust
tracker.record_access("note1", 0.4); // Below threshold?
tracker.record_access("note2", 0.8); // Above threshold?
let policy = ImportanceEvictionPolicy::with_threshold(0.6);
```

But `combined_importance` is NOT the same as base importance!

**Calculation**:
- combined = 0.5 × base + 0.3 × recency + 0.2 × frequency
- For freshly accessed: recency=1.0, frequency=0.5

So:
- note1: base 0.4 → combined = 0.2 + 0.3 + 0.1 = **0.6** (NOT below 0.6!)
- note2: base 0.8 → combined = 0.4 + 0.3 + 0.1 = **0.8**

Since note1's combined importance (0.6) equals the threshold (0.6), it won't be evicted.

**Fix**:
Update test to use correct base importance values:
```rust
// CORRECT: Account for combined_importance calculation
// note1: base 0.0 -> combined = 0.0 + 0.3 + 0.1 = 0.4 (below 0.6)
// note2: base 1.0 -> combined = 0.5 + 0.3 + 0.1 = 0.9 (above 0.6)
tracker.record_access("note1", 0.0); // Combined ~0.4 (below threshold)
tracker.record_access("note2", 1.0); // Combined ~0.9 (above threshold)
```

**Files Fixed**: `eviction.rs` lines 641-646

**Lesson**: This is the SECOND time this bug appeared (first in Phase 2). The combined_importance calculation is non-obvious and easy to misunderstand. Added detailed comments in the test to prevent future confusion.

---

## Test Results

### Before Fixes
```
running 17 tests
test result: FAILED. 15 passed; 2 failed
```

### After Fixes
```
running 17 tests
test result: ok. 17 passed; 0 failed; 0 ignored
```

### Full Test Suite
```
test result: ok. 604 passed; 0 failed; 2 ignored
```

**No regressions** - all existing 587 tests still pass, plus 17 new eviction tests.

---

## Components Implemented

### 1. EvictionPolicy Trait
Pluggable eviction logic for Core Memory management.

### 2. LRUEvictionPolicy
- Evicts least recently used entities
- Optional time threshold (only evict if not accessed within N milliseconds)
- Self_ entities never evicted (protected)

### 3. ImportanceEvictionPolicy
- Evicts lowest importance entities
- Optional minimum threshold (never evict above this importance)
- Self_ entities never evicted (protected)

### 4. HybridEvictionPolicy
- Combines importance (60%) and recency (40%)
- Calculates eviction score: lower score = higher priority to evict
- Optional minimum importance threshold
- Self_ entities never evicted (protected)
- Custom weights supported

### Key Features:
- ✅ All policies protect Self_ entities
- ✅ All policies deterministic
- ✅ All policies configurable
- ✅ All policies have comprehensive assertions (TigerStyle)

---

## Test Coverage

**17 comprehensive DST tests:**

1. `test_lru_eviction_basic` - Basic LRU functionality
2. `test_lru_eviction_protects_self` - Self_ protection
3. `test_lru_eviction_with_threshold` - Time threshold filtering
4. `test_importance_eviction_basic` - Basic importance-based
5. `test_importance_eviction_protects_self` - Self_ protection
6. `test_importance_eviction_with_threshold` - Importance threshold
7. `test_hybrid_eviction_basic` - Basic hybrid functionality
8. `test_hybrid_eviction_protects_self` - Self_ protection
9. `test_hybrid_eviction_custom_weights` - Custom weight configuration
10. `test_eviction_empty_entities` - Edge case: empty input
11. `test_eviction_all_protected` - Edge case: all Self_ entities
12. `test_lru_invalid_threshold` - Panic test: invalid threshold
13. `test_importance_invalid_threshold_high` - Panic test
14. `test_importance_invalid_threshold_low` - Panic test
15. `test_hybrid_invalid_weight_importance` - Panic test
16. `test_hybrid_weights_dont_sum_to_one` - Panic test
17. `test_eviction_determinism` - Deterministic behavior verification

---

## Lessons Learned

### 1. Recurring Bugs Reveal Design Issues
Bug #2 is the SAME bug as Phase 2 Bug #5:
- Misunderstanding combined_importance calculation
- This is the second time it appeared in tests
- Suggests the API might be confusing

**Action Taken**: Added detailed comments explaining the calculation in both tests.

### 2. Don't Approximate What You Can Measure
Bug #1 (using arbitrary entity's time as "current time") was a classic mistake:
- Had access to the real clock via `access_tracker.clock()`
- But tried to approximate "current time" from data
- **Always use the actual source of truth**

### 3. DST Harness Extension Works
During implementation, I discovered AccessTracker didn't expose its clock publicly.
**Extended the harness**: Added `pub fn clock(&self) -> &SimClock` method.
This is exactly what DST-first encourages - extend the harness when needed.

### 4. Edge Case Testing is Critical
Tests for empty entities and all-protected entities caught potential panics:
- Empty list → should return empty, not panic
- All Self_ → should return empty, not fail assertions

---

## Files Created/Modified

1. `umi-memory/src/orchestration/eviction.rs` - NEW (800+ lines)
   - EvictionPolicy trait
   - LRUEvictionPolicy implementation
   - ImportanceEvictionPolicy implementation
   - HybridEvictionPolicy implementation
   - 17 comprehensive tests
   - Fixed 2 bugs

2. `umi-memory/src/orchestration/access_tracker.rs` - MODIFIED
   - Added `pub fn clock(&self) -> &SimClock` method (harness extension)

3. `umi-memory/src/orchestration/mod.rs` - MODIFIED
   - Exported eviction module

4. `umi-memory/src/constants.rs` - MODIFIED
   - Added 8 eviction constants with TigerStyle naming

---

## Phase 3 Summary

**Completed:**
- ✅ EvictionPolicy trait designed
- ✅ LRUEvictionPolicy implemented
- ✅ ImportanceEvictionPolicy implemented
- ✅ HybridEvictionPolicy implemented
- ✅ 17 DST tests passing
- ✅ 8 TigerStyle constants defined
- ✅ Self_ entity protection (never evict)
- ✅ Configurable thresholds and weights
- ✅ All policies deterministic
- ✅ Extended DST harness (added clock() accessor)

**Bugs Found:** 2 (both runtime logic bugs)
**Tests Added:** 17 (12 functional + 5 panic tests)
**Test Pass Rate:** 100% (17/17 eviction tests, 604/604 total)

---

## Progress

**Completed:** 3/8 phases (37.5%)
- ✅ Phase 1: Access Tracking Foundation (14 tests, 4 bugs found)
- ✅ Phase 2: Promotion Policy System (14 tests, 5 bugs found)
- ✅ Phase 3: Eviction Policy System (17 tests, 2 bugs found)

**Next:** Phase 4: Unified Memory Orchestrator (the big integration phase)

**Total Bugs Found So Far:** 11 bugs across 3 phases
**All bugs caught before integration!**

---

## Conclusion

**DST-first development continues to work**:
- Found 2 bugs during implementation (not in production)
- One bug (current time approximation) was a logic error
- One bug (combined_importance) was a recurring design understanding issue
- Extended the DST harness when needed (clock accessor)
- All tests pass with 100% rate

This is **exactly** why UMI follows DST-first principles - catch bugs early, make them reproducible, and build confidence in correctness incrementally.

Phase 4 will integrate all these components into a UnifiedMemory orchestrator that manages all three memory tiers automatically.
