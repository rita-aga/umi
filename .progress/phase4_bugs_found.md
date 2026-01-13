# Phase 4: Bugs Found Through DST-First Development

## Summary

This document tracks bugs found during Phase 4: Unified Memory Orchestrator implementation.

## The Process

Following DST-first principles:
1. **Write tests FIRST** before implementing UnifiedMemory
2. **Use SimClock** from DST harness (lesson from Phases 1-3)
3. **Run tests iteratively** to find bugs
4. **Fix bugs incrementally** and re-run tests
5. **Document all bugs** found during development

## Key Learnings from Phases 1-3

Before starting, these are the critical lessons:
- **combined_importance = 0.5*base + 0.3*recency + 0.2*frequency** (not just base!)
- **Use SimClock from DST harness** (don't create manual time)
- **Self_ entities are always protected** from eviction
- **Extend harness when needed** (e.g., added clock() accessor in Phase 3)

---

## Bugs Found

### Bug #1: AccessTracker.clock() Was Test-Only But Used in Production

**What Happened**:
- Phase 3 added `clock()` accessor to AccessTracker with `#[cfg(test)]`
- eviction.rs uses `access_tracker.clock().now_ms()` in production code
- This worked during Phase 3 tests but fails when compiling UnifiedMemory

**Error Message**:
```
error[E0599]: no method named `clock` found for reference `&AccessTracker` in the current scope
   --> umi-memory/src/orchestration/eviction.rs:123:47
    |
123 |             let current_time = access_tracker.clock().now_ms();
    |                                               ^^^^^ private field, not a method
```

**Root Cause**:
- In Phase 3, `clock()` was added with `#[cfg(test)]` attribute
- The eviction.rs code that calls it was inside `impl` blocks that are always compiled
- Tests passed because in test builds, `#[cfg(test)]` code is included
- Production build failed because `#[cfg(test)]` code is excluded

**Fix**:
```rust
// Before (in access_tracker.rs):
/// Get clock reference (for testing).
#[cfg(test)]
pub fn clock(&self) -> &SimClock {
    &self.clock
}

// After:
/// Get clock reference.
///
/// Used by eviction policies to get current time for LRU calculations.
#[must_use]
pub fn clock(&self) -> &SimClock {
    &self.clock
}
```

**Files Fixed**: `umi-memory/src/orchestration/access_tracker.rs`

**Lesson**: When adding methods that will be used by other production code, don't use `#[cfg(test)]`. Only use it for methods that are truly test-only helpers.

---

### Bug #2: list_entities() Takes 3 Arguments, Not 2

**What Happened**:
- Called `storage.list_entities(None, LIMIT)` with 2 arguments
- The actual signature is `list_entities(entity_type, limit, offset)`

**Error Message**:
```
error[E0061]: this method takes 3 arguments but 2 arguments were supplied
   --> umi-memory/src/orchestration/unified.rs:562:45
    |
562 |         let candidates = match self.storage.list_entities(None, UNIFIED_MEMORY_PROMOTION_CANDIDATES_MAX).await {
    |                                             ^^^^^^^^^^^^^----------------------------------------------- argument #3 of type `usize` is missing
```

**Root Cause**:
- Forgot to check the StorageBackend trait signature before calling
- Common pattern in paginated APIs to have (filter, limit, offset)

**Fix**:
```rust
// Before:
self.storage.list_entities(None, UNIFIED_MEMORY_PROMOTION_CANDIDATES_MAX).await

// After:
self.storage.list_entities(None, UNIFIED_MEMORY_PROMOTION_CANDIDATES_MAX, 0).await
```

**Files Fixed**: `umi-memory/src/orchestration/unified.rs`

**Lesson**: Always check trait signatures before calling methods. Add a TigerStyle comment when signature is non-obvious.

---

## Test Results

### Final Status
```
running 29 tests (UnifiedMemory module)
test result: ok. 29 passed; 0 failed; 0 ignored

running 575 tests (full umi-memory lib)
test result: ok. 575 passed; 0 failed; 0 ignored
```

### Bugs Found: 2

| # | Bug | Root Cause | Fix |
|---|-----|------------|-----|
| 1 | `AccessTracker.clock()` test-only | `#[cfg(test)]` on method used in production | Removed `#[cfg(test)]` |
| 2 | `list_entities()` missing argument | Forgot offset parameter | Added `0` for offset |

### Bug #3 (via DST): Incorrect Fault Injection Method

**Discovery**: When running DST tests with `Simulation::new().with_fault()`, fault injection didn't affect `SimStorageBackend`.

**What Happened Initially**:
- Set `FaultType::StorageWriteFail` at 100% rate via `Simulation.with_fault()`
- Expected all `remember()` calls to fail
- Actually: all succeeded (`is_ok=true`)

**Root Cause**:
- `Simulation.with_fault()` configures faults for DST-internal components (`SimStorage`, `SimLLM` in `dst/` module)
- `SimStorageBackend` in `storage/sim.rs` has its own independent `FaultInjector`
- I was using the wrong fault injection API

**Fix**: Use `SimStorageBackend::new().with_faults()` to properly inject faults:

```rust
// WRONG: Simulation.with_fault() doesn't affect SimStorageBackend
let sim = Simulation::new(SimConfig::with_seed(42))
    .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 1.0));

// CORRECT: Use SimStorageBackend.with_faults()
let storage = SimStorageBackend::new(SimConfig::with_seed(seed))
    .with_faults(FaultConfig::new(FaultType::StorageWriteFail, 1.0));
```

**Result After Fix**:
- Storage write failure (100%): `is_ok=false` ✅
- Storage read failure (100%): `is_ok=false` ✅
- Probabilistic (50%): `success=5, failure=5` ✅

**Files Fixed**: `umi-memory/src/orchestration/unified.rs`

**Lesson**: Read the existing fault injection tests (in `storage/sim.rs`) to understand the correct API. DST has multiple fault injection points - use the right one for each component.

---

## Files Created/Modified

1. `umi-memory/src/orchestration/unified.rs` - NEW
2. `umi-memory/src/orchestration/mod.rs` - MODIFIED
3. `umi-memory/src/constants.rs` - MODIFIED (if needed)
4. `umi-memory/src/lib.rs` - MODIFIED (export)

---

## Progress

**Completed:** 3/8 phases (37.5%)
- ✅ Phase 1: Access Tracking Foundation (14 tests, 4 bugs found)
- ✅ Phase 2: Promotion Policy System (14 tests, 5 bugs found)
- ✅ Phase 3: Eviction Policy System (17 tests, 2 bugs found)
- 🔄 Phase 4: Unified Memory Orchestrator (IN PROGRESS)

**Total Bugs Found So Far:** 11 bugs across 3 phases
