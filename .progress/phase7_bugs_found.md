# Phase 7: Bugs Found Through DST-First Performance Testing

## Summary

This document tracks findings from Phase 7: Performance & Benchmarking using DST-first approach.

### Bug Classification

| Type | Description |
|------|-------------|
| **DST-FOUND** ⭐ | Bug/insight discovered only by running simulation with fault injection |
| **DESIGN-DOC** | Not a bug - documented behavior discovered through testing |
| **TEST-CONFIG** | Test setup issue, not a code bug |

---

## DST-Found Insights (Performance Testing)

### Finding #1: Compound Fault Probability ⭐
**Type:** DST-FOUND
**Severity:** Design Documentation
**Location:** `orchestration/tests/performance.rs:test_throughput_under_storage_faults`

**How Found:** Running DST performance test with 50% storage fault rate showed 27% success rate.
Initially looked like a bug (expected ~50%), but DST analysis revealed correct behavior.

**Issue:** Operation success rate ≠ (1 - fault_rate) when operations involve multiple storage calls.

**Math:**
```
With 50% per-store fault rate:
- Each remember() calls store_entity() twice (for 2 entities)
- P(both stores succeed) = 0.5 × 0.5 = 0.25 = 25%
- Observed: 27% ✓ (matches expected)
```

**Impact:**
- Fault rate configuration doesn't linearly map to operation success rate
- Operations with N sub-operations: P(success) = (1 - fault_rate)^N
- Must consider compound probability when setting fault rates

**Status:** Documented behavior. Updated test with correct expectations.

---

### Finding #2: Recall Also Tracks Accesses ⭐
**Type:** DST-FOUND
**Severity:** Design Documentation
**Location:** `orchestration/unified.rs:recall()` lines 654, 672, 677

**How Found:** DST test `test_concurrent_access_under_faults` showed 116 accesses when
expected bound was 60 (based on remember count only). Investigation revealed recall()
also tracks accesses.

**Issue:** Initial test bound calculation only counted remember() operations, but recall()
also calls `category_evolver.track_access()` for each entity retrieved.

**Code Path:**
```rust
// In recall():
for entity in core_results {
    self.access_tracker.record_access(&entity.id, 0.5);  // Line 654
    self.category_evolver.track_access(...);  // Line 658-659
    ...
}
```

**Impact:**
- Total accesses = remember_accesses + recall_accesses
- Access patterns reflect both write AND read behavior
- Evolution analysis captures full usage patterns

**Status:** By design. Updated test with correct access counting formula.

---

### Finding #3: API Surface Mismatch (Test Configuration)
**Type:** TEST-CONFIG
**Severity:** Test Issue
**Location:** `orchestration/tests/performance.rs`

**How Found:** Initial DST tests failed to compile, revealing non-existent API methods.

**Missing APIs (Not Bugs):**
1. `UnifiedMemoryConfig::with_working_memory_limit()` - doesn't exist
2. `UnifiedMemory::evict_from_working()` - doesn't exist (only `evict_from_core()`)
3. `UnifiedMemory::working_memory_count()` - use `working().entry_count()` instead
4. `crate::entity::EntityType` - use `crate::storage::EntityType`

**Status:** Test configuration issue. Fixed tests to use actual API.

---

## DST Verification Results

### What We Verified (No Bugs Found - System Is Robust)

| Test | Invariant | Result |
|------|-----------|--------|
| `test_throughput_under_storage_faults` | Success rate matches compound probability | ✓ 27% observed ≈ 25% expected |
| `test_latency_bounds_under_faults` | p99 latency ≤ 10x median | ✓ p99=14ms, median=10ms |
| `test_memory_bounds_under_load` | Core count ≤ limit | ✓ core_count=0, limit=10 |
| `test_recovery_after_fault_resolution` | 100% success after faults stop | ✓ 0 failures in recovery |
| `test_deterministic_timing` | Same seed = same results | ✓ Identical across runs |
| `test_llm_fallback_latency_bounds` | Fallback latency bounded | ✓ max=5ms |
| `test_concurrent_access_under_faults` | Access count bounded | ✓ 116 ≤ 142 expected |
| `test_eviction_under_memory_pressure` | Core ≤ limit, eviction triggers | ✓ core=3, eviction=true |

---

## Test Results

### Current Status (8 Performance Tests)
```
running 8 tests
test orchestration::tests::performance::test_concurrent_access_under_faults ... ok
test orchestration::tests::performance::test_deterministic_timing ... ok
test orchestration::tests::performance::test_eviction_under_memory_pressure ... ok
test orchestration::tests::performance::test_latency_bounds_under_faults ... ok
test orchestration::tests::performance::test_llm_fallback_latency_bounds ... ok
test orchestration::tests::performance::test_memory_bounds_under_load ... ok
test orchestration::tests::performance::test_recovery_after_fault_resolution ... ok
test orchestration::tests::performance::test_throughput_under_storage_faults ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured
```

---

## Files Created/Modified

1. `umi-memory/src/orchestration/tests/performance.rs` - NEW (8 DST performance tests)
2. `umi-memory/src/orchestration/tests/mod.rs` - MODIFIED (added performance module)
3. `.progress/007_20260113_phase7-performance-benchmarking.md` - NEW (plan)
4. `.progress/phase7_bugs_found.md` - NEW (this file)

---

## Phase 7 Summary

### Performance Testing Results

| Metric | Tested | Status |
|--------|--------|--------|
| Throughput under faults | 50% fault rate, 100 operations | ✓ Degrades proportionally |
| Latency bounds | 30% faults, 50 operations | ✓ p99 within 1.5x median |
| Memory bounds | 100 iterations, core limit 10 | ✓ Stays bounded |
| Recovery time | 80% fault then 0% | ✓ Immediate recovery |
| Determinism | Same seed comparison | ✓ Fully deterministic |
| LLM fallback | 100% LLM timeout | ✓ Bounded fallback path |
| Concurrent access | Mixed read/write | ✓ State consistent |
| Eviction pressure | 100 entities, limit 5 | ✓ Eviction triggers |

### DST-Found Insights Summary

| Finding | Classification | Impact |
|---------|---------------|--------|
| Compound fault probability | DST-FOUND ⭐ | Operation success = (1-rate)^N |
| Recall tracks accesses | DST-FOUND ⭐ | Total accesses include reads |
| API surface mismatch | TEST-CONFIG | Fixed test code |

### Key Takeaway

**No performance bugs found** - the system is robust under fault injection:
- Throughput degrades predictably (compound probability)
- Latency stays bounded (no infinite loops)
- Memory stays bounded (eviction works)
- Recovery is immediate (no lingering state)
- Behavior is fully deterministic

---

## Progress

**Completed:** 7/8 phases (87.5%)
- ✅ Phase 1: Access Tracking Foundation (14 tests, 4 bugs found)
- ✅ Phase 2: Promotion Policy System (14 tests, 5 bugs found)
- ✅ Phase 3: Eviction Policy System (17 tests, 2 bugs found)
- ✅ Phase 4: Unified Memory Orchestrator (37 tests, 3 bugs found)
- ✅ Phase 5: Self-Evolution CategoryEvolver (22 tests, 2 bugs found)
- ✅ Phase 6: Integration & Migration Path (27 tests, 6 bugs found)
- ✅ Phase 7: Performance & Benchmarking (8 tests, 2 DST-found insights)
- 🔲 Phase 8: Documentation & Examples

**Total Tests:** 139+ tests across all phases
**Total Bugs/Insights Found:** 24 across 7 phases
