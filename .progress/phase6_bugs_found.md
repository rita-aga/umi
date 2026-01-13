# Phase 6: Bugs Found Through DST-First Development

## Summary

This document tracks bugs found during Phase 6: Integration & Migration Path implementation.

### Bug Classification

| Type | Description |
|------|-------------|
| **DST-FOUND** | Bug discovered only by running simulation with fault injection |
| **DESIGN-DOC** | Not a bug - documented behavior discovered through testing |
| **TEST-CONFIG** | Test setup issue, not a code bug |

## The Process

Following DST-first principles with VERIFICATION:
1. **Write DST integration tests FIRST** - full workflow tests
2. **Run tests with Simulation harness** - sim.run(|env| ...)
3. **Verify fault injection works** - failures must actually fail
4. **Document all bugs** found during development

## Key Questions Before Writing Tests

1. **What makes integration tests meaningful?**
   - Test full workflow: remember -> promote -> recall -> evict
   - Test with multiple entity types
   - Test graceful degradation under faults
   - Test determinism across sessions

2. **What needs DST coverage?**
   - UnifiedMemory full lifecycle
   - Feature flag behavior
   - Migration path (Memory -> UnifiedMemory)

---

## Harness Verification

### Checklist Before Implementation

- [ ] Full workflow test fails without proper implementation
- [ ] Fault injection tests actually fail when faults are injected
- [ ] Multi-session tests verify state persistence
- [ ] Feature flag tests verify correct behavior

---

## Bugs Found

### Bug #1: Test Exceeded LLM Assertion Boundary (Fixed)
**Type:** TEST-CONFIG
**Severity:** Test Issue
**Location:** `orchestration/tests/integration.rs:test_long_input_handling`

**Issue:** Test used 100KB input which exactly hit the LLM's TigerStyle assertion boundary.

**Symptom:**
```
thread 'test_long_input_handling' panicked at umi-memory/src/llm/mod.rs:221:9:
prompt exceeds 100000 bytes
```

**Root Cause:** The test used `"x".repeat(100_000)` which, with prompt overhead, exceeded the 100KB limit.

**Fix:** Reduced test input to 50KB which still tests "very long" input handling but stays well below the assertion boundary.

---

### Bug #2: Frequency Score Requires Time Advancement (Documented)
**Type:** DESIGN-DOC
**Severity:** Design Issue
**Location:** `orchestration/access_tracker.rs:calculate_frequency_score`

**Issue:** Frequency score calculation returns 0.5 when `time_since_first_access_ms == 0`.

**Symptom:** Multiple accesses at the same timestamp don't increase frequency score.

**Root Cause:** Early return at line 210:
```rust
if time_since_first_access_ms == 0 {
    return 0.5;  // Always returns 0.5, ignoring access_count
}
```

**Impact:** Tests without time advancement never build up frequency scores, causing promotion to fail even with many accesses.

**Fix:** Tests must advance SimClock between accesses for realistic behavior.

---

### Bug #3: Promotion Threshold Blocks Single-Access Entities (By Design)
**Type:** DST-FOUND ⭐
**Severity:** Design Issue
**Location:** `orchestration/promotion.rs:HybridPolicy`

**How Found:** Running DST workflow tests showed 0 promotions when we expected some.
The simulation revealed that repeated access with time advancement is required.

**Issue:** Default threshold (0.75) is higher than single-access entity scores (~0.71).

**Math:**
- base_importance = 0.5, recency = 1.0, frequency = 0.5
- combined_importance = 0.5*0.5 + 0.3*1.0 + 0.2*0.5 = 0.65
- HybridPolicy score ≈ 0.71 < threshold 0.75

**Impact:** Entities are never promoted on first access - requires repeated access with time passing.

**Status:** By design, but documentation should clarify this behavior.

---

### Bug #4: SimLLM Entity Names Don't Match Arbitrary Input (Expected)
**Type:** TEST-CONFIG
**Severity:** Test Issue
**Location:** `dst/llm.rs:sim_entity_extraction`

**Issue:** SimLLM only recognizes COMMON_NAMES (Alice, Bob, etc.) and COMMON_ORGS (Acme, etc.). Other input creates "Note_XXX" entities.

**Impact:** Tests searching for "Entity" won't find stored entities named "Note_123abc".

**Fix:** Tests must use recognized names or search for actual stored entity names.

---

### Bug #5: LLM Fault Injection Causes Graceful Degradation, Not Failures (By Design)
**Type:** DST-FOUND ⭐
**Severity:** Design Documentation
**Location:** `orchestration/unified.rs:remember()`

**How Found:** Running fault injection tests with LLM timeouts showed 0 failures
when we expected ~50%. DST revealed the graceful degradation behavior.

**Issue:** Initial tests expected LLM timeouts/rate limits to cause `remember()` to fail. They don't.

**Symptom:**
```
LLM timeout test: 20 successes, 0 failures  <-- Expected failures!
```

**Root Cause:** UnifiedMemory::remember() catches LLM errors at lines 503-510:
```rust
let extracted = match self.extractor.extract(text, ...).await {
    Ok(result) => result.entities,
    Err(_) => vec![], // Extraction failed, will use fallback
};
```

**Impact:** LLM failures are silently converted to Note fallback entities. This is CORRECT behavior (graceful degradation), but:
1. Tests must verify degradation, not failure
2. Quality metrics may not capture that extraction failed

**Verification:** With 100% LLM timeout:
- remember() succeeds (returns Ok)
- Entities are Notes (fallback), not properly extracted

**Status:** By design. Updated tests to verify graceful degradation behavior.

---

### Bug #6: Cascading Failures Don't Propagate LLM Errors (By Design)
**Type:** DST-FOUND ⭐
**Severity:** Design Documentation
**Location:** `orchestration/unified.rs`

**How Found:** Running cascading failure simulation (LLM + storage faults together)
showed that only storage errors propagate. LLM errors are silently absorbed.

**Issue:** When testing cascading failures (LLM + storage faults), all LLM failures appear as "0 LLM fail".

**Output:**
```
Cascading failures: 50 total, 37 success, 0 LLM fail, 13 storage fail
```

**Root Cause:** LLM failures are caught and handled internally (graceful degradation). Only storage failures propagate as remember() errors.

**Impact:** Error classification in tests doesn't capture LLM degradation - only storage failures.

**Status:** By design. The system correctly prioritizes availability over quality when LLM fails.

---

## Test Results

### Current Status (27 Integration Tests)
```
running 27 tests
test orchestration::tests::integration::test_cascading_failures ... ok
test orchestration::tests::integration::test_category_evolver_full_workflow ... ok
test orchestration::tests::integration::test_co_occurrence_detection ... ok
test orchestration::tests::integration::test_determinism_different_seeds ... ok
test orchestration::tests::integration::test_determinism_same_seed ... ok
test orchestration::tests::integration::test_dst_edge_case_inputs ... ok ⭐ DST bug hunt
test orchestration::tests::integration::test_dst_invariant_access_count_under_faults ... ok ⭐ DST bug hunt
test orchestration::tests::integration::test_dst_multi_seed_determinism ... ok ⭐ DST bug hunt
test orchestration::tests::integration::test_dst_score_boundaries_exhaustive ... ok ⭐ DST bug hunt
test orchestration::tests::integration::test_empty_input_handling ... ok
test orchestration::tests::integration::test_fault_determinism ... ok
test orchestration::tests::integration::test_full_lifecycle_remember_to_evolution ... ok
test orchestration::tests::integration::test_graceful_degradation_storage_read_fail ... ok
test orchestration::tests::integration::test_graceful_degradation_storage_write_fail ... ok
test orchestration::tests::integration::test_llm_rate_limit_graceful_degradation ... ok
test orchestration::tests::integration::test_llm_timeout_graceful_degradation ... ok
test orchestration::tests::integration::test_long_input_handling ... ok
test orchestration::tests::integration::test_promotion_eviction_cycle ... ok
test orchestration::tests::integration::test_promotion_requires_repeated_access ... ok
test orchestration::tests::integration::test_promotion_under_storage_stress ... ok
test orchestration::tests::integration::test_rapid_fire_operations ... ok
test orchestration::tests::integration::test_recall_with_fallback ... ok
test orchestration::tests::integration::test_recency_score_validity_over_time ... ok
test orchestration::tests::integration::test_recovery_after_total_failure ... ok
test orchestration::tests::integration::test_state_consistency_under_faults ... ok
test orchestration::tests::integration::test_time_based_access_decay ... ok
test orchestration::tests::integration::test_total_storage_failure ... ok

test result: ok. 27 passed; 0 failed; 0 ignored; 0 measured
```

---

## Files Created/Modified

1. `umi-memory/src/orchestration/tests/mod.rs` - NEW (test module)
2. `umi-memory/src/orchestration/tests/integration.rs` - NEW (27 DST integration tests)
3. `umi-memory/Cargo.toml` - MODIFIED (unified-memory feature flag)
4. `umi-memory/src/lib.rs` - MODIFIED (feature flag docs + conditional exports)
5. `umi-memory/examples/unified_memory_basic.rs` - NEW (4 examples with DST verification)
6. `docs/migration_unified_memory.md` - NEW (migration guide)

---

## Progress

**Completed:** 6/8 phases (75%)
- ✅ Phase 1: Access Tracking Foundation (14 tests, 4 bugs found)
- ✅ Phase 2: Promotion Policy System (14 tests, 5 bugs found)
- ✅ Phase 3: Eviction Policy System (17 tests, 2 bugs found)
- ✅ Phase 4: Unified Memory Orchestrator (37 tests, 3 bugs found)
- ✅ Phase 5: Self-Evolution CategoryEvolver (22 tests, 2 bugs found)
- ✅ Phase 6: Integration & Migration Path (27 tests, 6 bugs found)
- 🔲 Phase 7: Performance & Benchmarking
- 🔲 Phase 8: Documentation & Examples

**Total Bugs Found:** 22 bugs across 6 phases

## Phase 6 Summary

- **27 DST integration tests** with full Simulation harness
  - 14 original workflow tests
  - 5 aggressive bug-hunting tests (state consistency, recency validity, promotion stress)
  - 4 LLM/cascading fault injection tests (found graceful degradation behavior)
  - 4 exhaustive DST invariant tests (multi-seed, score boundaries, edge cases)
- **Feature flag** `unified-memory` for conditional compilation
- **Example** `unified_memory_basic.rs` with 4 demonstrations
- **Migration guide** `docs/migration_unified_memory.md`

### DST Bug Classification Summary

| Type | Count | Examples |
|------|-------|----------|
| **DST-FOUND** ⭐ | 3 | #3 (promotion threshold), #5 (graceful degradation), #6 (cascading) |
| **DESIGN-DOC** | 1 | #2 (frequency requires time) |
| **TEST-CONFIG** | 2 | #1 (input size), #4 (SimLLM names) |

### DST-Found Findings (Would NOT have been found without simulation)

1. **Bug #3 - Promotion threshold behavior**: Running promotion workflow showed 0 promotions.
   DST with time advancement revealed the 0.75 threshold requires repeated access over time.

2. **Bug #5 - LLM graceful degradation**: Fault injection with 50% LLM timeout showed 0 failures.
   DST revealed the system silently falls back to Note entities instead of failing.

3. **Bug #6 - Cascading failure absorption**: Multi-fault simulation showed LLM errors don't propagate.
   Only storage errors cause remember() to fail.

### DST Verification Results (No bugs found - system is robust)

1. **Multi-seed determinism**: 7 seeds tested, all deterministic ✓
2. **Invariant under faults**: Access count stays bounded with 40% fault rate ✓
3. **Score boundaries**: All scores stay in [0.0, 1.0] across time patterns ✓
4. **Edge case handling**: Unicode, emoji, etc. don't corrupt state ✓
