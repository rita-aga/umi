# Phase 7: Performance & Benchmarking (DST-First)

## Objective

Verify performance characteristics under fault conditions using DST-first approach.

## DST-First Approach

**Write tests FIRST, then run simulations to find bugs.**

### What We're Testing

1. **Performance Invariants Under Faults**
   - Throughput degradation is bounded (not catastrophic)
   - Latency doesn't explode under partial failures
   - Memory usage stays bounded under stress

2. **Deterministic Performance**
   - Same seed = same timing sequence
   - SimClock-based latency injection is reproducible
   - Fault timing is deterministic

3. **Graceful Degradation Bounds**
   - With 50% storage faults, throughput drops proportionally (not worse)
   - With LLM faults, fallback path has bounded latency
   - Recovery time after fault resolution is bounded

## DST Test Plan

### Test 1: Throughput Under Fault Injection
```
Given: 50% storage write failures
When: 100 remember operations
Then: Success rate >= 45% (not worse than fault rate)
And: Total time <= 2x baseline (bounded degradation)
```

### Test 2: Latency Bounds Under Stress
```
Given: High concurrency (50 parallel operations)
When: Mixed read/write workload with 30% faults
Then: p99 latency <= 5x median (no extreme outliers)
And: No operations take infinite time (timeout works)
```

### Test 3: Memory Bounds Under Load
```
Given: Continuous remember operations for N iterations
When: Entity limits are reached
Then: Memory usage stabilizes (eviction works)
And: No unbounded growth
```

### Test 4: Recovery After Fault Resolution
```
Given: Storage faults enabled for 50 operations
When: Faults are disabled
Then: Next 10 operations all succeed
And: No lingering effects from fault period
```

### Test 5: Deterministic Timing
```
Given: Same seed and fault configuration
When: Run same workload twice
Then: Operation timing sequence is identical
And: Results match exactly
```

## Bugs to Hunt For

1. **Unbounded Retry Loops** - Do faults cause infinite retries?
2. **Memory Leaks Under Faults** - Do failed operations leak resources?
3. **State Corruption** - Does partial failure leave inconsistent state?
4. **Timing Non-Determinism** - Does SimClock fully control timing?
5. **Graceful Degradation Violations** - Does throughput drop worse than expected?

## Files to Create/Modify

1. `src/orchestration/tests/performance.rs` - DST performance tests
2. `src/orchestration/tests/mod.rs` - Add performance module
3. `.progress/phase7_bugs_found.md` - Bug documentation
4. `docs/adr/NNN-performance-testing.md` - ADR for decisions

## Status

- [ ] Write DST performance tests
- [ ] Run simulations with fault injection
- [ ] Document bugs found through DST
- [ ] Write ADR
- [ ] Update Criterion benchmarks if needed
