# ADR 019: DST Performance Testing

## Status

Accepted

## Context

Umi follows DST-first (Deterministic Simulation Testing) principles from TigerBeetle. Phase 7 extends this to performance testing, where we need to verify that the system maintains bounded performance characteristics under fault conditions.

Traditional performance testing uses random timing and non-deterministic workloads. This makes bugs hard to reproduce. We need deterministic performance tests that:

1. Verify throughput degradation is bounded
2. Verify latency bounds are maintained
3. Verify memory usage stays bounded
4. Ensure behavior is deterministic across runs

## Decision

We will implement DST performance tests that:

### 1. Use Compound Fault Probability Model

**Finding:** Operation success rate ≠ (1 - fault_rate) when operations involve multiple storage calls.

```
P(operation_success) = (1 - fault_rate)^N
where N = number of sub-operations
```

For `remember()` with 2 entities and 50% fault rate:
- P(success) = 0.5² = 0.25 = 25%

Tests must use this formula when setting expected bounds.

### 2. Track All Access Sources

**Finding:** Both `remember()` and `recall()` track accesses in CategoryEvolver.

Total accesses = remember_accesses + recall_accesses

Performance tests must account for ALL access sources when verifying bounds.

### 3. Test Invariants, Not Absolute Performance

Instead of "operation takes X ms", we test:
- "p99 latency ≤ 10x median" (no extreme outliers)
- "success rate ≥ expected compound probability"
- "memory usage ≤ configured limits"

This allows tests to pass across different hardware.

### 4. Verify Recovery After Faults

Systems must recover immediately when faults stop:
- No lingering corrupted state
- No backlog of failed operations
- 100% success rate after fault resolution

## Implementation

### Performance Test Suite

8 DST performance tests in `orchestration/tests/performance.rs`:

| Test | Invariant Verified |
|------|-------------------|
| `test_throughput_under_storage_faults` | Success rate matches compound probability |
| `test_latency_bounds_under_faults` | p99 ≤ 10x median |
| `test_memory_bounds_under_load` | Core count ≤ configured limit |
| `test_recovery_after_fault_resolution` | 100% success post-fault |
| `test_deterministic_timing` | Same seed = same results |
| `test_llm_fallback_latency_bounds` | Fallback path bounded |
| `test_concurrent_access_under_faults` | Access count bounded |
| `test_eviction_under_memory_pressure` | Eviction triggers correctly |

### Test Structure

```rust
#[tokio::test]
async fn test_performance_invariant() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = create_memory_with_faults(42, env.clock.clone(), 0.5);

        // Run workload
        for i in 0..N {
            let _ = memory.remember(...).await;
            let _ = env.clock.advance_ms(10);
        }

        // Verify invariants
        assert!(success_rate >= expected_compound_probability);
        assert!(latency_p99 <= latency_median * 10);
        assert!(memory_usage <= configured_limit);

        Ok(())
    }).await.unwrap();
}
```

## Consequences

### Positive

1. **Reproducible Performance Issues**: Deterministic tests catch performance regressions reliably
2. **Compound Probability Documented**: Developers understand fault rate ≠ operation rate
3. **Invariant-Based Testing**: Tests pass across different hardware
4. **Full Access Tracking Verified**: Confirms both read and write paths tracked

### Negative

1. **Simulation Overhead**: DST adds some overhead vs. raw benchmarks
2. **Not Real-World Timing**: SimClock doesn't capture actual latency characteristics
3. **Compound Probability Surprising**: Users may expect 50% faults = 50% failures

### Mitigation

- Use Criterion benchmarks for raw performance measurements
- DST tests verify correctness; Criterion measures speed
- Document compound probability in user-facing documentation

## Related Decisions

- ADR 004: Deterministic Simulation Testing
- ADR 012: SimLLM Implementation
- ADR 018: Lance Storage Backend

## References

- TigerBeetle TigerStyle: https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md
- FoundationDB Testing: https://www.foundationdb.org/files/fdb-paper.pdf
