# Umi-Memory Benchmarking & Testing Master Plan

## Status: Active

## Overview

Comprehensive plan to benchmark, test, and validate umi-memory before publication. This covers source reference support for multimedia, performance benchmarks, comparison with Kelpie, deterministic simulation testing with Bloodhound, accuracy benchmarks (LOCOMO), and hands-on validation.

---

## Phase 0: Source Reference Support (Multimedia)

**Goal**: Add source reference field to Entity for multimedia workflows.

### 0.1 Design

Agents processing multimedia (images, audio, video, PDFs) will:
1. Process the multimedia → extract text summary
2. Generate embedding from the content
3. Store in umi-memory with reference to original file

```rust
pub struct Entity {
    // ... existing fields ...

    /// Reference to source content (URL, file path, S3 URI, etc.)
    /// Used when entity was extracted from multimedia content
    pub source_ref: Option<SourceRef>,
}

pub struct SourceRef {
    /// URI to the source (file://, https://, s3://, etc.)
    pub uri: String,
    /// MIME type of the source (image/png, audio/mp3, application/pdf, etc.)
    pub mime_type: Option<String>,
    /// Size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Checksum for integrity (SHA-256)
    pub checksum: Option<String>,
}
```

### 0.2 Implementation Tasks
- [ ] Add `SourceRef` struct to `storage/entity.rs`
- [ ] Add `source_ref: Option<SourceRef>` field to `Entity`
- [ ] Update `EntityBuilder` with `.with_source_ref()`
- [ ] Update Lance schema to include source_ref columns
- [ ] Update SimStorageBackend serialization
- [ ] Add tests for source reference handling

### 0.3 Use Cases

| Scenario | source_ref.uri | source_ref.mime_type |
|----------|----------------|----------------------|
| Image analysis | `file:///photos/meeting.jpg` | `image/jpeg` |
| PDF extraction | `s3://docs/report.pdf` | `application/pdf` |
| Voice memo | `https://storage/memo.mp3` | `audio/mpeg` |
| Video summary | `file:///videos/demo.mp4` | `video/mp4` |
| Web page | `https://example.com/article` | `text/html` |
| No source (text input) | `None` | - |

### Deliverables
- Updated Entity struct with SourceRef
- Updated storage backends
- Tests for multimedia reference workflows

---

## Phase 1: Performance Micro-Benchmarks (Criterion)

**Goal**: Establish baseline performance metrics for umi-memory operations.

### 1.1 Setup Criterion Infrastructure
- [ ] Add criterion to dev-dependencies
- [ ] Create `benches/` directory structure
- [ ] Set up benchmark groups and reporting

### 1.2 Storage Backend Benchmarks
| Benchmark | Description | Scales |
|-----------|-------------|--------|
| `store_entity` | Single entity write | 1, 100, 1K, 10K, 100K entities |
| `get_entity` | Single entity read by ID | Same |
| `delete_entity` | Single entity delete | Same |
| `search_text` | Text search (LIKE query) | Same |
| `count_entities` | Count all entities | Same |
| `list_entities` | Paginated listing | Same |

### 1.3 Memory Tier Benchmarks
| Benchmark | Description |
|-----------|-------------|
| `core_memory_set_block` | Set a block in CoreMemory |
| `core_memory_render` | Render CoreMemory to XML |
| `working_memory_set` | Set KV pair |
| `working_memory_get` | Get KV pair |
| `working_memory_cleanup` | TTL cleanup |

### 1.4 Full Pipeline Benchmarks
| Benchmark | Description |
|-----------|-------------|
| `remember_full` | Extract → Store → Detect evolution |
| `recall_full` | Query expand → Search → Rank |
| `remember_no_extraction` | Store only (skip LLM) |
| `recall_fast_only` | Fast path only (skip LLM) |

### 1.5 Backend Comparison
- [ ] SimStorageBackend vs LanceStorageBackend
- [ ] Memory usage tracking
- [ ] Latency percentiles (p50, p95, p99)

### Deliverables
- Criterion benchmark suite in `benches/`
- HTML reports with graphs
- Baseline numbers documented

---

## Phase 2: Kelpie Memory Comparison

**Goal**: Compare umi-memory against Kelpie's memory implementation.

### 2.1 Understand Kelpie's Implementation
- [ ] Clone and explore https://github.com/nerdsane/kelpie
- [ ] Identify memory-related code paths
- [ ] Document Kelpie's storage backend
- [ ] Note API differences

### 2.2 Create Equivalent Benchmarks
- [ ] Same operations: store, get, search, count
- [ ] Same scales: 100, 1K, 10K entities
- [ ] Same hardware/environment

### 2.3 Feature Comparison Matrix
| Feature | umi-memory | Kelpie |
|---------|------------|--------|
| Core Memory (32KB) | ✅ | ✅ |
| Working Memory (KV+TTL) | ✅ | ✅ |
| Archival Memory | ✅ | ✅ |
| Source References | ✅ | ? |
| Entity Extraction | ✅ | ? |
| Evolution Tracking | ✅ | ? |
| Dual Retrieval | ✅ | ? |
| Vector Search | ✅ (Lance) | ? |
| DST Support | ✅ | ✅ |
| Fault Injection | ✅ | ✅ |

### 2.4 Performance Comparison
- [ ] Run identical workloads on both
- [ ] Compare latency, throughput, memory
- [ ] Document trade-offs

### Deliverables
- Comparison report
- Benchmark numbers for both systems
- Feature gap analysis

---

## Phase 3: Bloodhound Deterministic Testing

**Goal**: Run umi-memory through Bloodhound's deterministic simulation.

### 3.1 Understand Bloodhound
- [ ] Clone and explore https://github.com/nerdsane/bloodhound
- [ ] Understand QEMU-based approach
- [ ] Review fault injection capabilities
- [ ] Study snapshot tree architecture

### 3.2 Containerize umi-memory
- [ ] Create Dockerfile for umi-memory test harness
- [ ] Expose test endpoints for Bloodhound
- [ ] Set up Docker Compose if needed

### 3.3 Define Test Properties
| Property | Type | Description |
|----------|------|-------------|
| Memory consistency | Safety | Stored entities always retrievable |
| No data loss | Safety | Entities survive backend restarts |
| Eventual convergence | Liveness | Concurrent writes converge |
| Evolution correctness | Invariant | Evolution relations are valid |
| Source ref integrity | Safety | Source refs preserved on roundtrip |

### 3.4 Fault Injection Scenarios
- [ ] Storage failures during write
- [ ] Network partitions (if distributed)
- [ ] Process crashes mid-operation
- [ ] Clock skew effects on TTL

### 3.5 Run Bloodhound Campaigns
- [ ] Coverage-guided exploration
- [ ] Property violation detection
- [ ] Time-travel debugging for failures

### Deliverables
- Bloodhound test configuration
- Property specifications
- Bug reports (if any found)
- Coverage report

---

## Phase 4: Internal DST Validation

**Goal**: Thoroughly exercise umi-memory's built-in DST infrastructure.

### 4.1 Review Existing DST Tests
- [ ] Audit `dst/` module tests
- [ ] Identify coverage gaps
- [ ] Check fault injection scenarios

### 4.2 Property-Based Tests
- [ ] Store → Get roundtrip consistency
- [ ] Search returns stored entities
- [ ] Delete removes entities
- [ ] Evolution relations are symmetric
- [ ] TTL expiration is correct
- [ ] Source refs preserved correctly

### 4.3 Fault Injection Matrix
| Component | Fault Type | Expected Behavior |
|-----------|------------|-------------------|
| Storage | Write failure | Error returned, no partial state |
| Storage | Read failure | Error returned, retry possible |
| LLM | Timeout | Graceful degradation |
| LLM | Invalid response | Fallback to defaults |

### 4.4 Multi-Seed Testing
- [ ] Run property tests across 100+ seeds
- [ ] Verify determinism (same seed = same result)
- [ ] Find edge cases

### Deliverables
- Extended property test suite
- Fault injection coverage report
- Determinism verification

---

## Phase 5: LOCOMO Accuracy Benchmark

**Goal**: Measure umi-memory's accuracy on the standard benchmark.

### 5.1 Setup LOCOMO
- [ ] Clone https://github.com/snap-research/locomo
- [ ] Understand data format and evaluation
- [ ] Set up evaluation harness

### 5.2 Integrate with umi-memory
- [ ] Create LOCOMO → umi-memory adapter
- [ ] Map conversations to remember() calls
- [ ] Map questions to recall() calls

### 5.3 Run Evaluation
- [ ] Single-hop QA
- [ ] Multi-hop QA
- [ ] Temporal reasoning
- [ ] Event summarization
- [ ] (Skip multimodal - requires image support beyond references)

### 5.4 Compare Results
| System | F1 Score | Source |
|--------|----------|--------|
| Mem0 | 68.5% | Mem0 paper |
| Mem0g (graph) | ~70% | Mem0 paper |
| Letta | 74.0% | Letta blog |
| umi-memory | ? | This work |
| Human ceiling | 87.9% | LOCOMO paper |

### Deliverables
- LOCOMO integration code
- Accuracy numbers
- Analysis of failure cases

---

## Phase 6: Hands-On Validation

**Goal**: Manual testing to understand the full flow and catch UX issues.

### 6.1 Create Example Applications
- [ ] Simple CLI demo
- [ ] Chat agent with memory
- [ ] Note-taking assistant
- [ ] Image annotation agent (with source refs)

### 6.2 Full Flow Walkthrough
```
1. Create Memory instance with SimLLMProvider
2. remember("My name is Alice and I work at Acme Corp")
   → Observe entity extraction
   → Observe storage
3. remember("Alice got promoted to Senior Engineer")
   → Observe evolution detection (UPDATE)
4. recall("What do I know about Alice?")
   → Observe query expansion
   → Observe search
   → Observe ranking
5. Inspect stored entities
6. Test persistence (Lance backend)
7. Test source reference workflow:
   → Store entity with source_ref to image
   → Retrieve and verify source_ref preserved
```

### 6.3 Edge Case Testing
- [ ] Empty input handling
- [ ] Very long text (100KB)
- [ ] Unicode and special characters
- [ ] Concurrent operations
- [ ] Backend switching
- [ ] Invalid source URIs
- [ ] Missing source files

### 6.4 Documentation Gaps
- [ ] Note any confusing APIs
- [ ] Identify missing features
- [ ] Document workarounds

### Deliverables
- Example applications
- Walkthrough documentation
- UX improvement list

---

## Execution Order

```
Phase 0: Source Reference ──────────────────────────┐
                                                    │
Phase 1: Criterion Benchmarks ──────────────────────┼──→ Performance Report
                                                    │
Phase 2: Kelpie Comparison ─────────────────────────┘

Phase 3: Bloodhound Testing ────────────────────────┐
                                                    │
Phase 4: Internal DST ──────────────────────────────┼──→ Reliability Report
                                                    │
Phase 5: LOCOMO ────────────────────────────────────┼──→ Accuracy Report
                                                    │
Phase 6: Hands-On ──────────────────────────────────┘──→ UX Report
```

---

## Success Criteria

| Phase | Success Metric |
|-------|----------------|
| 0 | Source refs stored and retrieved correctly |
| 1 | Baseline numbers established, no regressions |
| 2 | Competitive with Kelpie on overlapping features |
| 3 | No property violations under fault injection |
| 4 | 100% determinism, all properties hold |
| 5 | F1 > 60% (reasonable for v0.1) |
| 6 | Can build working applications |

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| 0. Source Reference | 0.5 days |
| 1. Criterion | 1-2 days |
| 2. Kelpie | 1-2 days |
| 3. Bloodhound | 2-3 days |
| 4. Internal DST | 1 day |
| 5. LOCOMO | 2-3 days |
| 6. Hands-On | 1 day |

**Total: ~9-13 days**

---

## Progress Log

| Date | Phase | Work Done |
|------|-------|-----------|
| 2026-01-12 | Setup | Created master plan |
| 2026-01-12 | Setup | Added Phase 0 for source references |

---

## Instance Log

(For multi-Claude coordination if needed)

| Instance | Phase | Status |
|----------|-------|--------|
| - | - | - |
