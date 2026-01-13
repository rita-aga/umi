# UMI-memU Feature Parity Implementation

**Task**: Make UMI feature-complete compared to memU's automatic memory orchestration
**Approach**: DST-first, TigerStyle, incremental with integration tests after each phase
**Date**: 2026-01-13
**Priority**: HIGH - This transforms UMI from a library into a complete framework

## Overview

Transform UMI from a component library into a complete memory orchestration framework with automatic promotion, eviction, and self-evolution capabilities matching memU.

**Current State:**
- ✅ UMI has all components (CoreMemory, WorkingMemory, ArchivalMemory)
- ✅ UMI has mapping layer (EntityType ↔ KelpieBlockType)
- ❌ Memory orchestrator only uses ArchivalMemory
- ❌ No automatic promotion logic
- ❌ No access tracking
- ❌ No importance-based eviction
- ❌ No self-evolution

**Target State:**
- ✅ Unified orchestrator manages all three tiers
- ✅ Automatic promotion (Archival → Core)
- ✅ Access-pattern tracking
- ✅ Importance-based eviction from Core
- ✅ Self-evolving category structure
- ✅ Graceful fallback (Core → Archival)

## memU Architecture to Replicate

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: CATEGORY (What enters LLM context)               │
│  - Aggregated markdown summaries                            │
│  - Automatically promoted based on importance/access        │
│  - Forgotten (demoted) when not accessed                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Promotion/Demotion (automatic)
                     │
┌────────────────────┴────────────────────────────────────────┐
│  Layer 2: ITEMS (Discrete extracted units)                 │
│  - Individual entities                                      │
│  - Tracked for access patterns                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Extraction
                     │
┌────────────────────┴────────────────────────────────────────┐
│  Layer 1: RESOURCES (Raw data, never deleted)              │
│  - Original inputs                                          │
│  - Permanent storage                                        │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Access Tracking Foundation ✓ (Complete when checked)
**Goal**: Track when entities are accessed for importance scoring

**Components to Build:**
1. `AccessTracker` struct
2. `AccessPattern` metadata
3. Integration with existing Memory API

**Tasks:**
- [ ] Design AccessTracker API
  - Track entity access timestamps
  - Track entity access frequency
  - Calculate recency score (exponential decay)
  - Calculate importance score (frequency + recency + base importance)

- [ ] Create `umi-memory/src/orchestration/access_tracker.rs`
  - `AccessTracker::new()` - Initialize tracker
  - `record_access(entity_id, timestamp_ms)` - Record single access
  - `record_batch_access(entity_ids, timestamp_ms)` - Batch recording
  - `get_access_pattern(entity_id) -> AccessPattern` - Get access stats
  - `get_importance_score(entity_id) -> f64` - Calculate final score (0.0-1.0)
  - `prune_old_records(before_ms)` - Cleanup old access records

- [ ] Create `AccessPattern` struct
  - `first_access_ms: u64` - When first accessed
  - `last_access_ms: u64` - Most recent access
  - `access_count: u64` - Total number of accesses
  - `recency_score: f64` - Exponential decay score
  - `frequency_score: f64` - Normalized access frequency
  - `combined_importance: f64` - Final importance (0.0-1.0)

- [ ] Add TigerStyle constants to `constants.rs`
  ```rust
  /// Access tracking
  pub const ACCESS_TRACKER_DECAY_HALFLIFE_MS: u64 = 7 * 24 * 60 * 60 * 1000;  // 7 days
  pub const ACCESS_TRACKER_MIN_IMPORTANCE: f64 = 0.0;
  pub const ACCESS_TRACKER_MAX_IMPORTANCE: f64 = 1.0;
  pub const ACCESS_TRACKER_PRUNE_THRESHOLD_MS: u64 = 90 * 24 * 60 * 60 * 1000;  // 90 days
  pub const ACCESS_TRACKER_BATCH_SIZE_MAX: usize = 1000;
  ```

- [ ] Write DST tests (simulation-first)
  - Test access recording with SimClock
  - Test importance calculation with various patterns
  - Test recency decay over time
  - Test frequency scoring
  - Test batch operations
  - Test pruning old records
  - Test edge cases (no accesses, single access, etc.)

- [ ] Integration test: Add to existing Memory API
  - Record access during `recall()`
  - Record access during `remember()` (for evolution tracking)
  - Verify access patterns are tracked correctly

**Acceptance Criteria:**
- [ ] All DST tests pass (10+ tests)
- [ ] AccessTracker integrated into Memory without breaking existing tests
- [ ] Importance scores calculated correctly
- [ ] No performance regression (access tracking < 1ms per operation)

**Files to Create/Modify:**
- NEW: `umi-memory/src/orchestration/access_tracker.rs`
- NEW: `umi-memory/src/orchestration/mod.rs`
- MODIFY: `umi-memory/src/lib.rs` (export orchestration module)
- MODIFY: `umi-memory/src/constants.rs` (add access tracking constants)
- MODIFY: `umi-memory/src/umi/mod.rs` (integrate AccessTracker into Memory)

---

### Phase 2: Promotion Policy System ✓
**Goal**: Define rules for when entities are promoted to CoreMemory

**Components to Build:**
1. `PromotionPolicy` trait
2. `ImportanceBasedPolicy` implementation
3. `HybridPolicy` (importance + recency + entity type)

**Tasks:**
- [ ] Design PromotionPolicy trait
  ```rust
  pub trait PromotionPolicy {
      fn should_promote(&self, entity: &Entity, access_pattern: &AccessPattern) -> bool;
      fn calculate_priority(&self, entity: &Entity, access_pattern: &AccessPattern) -> f64;
  }
  ```

- [ ] Create `umi-memory/src/orchestration/promotion.rs`
  - `PromotionPolicy` trait
  - `ImportanceBasedPolicy` - Simple importance threshold
  - `HybridPolicy` - Combines importance, recency, access frequency, entity type
  - `PromotionConfig` - Configurable thresholds and weights

- [ ] Implement ImportanceBasedPolicy
  - Promote if `entity.importance >= threshold`
  - Simple and predictable

- [ ] Implement HybridPolicy
  - Score = `w1 * importance + w2 * recency + w3 * frequency + w4 * type_priority`
  - Entity type priorities (Self_ > Project > Task > Person > Topic > Note)
  - Configurable weights
  - Promote if score >= threshold

- [ ] Add TigerStyle constants
  ```rust
  /// Promotion thresholds
  pub const PROMOTION_IMPORTANCE_THRESHOLD_DEFAULT: f64 = 0.7;
  pub const PROMOTION_SCORE_THRESHOLD_DEFAULT: f64 = 0.75;
  pub const PROMOTION_CORE_MEMORY_ENTITIES_MAX: usize = 50;  // Max entities in core

  /// Hybrid policy weights
  pub const PROMOTION_WEIGHT_IMPORTANCE: f64 = 0.4;
  pub const PROMOTION_WEIGHT_RECENCY: f64 = 0.3;
  pub const PROMOTION_WEIGHT_FREQUENCY: f64 = 0.2;
  pub const PROMOTION_WEIGHT_TYPE_PRIORITY: f64 = 0.1;

  /// Entity type priorities (higher = more likely to promote)
  pub const ENTITY_TYPE_PRIORITY_SELF: f64 = 1.0;
  pub const ENTITY_TYPE_PRIORITY_PROJECT: f64 = 0.9;
  pub const ENTITY_TYPE_PRIORITY_TASK: f64 = 0.85;
  pub const ENTITY_TYPE_PRIORITY_PERSON: f64 = 0.7;
  pub const ENTITY_TYPE_PRIORITY_TOPIC: f64 = 0.6;
  pub const ENTITY_TYPE_PRIORITY_NOTE: f64 = 0.4;
  ```

- [ ] Write DST tests
  - Test ImportanceBasedPolicy with various thresholds
  - Test HybridPolicy with different entity types
  - Test HybridPolicy with different access patterns
  - Test priority calculation
  - Test edge cases (no accesses, brand new entities)
  - Test determinism (same inputs → same promotion decisions)

**Acceptance Criteria:**
- [ ] All DST tests pass (8+ tests)
- [ ] Policies are configurable
- [ ] Promotion decisions are deterministic
- [ ] Entity type priorities make sense

**Files to Create/Modify:**
- NEW: `umi-memory/src/orchestration/promotion.rs`
- MODIFY: `umi-memory/src/orchestration/mod.rs`
- MODIFY: `umi-memory/src/constants.rs`

---

### Phase 3: Eviction Policy System ✓
**Goal**: Define rules for when entities are demoted from CoreMemory

**Components to Build:**
1. `EvictionPolicy` trait
2. `LRUEvictionPolicy` - Least Recently Used
3. `ImportanceEvictionPolicy` - Evict lowest importance
4. `HybridEvictionPolicy` - Combines LRU + importance

**Tasks:**
- [ ] Design EvictionPolicy trait
  ```rust
  pub trait EvictionPolicy {
      fn select_eviction_candidates(
          &self,
          core_entities: &[Entity],
          access_tracker: &AccessTracker,
          count: usize,
      ) -> Vec<EntityId>;
  }
  ```

- [ ] Create `umi-memory/src/orchestration/eviction.rs`
  - `EvictionPolicy` trait
  - `LRUEvictionPolicy` - Evict least recently used
  - `ImportanceEvictionPolicy` - Evict lowest importance
  - `HybridEvictionPolicy` - Score-based (importance + recency)

- [ ] Implement LRUEvictionPolicy
  - Sort by `last_access_ms`
  - Evict oldest N entities

- [ ] Implement ImportanceEvictionPolicy
  - Sort by current importance score
  - Evict lowest N entities

- [ ] Implement HybridEvictionPolicy
  - Calculate eviction score = `importance * recency_multiplier`
  - Evict lowest-scoring N entities
  - Never evict Self_ entities (protect user context)

- [ ] Add TigerStyle constants
  ```rust
  /// Eviction thresholds
  pub const EVICTION_CORE_MEMORY_SIZE_BYTES_MAX: usize = 32 * 1024;  // 32KB
  pub const EVICTION_CORE_MEMORY_ENTITIES_MAX: usize = 50;
  pub const EVICTION_BATCH_SIZE: usize = 10;  // Evict in batches
  pub const EVICTION_IMPORTANCE_THRESHOLD_MIN: f64 = 0.5;  // Never evict above this
  pub const EVICTION_LAST_ACCESS_THRESHOLD_MS: u64 = 30 * 24 * 60 * 60 * 1000;  // 30 days

  /// Protected entity types (never evict)
  pub const EVICTION_PROTECTED_TYPES: &[EntityType] = &[EntityType::Self_];
  ```

- [ ] Write DST tests
  - Test LRU eviction with various access patterns
  - Test Importance eviction with various scores
  - Test Hybrid eviction
  - Test protected entity types (Self_ never evicted)
  - Test edge cases (empty core, all entities protected)
  - Test determinism

**Acceptance Criteria:**
- [ ] All DST tests pass (8+ tests)
- [ ] Self_ entities are never evicted
- [ ] Eviction is deterministic
- [ ] Eviction respects size limits

**Files to Create/Modify:**
- NEW: `umi-memory/src/orchestration/eviction.rs`
- MODIFY: `umi-memory/src/orchestration/mod.rs`
- MODIFY: `umi-memory/src/constants.rs`

---

### Phase 4: Unified Memory Orchestrator ✓
**Goal**: Create new orchestrator that manages all three tiers with automatic promotion/eviction

**Components to Build:**
1. `UnifiedMemory` struct
2. Integration with existing Memory API
3. Automatic promotion/eviction during remember/recall

**Tasks:**
- [ ] Design UnifiedMemory API
  ```rust
  pub struct UnifiedMemory<L, E, S, V> {
      // Existing components
      storage: S,                    // Archival (Layer 1)
      extractor: EntityExtractor<L>,
      retriever: DualRetriever<L, E, V, S>,
      evolution: EvolutionTracker<L, S>,
      embedder: E,
      vector: V,

      // New components
      core: CoreMemory,              // Layer 3 (what enters LLM context)
      working: WorkingMemory,        // Session state
      access_tracker: AccessTracker,
      promotion_policy: Box<dyn PromotionPolicy>,
      eviction_policy: Box<dyn EvictionPolicy>,

      config: UnifiedMemoryConfig,
  }
  ```

- [ ] Create `umi-memory/src/orchestration/unified.rs`
  - `UnifiedMemory::new()` - Initialize with all components
  - `UnifiedMemory::builder()` - Builder pattern for configuration
  - `remember()` - Store + extract + promote
  - `recall()` - Try core → fallback to archival
  - `promote_to_core()` - Execute promotion logic
  - `evict_from_core()` - Execute eviction logic
  - `get_core_memory()` - Access CoreMemory for LLM context
  - `sync()` - Manually trigger promotion/eviction

- [ ] Implement remember() flow
  ```rust
  async fn remember(&mut self, text: &str) -> Result<RememberResult> {
      // 1. Extract entities (existing logic)
      let entities = self.extractor.extract(text).await?;

      // 2. Store in archival (existing logic)
      for entity in &entities {
          self.storage.store_entity(entity).await?;
      }

      // 3. NEW: Automatic promotion
      self.promote_to_core().await?;

      // 4. NEW: Check if eviction needed
      if self.core.size_bytes() > EVICTION_CORE_MEMORY_SIZE_BYTES_MAX {
          self.evict_from_core().await?;
      }

      Ok(RememberResult::new(entities))
  }
  ```

- [ ] Implement recall() flow with fallback
  ```rust
  async fn recall(&self, query: &str) -> Result<Vec<Entity>> {
      // 1. NEW: Try core memory first (fast, in-context)
      let core_results = self.search_core_memory(query)?;
      if !core_results.is_empty() {
          // Record access
          self.access_tracker.record_batch_access(&core_results);
          return Ok(core_results);
      }

      // 2. Fallback to archival (existing logic)
      let archival_results = self.retriever.search(query).await?;

      // Record access
      self.access_tracker.record_batch_access(&archival_results);

      Ok(archival_results)
  }
  ```

- [ ] Implement promote_to_core()
  ```rust
  async fn promote_to_core(&mut self) -> Result<()> {
      // Get candidates from archival
      let candidates = self.get_promotion_candidates().await?;

      for entity in candidates {
          let access_pattern = self.access_tracker.get_access_pattern(&entity.id);

          // Check promotion policy
          if self.promotion_policy.should_promote(&entity, &access_pattern) {
              // Map entity type to Kelpie block type
              let block_type = KelpieBlockType::from(entity.entity_type);

              // Append to appropriate core memory block
              self.core.append_to_block(block_type, &entity.content)?;
              self.core.set_block_importance(block_type, entity.importance)?;
          }
      }

      Ok(())
  }
  ```

- [ ] Implement evict_from_core()
  ```rust
  async fn evict_from_core(&mut self) -> Result<()> {
      let core_entities = self.core.all_entities();

      // Select eviction candidates
      let to_evict = self.eviction_policy.select_eviction_candidates(
          &core_entities,
          &self.access_tracker,
          EVICTION_BATCH_SIZE,
      );

      // Remove from core (NOT from archival)
      for entity_id in to_evict {
          self.core.remove_entity(&entity_id)?;
      }

      Ok(())
  }
  ```

- [ ] Add UnifiedMemoryConfig
  ```rust
  pub struct UnifiedMemoryConfig {
      pub auto_promote: bool,              // Enable automatic promotion
      pub auto_evict: bool,                // Enable automatic eviction
      pub promotion_interval_ms: u64,      // How often to check for promotion
      pub eviction_interval_ms: u64,       // How often to check for eviction
      pub core_size_limit_bytes: usize,    // CoreMemory size limit
      pub core_entity_limit: usize,        // Max entities in core
  }
  ```

- [ ] Write DST tests
  - Test remember() with automatic promotion
  - Test recall() with core → archival fallback
  - Test promotion logic with various policies
  - Test eviction logic when core is full
  - Test access tracking integration
  - Test graceful degradation (LLM failures, storage failures)
  - Test determinism with SimConfig
  - Test that archival entities are never deleted (only demoted from core)

**Acceptance Criteria:**
- [ ] All DST tests pass (15+ tests)
- [ ] UnifiedMemory can be used as drop-in replacement for Memory
- [ ] Automatic promotion works correctly
- [ ] Automatic eviction maintains size limits
- [ ] Core → archival fallback works
- [ ] No breaking changes to existing Memory API
- [ ] All 501+ existing tests still pass

**Files to Create/Modify:**
- NEW: `umi-memory/src/orchestration/unified.rs`
- MODIFY: `umi-memory/src/orchestration/mod.rs`
- MODIFY: `umi-memory/src/lib.rs` (export UnifiedMemory)
- MODIFY: `umi-memory/src/umi/mod.rs` (optionally migrate to UnifiedMemory)

---

### Phase 5: Self-Evolution (Category Adaptation) ✓
**Goal**: Allow CoreMemory structure to adapt based on usage patterns

**Components to Build:**
1. `CategoryEvolver` - Detects patterns and suggests structure changes
2. Integration with UnifiedMemory

**Tasks:**
- [ ] Design self-evolution strategy
  - Track which entity types are frequently accessed together
  - Suggest new block types based on co-occurrence patterns
  - Suggest merging rarely-used block types
  - Suggest splitting frequently-used blocks

- [ ] Create `umi-memory/src/orchestration/evolution.rs` (different from EvolutionTracker)
  - `CategoryEvolver::new()` - Initialize evolver
  - `track_access_pattern(entity_type, block_type)` - Track usage
  - `analyze_patterns() -> EvolutionSuggestions` - Analyze and suggest changes
  - `apply_suggestion(suggestion) -> Result<()>` - Apply structure change

- [ ] Implement pattern detection
  - Co-occurrence matrix (which entity types accessed together)
  - Block type usage statistics
  - Entity type distribution per block

- [ ] Implement EvolutionSuggestions
  ```rust
  pub enum EvolutionSuggestion {
      CreateBlock {
          name: String,
          entity_types: Vec<EntityType>,
          reason: String,
      },
      MergeBlocks {
          block1: KelpieBlockType,
          block2: KelpieBlockType,
          into: String,
          reason: String,
      },
      SplitBlock {
          block: KelpieBlockType,
          into: Vec<String>,
          reason: String,
      },
  }
  ```

- [ ] Add TigerStyle constants
  ```rust
  /// Self-evolution thresholds
  pub const EVOLUTION_MIN_SAMPLES: usize = 100;  // Min accesses before suggesting
  pub const EVOLUTION_CO_OCCURRENCE_THRESHOLD: f64 = 0.7;  // Co-occurrence threshold
  pub const EVOLUTION_BLOCK_USAGE_THRESHOLD_MIN: f64 = 0.1;  // Merge if below
  pub const EVOLUTION_ANALYSIS_INTERVAL_MS: u64 = 7 * 24 * 60 * 60 * 1000;  // 7 days
  ```

- [ ] Write DST tests
  - Test pattern detection with various access patterns
  - Test suggestion generation
  - Test co-occurrence calculation
  - Test block usage statistics
  - Test edge cases (no patterns, all patterns, etc.)

- [ ] Integration with UnifiedMemory
  - Track patterns during promote_to_core()
  - Periodic analysis (every N days)
  - Optional: expose suggestions via API for user approval

**Acceptance Criteria:**
- [ ] All DST tests pass (10+ tests)
- [ ] Pattern detection works correctly
- [ ] Suggestions are sensible and actionable
- [ ] No automatic structure changes without user approval
- [ ] Integration doesn't impact performance

**Files to Create/Modify:**
- NEW: `umi-memory/src/orchestration/category_evolution.rs`
- MODIFY: `umi-memory/src/orchestration/unified.rs`
- MODIFY: `umi-memory/src/orchestration/mod.rs`
- MODIFY: `umi-memory/src/constants.rs`

---

### Phase 6: Integration & Migration Path ✓
**Goal**: Make UnifiedMemory available without breaking existing users

**Tasks:**
- [ ] Create migration guide
  - Document differences between Memory and UnifiedMemory
  - Show how to migrate existing code
  - Explain new configuration options

- [ ] Add feature flag
  ```toml
  [features]
  default = ["unified-memory"]
  unified-memory = []
  ```

- [ ] Update Memory to optionally use UnifiedMemory
  ```rust
  #[cfg(feature = "unified-memory")]
  pub type Memory<L, E, S, V> = UnifiedMemory<L, E, S, V>;

  #[cfg(not(feature = "unified-memory"))]
  pub struct Memory<L, E, S, V> {
      // Existing implementation
  }
  ```

- [ ] Add examples
  - `examples/unified_memory_basic.rs` - Simple usage
  - `examples/unified_memory_custom_policies.rs` - Custom promotion/eviction
  - `examples/unified_memory_evolution.rs` - Self-evolution demo

- [ ] Update documentation
  - Update README.md
  - Update VISION.md
  - Add new ADR for UnifiedMemory design
  - Update CLAUDE.md with new components

- [ ] Create comprehensive integration test
  - Test full workflow: remember → promote → recall → evict
  - Test with real data (not just simulation)
  - Test with multiple entity types
  - Test with various access patterns
  - Test graceful degradation

**Acceptance Criteria:**
- [ ] Migration guide is clear
- [ ] Examples run successfully
- [ ] Documentation is updated
- [ ] All existing tests pass with and without unified-memory feature
- [ ] New integration test passes

**Files to Create/Modify:**
- NEW: `docs/migration_unified_memory.md`
- NEW: `docs/adr/019-unified-memory.md`
- NEW: `examples/unified_memory_basic.rs`
- NEW: `examples/unified_memory_custom_policies.rs`
- NEW: `examples/unified_memory_evolution.rs`
- MODIFY: `README.md`
- MODIFY: `VISION.md`
- MODIFY: `CLAUDE.md`
- MODIFY: `umi-memory/Cargo.toml`

---

### Phase 7: Performance & Optimization ✓
**Goal**: Ensure UnifiedMemory meets performance targets

**Tasks:**
- [ ] Benchmark UnifiedMemory operations
  - Benchmark remember() with auto-promotion
  - Benchmark recall() with fallback
  - Benchmark promotion logic
  - Benchmark eviction logic
  - Compare to baseline Memory performance

- [ ] Optimize hot paths
  - Cache importance scores
  - Batch promotion/eviction
  - Optimize access tracking lookups
  - Minimize CoreMemory serialization overhead

- [ ] Add performance metrics
  - Track promotion count
  - Track eviction count
  - Track access pattern updates
  - Track core memory size

- [ ] Add TigerStyle assertions for performance
  ```rust
  assert!(promotion_latency_ms < PROMOTION_LATENCY_MS_MAX);
  assert!(eviction_latency_ms < EVICTION_LATENCY_MS_MAX);
  ```

- [ ] Profile with real workloads
  - 1000 entities, 100 promotions
  - 10000 entities, 1000 promotions
  - Stress test: 100000 entities

**Performance Targets:**
| Operation | Target | Notes |
|-----------|--------|-------|
| Promotion check | <5ms | Per entity |
| Eviction check | <10ms | Per batch |
| Access tracking | <1ms | Per record |
| Core memory render | <5ms | XML generation |
| remember() overhead | <10ms | vs baseline Memory |
| recall() overhead | <5ms | Core search first |

**Acceptance Criteria:**
- [ ] All performance targets met
- [ ] No regression vs baseline Memory
- [ ] Profiling shows no bottlenecks
- [ ] Memory usage within bounds

**Files to Create/Modify:**
- NEW: `umi-memory/benches/unified_memory.rs`
- MODIFY: `umi-memory/src/orchestration/unified.rs` (optimizations)

---

### Phase 8: Testing & Validation ✓
**Goal**: Comprehensive testing across all components

**Tasks:**
- [ ] Write property-based tests (proptest)
  - Promotion/eviction are inverse operations (promote then evict = noop)
  - Access tracking is monotonic (scores never decrease with more accesses)
  - Core memory size never exceeds limits
  - Archival entities are never deleted

- [ ] Write fault injection tests (DST)
  - Storage failures during promotion
  - LLM failures during extraction
  - Vector backend failures
  - CoreMemory full during promotion

- [ ] Write integration tests
  - Real-world usage patterns
  - Multiple sessions
  - Long-running scenarios
  - Evolution detection with promotion

- [ ] Write stress tests
  - 10000+ entities
  - 1000+ promotions
  - Rapid access pattern changes
  - Core memory churn

- [ ] Manual validation
  - Use UnifiedMemory in a real agent
  - Verify promotion makes sense
  - Verify eviction preserves important memories
  - Verify fallback works

**Acceptance Criteria:**
- [ ] All property tests pass
- [ ] All fault injection tests pass
- [ ] All integration tests pass
- [ ] Stress tests complete without crashes
- [ ] Manual validation confirms expected behavior
- [ ] Total test count: 600+ (400 existing + 200 new)

**Files to Create:**
- NEW: `umi-memory/src/orchestration/tests/proptest.rs`
- NEW: `umi-memory/src/orchestration/tests/fault_injection.rs`
- NEW: `umi-memory/src/orchestration/tests/integration.rs`
- NEW: `umi-memory/src/orchestration/tests/stress.rs`

---

## Success Criteria

### Functional Requirements
- [ ] UnifiedMemory manages all three tiers (Core, Working, Archival)
- [ ] Automatic promotion from archival to core based on importance/access
- [ ] Automatic eviction from core when size limits reached
- [ ] Access pattern tracking for all entities
- [ ] Graceful fallback (core → archival)
- [ ] Self-evolution suggestions based on usage patterns
- [ ] Compatible with existing Memory API (migration path)

### Non-Functional Requirements
- [ ] All 600+ tests pass (400 existing + 200 new)
- [ ] No performance regression vs baseline Memory
- [ ] Promotion overhead <10ms
- [ ] Eviction overhead <10ms
- [ ] Core memory size never exceeds 32KB
- [ ] Deterministic behavior with SimConfig
- [ ] Graceful degradation (LLM/storage failures)

### Documentation Requirements
- [ ] Migration guide complete
- [ ] New ADR for UnifiedMemory
- [ ] Examples for all major use cases
- [ ] Updated VISION.md and README.md
- [ ] Inline documentation (rustdoc)

### TigerStyle Compliance
- [ ] All constants have units in name
- [ ] 2+ assertions per non-trivial function
- [ ] No silent truncation
- [ ] Explicit error handling (no unwrap/expect)
- [ ] Debug assertions for expensive checks

---

## Testing Strategy

### DST Testing (Deterministic Simulation)
Every component MUST have DST coverage:
- [ ] AccessTracker with SimClock
- [ ] PromotionPolicy with deterministic scoring
- [ ] EvictionPolicy with deterministic selection
- [ ] UnifiedMemory with SimLLM, SimStorage, SimVector
- [ ] CategoryEvolver with pattern detection

### Fault Injection
Test graceful degradation:
- [ ] Storage write failures during promotion
- [ ] Storage read failures during recall
- [ ] LLM failures during extraction
- [ ] CoreMemory full (cannot promote)
- [ ] Network timeouts

### Property-Based Testing
Invariants to maintain:
- [ ] Core memory size ≤ EVICTION_CORE_MEMORY_SIZE_BYTES_MAX
- [ ] Core entity count ≤ EVICTION_CORE_MEMORY_ENTITIES_MAX
- [ ] Self_ entities never evicted
- [ ] Archival entities never deleted (only demoted from core)
- [ ] Access scores are monotonically non-decreasing with more accesses

### Integration Testing
Real-world scenarios:
- [ ] Multi-session agent with context preservation
- [ ] Evolving knowledge base (updates, contradictions)
- [ ] High-churn scenario (rapid promotion/eviction)
- [ ] Low-activity scenario (stable core memory)

---

## Architecture Decisions

### Key Design Choices

**1. Three-Tier Architecture**
- Layer 1 (Archival): Unlimited storage, never delete
- Layer 2 (Items): Extracted entities in archival
- Layer 3 (Core): What enters LLM context, actively managed

**2. Policy-Based Promotion/Eviction**
- Pluggable policies via traits
- Multiple implementations (simple, hybrid, custom)
- Configurable thresholds and weights

**3. Access-Pattern Driven**
- Track every entity access
- Calculate importance from frequency + recency + base importance
- Use for promotion and eviction decisions

**4. Self-Evolution**
- Optional feature
- Suggest structure changes, don't auto-apply
- User approval required

**5. Backward Compatibility**
- Feature flag for unified-memory
- Existing Memory API unchanged
- Migration path documented

---

## Risk Assessment

### High Risk
1. **Performance Regression** - Automatic promotion/eviction adds overhead
   - Mitigation: Benchmark early, optimize hot paths, batch operations

2. **Breaking Changes** - UnifiedMemory might break existing users
   - Mitigation: Feature flag, migration guide, comprehensive tests

3. **Complexity** - Adding policies and self-evolution increases complexity
   - Mitigation: Clear abstractions, simple defaults, extensive documentation

### Medium Risk
1. **Policy Tuning** - Default policies might not work for all use cases
   - Mitigation: Make policies configurable, provide multiple implementations

2. **CoreMemory Size Management** - Ensuring 32KB limit while maintaining quality
   - Mitigation: Aggressive eviction, entity summarization, block deduplication

### Low Risk
1. **Self-Evolution Correctness** - Suggestions might not be helpful
   - Mitigation: Require user approval, extensive testing, clear rationale

---

## Comparison to memU

After implementation, UMI will match memU's capabilities:

| Feature | memU | UMI (Current) | UMI (After) |
|---------|------|---------------|-------------|
| **Three-tier architecture** | ✅ | ⚠️ (components exist) | ✅ |
| **Automatic promotion** | ✅ | ❌ | ✅ |
| **Access-based eviction** | ✅ | ❌ | ✅ |
| **Self-evolution** | ✅ | ❌ | ✅ |
| **Graceful fallback** | ✅ | ⚠️ (LLM only) | ✅ |
| **Never delete raw data** | ✅ | ✅ | ✅ |
| **DST coverage** | ❌ | ✅ | ✅ |
| **TigerStyle safety** | ❌ | ✅ | ✅ |
| **Policy configurability** | ❌ | N/A | ✅ |
| **Rust + Python** | ❌ (Python only) | ✅ (Rust core) | ✅ |

**UMI's advantages over memU:**
- Rust core (faster, safer)
- DST coverage (deterministic testing)
- TigerStyle engineering (explicit limits, assertions)
- Pluggable policies (not hardcoded)
- Full control over promotion/eviction logic

---

## Timeline Estimate

| Phase | Estimated Time | Complexity |
|-------|----------------|------------|
| Phase 1: Access Tracking | 8-12 hours | Medium |
| Phase 2: Promotion Policy | 6-8 hours | Medium |
| Phase 3: Eviction Policy | 6-8 hours | Medium |
| Phase 4: Unified Orchestrator | 12-16 hours | High |
| Phase 5: Self-Evolution | 10-14 hours | High |
| Phase 6: Integration & Migration | 6-8 hours | Low |
| Phase 7: Performance | 8-10 hours | Medium |
| Phase 8: Testing & Validation | 10-12 hours | Medium |
| **Total** | **66-88 hours** | **~2-3 weeks** |

---

## Files to Create

### New Modules
```
umi-memory/src/orchestration/
├── mod.rs                      # Module exports
├── access_tracker.rs           # Phase 1
├── promotion.rs                # Phase 2
├── eviction.rs                 # Phase 3
├── unified.rs                  # Phase 4
├── category_evolution.rs       # Phase 5
└── tests/
    ├── proptest.rs             # Phase 8
    ├── fault_injection.rs      # Phase 8
    ├── integration.rs          # Phase 8
    └── stress.rs               # Phase 8
```

### Documentation
```
docs/
├── migration_unified_memory.md  # Phase 6
└── adr/
    └── 019-unified-memory.md    # Phase 6

examples/
├── unified_memory_basic.rs              # Phase 6
├── unified_memory_custom_policies.rs    # Phase 6
└── unified_memory_evolution.rs          # Phase 6

umi-memory/benches/
└── unified_memory.rs                    # Phase 7
```

---

## Constants to Define

All constants follow TigerStyle naming (units in name):

```rust
// Access tracking
pub const ACCESS_TRACKER_DECAY_HALFLIFE_MS: u64 = 7 * 24 * 60 * 60 * 1000;
pub const ACCESS_TRACKER_MIN_IMPORTANCE: f64 = 0.0;
pub const ACCESS_TRACKER_MAX_IMPORTANCE: f64 = 1.0;
pub const ACCESS_TRACKER_PRUNE_THRESHOLD_MS: u64 = 90 * 24 * 60 * 60 * 1000;
pub const ACCESS_TRACKER_BATCH_SIZE_MAX: usize = 1000;

// Promotion
pub const PROMOTION_IMPORTANCE_THRESHOLD_DEFAULT: f64 = 0.7;
pub const PROMOTION_SCORE_THRESHOLD_DEFAULT: f64 = 0.75;
pub const PROMOTION_CORE_MEMORY_ENTITIES_MAX: usize = 50;
pub const PROMOTION_WEIGHT_IMPORTANCE: f64 = 0.4;
pub const PROMOTION_WEIGHT_RECENCY: f64 = 0.3;
pub const PROMOTION_WEIGHT_FREQUENCY: f64 = 0.2;
pub const PROMOTION_WEIGHT_TYPE_PRIORITY: f64 = 0.1;

// Entity type priorities
pub const ENTITY_TYPE_PRIORITY_SELF: f64 = 1.0;
pub const ENTITY_TYPE_PRIORITY_PROJECT: f64 = 0.9;
pub const ENTITY_TYPE_PRIORITY_TASK: f64 = 0.85;
pub const ENTITY_TYPE_PRIORITY_PERSON: f64 = 0.7;
pub const ENTITY_TYPE_PRIORITY_TOPIC: f64 = 0.6;
pub const ENTITY_TYPE_PRIORITY_NOTE: f64 = 0.4;

// Eviction
pub const EVICTION_CORE_MEMORY_SIZE_BYTES_MAX: usize = 32 * 1024;
pub const EVICTION_CORE_MEMORY_ENTITIES_MAX: usize = 50;
pub const EVICTION_BATCH_SIZE: usize = 10;
pub const EVICTION_IMPORTANCE_THRESHOLD_MIN: f64 = 0.5;
pub const EVICTION_LAST_ACCESS_THRESHOLD_MS: u64 = 30 * 24 * 60 * 60 * 1000;

// Self-evolution
pub const EVOLUTION_MIN_SAMPLES: usize = 100;
pub const EVOLUTION_CO_OCCURRENCE_THRESHOLD: f64 = 0.7;
pub const EVOLUTION_BLOCK_USAGE_THRESHOLD_MIN: f64 = 0.1;
pub const EVOLUTION_ANALYSIS_INTERVAL_MS: u64 = 7 * 24 * 60 * 60 * 1000;
```

---

## Instance Log

- 2026-01-13 00:20 - Plan created
- Phase assignments: TBD

## Status

- **Current Phase**: Planning Complete
- **Next Phase**: Phase 1 (Access Tracking Foundation)
- **Completion**: 0% (0/8 phases)

## Notes

This plan transforms UMI from a component library into a complete memory orchestration framework matching memU's automatic management capabilities while maintaining UMI's advantages (Rust core, DST, TigerStyle, configurability).

Key principles:
1. **DST-first**: Every component has simulation tests
2. **TigerStyle**: Explicit constants, assertions, no silent failures
3. **Backward compatible**: Feature flag, migration path
4. **Graceful degradation**: Handle failures elegantly
5. **Performance**: No regression vs baseline Memory
