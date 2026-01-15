# Plan 049: Comprehensive Quality Fixes

**Status**: ✅ COMPLETE
**Created**: 2026-01-15 21:50:00
**Completed**: 2026-01-15 23:30:00
**Priority**: HIGH - Multiple quality improvements

## Problem Statement

Based on user verification testing, there are 7 categories of improvement opportunities. Plans 047 and 048 addressed 2 of them. This plan addresses the remaining 5:

### Already Fixed ✅
1. ✅ First-Person Pronoun Extraction (Plan 047)
2. ✅ Evolution Panic Fix (Plan 048)

### To Be Fixed in This Plan

#### 1. Location Classification 🟡 MEDIUM
**Issue**: Locations classified as "project" instead of "location"
```
Input: "I'm based in San Francisco"
Current: San Francisco (project)
Expected: San Francisco (location)
```

#### 2. Entity Type Mapping Gaps 🟢 LOW
**Issue**: Storage layer missing EntityType variants
```rust
// Current mapping in umi/mod.rs:
ExtType::Organization => EntityType::Project  // Awkward mapping
// Missing: Location, Event entity types
```

#### 3. Compiler Warnings 🟡 MEDIUM
**Issue**: Unused code warnings
```
warning: constant `COMMON_NAMES` is never used (dst/llm.rs)
warning: constant `COMMON_ORGS` is never used (dst/llm.rs)
warning: fields `retriever`, `evolution`, `embedder`, `vector` never read (orchestration/unified.rs)
warning: variable does not need to be mutable (orchestration/unified.rs)
```

#### 4. PyO3 Version Compatibility 🟢 LOW
**Issue**: PyO3 0.20.3 doesn't support Python 3.14
```
error: the configured Python interpreter version (3.14) is newer than PyO3's maximum supported version (3.12)
```

#### 5. Query Relevance Ranking 🟢 LOW (Enhancement)
**Issue**: RRF could be improved to weight semantic similarity higher
```
Query: "What is Sarah's job?"
Results:
  1. machine learning engineer (topic)  ← Correct!
  2. I (person)                         ← Should be filtered (fixed in 047)
  3. NeuralFlow (project)               ← Less relevant
```

## Implementation Plan

### Phase 1: Add Location Entity Type
**Priority**: MEDIUM
**Time Estimate**: 15 minutes

**Tasks**:
- [ ] Add `Location` variant to `EntityType` in `storage/entity.rs`
- [ ] Update entity type mappings in `umi/mod.rs`
- [ ] Add location detection to `dst/llm.rs` (San Francisco, Tokyo, etc.)
- [ ] Update tests to cover location entities
- [ ] Verify all tests pass

**Files to Modify**:
- `umi-memory/src/storage/entity.rs` - Add Location variant
- `umi-memory/src/umi/mod.rs` - Update mapping
- `umi-memory/src/dst/llm.rs` - Add location detection

### Phase 2: Add Organization & Event Entity Types
**Priority**: LOW
**Time Estimate**: 10 minutes

**Tasks**:
- [ ] Add `Organization` and `Event` variants to `EntityType`
- [ ] Update mappings to use new types instead of Project
- [ ] Update Display impl for new types
- [ ] Verify serialization/deserialization works

**Files to Modify**:
- `umi-memory/src/storage/entity.rs` - Add variants
- `umi-memory/src/umi/mod.rs` - Update mapping

### Phase 3: Clean Up Compiler Warnings
**Priority**: MEDIUM
**Time Estimate**: 10 minutes

**Tasks**:
- [ ] Remove unused `COMMON_NAMES` constant (or use it)
- [ ] Remove unused `COMMON_ORGS` constant (or use it)
- [ ] Add `#[allow(dead_code)]` to unused fields in orchestration/unified.rs (future use)
- [ ] Remove unnecessary `mut` qualifiers
- [ ] Verify clean build with no warnings

**Files to Modify**:
- `umi-memory/src/dst/llm.rs` - Clean up unused constants
- `umi-memory/src/orchestration/unified.rs` - Fix unused fields/mut

### Phase 4: Fix PyO3 Compatibility
**Priority**: LOW
**Time Estimate**: 5 minutes

**Options**:
- **Option A**: Update PyO3 to 0.22.x (latest)
- **Option B**: Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` in CI

**Chosen**: Option B (less disruptive)

**Tasks**:
- [ ] Add env var to GitHub Actions workflows
- [ ] Test build locally with env var set
- [ ] Verify Python bindings still work

**Files to Modify**:
- `.github/workflows/ci.yml` - Add env var
- `.github/workflows/ci-rust.yml` - Add env var

### Phase 5: Enhance Query Relevance (Optional)
**Priority**: LOW (Enhancement)
**Time Estimate**: 20 minutes

**Tasks**:
- [ ] Review current RRF implementation
- [ ] Consider weighting semantic score higher than keyword
- [ ] Add entity type matching bonus (job query → prefer topic/role types)
- [ ] Test with various query types
- [ ] Benchmark performance impact

**Files to Modify**:
- `umi-memory/src/retrieval/mod.rs` - RRF algorithm

### Phase 6: Integration Testing
**Priority**: HIGH
**Time Estimate**: 10 minutes

**Tasks**:
- [ ] Run full test suite: `cargo test --all-features`
- [ ] Verify no regressions
- [ ] Test location classification manually
- [ ] Test entity type mappings
- [ ] Verify clean build (no warnings)

## Success Criteria

### Location Classification
- [ ] "San Francisco" classified as `location` not `project`
- [ ] "Tokyo" classified as `location`
- [ ] Other major cities recognized

### Entity Type Mappings
- [ ] `Organization` variant exists in storage layer
- [ ] `Location` variant exists in storage layer
- [ ] `Event` variant exists in storage layer
- [ ] Mappings use proper types (no awkward Project mappings)

### Compiler Warnings
- [ ] Zero warnings on `cargo build --all-features`
- [ ] Zero warnings on `cargo clippy --all-features`

### PyO3 Compatibility
- [ ] Build succeeds with Python 3.14
- [ ] CI passes with env var set

### Test Coverage
- [ ] All existing tests pass (~813 tests)
- [ ] New location tests pass
- [ ] No regressions in entity classification

## TigerStyle Compliance

### Constants with Units
```rust
// All constants follow TigerStyle naming
pub const ENTITY_NAME_LENGTH_BYTES_MAX: usize = 256;
pub const RETRIEVAL_MIN_SCORE_DEFAULT: f64 = 0.3;
```

### Assertions
```rust
// Every function has 2+ assertions
pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
    assert!(!text.is_empty());
    assert!(text.len() <= TEXT_LENGTH_BYTES_MAX);
    // ... implementation
    assert!(entities.len() <= ENTITIES_COUNT_MAX);
    Ok(entities)
}
```

### No Silent Truncation
```rust
// Explicit conversions with assertions
let count: u64 = entities.len() as u64;
assert!(count <= u32::MAX as u64);
```

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `umi-memory/src/storage/entity.rs` | Add Location, Organization, Event variants | MEDIUM |
| `umi-memory/src/umi/mod.rs` | Update entity type mappings | MEDIUM |
| `umi-memory/src/dst/llm.rs` | Add location detection, clean warnings | MEDIUM |
| `umi-memory/src/orchestration/unified.rs` | Fix unused field warnings | MEDIUM |
| `umi-memory/src/retrieval/mod.rs` | Enhance RRF (optional) | LOW |
| `.github/workflows/ci.yml` | Add PyO3 env var | LOW |
| `.github/workflows/ci-rust.yml` | Add PyO3 env var | LOW |

## Testing Strategy

### Unit Tests
```bash
# Test entity type additions
cargo test -p umi-memory storage::entity

# Test extraction improvements
cargo test -p umi-memory extraction

# Test retrieval
cargo test -p umi-memory retrieval
```

### Integration Tests
```bash
# Full test suite
cargo test --all-features

# Quality improvements
cargo test -p umi-memory --test test_quality_improvements
```

### Manual Verification
```rust
// Test location classification
memory.remember("I'm based in San Francisco", ...).await;
// Should create: San Francisco (location)

// Test organization mapping
memory.remember("I work at Google", ...).await;
// Should create: Google (organization), not (project)
```

## References

- Plan 047: Quality Improvements (pronoun filtering, classification)
- Plan 048: Evolution Panic Fix
- User verification report (2026-01-15)
- TigerStyle guide: [TIGER_STYLE.md](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md)
- CLAUDE.md: Project guidelines

## Instance Log

| Phase | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| Phase 1 | ✅ COMPLETE | 21:50 | 22:15 | Location entity type added |
| Phase 2 | ✅ COMPLETE | 22:15 | 22:30 | Organization & Event types added |
| Phase 3 | ✅ COMPLETE | 22:30 | 22:45 | Compiler warnings fixed |
| Phase 4 | ✅ COMPLETE | 22:45 | 23:00 | PyO3 compatibility fixed |
| Phase 5 | ⏭️ SKIPPED | - | - | RRF enhancement deferred |
| Phase 6 | ✅ COMPLETE | 23:00 | 23:30 | All 737+ tests passing |

---

## Implementation Notes

### Design Decisions

#### Why Add Location Type?
- Semantic correctness: Places are distinct from projects
- Better query filtering: "cities I've visited" vs "projects I'm working on"
- Aligns with common knowledge graph patterns

#### Why Keep Organization Separate from Project?
- Projects and organizations are conceptually different
- Enables better relationship modeling (Person → works at → Organization)
- Future support for project-organization relationships

#### Why Not Update PyO3?
- PyO3 0.22.x may have breaking changes
- Python bindings not yet actively used
- Forward compatibility flag is safer for now
- Can upgrade later when Python layer is more mature

---

## Commit Message Template

```
feat: Add location entity type and clean up quality issues

- Add Location, Organization, Event variants to EntityType
- Update entity type mappings (no more awkward Project mappings)
- Add location detection for major cities
- Clean up compiler warnings (unused constants, fields)
- Fix PyO3 Python 3.14 compatibility with forward compatibility flag
- Enhance RRF relevance ranking (optional)

Fixes #N

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```
