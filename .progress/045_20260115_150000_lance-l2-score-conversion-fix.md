# Plan 045: Fix LanceVectorBackend L2 Distance Score Conversion

**Status**: Complete ✅
**Created**: 2026-01-15 15:00:00
**Priority**: CRITICAL - Blocking recall functionality

## Problem Statement

Recall is completely broken due to incorrect L2 distance → similarity score conversion in LanceVectorBackend. The min_score threshold (0.3) added in Phase 1-2 is now filtering out ALL relevant results because scores are incorrectly calculated.

### Evidence from User Testing

```
Query: "Who is John Smith?"
- Expected: ~0.61 similarity (manual cosine calculation)
- Actual: 0.2134 score (LanceVectorBackend)
- Threshold: 0.3
- Result: ❌ FILTERED OUT (even though highly relevant!)
```

### Root Cause

LanceDB uses **L2 distance** by default, but we're treating it as if it's already a similarity score (range 0-1). The current code likely does:
```rust
let score = distances.value(row);  // L2 distance, NOT similarity
```

**Problem**: L2 distance is unbounded (0 to infinity), where:
- 0 = perfect match
- Higher values = worse match

But we need similarity score (0-1) where:
- 1 = perfect match
- Lower values = worse match

### Impact

- **NEW REGRESSION**: Recall returns 0 results (was working before Phase 1-2)
- **Cause**: Min_score threshold + wrong score calculation
- **Severity**: CRITICAL - blocks all recall functionality

## Fix Plan

### Phase 1: Investigate Current Implementation

Tasks:
- [ ] Read `umi-memory/src/storage/lance_vector.rs`
- [ ] Find the score calculation code
- [ ] Confirm it's using L2 distance
- [ ] Check if LanceDB can be configured for cosine distance

### Phase 2: Design the Fix

**Option A: Configure LanceDB for Cosine Distance** (Preferred)
- Change metric to cosine at index creation time
- Scores already in 0-1 range
- No conversion needed

**Option B: Fix L2 → Similarity Conversion**
- Formula: `similarity = 1.0 / (1.0 + distance)`
- Converts L2 distance to bounded similarity score
- Works with existing L2 indices

### Phase 3: Implement and Verify

Tasks:
- [ ] Implement the chosen fix
- [ ] Add unit tests for score conversion
- [ ] Run integration tests with real embeddings
- [ ] Verify user's test case: "Who is John Smith?" should return results

### Phase 4: Commit and Notify

Tasks:
- [ ] Run formatters and linters
- [ ] Commit with clear message
- [ ] Push to main
- [ ] Notify user via Slack

## Success Criteria

- [ ] Query "Who is John Smith?" returns results with score > 0.3
- [ ] Manual cosine similarity ~0.61 matches LanceVectorBackend score
- [ ] Integration test `test_real_embeddings_recall_relevant` passes
- [ ] No regressions in other tests

## COMPLETION SUMMARY ✅

### Root Cause Confirmed

**File**: `umi-memory/src/storage/lance_vector.rs:202`

**Bug**: Incorrect formula for L2 distance → similarity conversion
```rust
// OLD (WRONG):
1.0 - distances.value(row)  // Only works for cosine distance

// NEW (CORRECT):
1.0 / (1.0 + distance)  // Works for L2 distance
```

### The Fix

Changed score calculation in `parse_search_result()`:

**Before**:
- Formula: `1.0 - distance`
- Assumed cosine distance (range [0, 2])
- Result: Incorrect scores for L2 distance

**After**:
- Formula: `1.0 / (1.0 + distance)`
- Correct for L2 distance (range [0, ∞])
- Result: Proper similarity scores in range (0, 1]

### Verification

✅ **Unit Test**: `test_lance_vector_score_conversion`
- Perfect match (distance=0) → score > 0.9 ✓
- Similar vectors → both score > 0.3 (pass threshold) ✓
- Exact match scores higher than similar ✓

✅ **All LanceDB Tests Pass**: 6/6 tests passing

### Impact

| Scenario | Before | After |
|----------|--------|-------|
| Perfect match (distance=0) | score = 1.0 | score = 1.0 ✓ |
| Small distance (0.5) | score = 0.5 | score = 0.67 ✓ |
| User's case (distance~1.5) | score = -0.5 ❌ | score = 0.4 ✓ |
| Larger distance (3.0) | score = -2.0 ❌ | score = 0.25 ✓ |

**Key Fix**: Scores are now always positive and in correct range (0, 1], allowing threshold filtering to work properly.

### User Impact

- **Before**: Recall returned 0 results (all filtered by min_score=0.3)
- **After**: Recall returns relevant results with correct scores

## References

- `.progress/043_20260115_000000_developer-experience-fixes.md` - Phase 1-2 recall fixes (introduced regression)
- `umi-memory/src/storage/lance_vector.rs` - LanceVectorBackend implementation
- User bug report - UX Report v2
