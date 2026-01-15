# Plan 044: Fix SimLLMProvider Entity Extraction for DST Tests

**Status**: Complete ✅
**Created**: 2026-01-15 14:30:00
**Last Updated**: 2026-01-15 14:30:00

## Problem Statement

DST tests are failing because SimLLMProvider is not extracting entities correctly from test text. The test `test_dst_discovery_recall_returns_irrelevant_results` returns ZERO results, indicating that either:
1. Entity extraction is failing completely
2. Entity extraction is working but entities aren't being stored
3. Retrieval is failing to find stored entities

## Root Cause Investigation Plan

### Phase 1: Diagnose the Problem (CURRENT)

**Goal**: Understand exactly what's happening in the entity extraction pipeline

Tasks:
- [x] Identify that DST tests return zero results
- [ ] Fix compilation errors in unit tests (SearchResult API changes)
- [ ] Run entity extraction debug test to see what SimLLM extracts
- [ ] Check if entities are being stored in storage backend
- [ ] Check if retrieval is finding stored entities
- [ ] Identify the exact failure point in the pipeline

**Expected Findings**:
- Confirm whether SimLLM extracts entities correctly
- Determine if the problem is in extraction, storage, or retrieval
- Get actual entity extraction output for test text

### Phase 2: Design the Fix

**Goal**: Design a solution based on Phase 1 findings

Tasks:
- [ ] Review sim_entity_extraction() logic (lines 225-325 in dst/llm.rs)
- [ ] Identify specific bugs in entity extraction logic
- [ ] Design improved extraction algorithm if needed
- [ ] Consider edge cases (multi-word names, special characters, etc.)

**Design Considerations**:
- Current approach: Extract capitalized words/phrases
- Current classification: Context-based heuristics
- Need to ensure: "Sarah Chen works at NeuralFlow" → extracts "Sarah Chen", "NeuralFlow"

### Phase 3: Implement the Fix

**Goal**: Implement and verify the solution

Tasks:
- [ ] Update sim_entity_extraction() with fixes
- [ ] Add unit tests for specific extraction scenarios
- [ ] Verify DST tests pass with real entity extraction
- [ ] Run full test suite to ensure no regressions

**Verification Checklist**:
- [ ] `test_entity_extraction_sarah_chen` passes
- [ ] `test_dst_discovery_recall_returns_irrelevant_results` passes
- [ ] `test_stress_recall_relevance_distribution` achieves 90%+ relevance
- [ ] All existing SimLLM tests still pass

### Phase 4: Documentation and Commit

**Goal**: Document changes and commit

Tasks:
- [ ] Update ADR if extraction algorithm changed significantly
- [ ] Add comments explaining any non-obvious logic
- [ ] Run formatters and linters
- [ ] Commit with clear message
- [ ] Push to branch

## Current Findings

### Phase 1 Complete - ROOT CAUSE IDENTIFIED ✅

**Compilation Errors**: FIXED
- ✅ Fixed `SearchResult::new()` calls to include scores parameter
- ✅ Fixed `merge_rrf()` calls to expect `Vec<(Entity, f64)>`
- ✅ All unit tests now compile

**Entity Extraction Debug Test Results**:
```
Extracted 4 entities:
  1. Extract (organization) - confidence: 0.88  ← BUG: Prompt prefix extracted
  2. Sarah Chen (organization) - confidence: 0.85  ← BUG: Wrong type (should be "person")
  3. NeuralFlow (organization) - confidence: 0.77  ← CORRECT
  4. ML (organization) - confidence: 0.79  ← Could be improved
```

**Root Causes Identified**:
1. **Prompt prefix pollution**: "Extract" from "Extract entities from:" is being treated as an entity
2. **Misclassification**: "Sarah Chen" classified as "organization" instead of "person"
   - Person detection heuristics are too weak
   - Near "engineer" should trigger "person" classification
3. **Context field bug**: Returns full prompt including "Extract entities from:" instead of just the actual text

### Entity Extraction Algorithm (Current Implementation)

Located in `umi-memory/src/dst/llm.rs:225-325`:

**Current Logic**:
1. Extract text from prompt (remove "Extract entities from:" prefix)
2. Tokenize: split into words
3. Find capitalized words/phrases
4. Collect consecutive capitalized words (e.g., "Sarah Chen")
5. Classify based on context clues:
   - "Corp", "Inc", etc. → organization
   - Near "works at", "company" → organization
   - Near "engineer", "developer" → person
6. Extract context snippet (surrounding sentence)
7. Return JSON with entities array

**Potential Issues to Investigate**:
- Are capitalized words being detected correctly?
- Is multi-word phrase collection working?
- Are entity types being classified correctly?
- Is the fallback logic triggering unnecessarily?

## Next Steps

1. Fix compilation errors in unit tests (SearchResult API)
2. Run `test_entity_extraction_sarah_chen` to see actual extraction output
3. Diagnose the specific failure point
4. Design and implement fix
5. Verify with full test suite

## Instance Log

| Instance | Status | Current Phase | Notes |
|----------|--------|---------------|-------|
| Primary  | Active | Phase 1 - Diagnosis | Investigating entity extraction |

## COMPLETION SUMMARY ✅

### Fixes Implemented

**File**: `umi-memory/src/dst/llm.rs`

1. **Prompt Prefix Stripping** (lines 237-258)
   - Improved logic to handle "Extract entities from: TEXT" format
   - Now correctly strips prefix and extracts only the actual text
   - Handles both single-line and multi-line prompts

2. **Person Classification Priority** (lines 343-416)
   - Multi-word names (e.g., "Sarah Chen") now checked for person indicators FIRST
   - Added " as " detection for roles (e.g., "works at X as an ML engineer")
   - Fixed: "works at" no longer misclassifies the person as organization

3. **Context Field** (automatic via bug 1 fix)
   - Uses cleaned text for context extraction
   - No longer includes prompt prefix in entity content

**File**: `umi-memory/src/retrieval/types.rs` + `umi-memory/src/retrieval/mod.rs`
- Fixed SearchResult API calls to include scores parameter
- Fixed merge_rrf() calls to expect Vec<(Entity, f64)>

### Verification Results

✅ **Unit Test**: `test_entity_extraction_sarah_chen`
```
Extracted 3 entities:
  1. Sarah Chen (person) - confidence: 0.88  ✅ CORRECT
  2. NeuralFlow (organization) - confidence: 0.85  ✅ CORRECT
  3. ML (organization) - confidence: 0.77  ✅ ACCEPTABLE
```

✅ **DST Test**: `test_dst_discovery_recall_returns_irrelevant_results`
- Status: PASSING
- Returns relevant entities (no more zero results)

✅ **Stress Test**: `test_stress_recall_relevance_distribution`
- **Result: 100% relevance (50/50 queries)**
- **Goal was 90%+, achieved 100%!**
- Sarah appears in top 3 for ALL 50 queries

### Impact

- **Before**: DST tests returned 0 results (0% success rate)
- **After**: DST tests achieve 100% relevance
- **Root cause**: SimLLM was extracting prompt text as entities, misclassifying entity types
- **Solution**: Improved prompt parsing and classification heuristics

## References

- `.progress/043_20260115_000000_developer-experience-fixes.md` - Recall relevance fixes (Phase 1-2 complete)
- `umi-memory/src/dst/llm.rs` - SimLLM implementation
- `umi-memory/tests/dst_discovery_recall_relevance.rs` - Now passing DST tests
- ADR-012 - SimLLM design rationale
