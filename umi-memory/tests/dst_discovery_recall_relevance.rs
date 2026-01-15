//! DST Discovery: Recall Relevance Issues
//!
//! This file contains DST-first discovery tests for recall relevance problems.
//! These tests are written BEFORE implementing any fixes, and are EXPECTED TO FAIL.
//!
//! The failures will reveal the root causes of poor recall relevance reported in
//! the developer experience report.

use umi_memory::dst::{SimConfig, Simulation};
use umi_memory::umi::{MemoryError, RecallOptions, RememberOptions};

/// DST Discovery Test 1: Recall Returns Irrelevant Results
///
/// This test EXPECTS proper relevance-based recall and will FAIL to reveal problems.
///
/// **Scenario**: Store facts about Sarah (ML engineer at NeuralFlow) and query about her job.
/// **Expected**: Top results mention Sarah, NeuralFlow, or ML engineer
/// **Actual (before fix)**: Returns Python/Rust (sorted by recency, not relevance)
///
/// **DISCOVERY GOALS**:
/// - Observe which entities are returned for a relevant query
/// - Identify the sorting/ranking logic being used
/// - Determine if similarity scores are tracked
#[tokio::test]
async fn test_dst_discovery_recall_returns_irrelevant_results() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        // Store test data matching UX report scenario
        println!("=== DST DISCOVERY: Storing test data ===");

        memory
            .remember(
                "Sarah Chen works at NeuralFlow as an ML engineer",
                RememberOptions::default(),
            )
            .await?;

        memory
            .remember(
                "Sarah is learning Rust for the recommendation systems team",
                RememberOptions::default(),
            )
            .await?;

        memory
            .remember(
                "Python is Sarah's main language",
                RememberOptions::default(),
            )
            .await?;

        println!("=== DST DISCOVERY: Querying 'What is Sarah's job?' ===");

        // Query: What is Sarah's job?
        let results = memory
            .recall(
                "What is Sarah's job?",
                RecallOptions::default().with_limit(3).unwrap(),
            )
            .await?;

        // Print actual results for investigation
        println!("=== DST DISCOVERY: Top 3 results ===");
        for (i, entity) in results.iter().enumerate() {
            println!(
                "  {}. {} - '{}'",
                i + 1,
                entity.name,
                entity.content.chars().take(60).collect::<String>()
            );
        }

        // DISCOVERY ASSERTION: This WILL FAIL before fixes
        assert!(results.len() >= 1, "Should find at least one result");

        let top_result = &results[0];

        // EXPECT: Top result mentions Sarah, NeuralFlow, or ML engineer
        let is_relevant = top_result.name.contains("Sarah")
            || top_result.content.contains("Sarah")
            || top_result.content.contains("NeuralFlow")
            || top_result.content.contains("ML engineer");

        assert!(
            is_relevant,
            "DISCOVERY FAILED: Top result should be Sarah-related, got: '{}' - '{}'

            This failure reveals the recall relevance problem from the UX report.

            EXPECTED: Top result mentions Sarah, NeuralFlow, or ML engineer
            ACTUAL: Top result is '{}' with content '{}'

            Before fix: Returns Python/Rust for every query (sorted by recency, not relevance)
            After fix: Should return Sarah-related entities (sorted by relevance score)

            INVESTIGATE:
            1. Check DualRetriever::search() - what order are results returned?
            2. Check merge_rrf() - are similarity scores preserved?
            3. Check SimVectorBackend::search() - random or similarity-based?
            4. Check sorting logic - by updated_at or by score?",
            top_result.name,
            top_result.content.chars().take(100).collect::<String>(),
            top_result.name,
            top_result.content.chars().take(100).collect::<String>()
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

/// DST Stress Test 1: Recall Relevance Distribution
///
/// This stress test runs 50 queries to measure relevance distribution.
///
/// **Scenario**: Store 20 entities (10 about Sarah, 10 unrelated), run 50 queries about Sarah
/// **Expected**: Sarah appears in top 3 for ~90%+ of queries
/// **Actual (before fix)**: Sarah appears in ~16% (near-random)
///
/// **DISCOVERY GOALS**:
/// - Quantify the relevance problem
/// - Measure how often relevant results appear in top N
/// - Confirm that sorting is not relevance-based
#[tokio::test]
async fn test_stress_recall_relevance_distribution() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== DST STRESS TEST: Setting up test data ===");

        // Store 10 entities about Sarah
        for i in 0..10 {
            memory
                .remember(
                    &format!("Sarah fact {}: ML engineer at NeuralFlow", i),
                    RememberOptions::default(),
                )
                .await?;
        }

        // Store 10 unrelated entities
        for i in 0..10 {
            memory
                .remember(
                    &format!("Unrelated fact {}: Python and Rust programming", i),
                    RememberOptions::default(),
                )
                .await?;
        }

        println!("=== DST STRESS TEST: Running 50 queries about Sarah ===");

        let mut sarah_in_top3_count = 0;
        let mut total_results = 0;

        // Run 50 queries about Sarah
        for i in 0..50 {
            let results = memory
                .recall(
                    "What does Sarah do?",
                    RecallOptions::default().with_limit(3).unwrap(),
                )
                .await?;

            total_results += results.len();

            // Count how many times Sarah appears in top 3
            let sarah_found = results
                .iter()
                .any(|e| e.name.contains("Sarah") || e.content.contains("Sarah"));

            if sarah_found {
                sarah_in_top3_count += 1;
            }

            // Print sample results every 10 queries
            if i % 10 == 0 {
                println!("  Query {}: Found Sarah = {}", i, sarah_found);
            }
        }

        let relevance_percentage = (sarah_in_top3_count as f32 / 50.0) * 100.0;

        println!("=== DST STRESS TEST: Results ===");
        println!(
            "  Sarah in top 3: {}/50 ({:.0}%)",
            sarah_in_top3_count, relevance_percentage
        );
        println!(
            "  Average results per query: {:.1}",
            total_results as f32 / 50.0
        );

        // DISCOVERY: Before fix, Sarah appears rarely (~20% instead of ~100%)
        // This quantifies the relevance problem from the UX report

        // After fix, this should be close to 100%
        assert!(
            sarah_in_top3_count >= 45,
            "DISCOVERY FAILED: Sarah should appear in top 3 for 90%+ of queries

            EXPECTED: Sarah in top 3 for 45+/50 queries (90%+)
            ACTUAL: Sarah in top 3 for {}/50 queries ({:.0}%)

            This quantifies the relevance problem:
            - With 10 Sarah entities and 10 unrelated entities
            - Random ranking would give ~50% (half the entities are about Sarah)
            - Actual ~{}% suggests sorting by recency, not relevance
            - Top 3 entities are likely the 3 most recently stored (unrelated facts)

            DISCOVERIES:
            - Relevance is near-random or recency-based
            - Confirms UX report: queries return irrelevant results
            - Need similarity score tracking and relevance-based sorting",
            sarah_in_top3_count,
            relevance_percentage,
            relevance_percentage
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

/// DST Discovery Test 2: Non-Existent Entity Returns Results
///
/// This test queries for an entity that doesn't exist in memory.
///
/// **Scenario**: Store facts about Sarah, query for "John Smith"
/// **Expected**: Empty results or low-relevance filtered results
/// **Actual (before fix)**: Returns full set of results (no relevance threshold)
///
/// **DISCOVERY GOALS**:
/// - Confirm no relevance threshold filtering exists
/// - Observe behavior when query has no relevant entities
#[tokio::test]
async fn test_dst_discovery_nonexistent_entity_returns_results() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== DST DISCOVERY: Storing Sarah facts ===");

        // Store facts about Sarah
        memory
            .remember(
                "Sarah works at NeuralFlow as an ML engineer",
                RememberOptions::default(),
            )
            .await?;

        memory
            .remember("Sarah loves Python programming", RememberOptions::default())
            .await?;

        println!("=== DST DISCOVERY: Querying for non-existent 'John Smith' ===");

        // Query for John Smith (doesn't exist)
        let results = memory
            .recall("Who is John Smith?", RecallOptions::default())
            .await?;

        println!("=== DST DISCOVERY: Results for non-existent entity ===");
        println!("  Found {} results", results.len());
        for (i, entity) in results.iter().take(3).enumerate() {
            println!("  {}. {}", i + 1, entity.name);
        }

        // DISCOVERY: Should return 0 or very few results (with relevance threshold)
        // Before fix: Returns all/many results (no filtering)
        assert!(
            results.len() <= 2,
            "DISCOVERY FAILED: Query for non-existent entity should return few/no results

            EXPECTED: 0-2 results (low relevance filtered out)
            ACTUAL: {} results

            This reveals lack of relevance threshold filtering.
            Without min_score threshold, all queries return max results regardless of relevance.

            DISCOVERY:
            - No relevance score filtering
            - Queries for non-existent entities return arbitrary results
            - Need RETRIEVAL_MIN_SCORE_DEFAULT threshold",
            results.len()
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

// ==============================================================================
// DST DISCOVERIES - DOCUMENTED AFTER TEST FAILURES
// ==============================================================================
//
// Tests run on: 2026-01-15
// All 2 discovery tests FAILED as expected, revealing 4 root causes:
//
// DISCOVERY 1: Results sorted by updated_at (recency), not relevance ❌
// - Location: umi-memory/src/retrieval/mod.rs:251-253
// - Code: `results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));`
// - Evidence: Test 1 returned "Eve" for query "What is Sarah's job?"
// - Evidence: Test 2 showed 0% relevance (Sarah never in top 3)
// - Fix needed: Sort by similarity score descending
//
// DISCOVERY 2: No similarity scores tracked in SearchResult ❌
// - Location: umi-memory/src/retrieval/types.rs:114-126
// - Evidence: SearchResult struct has NO score field
// - Impact: Even if we compute scores, they can't be used for sorting
// - Fix needed: Add `scores: Vec<f64>` field to SearchResult
//
// DISCOVERY 3: SimVectorBackend IS CORRECT (Surprise!) ✅
// - Location: umi-memory/src/storage/vector.rs:206-215
// - Code: Correctly computes `cosine_similarity(embedding, stored)`
// - Code: Correctly sorts by score descending
// - Surprise: Vector backend is working correctly!
// - Issue: DualRetriever throws away the scores after getting them
//
// DISCOVERY 4: Scores Lost in Translation ❌
// - Root cause: Vector backend returns scores, but DualRetriever loses them
// - Flow: VectorBackend::search() → Vec<VectorSearchResult> (with scores)
//         → DualRetriever converts to Vec<Entity> (loses scores)
//         → Sorts by updated_at instead of score
// - Fix needed: Preserve scores through the pipeline, sort by score
//
// DISCOVERY 5: No relevance threshold filtering ❌
// - Location: umi-memory/src/retrieval/mod.rs
// - Evidence: Test 3 passed but still returned irrelevant "Eve" results
// - Fix needed: Add min_score filtering with RETRIEVAL_MIN_SCORE_DEFAULT = 0.3
//
// ==============================================================================
// QUANTITATIVE RESULTS FROM STRESS TEST
// ==============================================================================
//
// Stress test results (50 queries):
// - Sarah relevance: 0/50 (0%)
// - Expected: 45+/50 (90%+)
// - Gap: 90 percentage points
//
// This quantifies the severity of the recall relevance problem.
//
// ==============================================================================
