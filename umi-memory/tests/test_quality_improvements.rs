//! Test Quality Improvements
//!
//! Tests for pronoun filtering and entity classification improvements.

use umi_memory::dst::{SimConfig, Simulation};
use umi_memory::umi::{MemoryError, RecallOptions, RememberOptions};

/// Test that pronouns are filtered out from entity extraction.
#[tokio::test]
async fn test_pronoun_filtering() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== Testing: Pronoun Filtering ===");

        let result = memory
            .remember("I work at Google", RememberOptions::default())
            .await?;

        println!("Entities created from 'I work at Google':");
        for (i, e) in result.entities.iter().enumerate() {
            println!("  {}. {} ({})", i + 1, e.name, e.entity_type);
        }

        // ASSERTION: "I" should NOT be extracted as an entity
        let has_pronoun = result.entities.iter().any(|e| e.name == "I");
        assert!(
            !has_pronoun,
            "'I' pronoun should be filtered out, but was found in entities"
        );

        println!("✓ Pronoun 'I' successfully filtered");

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

/// Test that Google is classified as an organization.
#[tokio::test]
async fn test_google_classification() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== Testing: Google Classification ===");

        let result = memory
            .remember("I work at Google", RememberOptions::default())
            .await?;

        println!("Entities created:");
        for (i, e) in result.entities.iter().enumerate() {
            println!("  {}. {} ({})", i + 1, e.name, e.entity_type);
        }

        // ASSERTION: Google should be classified as "organization"
        let google = result.entities.iter().find(|e| e.name == "Google");

        assert!(google.is_some(), "Google entity should be extracted");

        let google_type = &google.unwrap().entity_type;
        assert_eq!(
            google_type.as_str(),
            "organization",
            "Google should be classified as 'organization', got '{}'",
            google_type.as_str()
        );

        println!("✓ Google correctly classified as organization");

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

/// Test that unrelated queries return fewer results (relevance filtering).
#[tokio::test]
async fn test_unrelated_query_filtering() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== Testing: Unrelated Query Filtering ===");

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

        // Query for completely unrelated entity
        let results = memory
            .recall("Who is John Smith?", RecallOptions::default())
            .await?;

        println!(
            "Results for 'Who is John Smith?': {} entities",
            results.len()
        );
        for (i, e) in results.iter().take(5).enumerate() {
            println!("  {}. {}", i + 1, e.name);
        }

        // With relevance filtering (threshold 0.3), we should get fewer false positives
        // Acceptable range: 0-5 results (token-based embedding has limitations)
        // Note: SimEmbedding uses token overlap, so completely unrelated queries
        // may still return some results with low scores above the threshold
        assert!(
            results.len() <= 5,
            "Query for non-existent entity returned too many results: {}",
            results.len()
        );

        println!(
            "✓ Relevance filtering working (got {} results)",
            results.len()
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}
