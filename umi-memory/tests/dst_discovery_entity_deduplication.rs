//! DST Discovery: Entity Deduplication
//!
//! This file contains DST-first discovery tests for entity deduplication problems.
//! These tests are written BEFORE implementing any fixes, and are EXPECTED TO FAIL.
//!
//! The failures will reveal the root causes of duplicate entity creation reported in
//! the developer experience report.

use umi_memory::dst::{SimConfig, Simulation};
use umi_memory::umi::{MemoryError, RememberOptions};

/// DST Discovery Test 1: No Duplicate Entities from Single Text
///
/// This test EXPECTS no duplicate entities and will FAIL to reveal problems.
///
/// **Scenario**: Store "Sarah Chen works at NeuralFlow" once
/// **Expected**: Creates 2-3 unique entities (Sarah Chen, NeuralFlow, maybe ML)
/// **Actual (before fix)**: Creates 4-6 entities with duplicates
///
/// **DISCOVERY GOALS**:
/// - Observe actual entity extraction (how many entities created?)
/// - Identify duplicate patterns (Sarah + Chen + "Sarah Chen"?)
/// - Determine where duplicates are created (extraction? storage?)
#[tokio::test]
async fn test_dst_discovery_no_duplicate_entities_single_text() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== DST DISCOVERY: Storing single fact ===");

        let result = memory
            .remember(
                "Sarah Chen works at NeuralFlow as an ML engineer",
                RememberOptions::default(),
            )
            .await?;

        println!("=== DST DISCOVERY: Entities created ===");
        println!("  Total count: {}", result.entities.len());
        for (i, entity) in result.entities.iter().enumerate() {
            println!("  {}. {} ({})", i + 1, entity.name, entity.entity_type);
        }

        // DISCOVERY: Check for duplicates
        let unique_names: std::collections::HashSet<_> =
            result.entities.iter().map(|e| &e.name).collect();
        let duplicate_count = result.entities.len() - unique_names.len();

        println!("=== DST DISCOVERY: Duplication analysis ===");
        println!("  Unique names: {}", unique_names.len());
        println!("  Duplicate count: {}", duplicate_count);

        // DISCOVERY ASSERTION: This WILL FAIL before fixes
        assert_eq!(
            duplicate_count, 0,
            "DISCOVERY FAILED: Found {} duplicate entities

            This reveals entity deduplication is not working.

            Expected: No duplicates (each entity name appears once)
            Actual: {} duplicates detected

            Common patterns:
            - 'Sarah' + 'Chen' + 'Sarah Chen' (name splitting)
            - Same entity extracted multiple times

            INVESTIGATE:
            1. Check EntityExtractor - does it create duplicates?
            2. Check if same entity name appears multiple times
            3. Check if storage allows duplicate names",
            duplicate_count, duplicate_count
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

/// DST Discovery Test 2: No Duplicate Entities from Repeated Storage
///
/// This test stores the same fact TWICE and expects no additional duplicates.
///
/// **Scenario**: Store "Sarah works at Acme" twice
/// **Expected**: Still 2 entities (Sarah, Acme) - updates existing
/// **Actual (before fix)**: 4 entities (Sarah, Acme, Sarah, Acme)
///
/// **DISCOVERY GOALS**:
/// - Observe what happens when same fact is stored twice
/// - Check if storage layer deduplicates by entity ID/name
/// - Determine if entities are updated or duplicated
#[tokio::test]
async fn test_dst_discovery_no_duplicate_entities_repeated_storage() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== DST DISCOVERY: Storing fact first time ===");

        let result1 = memory
            .remember("Sarah works at Acme", RememberOptions::default())
            .await?;

        println!("  First storage: {} entities", result1.entities.len());

        println!("=== DST DISCOVERY: Storing same fact second time ===");

        let result2 = memory
            .remember("Sarah works at Acme", RememberOptions::default())
            .await?;

        println!("  Second storage: {} entities", result2.entities.len());

        // Check total entities in memory
        let all_entities = memory.recall("Sarah Acme", Default::default()).await?;

        println!("=== DST DISCOVERY: Total entities in memory ===");
        println!("  Total count: {}", all_entities.len());
        for (i, entity) in all_entities.iter().enumerate() {
            println!("  {}. {} ({})", i + 1, entity.name, entity.entity_type);
        }

        // DISCOVERY: Check for duplicates by name
        let mut name_counts = std::collections::HashMap::new();
        for entity in &all_entities {
            *name_counts.entry(&entity.name).or_insert(0) += 1;
        }

        let duplicates: Vec<_> = name_counts
            .iter()
            .filter(|(_, count)| **count > 1)
            .collect();

        println!("=== DST DISCOVERY: Duplicate analysis ===");
        for (name, count) in &duplicates {
            println!("  '{}' appears {} times", name, count);
        }

        // DISCOVERY ASSERTION: This WILL FAIL before fixes
        assert!(
            duplicates.is_empty(),
            "DISCOVERY FAILED: Found {} duplicate entity names after storing same fact twice

            Duplicates found:
            {:?}

            Expected: Each entity name appears once (storage should update, not duplicate)
            Actual: {} entity names appear multiple times

            This reveals storage does not deduplicate by entity name/ID.

            INVESTIGATE:
            1. Check storage backend - does it allow duplicate entity names?
            2. Check if entity IDs are deterministic (same name = same ID?)
            3. Check if remember() updates or creates new entities",
            duplicates.len(),
            duplicates,
            duplicates.len()
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}

/// DST Discovery Test 3: Entity Name Consistency
///
/// This test checks if multi-word names are preserved vs. split.
///
/// **Scenario**: Store "Sarah Chen is a person"
/// **Expected**: Creates "Sarah Chen" entity (NOT "Sarah" + "Chen")
/// **Actual (before fix)**: May create both "Sarah", "Chen", AND "Sarah Chen"
#[tokio::test]
async fn test_dst_discovery_multiword_name_consistency() {
    let sim = Simulation::new(SimConfig::with_seed(42));

    sim.run(|env| async move {
        let mut memory = env.create_memory();

        println!("=== DST DISCOVERY: Storing fact with multi-word name ===");

        let result = memory
            .remember("Sarah Chen is a developer", RememberOptions::default())
            .await?;

        println!("=== DST DISCOVERY: Entities created ===");
        for (i, entity) in result.entities.iter().enumerate() {
            println!("  {}. '{}' ({})", i + 1, entity.name, entity.entity_type);
        }

        // Check if we have both "Sarah Chen" and individual "Sarah"/"Chen"
        let has_full_name = result.entities.iter().any(|e| e.name == "Sarah Chen");
        let has_sarah = result.entities.iter().any(|e| e.name == "Sarah");
        let has_chen = result.entities.iter().any(|e| e.name == "Chen");

        println!("=== DST DISCOVERY: Name pattern analysis ===");
        println!("  Has 'Sarah Chen': {}", has_full_name);
        println!("  Has 'Sarah': {}", has_sarah);
        println!("  Has 'Chen': {}", has_chen);

        // DISCOVERY: If we have both full name AND parts, that's a problem
        let name_splitting_issue = has_full_name && (has_sarah || has_chen);

        // DISCOVERY ASSERTION: This WILL FAIL before fixes
        assert!(
            !name_splitting_issue,
            "DISCOVERY FAILED: Multi-word name 'Sarah Chen' was split into parts

            Found:
            - Full name 'Sarah Chen': {}
            - Part 'Sarah': {}
            - Part 'Chen': {}

            This reveals entity extraction creates both the full name AND its parts.

            Expected: Either 'Sarah Chen' (preferred) OR 'Sarah' + 'Chen', but not BOTH
            Actual: All three entities created simultaneously

            INVESTIGATE:
            1. Check SimLLM entity extraction logic
            2. Check if consecutive capitalized words are handled correctly
            3. Consider deduplication at extraction time",
            has_full_name, has_sarah, has_chen
        );

        Ok::<(), MemoryError>(())
    })
    .await
    .unwrap();
}
