//! DST Tests for PostgresVectorBackend
//!
//! TigerStyle: Test production backend behavior deterministically.
//!
//! These tests verify that PostgresVectorBackend:
//! 1. Persists data across connections
//! 2. Behaves consistently with SimVectorBackend
//! 3. Handles concurrent operations correctly
//! 4. Works with pgvector extension
//!
//! **IMPORTANT**: This test file is written FIRST, before implementation.
//! PostgresVectorBackend doesn't exist yet - this defines the contract.

use std::sync::Arc;

use tempfile::TempDir;

use umi_memory::constants::EMBEDDING_DIMENSIONS_COUNT;
use umi_memory::storage::{SimVectorBackend, PostgresVectorBackend, VectorBackend};

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a SimVectorBackend with deterministic seed.
fn create_sim_backend(seed: u64) -> SimVectorBackend {
    SimVectorBackend::new(seed)
}

/// Create a PostgresVectorBackend with test database.
async fn create_postgres_backend() -> PostgresVectorBackend {
    // Use test database URL from environment or default
    let db_url = std::env::var("TEST_POSTGRES_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/umi_test".to_string());

    PostgresVectorBackend::connect(&db_url)
        .await
        .expect("Failed to connect to test Postgres database")
}

/// Generate deterministic embedding from seed and index.
fn generate_embedding(seed: u64, index: usize) -> Vec<f32> {
    let mut emb = vec![0.0; EMBEDDING_DIMENSIONS_COUNT];
    for i in 0..EMBEDDING_DIMENSIONS_COUNT {
        emb[i] = ((seed + i as u64 + index as u64) % 1000) as f32 / 1000.0;
    }
    emb
}

// =============================================================================
// DST Test 1: Persistence Across Connections
// =============================================================================

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_persistence_store_and_retrieve() {
    let backend1 = create_postgres_backend().await;

    // Phase 1: Store data
    let emb1 = generate_embedding(42, 1);
    let emb2 = generate_embedding(42, 2);

    backend1.store("entity1", &emb1).await.unwrap();
    backend1.store("entity2", &emb2).await.unwrap();

    assert!(backend1.exists("entity1").await.unwrap());
    assert!(backend1.exists("entity2").await.unwrap());
    assert_eq!(backend1.count().await.unwrap(), 2);

    // Drop first backend
    drop(backend1);

    // Phase 2: Reopen and verify data persists
    let backend2 = create_postgres_backend().await;

    assert!(backend2.exists("entity1").await.unwrap());
    assert!(backend2.exists("entity2").await.unwrap());
    assert_eq!(backend2.count().await.unwrap(), 2);

    let emb1_retrieved = backend2.get("entity1").await.unwrap().unwrap();
    let emb1_expected = generate_embedding(42, 1);
    assert_eq!(emb1_retrieved, emb1_expected);

    // Cleanup
    backend2.delete("entity1").await.unwrap();
    backend2.delete("entity2").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_persistence_update_across_restarts() {
    let backend1 = create_postgres_backend().await;

    // Phase 1: Store initial data
    let emb1 = generate_embedding(42, 1);
    backend1.store("entity_update", &emb1).await.unwrap();

    drop(backend1);

    // Phase 2: Update data
    let backend2 = create_postgres_backend().await;
    let emb2 = generate_embedding(42, 2);
    backend2.store("entity_update", &emb2).await.unwrap();

    drop(backend2);

    // Phase 3: Verify updated data persists
    let backend3 = create_postgres_backend().await;
    let emb_retrieved = backend3.get("entity_update").await.unwrap().unwrap();
    let emb_expected = generate_embedding(42, 2);
    assert_eq!(emb_retrieved, emb_expected);

    // Cleanup
    backend3.delete("entity_update").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_persistence_delete_across_restarts() {
    let backend1 = create_postgres_backend().await;

    // Phase 1: Store data
    let emb1 = generate_embedding(42, 1);
    let emb2 = generate_embedding(42, 2);
    backend1.store("entity_del1", &emb1).await.unwrap();
    backend1.store("entity_del2", &emb2).await.unwrap();

    drop(backend1);

    // Phase 2: Delete one entity
    let backend2 = create_postgres_backend().await;
    backend2.delete("entity_del1").await.unwrap();

    drop(backend2);

    // Phase 3: Verify deletion persists
    let backend3 = create_postgres_backend().await;
    assert!(!backend3.exists("entity_del1").await.unwrap());
    assert!(backend3.exists("entity_del2").await.unwrap());

    // Cleanup
    backend3.delete("entity_del2").await.unwrap();
}

// =============================================================================
// DST Test 2: Behavior Consistency with SimVectorBackend
// =============================================================================

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_behavior_matches_sim_empty_search() {
    let sim = create_sim_backend(42);
    let postgres = create_postgres_backend().await;

    let query = generate_embedding(42, 999);

    let sim_results = sim.search(&query, 10).await.unwrap();
    let postgres_results = postgres.search(&query, 10).await.unwrap();

    assert_eq!(sim_results.len(), 0);
    assert_eq!(postgres_results.len(), 0);
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_behavior_matches_sim_store_and_count() {
    let sim = create_sim_backend(42);
    let postgres = create_postgres_backend().await;

    let emb1 = generate_embedding(42, 1);
    let emb2 = generate_embedding(42, 2);

    sim.store("entity_sim1", &emb1).await.unwrap();
    sim.store("entity_sim2", &emb2).await.unwrap();

    postgres.store("entity_pg1", &emb1).await.unwrap();
    postgres.store("entity_pg2", &emb2).await.unwrap();

    // Both should have 2 entities
    assert_eq!(sim.count().await.unwrap(), 2);
    assert_eq!(postgres.count().await.unwrap(), 2);

    assert_eq!(sim.exists("entity_sim1").await.unwrap(), postgres.exists("entity_pg1").await.unwrap());
    assert_eq!(sim.exists("entity_sim2").await.unwrap(), postgres.exists("entity_pg2").await.unwrap());

    // Cleanup
    postgres.delete("entity_pg1").await.unwrap();
    postgres.delete("entity_pg2").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_behavior_matches_sim_search_returns_stored() {
    let sim = create_sim_backend(42);
    let postgres = create_postgres_backend().await;

    let emb1 = generate_embedding(42, 1);

    sim.store("entity_search_sim", &emb1).await.unwrap();
    postgres.store("entity_search_pg", &emb1).await.unwrap();

    let sim_results = sim.search(&emb1, 10).await.unwrap();
    let postgres_results = postgres.search(&emb1, 10).await.unwrap();

    assert_eq!(sim_results.len(), 1);
    assert_eq!(postgres_results.len(), 1);
    assert_eq!(sim_results[0].id, "entity_search_sim");
    assert_eq!(postgres_results[0].id, "entity_search_pg");

    // Cleanup
    postgres.delete("entity_search_pg").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_behavior_matches_sim_delete() {
    let sim = create_sim_backend(42);
    let postgres = create_postgres_backend().await;

    let emb1 = generate_embedding(42, 1);

    sim.store("entity_delete_sim", &emb1).await.unwrap();
    postgres.store("entity_delete_pg", &emb1).await.unwrap();

    sim.delete("entity_delete_sim").await.unwrap();
    postgres.delete("entity_delete_pg").await.unwrap();

    assert_eq!(sim.exists("entity_delete_sim").await.unwrap(), false);
    assert_eq!(postgres.exists("entity_delete_pg").await.unwrap(), false);
}

// =============================================================================
// DST Test 3: Concurrent Operations
// =============================================================================

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_concurrent_stores() {
    let postgres = create_postgres_backend().await;
    let postgres = Arc::new(postgres);

    let mut handles = vec![];

    // Spawn concurrent writes
    for i in 0..3 {
        let postgres_clone = Arc::clone(&postgres);

        let handle = tokio::spawn(async move {
            let emb = generate_embedding(42, i);
            let id = format!("entity_concurrent{}", i);

            // Store concurrently
            postgres_clone.store(&id, &emb).await.unwrap();
        });

        handles.push(handle);
    }

    // Wait for all stores to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all entities were stored
    assert_eq!(postgres.count().await.unwrap(), 3);
    assert!(postgres.exists("entity_concurrent0").await.unwrap());
    assert!(postgres.exists("entity_concurrent1").await.unwrap());
    assert!(postgres.exists("entity_concurrent2").await.unwrap());

    // Cleanup
    for i in 0..3 {
        let id = format!("entity_concurrent{}", i);
        postgres.delete(&id).await.unwrap();
    }
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_concurrent_reads() {
    let postgres = create_postgres_backend().await;

    // Setup: Store some data
    let emb1 = generate_embedding(42, 1);
    postgres.store("entity_concurrent_read", &emb1).await.unwrap();

    let postgres = Arc::new(postgres);
    let mut handles = vec![];

    // Spawn 5 concurrent readers
    for _ in 0..5 {
        let postgres_clone = Arc::clone(&postgres);

        let handle = tokio::spawn(async move {
            let exists = postgres_clone.exists("entity_concurrent_read").await.unwrap();
            let emb = postgres_clone.get("entity_concurrent_read").await.unwrap();

            (exists, emb)
        });

        handles.push(handle);
    }

    // All readers should see the same data
    for handle in handles {
        let (exists, emb) = handle.await.unwrap();
        assert!(exists);
        assert!(emb.is_some());
        assert_eq!(emb.unwrap(), generate_embedding(42, 1));
    }

    // Cleanup
    postgres.delete("entity_concurrent_read").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_concurrent_mixed_operations() {
    let postgres = create_postgres_backend().await;
    let postgres = Arc::new(postgres);

    let mut handles = vec![];

    // 2 writers
    for i in 0..2 {
        let postgres_clone = Arc::clone(&postgres);

        let handle = tokio::spawn(async move {
            let emb = generate_embedding(42, i);
            let id = format!("writer_mixed{}", i);
            postgres_clone.store(&id, &emb).await.unwrap();
        });

        handles.push(handle);
    }

    // 2 readers
    for _ in 0..2 {
        let postgres_clone = Arc::clone(&postgres);

        let handle = tokio::spawn(async move {
            let _ = postgres_clone.count().await.unwrap();
        });

        handles.push(handle);
    }

    // 2 searchers
    for _ in 0..2 {
        let postgres_clone = Arc::clone(&postgres);

        let handle = tokio::spawn(async move {
            let query = generate_embedding(42, 999);
            let _ = postgres_clone.search(&query, 10).await.unwrap();
        });

        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify writers succeeded
    assert!(postgres.exists("writer_mixed0").await.unwrap());
    assert!(postgres.exists("writer_mixed1").await.unwrap());

    // Cleanup
    postgres.delete("writer_mixed0").await.unwrap();
    postgres.delete("writer_mixed1").await.unwrap();
}

// =============================================================================
// DST Test 4: Edge Cases and Boundary Conditions
// =============================================================================

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_large_batch_operations() {
    let postgres = create_postgres_backend().await;

    // Store 100 embeddings
    for i in 0..100 {
        let emb = generate_embedding(42, i);
        let id = format!("batch_entity{}", i);
        postgres.store(&id, &emb).await.unwrap();
    }

    assert_eq!(postgres.count().await.unwrap(), 100);

    // Search should return results
    let query = generate_embedding(42, 50);
    let results = postgres.search(&query, 10).await.unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 10);

    // Cleanup
    for i in 0..100 {
        let id = format!("batch_entity{}", i);
        postgres.delete(&id).await.unwrap();
    }
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_update_same_id_multiple_times() {
    let postgres = create_postgres_backend().await;

    // Update the same ID 10 times
    for i in 0..10 {
        let emb = generate_embedding(42, i);
        postgres.store("entity_multi_update", &emb).await.unwrap();
    }

    // Should still have only 1 entity
    assert_eq!(postgres.count().await.unwrap(), 1);

    // Should have the last embedding
    let emb_retrieved = postgres.get("entity_multi_update").await.unwrap().unwrap();
    let emb_expected = generate_embedding(42, 9);
    assert_eq!(emb_retrieved, emb_expected);

    // Cleanup
    postgres.delete("entity_multi_update").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_delete_nonexistent() {
    let postgres = create_postgres_backend().await;

    // Deleting non-existent entity should not error
    postgres.delete("nonexistent_pg").await.unwrap();

    assert!(!postgres.exists("nonexistent_pg").await.unwrap());
    assert_eq!(postgres.count().await.unwrap(), 0);
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_get_nonexistent() {
    let postgres = create_postgres_backend().await;

    let result = postgres.get("nonexistent_get_pg").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_search_limit_respected() {
    let postgres = create_postgres_backend().await;

    // Store 20 embeddings
    for i in 0..20 {
        let emb = generate_embedding(42, i);
        let id = format!("limit_entity{}", i);
        postgres.store(&id, &emb).await.unwrap();
    }

    // Search with limit 5
    let query = generate_embedding(42, 10);
    let results = postgres.search(&query, 5).await.unwrap();

    assert!(results.len() <= 5);

    // Cleanup
    for i in 0..20 {
        let id = format!("limit_entity{}", i);
        postgres.delete(&id).await.unwrap();
    }
}

// =============================================================================
// DST Test 5: pgvector-Specific Tests
// =============================================================================

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_pgvector_cosine_similarity() {
    let postgres = create_postgres_backend().await;

    // Store embeddings with known similarity
    let emb1 = vec![1.0, 0.0, 0.0]; // Padded to EMBEDDING_DIMENSIONS_COUNT
    let mut emb1_full = vec![0.0; EMBEDDING_DIMENSIONS_COUNT];
    emb1_full[0] = 1.0;

    let emb2 = vec![0.0, 1.0, 0.0]; // Orthogonal to emb1
    let mut emb2_full = vec![0.0; EMBEDDING_DIMENSIONS_COUNT];
    emb2_full[1] = 1.0;

    postgres.store("emb1_cos", &emb1_full).await.unwrap();
    postgres.store("emb2_cos", &emb2_full).await.unwrap();

    // Query with emb1 should rank emb1 higher
    let results = postgres.search(&emb1_full, 10).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "emb1_cos"); // Most similar
    assert!(results[0].score > results[1].score);

    // Cleanup
    postgres.delete("emb1_cos").await.unwrap();
    postgres.delete("emb2_cos").await.unwrap();
}

#[tokio::test]
#[ignore] // Requires Postgres + pgvector
async fn dst_postgres_deterministic_search_ranking() {
    let postgres = create_postgres_backend().await;

    // Store embeddings
    for i in 0..5 {
        let emb = generate_embedding(42, i);
        let id = format!("rank_entity{}", i);
        postgres.store(&id, &emb).await.unwrap();
    }

    // Run same search twice
    let query = generate_embedding(42, 2);
    let results1 = postgres.search(&query, 10).await.unwrap();
    let results2 = postgres.search(&query, 10).await.unwrap();

    // Results should be identical
    assert_eq!(results1.len(), results2.len());
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.id, r2.id);
        assert_eq!(r1.score, r2.score);
    }

    // Cleanup
    for i in 0..5 {
        let id = format!("rank_entity{}", i);
        postgres.delete(&id).await.unwrap();
    }
}
