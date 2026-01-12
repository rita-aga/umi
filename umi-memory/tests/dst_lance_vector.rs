//! DST Tests for LanceVectorBackend
//!
//! TigerStyle: Test production backend behavior deterministically.
//!
//! These tests verify that LanceVectorBackend:
//! 1. Persists data across connections
//! 2. Behaves consistently with SimVectorBackend
//! 3. Handles concurrent operations correctly
//! 4. Recovers from failure scenarios

use std::sync::Arc;

use tempfile::TempDir;

use umi_memory::constants::EMBEDDING_DIMENSIONS_COUNT;
use umi_memory::storage::{SimVectorBackend, LanceVectorBackend, VectorBackend};

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a SimVectorBackend with deterministic seed.
fn create_sim_backend(seed: u64) -> SimVectorBackend {
    SimVectorBackend::new(seed)
}

/// Create a LanceVectorBackend with temp directory.
async fn create_lance_backend() -> (LanceVectorBackend, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let backend = LanceVectorBackend::connect(temp_dir.path().to_str().unwrap())
        .await
        .unwrap();
    (backend, temp_dir)
}

/// Create a LanceVectorBackend at a specific path.
async fn create_lance_backend_at_path(path: &str) -> LanceVectorBackend {
    LanceVectorBackend::connect(path).await.unwrap()
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
async fn dst_lance_persistence_store_and_retrieve() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();

    // Phase 1: Store data
    {
        let backend = create_lance_backend_at_path(path).await;

        let emb1 = generate_embedding(42, 1);
        let emb2 = generate_embedding(42, 2);

        backend.store("entity1", &emb1).await.unwrap();
        backend.store("entity2", &emb2).await.unwrap();

        assert!(backend.exists("entity1").await.unwrap());
        assert!(backend.exists("entity2").await.unwrap());
        assert_eq!(backend.count().await.unwrap(), 2);
    }
    // Backend dropped here - connection closed

    // Phase 2: Reopen and verify data persists
    {
        let backend = create_lance_backend_at_path(path).await;

        assert!(backend.exists("entity1").await.unwrap());
        assert!(backend.exists("entity2").await.unwrap());
        assert_eq!(backend.count().await.unwrap(), 2);

        let emb1_retrieved = backend.get("entity1").await.unwrap().unwrap();
        let emb1_expected = generate_embedding(42, 1);
        assert_eq!(emb1_retrieved, emb1_expected);
    }
}

#[tokio::test]
async fn dst_lance_persistence_update_across_restarts() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();

    // Phase 1: Store initial data
    {
        let backend = create_lance_backend_at_path(path).await;
        let emb1 = generate_embedding(42, 1);
        backend.store("entity1", &emb1).await.unwrap();
    }

    // Phase 2: Update data
    {
        let backend = create_lance_backend_at_path(path).await;
        let emb2 = generate_embedding(42, 2);
        backend.store("entity1", &emb2).await.unwrap();
    }

    // Phase 3: Verify updated data persists
    {
        let backend = create_lance_backend_at_path(path).await;
        let emb_retrieved = backend.get("entity1").await.unwrap().unwrap();
        let emb_expected = generate_embedding(42, 2);
        assert_eq!(emb_retrieved, emb_expected);
    }
}

#[tokio::test]
async fn dst_lance_persistence_delete_across_restarts() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();

    // Phase 1: Store data
    {
        let backend = create_lance_backend_at_path(path).await;
        let emb1 = generate_embedding(42, 1);
        let emb2 = generate_embedding(42, 2);
        backend.store("entity1", &emb1).await.unwrap();
        backend.store("entity2", &emb2).await.unwrap();
    }

    // Phase 2: Delete one entity
    {
        let backend = create_lance_backend_at_path(path).await;
        backend.delete("entity1").await.unwrap();
    }

    // Phase 3: Verify deletion persists
    {
        let backend = create_lance_backend_at_path(path).await;
        assert!(!backend.exists("entity1").await.unwrap());
        assert!(backend.exists("entity2").await.unwrap());
        assert_eq!(backend.count().await.unwrap(), 1);
    }
}

// =============================================================================
// DST Test 2: Behavior Consistency with SimVectorBackend
// =============================================================================

#[tokio::test]
async fn dst_lance_behavior_matches_sim_empty_search() {
    let sim = create_sim_backend(42);
    let (lance, _temp) = create_lance_backend().await;

    let query = generate_embedding(42, 999);

    let sim_results = sim.search(&query, 10).await.unwrap();
    let lance_results = lance.search(&query, 10).await.unwrap();

    assert_eq!(sim_results.len(), 0);
    assert_eq!(lance_results.len(), 0);
}

#[tokio::test]
async fn dst_lance_behavior_matches_sim_store_and_count() {
    let sim = create_sim_backend(42);
    let (lance, _temp) = create_lance_backend().await;

    let emb1 = generate_embedding(42, 1);
    let emb2 = generate_embedding(42, 2);

    sim.store("entity1", &emb1).await.unwrap();
    sim.store("entity2", &emb2).await.unwrap();

    lance.store("entity1", &emb1).await.unwrap();
    lance.store("entity2", &emb2).await.unwrap();

    assert_eq!(sim.count().await.unwrap(), lance.count().await.unwrap());
    assert_eq!(sim.exists("entity1").await.unwrap(), lance.exists("entity1").await.unwrap());
    assert_eq!(sim.exists("entity2").await.unwrap(), lance.exists("entity2").await.unwrap());
}

#[tokio::test]
async fn dst_lance_behavior_matches_sim_search_returns_stored() {
    let sim = create_sim_backend(42);
    let (lance, _temp) = create_lance_backend().await;

    let emb1 = generate_embedding(42, 1);

    sim.store("entity1", &emb1).await.unwrap();
    lance.store("entity1", &emb1).await.unwrap();

    let sim_results = sim.search(&emb1, 10).await.unwrap();
    let lance_results = lance.search(&emb1, 10).await.unwrap();

    assert_eq!(sim_results.len(), 1);
    assert_eq!(lance_results.len(), 1);
    assert_eq!(sim_results[0].id, "entity1");
    assert_eq!(lance_results[0].id, "entity1");
}

#[tokio::test]
async fn dst_lance_behavior_matches_sim_delete() {
    let sim = create_sim_backend(42);
    let (lance, _temp) = create_lance_backend().await;

    let emb1 = generate_embedding(42, 1);

    sim.store("entity1", &emb1).await.unwrap();
    lance.store("entity1", &emb1).await.unwrap();

    sim.delete("entity1").await.unwrap();
    lance.delete("entity1").await.unwrap();

    assert_eq!(sim.exists("entity1").await.unwrap(), false);
    assert_eq!(lance.exists("entity1").await.unwrap(), false);
    assert_eq!(sim.count().await.unwrap(), 0);
    assert_eq!(lance.count().await.unwrap(), 0);
}

// =============================================================================
// DST Test 3: Concurrent Operations
// =============================================================================

#[tokio::test]
async fn dst_lance_concurrent_stores() {
    let (lance, _temp) = create_lance_backend().await;

    // Pre-create table to avoid initialization conflicts
    let init_emb = generate_embedding(42, 999);
    lance.store("_init", &init_emb).await.unwrap();
    lance.delete("_init").await.unwrap();

    let lance = Arc::new(lance);
    let mut handles = vec![];

    // Spawn concurrent writes (retry logic handles conflicts)
    for i in 0..3 {
        let lance_clone = Arc::clone(&lance);

        let handle = tokio::spawn(async move {
            let emb = generate_embedding(42, i);
            let id = format!("entity{}", i);

            // Store concurrently
            lance_clone.store(&id, &emb).await.unwrap();
        });

        handles.push(handle);
    }

    // Wait for all stores to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all entities were stored
    assert_eq!(lance.count().await.unwrap(), 3);
    assert!(lance.exists("entity0").await.unwrap());
    assert!(lance.exists("entity1").await.unwrap());
    assert!(lance.exists("entity2").await.unwrap());
}

#[tokio::test]
async fn dst_lance_concurrent_reads() {
    let (lance, _temp) = create_lance_backend().await;

    // Setup: Store some data
    let emb1 = generate_embedding(42, 1);
    lance.store("entity1", &emb1).await.unwrap();

    let lance = Arc::new(lance);
    let mut handles = vec![];

    // Spawn 5 concurrent readers (reads don't conflict)
    for _ in 0..5 {
        let lance_clone = Arc::clone(&lance);

        let handle = tokio::spawn(async move {
            let exists = lance_clone.exists("entity1").await.unwrap();
            let emb = lance_clone.get("entity1").await.unwrap();

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
}

#[tokio::test]
async fn dst_lance_concurrent_mixed_operations() {
    let (lance, _temp) = create_lance_backend().await;

    // Pre-create table to avoid initialization conflicts
    let init_emb = generate_embedding(42, 999);
    lance.store("_init", &init_emb).await.unwrap();
    lance.delete("_init").await.unwrap();

    let lance = Arc::new(lance);
    let mut handles = vec![];

    // 2 writers (concurrent writes with retry logic)
    for i in 0..2 {
        let lance_clone = Arc::clone(&lance);

        let handle = tokio::spawn(async move {
            let emb = generate_embedding(42, i);
            let id = format!("writer{}", i);
            lance_clone.store(&id, &emb).await.unwrap();
        });

        handles.push(handle);
    }

    // 2 readers (reads don't conflict)
    for _ in 0..2 {
        let lance_clone = Arc::clone(&lance);

        let handle = tokio::spawn(async move {
            let _ = lance_clone.count().await.unwrap();
        });

        handles.push(handle);
    }

    // 2 searchers (searches don't conflict)
    for _ in 0..2 {
        let lance_clone = Arc::clone(&lance);

        let handle = tokio::spawn(async move {
            let query = generate_embedding(42, 999);
            let _ = lance_clone.search(&query, 10).await.unwrap();
        });

        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify writers succeeded
    assert!(lance.exists("writer0").await.unwrap());
    assert!(lance.exists("writer1").await.unwrap());
}

// =============================================================================
// DST Test 4: Edge Cases and Boundary Conditions
// =============================================================================

#[tokio::test]
async fn dst_lance_large_batch_operations() {
    let (lance, _temp) = create_lance_backend().await;

    // Store 100 embeddings
    for i in 0..100 {
        let emb = generate_embedding(42, i);
        let id = format!("entity{}", i);
        lance.store(&id, &emb).await.unwrap();
    }

    assert_eq!(lance.count().await.unwrap(), 100);

    // Search should return results
    let query = generate_embedding(42, 50);
    let results = lance.search(&query, 10).await.unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
}

#[tokio::test]
async fn dst_lance_update_same_id_multiple_times() {
    let (lance, _temp) = create_lance_backend().await;

    // Update the same ID 10 times
    for i in 0..10 {
        let emb = generate_embedding(42, i);
        lance.store("entity1", &emb).await.unwrap();
    }

    // Should still have only 1 entity
    assert_eq!(lance.count().await.unwrap(), 1);

    // Should have the last embedding
    let emb_retrieved = lance.get("entity1").await.unwrap().unwrap();
    let emb_expected = generate_embedding(42, 9);
    assert_eq!(emb_retrieved, emb_expected);
}

#[tokio::test]
async fn dst_lance_delete_nonexistent() {
    let (lance, _temp) = create_lance_backend().await;

    // Deleting non-existent entity should not error
    lance.delete("nonexistent").await.unwrap();

    assert!(!lance.exists("nonexistent").await.unwrap());
    assert_eq!(lance.count().await.unwrap(), 0);
}

#[tokio::test]
async fn dst_lance_get_nonexistent() {
    let (lance, _temp) = create_lance_backend().await;

    let result = lance.get("nonexistent").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn dst_lance_search_limit_respected() {
    let (lance, _temp) = create_lance_backend().await;

    // Store 20 embeddings
    for i in 0..20 {
        let emb = generate_embedding(42, i);
        let id = format!("entity{}", i);
        lance.store(&id, &emb).await.unwrap();
    }

    // Search with limit 5
    let query = generate_embedding(42, 10);
    let results = lance.search(&query, 5).await.unwrap();

    assert!(results.len() <= 5);
}

// =============================================================================
// DST Test 5: Deterministic Behavior
// =============================================================================

#[tokio::test]
async fn dst_lance_deterministic_storage_retrieval() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();

    // Run 1: Store data
    {
        let lance = create_lance_backend_at_path(path).await;
        for i in 0..10 {
            let emb = generate_embedding(42, i);
            let id = format!("entity{}", i);
            lance.store(&id, &emb).await.unwrap();
        }
    }

    // Run 2: Retrieve and verify
    let retrieved_run1 = {
        let lance = create_lance_backend_at_path(path).await;
        let mut retrieved = Vec::new();
        for i in 0..10 {
            let id = format!("entity{}", i);
            let emb = lance.get(&id).await.unwrap().unwrap();
            retrieved.push((id, emb));
        }
        retrieved
    };

    // Run 3: Retrieve again - should be identical
    let retrieved_run2 = {
        let lance = create_lance_backend_at_path(path).await;
        let mut retrieved = Vec::new();
        for i in 0..10 {
            let id = format!("entity{}", i);
            let emb = lance.get(&id).await.unwrap().unwrap();
            retrieved.push((id, emb));
        }
        retrieved
    };

    // Both runs should return identical data
    assert_eq!(retrieved_run1, retrieved_run2);
}

#[tokio::test]
async fn dst_lance_deterministic_count_after_operations() {
    let (lance, _temp) = create_lance_backend().await;

    // Perform sequence of operations
    let emb1 = generate_embedding(42, 1);
    let emb2 = generate_embedding(42, 2);

    lance.store("e1", &emb1).await.unwrap();
    assert_eq!(lance.count().await.unwrap(), 1);

    lance.store("e2", &emb2).await.unwrap();
    assert_eq!(lance.count().await.unwrap(), 2);

    lance.store("e1", &emb2).await.unwrap(); // Update
    assert_eq!(lance.count().await.unwrap(), 2);

    lance.delete("e1").await.unwrap();
    assert_eq!(lance.count().await.unwrap(), 1);

    lance.delete("e2").await.unwrap();
    assert_eq!(lance.count().await.unwrap(), 0);
}
