//! Storage Backend Benchmarks
//!
//! Benchmarks for storage operations at various scales.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use umi_memory::dst::SimConfig;
use umi_memory::storage::{Entity, EntityType, SimStorageBackend, StorageBackend};

#[cfg(feature = "lance")]
use umi_memory::storage::LanceStorageBackend;

use std::time::Duration;

// =============================================================================
// Setup Helpers
// =============================================================================

/// Create test entities for benchmarking.
fn create_test_entities(count: usize) -> Vec<Entity> {
    (0..count)
        .map(|i| {
            Entity::new(
                EntityType::Note,
                format!("Test Entity {}", i),
                format!("This is test content for entity number {}. It contains some sample text to make the benchmark realistic.", i),
            )
        })
        .collect()
}

// =============================================================================
// SimStorageBackend Benchmarks
// =============================================================================

fn bench_sim_store_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_storage/store_entity");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let entities = create_test_entities(size);

            b.to_async(&rt).iter(|| async {
                let entity = &entities[0];
                black_box(storage.store_entity(entity).await.unwrap());
            });
        });
    }
    group.finish();
}

fn bench_sim_get_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_storage/get_entity");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let entities = create_test_entities(size);

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            let entity_id = entities[size / 2].id.clone();

            b.to_async(&rt).iter(|| async {
                black_box(storage.get_entity(&entity_id).await.unwrap());
            });
        });
    }
    group.finish();
}

fn bench_sim_delete_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_storage/delete_entity");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let entities = create_test_entities(size);

            // Pre-populate once
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            let entity_id = entities[0].id.clone();

            b.to_async(&rt).iter(|| async {
                // Delete and re-add to make benchmark repeatable
                storage.delete_entity(&entity_id).await.unwrap();
                storage.store_entity(&entities[0]).await.unwrap();
                black_box(());
            });
        });
    }
    group.finish();
}

fn bench_sim_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_storage/search");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let entities = create_test_entities(size);

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            b.to_async(&rt).iter(|| async {
                black_box(storage.search("test content", 10).await.unwrap());
            });
        });
    }
    group.finish();
}

fn bench_sim_count_entities(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_storage/count_entities");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let entities = create_test_entities(size);

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            b.to_async(&rt).iter(|| async {
                black_box(storage.count_entities(None).await.unwrap());
            });
        });
    }
    group.finish();
}

fn bench_sim_list_entities(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_storage/list_entities");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let entities = create_test_entities(size);

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            b.to_async(&rt).iter(|| async {
                black_box(storage.list_entities(None, 100, 0).await.unwrap());
            });
        });
    }
    group.finish();
}

// =============================================================================
// LanceStorageBackend Benchmarks
// =============================================================================

#[cfg(feature = "lance")]
fn bench_lance_store_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("lance_storage/store_entity");
    group.measurement_time(Duration::from_secs(15));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let entities = create_test_entities(size);

            b.to_async(&rt).iter_batched(
                || {
                    // Setup: Create fresh Lance storage for each iteration
                    let temp_dir = tempfile::tempdir().unwrap();
                    let storage =
                        rt.block_on(LanceStorageBackend::connect(temp_dir.path().to_str().unwrap())).unwrap();
                    (storage, temp_dir)
                },
                |(storage, _temp_dir)| async move {
                    let entity = &entities[0];
                    black_box(storage.store_entity(entity).await.unwrap());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

#[cfg(feature = "lance")]
fn bench_lance_get_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("lance_storage/get_entity");
    group.measurement_time(Duration::from_secs(15));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let entities = create_test_entities(size);
            let temp_dir = tempfile::tempdir().unwrap();
            let storage = rt.block_on(LanceStorageBackend::connect(temp_dir.path().to_str().unwrap())).unwrap();

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            let entity_id = entities[size / 2].id.clone();

            b.to_async(&rt).iter(|| async {
                black_box(storage.get_entity(&entity_id).await.unwrap());
            });
        });
    }
    group.finish();
}

#[cfg(feature = "lance")]
fn bench_lance_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("lance_storage/search");
    group.measurement_time(Duration::from_secs(15));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let entities = create_test_entities(size);
            let temp_dir = tempfile::tempdir().unwrap();
            let storage = rt.block_on(LanceStorageBackend::connect(temp_dir.path().to_str().unwrap())).unwrap();

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            b.to_async(&rt).iter(|| async {
                black_box(storage.search("test content", 10).await.unwrap());
            });
        });
    }
    group.finish();
}

#[cfg(feature = "lance")]
fn bench_lance_count_entities(c: &mut Criterion) {
    let mut group = c.benchmark_group("lance_storage/count_entities");
    group.measurement_time(Duration::from_secs(15));

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let entities = create_test_entities(size);
            let temp_dir = tempfile::tempdir().unwrap();
            let storage = rt.block_on(LanceStorageBackend::connect(temp_dir.path().to_str().unwrap())).unwrap();

            // Pre-populate storage
            rt.block_on(async {
                for entity in &entities {
                    storage.store_entity(entity).await.unwrap();
                }
            });

            b.to_async(&rt).iter(|| async {
                black_box(storage.count_entities(None).await.unwrap());
            });
        });
    }
    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    sim_benches,
    bench_sim_store_entity,
    bench_sim_get_entity,
    bench_sim_delete_entity,
    bench_sim_search,
    bench_sim_count_entities,
    bench_sim_list_entities,
);

// NOTE: Lance benchmarks are commented out for now due to complexity
// Will be added in Phase 2 when comparing with Kelpie

// #[cfg(feature = "lance")]
// criterion_group!(
//     lance_benches,
//     bench_lance_store_entity,
//     bench_lance_get_entity,
//     bench_lance_search,
//     bench_lance_count_entities,
// );

// #[cfg(feature = "lance")]
// criterion_main!(sim_benches, lance_benches);

// #[cfg(not(feature = "lance"))]
criterion_main!(sim_benches);
