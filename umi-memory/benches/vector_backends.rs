//! Vector Backend Performance Benchmarks
//!
//! Compares performance across SimVectorBackend, LanceVectorBackend, and PostgresVectorBackend.
//!
//! Run with: `cargo bench --bench vector_backends`
//! With specific backend: `cargo bench --bench vector_backends --features lance`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use umi_memory::constants::EMBEDDING_DIMENSIONS_COUNT;
use umi_memory::storage::{SimVectorBackend, VectorBackend};

#[cfg(feature = "lance")]
use umi_memory::storage::LanceVectorBackend;

// =============================================================================
// Helper Functions
// =============================================================================

/// Generate deterministic embedding from seed and index.
fn generate_embedding(seed: u64, index: usize) -> Vec<f32> {
    let mut emb = vec![0.0; EMBEDDING_DIMENSIONS_COUNT];
    for i in 0..EMBEDDING_DIMENSIONS_COUNT {
        emb[i] = ((seed + i as u64 + index as u64) % 1000) as f32 / 1000.0;
    }
    emb
}

/// Create SimVectorBackend for baseline.
fn create_sim_backend() -> SimVectorBackend {
    SimVectorBackend::new(42)
}

/// Create LanceVectorBackend for benchmarking.
#[cfg(feature = "lance")]
async fn create_lance_backend() -> (LanceVectorBackend, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let backend = LanceVectorBackend::connect(temp_dir.path().to_str().unwrap())
        .await
        .unwrap();
    (backend, temp_dir)
}

// =============================================================================
// Benchmark: Single Insert
// =============================================================================

fn bench_single_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_single_insert");
    group.measurement_time(Duration::from_secs(10));

    // Baseline: SimVectorBackend
    group.bench_function("sim", |b| {
        let backend = create_sim_backend();
        let emb = generate_embedding(42, 0);

        b.to_async(&rt).iter(|| async {
            backend.store("entity_bench", &emb).await.unwrap();
        });
    });

    // LanceVectorBackend - include setup overhead in benchmark
    #[cfg(feature = "lance")]
    group.bench_function("lance", |b| {
        b.to_async(&rt).iter(|| async {
            let (backend, _temp) = create_lance_backend().await;
            let emb = generate_embedding(42, 0);
            backend.store("entity_bench", &emb).await.unwrap();
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: Batch Insert
// =============================================================================

fn bench_batch_insert(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_batch_insert");
    group.measurement_time(Duration::from_secs(15));

    for batch_size in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        // Baseline: SimVectorBackend
        group.bench_with_input(
            BenchmarkId::new("sim", batch_size),
            batch_size,
            |b, &size| {
                let backend = create_sim_backend();

                b.to_async(&rt).iter(|| async {
                    for i in 0..size {
                        let emb = generate_embedding(42, i);
                        let id = format!("entity_{}", i);
                        backend.store(&id, &emb).await.unwrap();
                    }
                });
            },
        );

        // LanceVectorBackend - include setup overhead in benchmark
        #[cfg(feature = "lance")]
        group.bench_with_input(
            BenchmarkId::new("lance", batch_size),
            batch_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async move {
                    let (backend, _temp) = create_lance_backend().await;
                    for i in 0..size {
                        let emb = generate_embedding(42, i);
                        let id = format!("entity_{}", i);
                        backend.store(&id, &emb).await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: Search
// =============================================================================

fn bench_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_search");
    group.measurement_time(Duration::from_secs(10));

    // Setup: Prepopulate backends with data
    let sim_backend = {
        let backend = create_sim_backend();
        rt.block_on(async {
            for i in 0..100 {
                let emb = generate_embedding(42, i);
                let id = format!("entity_{}", i);
                backend.store(&id, &emb).await.unwrap();
            }
        });
        backend
    };

    #[cfg(feature = "lance")]
    let (lance_backend, _lance_temp) = rt.block_on(async {
        let (backend, temp) = create_lance_backend().await;
        for i in 0..100 {
            let emb = generate_embedding(42, i);
            let id = format!("entity_{}", i);
            backend.store(&id, &emb).await.unwrap();
        }
        (backend, temp)
    });

    for limit in [1, 5, 10, 20].iter() {
        group.throughput(Throughput::Elements(*limit as u64));

        // Baseline: SimVectorBackend
        group.bench_with_input(BenchmarkId::new("sim", limit), limit, |b, &lim| {
            let query = generate_embedding(42, 50);

            b.to_async(&rt)
                .iter(|| async { sim_backend.search(&query, lim).await.unwrap() });
        });

        // LanceVectorBackend
        #[cfg(feature = "lance")]
        group.bench_with_input(BenchmarkId::new("lance", limit), limit, |b, &lim| {
            let query = generate_embedding(42, 50);

            b.to_async(&rt)
                .iter(|| async { lance_backend.search(&query, lim).await.unwrap() });
        });
    }

    group.finish();
}

// =============================================================================
// Benchmark: Update (Upsert)
// =============================================================================

fn bench_update(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_update");
    group.measurement_time(Duration::from_secs(10));

    // Baseline: SimVectorBackend
    group.bench_function("sim", |b| {
        let backend = create_sim_backend();
        let emb1 = generate_embedding(42, 1);

        // Pre-insert
        rt.block_on(async {
            backend.store("entity_update", &emb1).await.unwrap();
        });

        let emb2 = generate_embedding(42, 2);

        b.to_async(&rt).iter(|| async {
            backend.store("entity_update", &emb2).await.unwrap();
        });
    });

    // LanceVectorBackend - include pre-insert and update in benchmark
    #[cfg(feature = "lance")]
    group.bench_function("lance", |b| {
        b.to_async(&rt).iter(|| async {
            let (backend, _temp) = create_lance_backend().await;
            let emb1 = generate_embedding(42, 1);
            backend.store("entity_update", &emb1).await.unwrap();

            let emb2 = generate_embedding(42, 2);
            backend.store("entity_update", &emb2).await.unwrap();
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: CRUD Operations
// =============================================================================

fn bench_crud_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_crud");
    group.measurement_time(Duration::from_secs(10));

    // Baseline: SimVectorBackend - exists()
    group.bench_function("sim_exists", |b| {
        let backend = create_sim_backend();
        let emb = generate_embedding(42, 0);

        rt.block_on(async {
            backend.store("entity_crud", &emb).await.unwrap();
        });

        b.to_async(&rt)
            .iter(|| async { backend.exists("entity_crud").await.unwrap() });
    });

    // Baseline: SimVectorBackend - get()
    group.bench_function("sim_get", |b| {
        let backend = create_sim_backend();
        let emb = generate_embedding(42, 0);

        rt.block_on(async {
            backend.store("entity_crud", &emb).await.unwrap();
        });

        b.to_async(&rt)
            .iter(|| async { backend.get("entity_crud").await.unwrap() });
    });

    // Baseline: SimVectorBackend - delete()
    group.bench_function("sim_delete", |b| {
        b.to_async(&rt).iter_with_setup(
            || {
                let backend = create_sim_backend();
                let emb = generate_embedding(42, 0);
                let rt2 = Runtime::new().unwrap();
                rt2.block_on(async {
                    backend.store("entity_delete", &emb).await.unwrap();
                });
                backend
            },
            |backend| async move {
                backend.delete("entity_delete").await.unwrap();
            },
        );
    });

    #[cfg(feature = "lance")]
    {
        // LanceVectorBackend - exists() - include store setup in benchmark
        group.bench_function("lance_exists", |b| {
            b.to_async(&rt).iter(|| async {
                let (backend, _temp) = create_lance_backend().await;
                let emb = generate_embedding(42, 0);
                backend.store("entity_crud", &emb).await.unwrap();
                backend.exists("entity_crud").await.unwrap()
            });
        });

        // LanceVectorBackend - get() - include store setup in benchmark
        group.bench_function("lance_get", |b| {
            b.to_async(&rt).iter(|| async {
                let (backend, _temp) = create_lance_backend().await;
                let emb = generate_embedding(42, 0);
                backend.store("entity_crud", &emb).await.unwrap();
                backend.get("entity_crud").await.unwrap()
            });
        });

        // LanceVectorBackend - delete() - include store setup in benchmark
        group.bench_function("lance_delete", |b| {
            b.to_async(&rt).iter(|| async {
                let (backend, _temp) = create_lance_backend().await;
                let emb = generate_embedding(42, 0);
                backend.store("entity_delete", &emb).await.unwrap();
                backend.delete("entity_delete").await.unwrap();
            });
        });
    }

    group.finish();
}

// =============================================================================
// Benchmark Groups
// =============================================================================

criterion_group!(
    benches,
    bench_single_insert,
    bench_batch_insert,
    bench_search,
    bench_update,
    bench_crud_operations,
);

criterion_main!(benches);
