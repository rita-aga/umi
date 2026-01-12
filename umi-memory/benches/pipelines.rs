//! Full Pipeline Benchmarks
//!
//! Benchmarks for end-to-end remember/recall operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use umi_memory::dst::SimConfig;
use umi_memory::llm::SimLLMProvider;
use umi_memory::storage::SimStorageBackend;
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

use std::time::Duration;

// =============================================================================
// Test Data
// =============================================================================

fn sample_texts() -> Vec<&'static str> {
    vec![
        "Alice works at Acme Corporation as a Senior Engineer. She joined in 2021.",
        "Bob is working on the Phoenix project with a deadline of March 2024.",
        "The quarterly review meeting is scheduled for next Tuesday at 2pm.",
        "Implementation of the new authentication system is complete. Tests are passing.",
        "Customer feedback indicates the dashboard needs performance improvements.",
        "The team decided to use Rust for the memory library after evaluating Go and Python.",
    ]
}

fn sample_queries() -> Vec<&'static str> {
    vec![
        "Alice",
        "project deadline",
        "meeting schedule",
        "authentication",
        "performance",
        "Rust",
    ]
}

// =============================================================================
// Remember Benchmarks
// =============================================================================

fn bench_remember_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/remember_full");
    group.measurement_time(Duration::from_secs(10));

    let texts = sample_texts();

    group.bench_function("single", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.to_async(&rt).iter_batched(
            || {
                // Setup: Create fresh memory for each iteration
                let llm = SimLLMProvider::with_seed(42);
                let storage = SimStorageBackend::new(SimConfig::with_seed(42));
                let memory = Memory::new(llm, storage);
                (memory, texts[0])
            },
            |(mut memory, text)| async move {
                black_box(
                    memory
                        .remember(text, RememberOptions::default())
                        .await
                        .unwrap(),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_remember_no_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/remember_no_extraction");
    group.measurement_time(Duration::from_secs(10));

    let texts = sample_texts();

    group.bench_function("single", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.to_async(&rt).iter_batched(
            || {
                // Setup: Create fresh memory for each iteration
                let llm = SimLLMProvider::with_seed(42);
                let storage = SimStorageBackend::new(SimConfig::with_seed(42));
                let memory = Memory::new(llm, storage);
                (memory, texts[0])
            },
            |(mut memory, text)| async move {
                black_box(
                    memory
                        .remember(text, RememberOptions::new().without_extraction())
                        .await
                        .unwrap(),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_remember_no_evolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/remember_no_evolution");
    group.measurement_time(Duration::from_secs(10));

    let texts = sample_texts();

    group.bench_function("single", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.to_async(&rt).iter_batched(
            || {
                // Setup: Create fresh memory for each iteration
                let llm = SimLLMProvider::with_seed(42);
                let storage = SimStorageBackend::new(SimConfig::with_seed(42));
                let memory = Memory::new(llm, storage);
                (memory, texts[0])
            },
            |(mut memory, text)| async move {
                black_box(
                    memory
                        .remember(text, RememberOptions::new().without_evolution())
                        .await
                        .unwrap(),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_remember_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/remember_batch");
    group.measurement_time(Duration::from_secs(15));

    for count in [5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            let rt = tokio::runtime::Runtime::new().unwrap();

            b.to_async(&rt).iter_batched(
                || {
                    // Setup: Create fresh memory for each iteration
                    let llm = SimLLMProvider::with_seed(42);
                    let storage = SimStorageBackend::new(SimConfig::with_seed(42));
                    Memory::new(llm, storage)
                },
                |mut memory| async move {
                    let texts = sample_texts();
                    for i in 0..count {
                        let text = texts[i % texts.len()];
                        black_box(
                            memory
                                .remember(text, RememberOptions::default())
                                .await
                                .unwrap(),
                        );
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// =============================================================================
// Recall Benchmarks
// =============================================================================

fn bench_recall_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/recall_full");
    group.measurement_time(Duration::from_secs(10));

    let texts = sample_texts();
    let queries = sample_queries();

    group.bench_function("single_query", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Pre-populate memory once
        let llm = SimLLMProvider::with_seed(42);
        let storage = SimStorageBackend::new(SimConfig::with_seed(42));
        let mut memory = Memory::new(llm, storage);

        rt.block_on(async {
            for text in &texts {
                memory
                    .remember(text, RememberOptions::default())
                    .await
                    .unwrap();
            }
        });

        b.to_async(&rt).iter(|| async {
            black_box(
                memory
                    .recall(queries[0], RecallOptions::default())
                    .await
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_recall_fast_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/recall_fast_only");
    group.measurement_time(Duration::from_secs(10));

    let texts = sample_texts();
    let queries = sample_queries();

    group.bench_function("single_query", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        // Pre-populate memory once
        let llm = SimLLMProvider::with_seed(42);
        let storage = SimStorageBackend::new(SimConfig::with_seed(42));
        let mut memory = Memory::new(llm, storage);

        rt.block_on(async {
            for text in &texts {
                memory
                    .remember(text, RememberOptions::default())
                    .await
                    .unwrap();
            }
        });

        b.to_async(&rt).iter(|| async {
            black_box(
                memory
                    .recall(queries[0], RecallOptions::new().fast_only())
                    .await
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_recall_varying_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/recall_varying_limit");
    group.measurement_time(Duration::from_secs(10));

    let texts = sample_texts();
    let queries = sample_queries();

    for limit in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(limit), limit, |b, &limit| {
            let rt = tokio::runtime::Runtime::new().unwrap();

            // Pre-populate memory once
            let llm = SimLLMProvider::with_seed(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, storage);

            rt.block_on(async {
                for text in &texts {
                    memory
                        .remember(text, RememberOptions::default())
                        .await
                        .unwrap();
                }
            });

            b.to_async(&rt).iter(|| async {
                black_box(
                    memory
                        .recall(queries[0], RecallOptions::new().with_limit(limit).unwrap())
                        .await
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

fn bench_recall_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/recall_batch");
    group.measurement_time(Duration::from_secs(15));

    let texts = sample_texts();

    for count in [5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            let rt = tokio::runtime::Runtime::new().unwrap();

            // Pre-populate memory once
            let llm = SimLLMProvider::with_seed(42);
            let storage = SimStorageBackend::new(SimConfig::with_seed(42));
            let mut memory = Memory::new(llm, storage);

            rt.block_on(async {
                for text in &texts {
                    memory
                        .remember(text, RememberOptions::default())
                        .await
                        .unwrap();
                }
            });

            let query = "Alice"; // Use constant query to avoid closure issues

            b.to_async(&rt).iter(|| async {
                for _ in 0..count {
                    black_box(
                        memory
                            .recall(query, RecallOptions::default())
                            .await
                            .unwrap(),
                    );
                }
            });
        });
    }

    group.finish();
}

// =============================================================================
// Combined Workload Benchmarks
// =============================================================================

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/mixed_workload");
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("interleaved", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();

        b.to_async(&rt).iter_batched(
            || {
                // Setup: Create fresh memory for each iteration
                let llm = SimLLMProvider::with_seed(42);
                let storage = SimStorageBackend::new(SimConfig::with_seed(42));
                Memory::new(llm, storage)
            },
            |mut memory| async move {
                // Interleave remember and recall operations
                let texts = sample_texts();
                for i in 0..6 {
                    // Remember
                    black_box(
                        memory
                            .remember(texts[i], RememberOptions::default())
                            .await
                            .unwrap(),
                    );

                    // Recall
                    if i > 0 {
                        black_box(
                            memory
                                .recall("Alice", RecallOptions::default())
                                .await
                                .unwrap(),
                        );
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    remember_benches,
    bench_remember_full,
    bench_remember_no_extraction,
    bench_remember_no_evolution,
    bench_remember_batch,
);

criterion_group!(
    recall_benches,
    bench_recall_full,
    bench_recall_fast_only,
    bench_recall_varying_limit,
    bench_recall_batch,
);

criterion_group!(combined_benches, bench_mixed_workload,);

criterion_main!(remember_benches, recall_benches, combined_benches);
