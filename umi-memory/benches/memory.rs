//! Memory API Benchmarks
//!
//! Benchmarks for core Memory operations using Criterion.
//!
//! These benchmarks validate performance of:
//! - Remember operations (single and multiple entities)
//! - Recall operations with pre-populated data
//!
//! Run with: cargo bench --bench memory

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tokio::runtime::Runtime;
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

// =============================================================================
// Remember Benchmarks
// =============================================================================

fn bench_remember_single_entity(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("remember_single_entity", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                memory
                    .remember(
                        black_box("Alice is a software engineer at Acme Corp"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap()
            })
        });
    });
}

fn bench_remember_multiple_entities(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("remember_multiple_entities", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                memory
                    .remember(
                        black_box("Alice and Bob work together at Acme Corp on the new project"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap()
            })
        });
    });
}

fn bench_remember_complex_text(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let complex_text = "Alice manages the engineering team at Acme Corp. \
                        She works closely with Bob, the lead developer, \
                        and Carol, the product manager. Together they are \
                        building a new authentication system for the platform.";

    c.bench_function("remember_complex_text", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                memory
                    .remember(black_box(complex_text), RememberOptions::default())
                    .await
                    .unwrap()
            })
        });
    });
}

// =============================================================================
// Recall Benchmarks
// =============================================================================

fn bench_recall_with_results(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("recall_with_100_entities", |b| {
        // Setup: pre-populate memory once
        let mut memory = Memory::sim(42);
        rt.block_on(async {
            for i in 0..100 {
                memory
                    .remember(
                        &format!("Person {} is a software engineer at Company {}", i, i % 10),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap();
            }
        });

        // Benchmark: recall from pre-populated memory
        b.to_async(&rt).iter(|| async {
            memory
                .recall(black_box("engineer"), RecallOptions::default())
                .await
                .unwrap()
        });
    });
}

fn bench_recall_semantic_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("recall_semantic_query", |b| {
        // Setup: pre-populate with diverse data
        let mut memory = Memory::sim(42);
        rt.block_on(async {
            memory
                .remember("Alice is a software engineer", RememberOptions::default())
                .await
                .unwrap();
            memory
                .remember("Bob works as a developer", RememberOptions::default())
                .await
                .unwrap();
            memory
                .remember("Carol is a product manager", RememberOptions::default())
                .await
                .unwrap();
            memory
                .remember("The weather is sunny today", RememberOptions::default())
                .await
                .unwrap();
        });

        // Benchmark: semantic search
        b.to_async(&rt).iter(|| async {
            memory
                .recall(black_box("Who are the programmers?"), RecallOptions::default())
                .await
                .unwrap()
        });
    });
}

fn bench_recall_with_limit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("recall_with_limit");

    // Setup: pre-populate memory once
    let mut memory = Memory::sim(42);
    rt.block_on(async {
        for i in 0..200 {
            memory
                .remember(
                    &format!("Entity {} is a test item", i),
                    RememberOptions::default(),
                )
                .await
                .unwrap();
        }
    });

    // Benchmark different limits
    for limit in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(limit), limit, |b, &limit| {
            b.to_async(&rt).iter(|| async {
                memory
                    .recall(
                        black_box("entity"),
                        RecallOptions::default().with_limit(limit),
                    )
                    .await
                    .unwrap()
            });
        });
    }

    group.finish();
}

// =============================================================================
// Full Workflow Benchmarks
// =============================================================================

fn bench_remember_recall_cycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("remember_recall_cycle", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                // Remember
                memory
                    .remember(
                        black_box("Alice works at Acme Corp"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap();

                // Recall
                memory
                    .recall(black_box("Alice"), RecallOptions::default())
                    .await
                    .unwrap()
            })
        });
    });
}

fn bench_evolution_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("evolution_detection", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                // First remember
                memory
                    .remember(
                        black_box("Alice works at Acme Corp"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap();

                // Second remember (potential evolution)
                memory
                    .remember(
                        black_box("Alice now works at TechCo"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap()
            })
        });
    });
}

// =============================================================================
// Configuration Effects Benchmarks
// =============================================================================

fn bench_remember_without_embeddings(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("config_effects");

    // With embeddings (default)
    group.bench_function("with_embeddings", |b| {
        b.iter(|| {
            let mut memory = Memory::sim(42);
            rt.block_on(async {
                memory
                    .remember(
                        black_box("Test entity"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap()
            })
        });
    });

    // Without embeddings
    group.bench_function("without_embeddings", |b| {
        b.iter(|| {
            let config = umi_memory::umi::MemoryConfig::default().without_embeddings();
            let mut memory = Memory::sim_with_config(42, config);
            rt.block_on(async {
                memory
                    .remember(
                        black_box("Test entity"),
                        RememberOptions::default(),
                    )
                    .await
                    .unwrap()
            })
        });
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_remember_single_entity,
    bench_remember_multiple_entities,
    bench_remember_complex_text,
    bench_recall_with_results,
    bench_recall_semantic_query,
    bench_recall_with_limit,
    bench_remember_recall_cycle,
    bench_evolution_detection,
    bench_remember_without_embeddings,
);
criterion_main!(benches);
