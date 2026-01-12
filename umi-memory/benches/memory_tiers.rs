//! Memory Tier Benchmarks
//!
//! Benchmarks for CoreMemory and WorkingMemory operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use umi_memory::memory::{CoreMemory, MemoryBlockType, WorkingMemory};

use std::time::Duration;

// =============================================================================
// CoreMemory Benchmarks
// =============================================================================

fn bench_core_memory_set_block(c: &mut Criterion) {
    c.bench_function("core_memory/set_block", |b| {
        let mut core = CoreMemory::new();
        let content = "This is a test block with some content that simulates real usage.";

        b.iter(|| {
            black_box(
                core.set_block(MemoryBlockType::Human, content)
                    .unwrap()
            );
        });
    });
}

fn bench_core_memory_get_block(c: &mut Criterion) {
    c.bench_function("core_memory/get_block", |b| {
        let mut core = CoreMemory::new();
        core.set_block(
            MemoryBlockType::Human,
            "This is a test block with some content.",
        )
        .unwrap();

        b.iter(|| {
            black_box(core.get_block(MemoryBlockType::Human));
        });
    });
}

fn bench_core_memory_remove_block(c: &mut Criterion) {
    c.bench_function("core_memory/remove_block", |b| {
        b.iter_batched(
            || {
                // Setup: Create a core memory with a block
                let mut core = CoreMemory::new();
                core.set_block(
                    MemoryBlockType::Human,
                    "This is a test block with some content.",
                )
                .unwrap();
                core
            },
            |mut core| {
                black_box(core.remove_block(MemoryBlockType::Human).ok());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_core_memory_render(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_memory/render");
    group.measurement_time(Duration::from_secs(5));

    // Test with varying numbers of blocks
    for block_count in [1, 3, 5].iter() {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(block_count),
            block_count,
            |b, &block_count| {
                let mut core = CoreMemory::new();
                let block_types = [
                    MemoryBlockType::System,
                    MemoryBlockType::Persona,
                    MemoryBlockType::Human,
                    MemoryBlockType::Facts,
                    MemoryBlockType::Goals,
                ];

                for i in 0..block_count {
                    core.set_block(
                        block_types[i],
                        &format!("Block {} content with some text to make it realistic.", i),
                    )
                    .unwrap();
                }

                b.iter(|| {
                    black_box(core.render());
                });
            },
        );
    }
    group.finish();
}

fn bench_core_memory_used_bytes(c: &mut Criterion) {
    c.bench_function("core_memory/used_bytes", |b| {
        let mut core = CoreMemory::new();
        core.set_block(MemoryBlockType::System, "System content")
            .unwrap();
        core.set_block(MemoryBlockType::Human, "Human content")
            .unwrap();
        core.set_block(MemoryBlockType::Facts, "Facts content")
            .unwrap();

        b.iter(|| {
            black_box(core.used_bytes());
        });
    });
}

// =============================================================================
// WorkingMemory Benchmarks
// =============================================================================

fn bench_working_memory_set(c: &mut Criterion) {
    c.bench_function("working_memory/set", |b| {
        let mut working = WorkingMemory::new();
        let key = "test_key";
        let value = b"This is a test value with some content.";

        b.iter(|| {
            black_box(working.set(key, value, None).unwrap());
        });
    });
}

fn bench_working_memory_get(c: &mut Criterion) {
    c.bench_function("working_memory/get", |b| {
        let mut working = WorkingMemory::new();
        working.set("test_key", b"Test value", None).unwrap();

        b.iter(|| {
            black_box(working.get("test_key"));
        });
    });
}

fn bench_working_memory_contains_key(c: &mut Criterion) {
    c.bench_function("working_memory/contains_key", |b| {
        let mut working = WorkingMemory::new();
        working.set("test_key", b"Test value", None).unwrap();

        b.iter(|| {
            black_box(working.exists("test_key"));
        });
    });
}

fn bench_working_memory_cleanup(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_memory/cleanup");
    group.measurement_time(Duration::from_secs(5));

    // Test cleanup with varying numbers of entries
    for entry_count in [10, 100, 1000].iter() {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(entry_count),
            entry_count,
            |b, &entry_count| {
                b.iter_batched(
                    || {
                        // Setup: Create working memory with many entries
                        let mut working = WorkingMemory::new();
                        for i in 0..entry_count {
                            working
                                .set(&format!("key_{}", i), format!("value_{}", i).as_bytes(), None)
                                .unwrap();
                        }
                        working
                    },
                    |mut working| {
                        black_box(working.cleanup_expired());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_working_memory_bulk_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("working_memory/bulk_set");
    group.measurement_time(Duration::from_secs(10));

    for count in [10, 100, 1000].iter() {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(count),
            count,
            |b, &count| {
                b.iter_batched(
                    || {
                        // Setup: Create fresh working memory
                        WorkingMemory::new()
                    },
                    |mut working| {
                        for i in 0..count {
                            black_box(
                                working
                                    .set(&format!("key_{}", i), format!("value_{}", i).as_bytes(), None)
                                    .unwrap(),
                            );
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_working_memory_count(c: &mut Criterion) {
    c.bench_function("working_memory/count", |b| {
        let mut working = WorkingMemory::new();
        for i in 0..100 {
            working
                .set(&format!("key_{}", i), format!("value_{}", i).as_bytes(), None)
                .unwrap();
        }

        b.iter(|| {
            black_box(working.entry_count());
        });
    });
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    core_memory_benches,
    bench_core_memory_set_block,
    bench_core_memory_get_block,
    bench_core_memory_remove_block,
    bench_core_memory_render,
    bench_core_memory_used_bytes,
);

criterion_group!(
    working_memory_benches,
    bench_working_memory_set,
    bench_working_memory_get,
    bench_working_memory_contains_key,
    bench_working_memory_cleanup,
    bench_working_memory_bulk_operations,
    bench_working_memory_count,
);

criterion_main!(core_memory_benches, working_memory_benches);
