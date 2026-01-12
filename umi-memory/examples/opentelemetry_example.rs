//! OpenTelemetry Integration Example
//!
//! This example demonstrates how to use UMI with OpenTelemetry for distributed tracing.
//!
//! # Prerequisites
//!
//! You need an OpenTelemetry collector running. The easiest way is with Docker:
//!
//! ```bash
//! docker run -d --name jaeger \
//!   -e COLLECTOR_OTLP_ENABLED=true \
//!   -p 4317:4317 \
//!   -p 4318:4318 \
//!   -p 16686:16686 \
//!   jaegertracing/all-in-one:latest
//! ```
//!
//! Then view traces at: http://localhost:16686
//!
//! # Run Example
//!
//! ```bash
//! cargo run --example opentelemetry_example --features opentelemetry
//! ```

use umi_memory::telemetry::{init_telemetry, TelemetryConfig};
use umi_memory::umi::{Memory, RecallOptions, RememberOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OpenTelemetry with OTLP exporter
    let telemetry_config = TelemetryConfig::builder()
        .service_name("umi-example")
        .endpoint("http://localhost:4317")
        .sampling_rate(1.0) // Sample all traces
        .build();

    // Initialize telemetry - keep guard alive for the program lifetime
    let _guard = init_telemetry(telemetry_config)?;

    println!("✅ OpenTelemetry initialized");
    println!("📊 Traces will be exported to http://localhost:4317");
    println!("🔍 View traces at http://localhost:16686 (Jaeger UI)\n");

    // Create memory with simulation providers
    let mut memory = Memory::sim(42);

    // Remember some information - these will generate spans
    println!("📝 Remembering information...");
    memory
        .remember(
            "Alice is a senior software engineer at Acme Corp",
            RememberOptions::default(),
        )
        .await?;

    memory
        .remember(
            "Bob leads the infrastructure team at Acme Corp",
            RememberOptions::default(),
        )
        .await?;

    memory
        .remember(
            "The Acme Corp main office is in San Francisco",
            RememberOptions::default(),
        )
        .await?;

    // Recall information - this will generate retrieval spans
    println!("🔍 Recalling information...\n");
    let results = memory
        .recall("Who works at Acme Corp?", RecallOptions::default())
        .await?;

    println!("📌 Found {} results:", results.len());
    for (i, entity) in results.iter().enumerate() {
        println!("  {}. {} - {}", i + 1, entity.name, entity.content);
    }

    println!("\n✨ Done! Check Jaeger UI for traces:");
    println!("   http://localhost:16686");
    println!("   Service: umi-example");
    println!("\n📋 Spans you'll see:");
    println!("   - remember (with text_len, stored_count, evolution_count)");
    println!("   - recall (with query_len, limit)");
    println!("   - extract (entity extraction)");
    println!("   - search (dual retrieval)");
    println!("   - store_entity (storage operations)");
    println!("   - complete (LLM calls with prompt_len)");
    println!("   - embed_batch (embedding generation)");
    println!("   - detect (evolution tracking)");

    // Keep the guard alive to flush remaining spans
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    Ok(())
}
