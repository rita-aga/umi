//! User Experience Test v3 - Debug score values
//!
//! Testing the retrieval scores directly

use std::env;
use umi_memory::embedding::EmbeddingProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Umi Memory - Score Investigation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let has_anthropic = env::var("ANTHROPIC_API_KEY").is_ok();
    let has_openai = env::var("OPENAI_API_KEY").is_ok();

    if !has_anthropic || !has_openai {
        println!("⚠️  Missing API keys");
        return Ok(());
    }

    let temp_dir = tempfile::tempdir()?;
    let db_path = temp_dir.path().join("test_memory");
    std::fs::create_dir_all(&db_path)?;

    use umi_memory::embedding::OpenAIEmbeddingProvider;
    use umi_memory::llm::AnthropicProvider;
    use umi_memory::retrieval::{DualRetriever, SearchOptions};
    use umi_memory::storage::{
        Entity, EntityType, LanceStorageBackend, LanceVectorBackend, StorageBackend, VectorBackend,
    };

    let llm = AnthropicProvider::new(env::var("ANTHROPIC_API_KEY").unwrap());
    let embedder = OpenAIEmbeddingProvider::new(env::var("OPENAI_API_KEY").unwrap());
    let vector = LanceVectorBackend::connect(db_path.join("vectors").to_str().unwrap()).await?;
    let storage = LanceStorageBackend::connect(db_path.join("entities").to_str().unwrap()).await?;

    // =========================================================================
    // Step 1: Store an entity directly with embedding
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("Step 1: Direct Storage with Embedding");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create entity content
    let content = "John Smith is a software developer at Google";

    // Generate embedding
    println!("   Generating embedding for: \"{}\"", content);
    let embedding = embedder.embed(content).await?;
    println!("   → Embedding dims: {}", embedding.len());

    // Create entity
    let mut entity = Entity::new(
        EntityType::Person,
        "John Smith".to_string(),
        content.to_string(),
    );
    entity.set_embedding(embedding.clone());

    let entity_id = entity.id.clone();
    println!("   → Entity ID: {}", &entity_id[..8]);

    // Store in both backends
    println!("\n   Storing in storage backend...");
    storage.store_entity(&entity).await?;
    println!("   → Stored in storage");

    println!("   Storing in vector backend...");
    vector.store(&entity_id, &embedding).await?;
    println!("   → Stored in vector");

    // =========================================================================
    // Step 2: Test vector search directly
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 2: Direct Vector Search");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Search with same embedding (should have score ~1.0)
    println!("   Searching with exact same embedding...");
    let results = vector.search(&embedding, 10).await?;
    println!("   → Found {} results:", results.len());
    for r in &results {
        println!("      - {} (score: {:.4})", &r.id[..8], r.score);
    }

    // Search with a query embedding
    let query = "Who is John Smith?";
    println!("\n   Searching with query embedding: \"{}\"", query);
    let query_embedding = embedder.embed(query).await?;
    let results2 = vector.search(&query_embedding, 10).await?;
    println!("   → Found {} results:", results2.len());
    for r in &results2 {
        println!("      - {} (score: {:.4})", &r.id[..8], r.score);
    }

    // =========================================================================
    // Step 3: Test DualRetriever search
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 3: DualRetriever Search");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Need to create new instances since we consumed them
    let vector2 = LanceVectorBackend::connect(db_path.join("vectors").to_str().unwrap()).await?;
    let storage2 = LanceStorageBackend::connect(db_path.join("entities").to_str().unwrap()).await?;
    let embedder2 = OpenAIEmbeddingProvider::new(env::var("OPENAI_API_KEY").unwrap());
    let llm2 = AnthropicProvider::new(env::var("ANTHROPIC_API_KEY").unwrap());

    let retriever = DualRetriever::new(
        Box::new(llm2),
        Box::new(embedder2),
        Box::new(vector2),
        Box::new(storage2),
    );

    println!("   Searching via DualRetriever: \"{}\"", query);
    let search_result = retriever.search(query, SearchOptions::default()).await?;

    println!(
        "   → Found {} results (min_score=0.3 filter applied)",
        search_result.len()
    );
    println!("   → Deep search used: {}", search_result.deep_search_used);
    println!(
        "   → Query variations: {:?}",
        search_result.query_variations
    );

    for entity in search_result.iter() {
        println!(
            "      - {} ({}): {}",
            entity.name,
            entity.entity_type.as_str(),
            entity.content
        );
    }

    // =========================================================================
    // Step 4: Cosine similarity analysis
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 4: Cosine Similarity Analysis");
    println!("═══════════════════════════════════════════════════════════════\n");

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    let queries = vec![
        "John Smith",
        "Who is John Smith?",
        "software developer",
        "Google",
        "programmer at tech company",
        "completely unrelated query about cooking",
    ];

    println!("   Content: \"{}\"", content);
    println!();

    for q in queries {
        let q_emb = embedder.embed(q).await?;
        let sim = cosine_similarity(&embedding, &q_emb);
        let status = if sim >= 0.3 {
            "✓ passes"
        } else {
            "✗ filtered"
        };
        println!("   Query: \"{}\"", q);
        println!("      Similarity: {:.4} {}", sim, status);
    }

    println!("\n✅ Investigation complete!\n");
    println!("   KEY FINDING: RETRIEVAL_MIN_SCORE_DEFAULT = 0.3");
    println!("   Results with similarity < 0.3 are filtered out.");

    Ok(())
}
