//! Kelpie Integration - User-Level Test
//!
//! This example demonstrates all three Kelpie integration features working together
//! in a realistic scenario. It simulates a conversational AI agent that:
//!
//! 1. Uses core memory with importance scoring for context windows
//! 2. Tracks session state with atomic KV operations
//! 3. Maps UMI entity types to Kelpie block types for XML rendering

use std::error::Error;
use umi_memory::{
    memory::{CoreMemory, KelpieBlockType, MemoryBlockType, WorkingMemory},
    storage::EntityType,
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== UMI-Kelpie Integration Test ===\n");

    // =========================================================================
    // Feature 1: Core Memory with Importance Scoring
    // =========================================================================
    println!("📝 Feature 1: Core Memory with Importance");
    println!("{}", "─".repeat(50));

    let mut core = CoreMemory::new();

    // Set up core memory blocks with varying importance
    core.set_block(
        MemoryBlockType::System,
        "You are a helpful AI assistant with expertise in software development.",
    )?;
    core.set_block_importance(MemoryBlockType::System, 1.0)?; // Highest importance

    core.set_block(
        MemoryBlockType::Persona,
        "You are friendly, concise, and prefer practical examples over theory.",
    )?;
    core.set_block_importance(MemoryBlockType::Persona, 0.9)?;

    core.set_block_with_label(
        MemoryBlockType::Human,
        "user_profile",
        "Name: Alex\nRole: Senior Engineer\nPreferences: Rust, TDD, functional programming",
    )?;
    core.set_block_importance(MemoryBlockType::Human, 0.85)?;

    core.set_block(
        MemoryBlockType::Facts,
        "Current project: Building a memory system called UMI\nTech stack: Rust, PyO3",
    )?;
    core.set_block_importance(MemoryBlockType::Facts, 0.75)?;

    core.set_block(
        MemoryBlockType::Goals,
        "1. Complete Kelpie integration\n2. Test all features from user perspective\n3. Write documentation",
    )?;
    core.set_block_importance(MemoryBlockType::Goals, 0.70)?;

    core.set_block(
        MemoryBlockType::Scratch,
        "Temporary notes: Testing importance scoring, checking XML rendering",
    )?;
    core.set_block_importance(MemoryBlockType::Scratch, 0.50)?; // Lowest importance

    // Verify importance values
    println!("✓ Created core memory with 6 blocks");
    println!(
        "  System:  {:.2}",
        core.get_block_importance(MemoryBlockType::System)?
    );
    println!(
        "  Persona: {:.2}",
        core.get_block_importance(MemoryBlockType::Persona)?
    );
    println!(
        "  Human:   {:.2}",
        core.get_block_importance(MemoryBlockType::Human)?
    );
    println!(
        "  Facts:   {:.2}",
        core.get_block_importance(MemoryBlockType::Facts)?
    );
    println!(
        "  Goals:   {:.2}",
        core.get_block_importance(MemoryBlockType::Goals)?
    );
    println!(
        "  Scratch: {:.2}",
        core.get_block_importance(MemoryBlockType::Scratch)?
    );

    // Render to XML (Kelpie-compatible format)
    println!("\n📄 Rendered XML (first 500 chars):");
    let xml = core.render();
    assert!(xml.contains("<core_memory>"));
    assert!(xml.contains("importance=\"1.00\""));
    assert!(xml.contains("importance=\"0.50\""));
    println!("{}", &xml[..xml.len().min(500)]);
    println!("...\n");

    println!("✅ Feature 1: Core memory with importance - VERIFIED\n");

    // =========================================================================
    // Feature 2: Atomic KV Operations
    // =========================================================================
    println!("🔢 Feature 2: Atomic KV Operations");
    println!("{}", "─".repeat(50));

    let mut working = WorkingMemory::new();

    // Test incr() - track conversation metrics
    println!("Testing incr() - atomic integer increments:");
    working.incr("message_count", 1)?;
    working.incr("message_count", 1)?;
    working.incr("message_count", 1)?;

    let count_bytes = working
        .get("message_count")
        .expect("message_count should exist");
    let count = std::str::from_utf8(count_bytes)?.parse::<i64>()?;
    assert_eq!(count, 3);
    println!("  ✓ message_count after 3 increments: {}", count);

    working.incr("tokens_used", 150)?;
    working.incr("tokens_used", 200)?;
    working.incr("tokens_used", 75)?;

    let tokens_bytes = working
        .get("tokens_used")
        .expect("tokens_used should exist");
    let tokens = std::str::from_utf8(tokens_bytes)?.parse::<i64>()?;
    assert_eq!(tokens, 425);
    println!("  ✓ tokens_used after 3 calls: {}", tokens);

    // Test negative increment
    working.incr("errors", -5)?; // Initialize to -5
    working.incr("errors", 3)?; // Add 3
    let errors_bytes = working.get("errors").expect("errors should exist");
    let errors = std::str::from_utf8(errors_bytes)?.parse::<i64>()?;
    assert_eq!(errors, -2);
    println!("  ✓ errors with negative deltas: {}", errors);

    // Test append() - build conversation log
    println!("\nTesting append() - atomic string appends:");
    working.append("conversation", b"[User]: What is UMI?\n")?;
    working.append(
        "conversation",
        b"[AI]: UMI is a memory system for agents.\n",
    )?;
    working.append("conversation", b"[User]: How does it work?\n")?;

    let log_bytes = working
        .get("conversation")
        .expect("conversation should exist");
    let log = std::str::from_utf8(log_bytes)?;
    assert!(log.contains("[User]: What is UMI?"));
    assert!(log.contains("[AI]: UMI is a memory system"));
    println!("  ✓ conversation log (first 100 chars):");
    println!("    {}", &log[..log.len().min(100)]);

    // Test touch() - refresh TTL
    println!("\nTesting touch() - TTL refresh:");
    working.set("session_id", b"abc-123-def", None)?;
    working.touch("session_id")?;
    assert!(working.exists("session_id"));
    println!("  ✓ session_id TTL refreshed successfully");

    // Test expired key behavior
    println!("\nTesting expired key handling:");
    working.set_clock_ms(0);
    working.set("temp_key", b"100", Some(1000))?; // 1 second TTL

    // Advance clock past expiration
    working.set_clock_ms(1500);

    // incr() should treat expired key as 0
    let result = working.incr("temp_key", 5)?;
    assert_eq!(result, 5);
    println!("  ✓ incr() on expired key treated as 0, result: {}", result);

    // Reset clock and test append
    working.set_clock_ms(0);
    working.set("expired_log", b"old data", Some(1000))?;
    working.set_clock_ms(1500);

    working.append("expired_log", b"new data")?;
    let new_log = std::str::from_utf8(
        working
            .get("expired_log")
            .expect("expired_log should exist"),
    )?;
    assert_eq!(new_log, "new data");
    println!("  ✓ append() on expired key started fresh: {}", new_log);

    println!("\n✅ Feature 2: Atomic KV operations - VERIFIED\n");

    // =========================================================================
    // Feature 3: Block Type Mapping
    // =========================================================================
    println!("🔄 Feature 3: Entity Type → Kelpie Block Type Mapping");
    println!("{}", "─".repeat(50));

    // Test all entity type mappings
    let mappings = vec![
        (EntityType::Self_, KelpieBlockType::Persona),
        (EntityType::Person, KelpieBlockType::Facts),
        (EntityType::Project, KelpieBlockType::Goals),
        (EntityType::Topic, KelpieBlockType::Facts),
        (EntityType::Note, KelpieBlockType::Scratch),
        (EntityType::Task, KelpieBlockType::Goals),
    ];

    println!("Testing EntityType → KelpieBlockType conversions:");
    for (entity_type, expected_kelpie) in &mappings {
        let kelpie = KelpieBlockType::from(*entity_type);
        assert_eq!(kelpie, *expected_kelpie);
        println!(
            "  ✓ {:?} → {} ({})",
            entity_type,
            kelpie.as_str(),
            if kelpie == *expected_kelpie {
                "✓"
            } else {
                "✗"
            }
        );
    }

    // Test reverse mapping (lossy)
    println!("\nTesting KelpieBlockType → EntityType (reverse, lossy):");
    let reverse_mappings = vec![
        (KelpieBlockType::Persona, EntityType::Self_),
        (KelpieBlockType::Facts, EntityType::Topic), // Default for facts
        (KelpieBlockType::Goals, EntityType::Project), // Default for goals
        (KelpieBlockType::Scratch, EntityType::Note),
    ];

    for (kelpie_type, expected_entity) in &reverse_mappings {
        let entity = EntityType::from(*kelpie_type);
        assert_eq!(entity, *expected_entity);
        println!("  ✓ {} → {:?} (default)", kelpie_type.as_str(), entity);
    }

    // Test round-trip for 1:1 mappings
    println!("\nTesting round-trip conversions (1:1 mappings only):");
    let self_type = EntityType::Self_;
    let kelpie = KelpieBlockType::from(self_type);
    let back = EntityType::from(kelpie);
    assert_eq!(back, EntityType::Self_);
    println!("  ✓ Self → Persona → Self (preserved)");

    let note_type = EntityType::Note;
    let kelpie = KelpieBlockType::from(note_type);
    let back = EntityType::from(kelpie);
    assert_eq!(back, EntityType::Note);
    println!("  ✓ Note → Scratch → Note (preserved)");

    // Test lossy round-trip
    println!("\nTesting lossy round-trip (N:1 mappings):");
    let person = EntityType::Person;
    let kelpie = KelpieBlockType::from(person);
    let back = EntityType::from(kelpie);
    println!("  ⚠ Person → Facts → {:?} (lossy, defaults to Topic)", back);
    assert_eq!(back, EntityType::Topic); // Lossy conversion

    println!("\n✅ Feature 3: Block type mapping - VERIFIED\n");

    // =========================================================================
    // Integration Test: All Features Together
    // =========================================================================
    println!("🎯 Integration Test: All Features Working Together");
    println!("{}", "─".repeat(50));

    // Simulate a realistic agent workflow
    println!("Simulating conversational AI agent with UMI-Kelpie integration...\n");

    // 1. Set up core memory with importance
    let mut agent_core = CoreMemory::new();
    agent_core.set_block(
        MemoryBlockType::System,
        "You are an expert Rust developer helping users build memory systems.",
    )?;
    agent_core.set_block_importance(MemoryBlockType::System, 0.95)?;

    // 2. Track session state with atomic operations
    let mut session = WorkingMemory::new();
    session.set_clock_ms(0);

    // Simulate conversation turns
    for turn in 1..=5 {
        // Increment message counter
        session.incr("turn_count", 1)?;

        // Append to interaction log
        let log_entry = format!("Turn {}: User asked about Kelpie integration\n", turn);
        session.append("interaction_log", log_entry.as_bytes())?;

        // Track tokens
        let tokens_used = 50 + (turn * 10); // Simulated
        session.incr("total_tokens", tokens_used)?;

        // Refresh session TTL
        session.touch("turn_count")?;
    }

    // 3. Use entity type mapping for archival storage
    let entity_types = vec![
        EntityType::Person,  // User profile
        EntityType::Project, // Current work
        EntityType::Task,    // Action items
    ];

    println!("Conversation summary after 5 turns:");
    let turn_count = std::str::from_utf8(session.get("turn_count").unwrap())?.parse::<i64>()?;
    let total_tokens = std::str::from_utf8(session.get("total_tokens").unwrap())?.parse::<i64>()?;
    println!("  - Turns: {}", turn_count);
    println!("  - Total tokens: {}", total_tokens);
    println!(
        "  - Log length: {} bytes",
        session.get("interaction_log").unwrap().len()
    );

    println!("\nEntity types to be archived:");
    for entity_type in entity_types {
        let kelpie_block = KelpieBlockType::from(entity_type);
        println!(
            "  - {:?} will be stored in Kelpie {} block",
            entity_type,
            kelpie_block.as_str()
        );
    }

    // Final verification
    assert_eq!(turn_count, 5);
    assert!(total_tokens > 0);
    assert!(session.get("interaction_log").unwrap().len() > 0);
    assert_eq!(
        agent_core.get_block_importance(MemoryBlockType::System)?,
        0.95
    );

    println!("\n✅ Integration test - ALL FEATURES WORKING TOGETHER\n");

    // =========================================================================
    // Edge Cases & Error Handling
    // =========================================================================
    println!("🧪 Testing Edge Cases");
    println!("{}", "─".repeat(50));

    // Test importance boundaries
    let mut edge_core = CoreMemory::new();
    edge_core.set_block(MemoryBlockType::System, "test")?;
    edge_core.set_block_importance(MemoryBlockType::System, 0.0)?; // Min
    assert_eq!(
        edge_core.get_block_importance(MemoryBlockType::System)?,
        0.0
    );
    edge_core.set_block_importance(MemoryBlockType::System, 1.0)?; // Max
    assert_eq!(
        edge_core.get_block_importance(MemoryBlockType::System)?,
        1.0
    );
    println!("  ✓ Importance boundaries (0.0, 1.0) work correctly");

    // Test overflow protection in incr
    let mut edge_working = WorkingMemory::new();
    edge_working.set("near_max", (i64::MAX - 10).to_string().as_bytes(), None)?;
    let result = edge_working.incr("near_max", 100);
    assert!(result.is_err()); // Should overflow
    println!("  ✓ incr() overflow protection works");

    // Test append size limit (max entry size is 64KB)
    let large_data = vec![b'x'; 60_000]; // 60KB
    edge_working.set("log", &large_data, None)?;
    let more_data = vec![b'y'; 10_000]; // Adding 10KB would exceed 64KB limit
    let result = edge_working.append("log", &more_data); // Would exceed limit
    assert!(result.is_err());
    println!("  ✓ append() size limit protection works");

    // Test touch on non-existent key
    let result = edge_working.touch("nonexistent");
    assert!(result.is_err());
    println!("  ✓ touch() error handling works");

    println!("\n✅ Edge cases handled correctly\n");

    // =========================================================================
    // Final Summary
    // =========================================================================
    println!("{}", "=".repeat(50));
    println!("🎉 ALL KELPIE INTEGRATION FEATURES VERIFIED!");
    println!("{}", "=".repeat(50));
    println!();
    println!("Summary:");
    println!("  ✅ Phase 1: XML rendering with importance (0.0-1.0)");
    println!("  ✅ Phase 2: Atomic KV operations (incr, append, touch)");
    println!("  ✅ Phase 3: Entity type → Kelpie block mapping");
    println!("  ✅ Integration: All features working together");
    println!("  ✅ Edge cases: Proper error handling");
    println!();
    println!("🚀 UMI is fully Kelpie-compatible!");
    println!();

    Ok(())
}
