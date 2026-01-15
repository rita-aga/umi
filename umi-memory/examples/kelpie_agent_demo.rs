//! Kelpie Agent Demo - Realistic Integration Example
//!
//! This example shows how a developer would integrate UMI with Kelpie
//! to build a conversational AI agent with memory.

use std::error::Error;
use umi_memory::{
    memory::{CoreMemory, KelpieBlockType, MemoryBlockType, WorkingMemory},
    storage::{Entity, EntityType},
};

/// A simple conversational agent using UMI with Kelpie integration
struct KelpieAgent {
    core: CoreMemory,
    session: WorkingMemory,
    #[allow(dead_code)]
    conversation_id: String,
}

impl KelpieAgent {
    fn new(conversation_id: String) -> Self {
        let mut core = CoreMemory::new();
        let session = WorkingMemory::new();

        // Initialize core memory with system prompt
        core.set_block(
            MemoryBlockType::System,
            "You are a helpful AI assistant. Be concise and friendly.",
        )
        .expect("Failed to set system block");
        core.set_block_importance(MemoryBlockType::System, 1.0)
            .expect("Failed to set importance");

        Self {
            core,
            session,
            conversation_id,
        }
    }

    /// Load user profile into core memory
    fn load_user_profile(&mut self, name: &str, preferences: &str) -> Result<(), Box<dyn Error>> {
        let profile = format!("User: {}\nPreferences: {}", name, preferences);
        self.core.set_block(MemoryBlockType::Human, profile)?;
        self.core
            .set_block_importance(MemoryBlockType::Human, 0.9)?;
        Ok(())
    }

    /// Update persona based on conversation style
    fn update_persona(&mut self, style: &str) -> Result<(), Box<dyn Error>> {
        let persona = format!("Conversation style: {}", style);
        self.core.set_block(MemoryBlockType::Persona, persona)?;
        self.core
            .set_block_importance(MemoryBlockType::Persona, 0.85)?;
        Ok(())
    }

    /// Add a fact to core memory
    fn remember_fact(&mut self, fact: &str) -> Result<(), Box<dyn Error>> {
        // Get existing facts or start fresh
        let existing = self.core.get_content(MemoryBlockType::Facts).unwrap_or("");
        let facts = if existing.is_empty() {
            fact.to_string()
        } else {
            format!("{}\n• {}", existing, fact)
        };

        self.core.set_block(MemoryBlockType::Facts, facts)?;
        self.core
            .set_block_importance(MemoryBlockType::Facts, 0.8)?;
        Ok(())
    }

    /// Add a goal/task
    fn add_goal(&mut self, goal: &str) -> Result<(), Box<dyn Error>> {
        let existing = self.core.get_content(MemoryBlockType::Goals).unwrap_or("");
        let goals = if existing.is_empty() {
            format!("• {}", goal)
        } else {
            format!("{}\n• {}", existing, goal)
        };

        self.core.set_block(MemoryBlockType::Goals, goals)?;
        self.core
            .set_block_importance(MemoryBlockType::Goals, 0.75)?;
        Ok(())
    }

    /// Process a conversation turn
    fn process_turn(
        &mut self,
        user_message: &str,
        ai_response: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Increment turn counter
        let turn_num = self.session.incr("turn_count", 1)?;

        // Append to conversation log
        let log_entry = format!(
            "Turn {}: User: {} | AI: {}\n",
            turn_num, user_message, ai_response
        );
        self.session
            .append("conversation_log", log_entry.as_bytes())?;

        // Track message lengths (simulated token counts)
        let tokens = (user_message.len() + ai_response.len()) as i64;
        self.session.incr("total_tokens", tokens)?;

        // Update scratch space with current context
        let scratch = format!(
            "Processing turn {}: User asked about memory systems",
            turn_num
        );
        self.core.set_block(MemoryBlockType::Scratch, scratch)?;
        self.core
            .set_block_importance(MemoryBlockType::Scratch, 0.5)?;

        // Refresh session TTL
        self.session.touch("turn_count")?;

        Ok(())
    }

    /// Get the current context for the LLM (Kelpie-compatible XML)
    fn get_context(&self) -> String {
        self.core.render()
    }

    /// Get conversation statistics
    fn get_stats(&self) -> ConversationStats {
        let turns = if let Some(bytes) = self.session.get("turn_count") {
            std::str::from_utf8(bytes)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        } else {
            0
        };

        let tokens = if let Some(bytes) = self.session.get("total_tokens") {
            std::str::from_utf8(bytes)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        } else {
            0
        };

        let log_size = self
            .session
            .get("conversation_log")
            .map(|b| b.len())
            .unwrap_or(0);

        ConversationStats {
            turns,
            tokens,
            log_size,
            core_bytes: self.core.used_bytes(),
            session_bytes: self.session.used_bytes(),
        }
    }

    /// Archive entities to long-term storage (using entity type mapping)
    fn archive_entities(&self, entities: Vec<Entity>) -> Vec<(Entity, KelpieBlockType)> {
        entities
            .into_iter()
            .map(|entity| {
                let kelpie_block = KelpieBlockType::from(entity.entity_type);
                (entity, kelpie_block)
            })
            .collect()
    }
}

#[derive(Debug)]
struct ConversationStats {
    turns: i64,
    tokens: i64,
    log_size: usize,
    core_bytes: usize,
    session_bytes: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🤖 Kelpie Agent Demo - Realistic Integration\n");
    println!("{}", "=".repeat(60));

    // =========================================================================
    // Setup Phase
    // =========================================================================
    println!("\n📋 Phase 1: Agent Setup");
    println!("{}", "-".repeat(60));

    let mut agent = KelpieAgent::new("conv-123-abc".to_string());
    agent.load_user_profile("Alice Chen", "Prefers technical details, Python developer")?;
    agent.update_persona("Technical and precise")?;

    println!("✓ Agent initialized");
    println!("✓ User profile loaded (importance: 0.9)");
    println!("✓ Persona configured (importance: 0.85)");

    // =========================================================================
    // Conversation Simulation
    // =========================================================================
    println!("\n💬 Phase 2: Conversation");
    println!("{}", "-".repeat(60));

    // Turn 1
    agent.process_turn(
        "What is UMI?",
        "UMI is a memory system for AI agents with entity extraction and dual retrieval.",
    )?;
    agent.remember_fact("User asked about UMI basics")?;
    println!("✓ Turn 1: Introduction to UMI");

    // Turn 2
    agent.process_turn(
        "How does it integrate with Kelpie?",
        "UMI provides Kelpie-compatible XML rendering, atomic KV operations, and entity type mapping.",
    )?;
    agent.remember_fact("User interested in Kelpie integration")?;
    agent.add_goal("Help user integrate UMI with their Kelpie agent")?;
    println!("✓ Turn 2: Kelpie integration details");

    // Turn 3
    agent.process_turn(
        "Can you show me the importance scoring?",
        "Sure! UMI allows setting importance (0.0-1.0) on memory blocks. System prompts are typically 1.0.",
    )?;
    agent.remember_fact("User wants to understand importance scoring")?;
    println!("✓ Turn 3: Importance scoring explanation");

    // Turn 4
    agent.process_turn(
        "What about atomic operations?",
        "UMI provides incr(), append(), and touch() for atomic updates to working memory with TTL.",
    )?;
    agent.add_goal("Provide example code for atomic operations")?;
    println!("✓ Turn 4: Atomic operations overview");

    // =========================================================================
    // Context Generation
    // =========================================================================
    println!("\n📝 Phase 3: Context Generation");
    println!("{}", "-".repeat(60));

    let context = agent.get_context();
    println!("Generated Kelpie-compatible XML context:");
    println!();

    // Show context with line numbers for clarity
    for (i, line) in context.lines().take(25).enumerate() {
        println!("{:3}: {}", i + 1, line);
    }

    if context.lines().count() > 25 {
        println!("... ({} more lines)", context.lines().count() - 25);
    }

    // Verify XML structure
    assert!(context.contains("<core_memory>"));
    assert!(context.contains("importance=\"1.00\""));
    assert!(context.contains("importance=\"0.50\""));
    println!("\n✓ XML structure validated");

    // =========================================================================
    // Statistics
    // =========================================================================
    println!("\n📊 Phase 4: Conversation Statistics");
    println!("{}", "-".repeat(60));

    let stats = agent.get_stats();
    println!("Session Statistics:");
    println!("  Turns:              {}", stats.turns);
    println!("  Tokens:             {}", stats.tokens);
    println!("  Log size:           {} bytes", stats.log_size);
    println!("  Core memory:        {} bytes", stats.core_bytes);
    println!("  Working memory:     {} bytes", stats.session_bytes);
    println!();

    // =========================================================================
    // Entity Archival
    // =========================================================================
    println!("💾 Phase 5: Entity Archival");
    println!("{}", "-".repeat(60));

    // Create sample entities that would be archived
    let entities_to_archive = vec![
        Entity::new(
            EntityType::Person,
            "Alice Chen".to_string(),
            "Python developer interested in AI memory systems".to_string(),
        ),
        Entity::new(
            EntityType::Topic,
            "Kelpie Integration".to_string(),
            "UMI provides Kelpie-compatible features".to_string(),
        ),
        Entity::new(
            EntityType::Project,
            "UMI-Kelpie Integration".to_string(),
            "Integrate UMI memory system with Kelpie agent framework".to_string(),
        ),
    ];

    let archived = agent.archive_entities(entities_to_archive);
    println!("Archiving {} entities:", archived.len());
    for (entity, kelpie_block) in archived {
        println!(
            "  • {:?} '{}' → {} block",
            entity.entity_type,
            entity.name,
            kelpie_block.as_str()
        );
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("{}", "=".repeat(60));
    println!("✅ Demo Complete - All Features Demonstrated:");
    println!("{}", "=".repeat(60));
    println!();
    println!("1. ✅ Core Memory with Importance Scoring");
    println!("   - System, Persona, Human, Facts, Goals, Scratch blocks");
    println!("   - Importance values from 0.5 to 1.0");
    println!();
    println!("2. ✅ Atomic KV Operations");
    println!("   - incr(): Message counting, token tracking");
    println!("   - append(): Conversation logging");
    println!("   - touch(): TTL refresh");
    println!();
    println!("3. ✅ Entity Type Mapping");
    println!("   - Person → facts");
    println!("   - Topic → facts");
    println!("   - Project → goals");
    println!();
    println!("4. ✅ Kelpie-Compatible XML Rendering");
    println!("   - Proper XML structure with importance attributes");
    println!("   - Deterministic render order");
    println!();
    println!("🚀 Ready for production use with Kelpie agents!");
    println!();

    Ok(())
}
