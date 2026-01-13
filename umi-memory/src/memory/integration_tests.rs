//! Integration Tests for Kelpie Features
//!
//! `TigerStyle`: Test all features together with DST and fault injection.
//!
//! Tests the integration of:
//! 1. CoreMemory XML rendering with importance
//! 2. WorkingMemory atomic operations (incr, append, touch)
//! 3. KelpieBlockType mapping between UMI and Kelpie types

#[cfg(test)]
mod tests {
    use crate::memory::{CoreMemory, KelpieBlockType, MemoryBlockType, WorkingMemory};
    use crate::storage::EntityType;

    /// Integration test: All Kelpie features working together.
    ///
    /// TigerStyle: Tests the complete workflow with all 3 features.
    #[test]
    fn test_kelpie_integration_full_workflow() {
        // Phase 1: CoreMemory with importance
        let mut core = CoreMemory::new();

        // Set blocks with different importance levels
        core.set_block(MemoryBlockType::System, "You are a helpful AI assistant.")
            .unwrap();
        core.set_block_importance(MemoryBlockType::System, 1.0)
            .unwrap();

        core.set_block(MemoryBlockType::Persona, "I am friendly and concise.")
            .unwrap();
        core.set_block_importance(MemoryBlockType::Persona, 0.95)
            .unwrap();

        core.set_block(MemoryBlockType::Human, "User prefers technical details.")
            .unwrap();
        core.set_block_importance(MemoryBlockType::Human, 0.85)
            .unwrap();

        // Verify importance values
        assert_eq!(
            core.get_block_importance(MemoryBlockType::System).unwrap(),
            1.0
        );
        assert_eq!(
            core.get_block_importance(MemoryBlockType::Persona).unwrap(),
            0.95
        );

        // Verify XML rendering includes importance
        let xml = core.render();
        assert!(xml.contains(r#"type="system" importance="1.00""#));
        assert!(xml.contains(r#"type="persona" importance="0.95""#));
        assert!(xml.contains(r#"type="human" importance="0.85""#));

        // Phase 2: WorkingMemory atomic operations
        let mut working = WorkingMemory::new();

        // Test incr operation
        let counter1 = working.incr("request_count", 1).unwrap();
        assert_eq!(counter1, 1);
        let counter2 = working.incr("request_count", 5).unwrap();
        assert_eq!(counter2, 6);

        // Test append operation
        working.append("log", b"Event1 ").unwrap();
        working.append("log", b"Event2 ").unwrap();
        working.append("log", b"Event3").unwrap();
        assert_eq!(working.get("log").unwrap(), b"Event1 Event2 Event3");

        // Test touch operation (refresh TTL)
        working.set("session_id", b"abc123", None).unwrap();
        assert!(working.exists("session_id"));
        working.touch("session_id").unwrap();

        // Phase 3: KelpieBlockType mapping
        // Test UMI EntityType → Kelpie block type conversion
        let self_entity = EntityType::Self_;
        let kelpie_persona = KelpieBlockType::from(self_entity);
        assert_eq!(kelpie_persona, KelpieBlockType::Persona);

        let person_entity = EntityType::Person;
        let kelpie_facts = KelpieBlockType::from(person_entity);
        assert_eq!(kelpie_facts, KelpieBlockType::Facts);

        let project_entity = EntityType::Project;
        let kelpie_goals = KelpieBlockType::from(project_entity);
        assert_eq!(kelpie_goals, KelpieBlockType::Goals);

        // Test reverse mapping
        let back_to_entity = EntityType::from(kelpie_persona);
        assert_eq!(back_to_entity, EntityType::Self_);

        // TigerStyle: Postconditions - verify all features work together
        assert!(core.render().contains("importance="));
        assert_eq!(working.incr("request_count", 1).unwrap(), 7);
        assert_eq!(
            KelpieBlockType::from(EntityType::Note),
            KelpieBlockType::Scratch
        );
    }

    /// Integration test: Graceful degradation under stress.
    ///
    /// TigerStyle: Verify system handles edge cases gracefully.
    #[test]
    fn test_kelpie_integration_edge_cases() {
        // Test 1: CoreMemory with boundary importance values
        let mut core = CoreMemory::new();
        core.set_block(MemoryBlockType::Facts, "Test content")
            .unwrap();
        core.set_block_importance(MemoryBlockType::Facts, 0.0)
            .unwrap();
        assert_eq!(
            core.get_block_importance(MemoryBlockType::Facts).unwrap(),
            0.0
        );

        core.set_block_importance(MemoryBlockType::Facts, 1.0)
            .unwrap();
        assert_eq!(
            core.get_block_importance(MemoryBlockType::Facts).unwrap(),
            1.0
        );

        // Test 2: WorkingMemory atomic ops with expired keys
        let mut working = WorkingMemory::new();
        let start_ms = 1000;
        working.set_clock_ms(start_ms);

        // Set a key and let it expire
        working.set("temp", b"value", Some(100)).unwrap();
        working.set_clock_ms(start_ms + 200);

        // incr on expired key should start from 0
        let val = working.incr("temp", 10).unwrap();
        assert_eq!(val, 10);

        // append on expired key should start empty
        working.set("temp2", b"old", Some(100)).unwrap();
        working.set_clock_ms(start_ms + 400);
        working.append("temp2", b"new").unwrap();
        assert_eq!(working.get("temp2").unwrap(), b"new");

        // touch on expired key should error
        working.set("temp3", b"val", Some(100)).unwrap();
        working.set_clock_ms(start_ms + 600);
        assert!(working.touch("temp3").is_err());

        // Test 3: All entity types map correctly
        for entity_type in EntityType::all() {
            let kelpie = KelpieBlockType::from(*entity_type);
            // Should produce a valid block type
            assert!(KelpieBlockType::all_ordered().contains(&kelpie));
        }
    }

    /// Integration test: Determinism across features.
    ///
    /// TigerStyle: Same configuration should produce identical behavior.
    #[test]
    fn test_kelpie_integration_determinism() {
        // Run 1
        let mut core1 = CoreMemory::new();
        core1.set_block(MemoryBlockType::Scratch, "Test").unwrap();
        core1
            .set_block_importance(MemoryBlockType::Scratch, 0.7)
            .unwrap();

        let mut working1 = WorkingMemory::new();
        let counter1a = working1.incr("counter", 5).unwrap();
        let counter1b = working1.incr("counter", 3).unwrap();
        let xml1 = core1.render();

        // Run 2 with same operations
        let mut core2 = CoreMemory::new();
        core2.set_block(MemoryBlockType::Scratch, "Test").unwrap();
        core2
            .set_block_importance(MemoryBlockType::Scratch, 0.7)
            .unwrap();

        let mut working2 = WorkingMemory::new();
        let counter2a = working2.incr("counter", 5).unwrap();
        let counter2b = working2.incr("counter", 3).unwrap();
        let xml2 = core2.render();

        // TigerStyle: Verify identical results
        assert_eq!(counter1a, counter2a);
        assert_eq!(counter1b, counter2b);
        assert_eq!(xml1, xml2);
        assert_eq!(
            core1
                .get_block_importance(MemoryBlockType::Scratch)
                .unwrap(),
            core2
                .get_block_importance(MemoryBlockType::Scratch)
                .unwrap()
        );
    }

    /// Integration test: Complete Kelpie-style workflow.
    ///
    /// TigerStyle: Simulates how Kelpie would use these features.
    #[test]
    fn test_kelpie_integration_realistic_workflow() {
        // 1. Setup core memory (what Kelpie keeps in context)
        let mut core = CoreMemory::new();

        core.set_block(
            MemoryBlockType::System,
            "You are a helpful AI assistant for software development.",
        )
        .unwrap();
        core.set_block_importance(MemoryBlockType::System, 1.0)
            .unwrap();

        core.set_block(
            MemoryBlockType::Persona,
            "I am pragmatic and provide concrete examples.",
        )
        .unwrap();
        core.set_block_importance(MemoryBlockType::Persona, 0.95)
            .unwrap();

        core.set_block(
            MemoryBlockType::Human,
            "User is working on the Umi memory system.",
        )
        .unwrap();
        core.set_block_importance(MemoryBlockType::Human, 0.9)
            .unwrap();

        core.set_block(
            MemoryBlockType::Facts,
            "Umi uses Rust and Python. It has 3 memory tiers.",
        )
        .unwrap();
        core.set_block_importance(MemoryBlockType::Facts, 0.8)
            .unwrap();

        core.set_block(
            MemoryBlockType::Goals,
            "Implement Kelpie integration features.",
        )
        .unwrap();
        core.set_block_importance(MemoryBlockType::Goals, 0.85)
            .unwrap();

        // 2. Use working memory for session state
        let mut working = WorkingMemory::new();

        // Track conversation metrics
        working.incr("message_count", 1).unwrap();
        working.incr("message_count", 1).unwrap();
        working.incr("message_count", 1).unwrap();
        assert_eq!(working.incr("message_count", 1).unwrap(), 4);

        // Append to activity log
        working
            .append("activity", b"[10:00] Started session. ")
            .unwrap();
        working
            .append("activity", b"[10:05] Discussed features. ")
            .unwrap();
        working
            .append("activity", b"[10:10] Reviewed code.")
            .unwrap();

        // Keep session alive
        working
            .set("session_id", b"kelpie-session-123", None)
            .unwrap();
        working.touch("session_id").unwrap();

        // 3. Map between UMI and Kelpie types
        // Kelpie receives entities from archival memory and maps them to block types
        let entities_from_archival = vec![
            EntityType::Self_,   // User's self-representation
            EntityType::Person,  // Other people mentioned
            EntityType::Project, // Current projects
            EntityType::Task,    // Active tasks
            EntityType::Note,    // General notes
        ];

        let kelpie_blocks: Vec<KelpieBlockType> = entities_from_archival
            .iter()
            .map(|e| KelpieBlockType::from(*e))
            .collect();

        // Verify mapping
        assert_eq!(kelpie_blocks[0], KelpieBlockType::Persona);
        assert_eq!(kelpie_blocks[1], KelpieBlockType::Facts);
        assert_eq!(kelpie_blocks[2], KelpieBlockType::Goals);
        assert_eq!(kelpie_blocks[3], KelpieBlockType::Goals);
        assert_eq!(kelpie_blocks[4], KelpieBlockType::Scratch);

        // 4. Generate final context for LLM
        let context_xml = core.render();
        assert!(context_xml.contains("<core_memory>"));
        assert!(context_xml.contains(r#"importance="1.00""#));
        assert!(context_xml.contains(r#"importance="0.95""#));
        assert!(context_xml.contains(r#"importance="0.90""#));
        assert!(context_xml.contains(r#"importance="0.85""#));
        assert!(context_xml.contains(r#"importance="0.80""#));
        assert!(context_xml.contains("</core_memory>"));

        // 5. Verify session state
        assert_eq!(working.get("message_count").unwrap(), b"4");
        assert!(working.get("activity").unwrap().len() > 50);
        assert!(working.exists("session_id"));

        // TigerStyle: Postconditions - complete workflow succeeded
        assert!(context_xml.len() > 200);
        assert_eq!(kelpie_blocks.len(), 5);
    }

    /// Integration test: Memory operations under time pressure.
    ///
    /// TigerStyle: Verify correct behavior as time advances.
    #[test]
    fn test_kelpie_integration_temporal_behavior() {
        let mut working = WorkingMemory::new();
        let start_ms = 1000;
        working.set_clock_ms(start_ms);

        // Set multiple keys with different TTLs
        working
            .set("short_lived", b"expires_soon", Some(500))
            .unwrap();
        working
            .set("medium_lived", b"expires_later", Some(2000))
            .unwrap();
        working
            .set("long_lived", b"expires_much_later", Some(5000))
            .unwrap();

        // Initial state - all exist
        assert!(working.exists("short_lived"));
        assert!(working.exists("medium_lived"));
        assert!(working.exists("long_lived"));

        // Advance past first expiry
        working.set_clock_ms(start_ms + 600);
        assert!(!working.exists("short_lived"));
        assert!(working.exists("medium_lived"));
        assert!(working.exists("long_lived"));

        // Advance past medium_lived expiry
        working.set_clock_ms(start_ms + 2100);
        assert!(!working.exists("medium_lived"));
        assert!(working.exists("long_lived"));

        // Touch long_lived to extend its life (gets default TTL)
        working.touch("long_lived").unwrap();

        // Advance past original long_lived expiry
        working.set_clock_ms(start_ms + 6000);
        assert!(working.exists("long_lived")); // Still alive due to touch with long default TTL

        // TigerStyle: Postcondition - verify touch extended TTL
        assert!(working.exists("long_lived"));
    }
}
