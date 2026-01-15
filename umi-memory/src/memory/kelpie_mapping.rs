//! Kelpie Block Type Mapping - Translation Layer for Kelpie Integration
//!
//! `TigerStyle`: Explicit mapping with bidirectional conversion.
//!
//! This module provides mapping between UMI's `EntityType` (used in archival memory)
//! and Kelpie's block type terminology (used in core memory XML rendering).
//!
//! # Mapping Rationale
//!
//! - **Self → Persona**: User's self-representation maps to personality/behavior
//! - **Person → Facts**: Information about other people is factual knowledge
//! - **Project → Goals**: Projects are goals/objectives
//! - **Topic → Facts**: Topics and concepts are factual knowledge
//! - **Note → Scratch**: General notes map to scratch/working space
//! - **Task → Goals**: Tasks are goals/action items
//!
//! The mapping is N:1 (many UMI types → one Kelpie type) because Kelpie has
//! fewer, broader categories while UMI has more specific entity types.

use std::fmt;

use crate::storage::EntityType;

/// Kelpie block types for core memory.
///
/// `TigerStyle`: Exhaustive enum matching Kelpie's core memory structure.
///
/// These types correspond to Kelpie's memory block types and are used when
/// rendering UMI entities as Kelpie-compatible XML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KelpieBlockType {
    /// System instructions and prompts (not mapped from entities)
    System,
    /// AI personality and behavior guidelines (from Self entities)
    Persona,
    /// Information about the human user (from Human block only)
    Human,
    /// Key facts and knowledge (from Person, Topic entities)
    Facts,
    /// Current objectives and tasks (from Project, Task entities)
    Goals,
    /// Temporary working space (from Note entities)
    Scratch,
}

impl KelpieBlockType {
    /// Get the string representation for XML rendering.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Persona => "persona",
            Self::Human => "human",
            Self::Facts => "facts",
            Self::Goals => "goals",
            Self::Scratch => "scratch",
        }
    }

    /// Get all block types in render order.
    #[must_use]
    pub fn all_ordered() -> &'static [KelpieBlockType] {
        &[
            Self::System,
            Self::Persona,
            Self::Human,
            Self::Facts,
            Self::Goals,
            Self::Scratch,
        ]
    }

    /// Get render priority (lower = rendered first).
    #[must_use]
    pub fn priority(&self) -> u8 {
        match self {
            Self::System => 0,
            Self::Persona => 1,
            Self::Human => 2,
            Self::Facts => 3,
            Self::Goals => 4,
            Self::Scratch => 5,
        }
    }
}

impl fmt::Display for KelpieBlockType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Convert from UMI EntityType to Kelpie block type.
///
/// # Mapping Rules
///
/// - Self → Persona (1:1)
/// - Person → Facts (N:1, information about people)
/// - Project → Goals (1:1, projects are goals)
/// - Topic → Facts (N:1, topics are factual knowledge)
/// - Note → Scratch (1:1, notes are scratch space)
/// - Task → Goals (N:1, tasks are goals)
///
/// `TigerStyle`: Explicit mapping with clear rationale.
impl From<EntityType> for KelpieBlockType {
    fn from(entity_type: EntityType) -> Self {
        match entity_type {
            EntityType::Self_ => KelpieBlockType::Persona,
            EntityType::Person => KelpieBlockType::Facts,
            EntityType::Organization => KelpieBlockType::Facts,
            EntityType::Project => KelpieBlockType::Goals,
            EntityType::Topic => KelpieBlockType::Facts,
            EntityType::Location => KelpieBlockType::Facts,
            EntityType::Event => KelpieBlockType::Goals,
            EntityType::Note => KelpieBlockType::Scratch,
            EntityType::Task => KelpieBlockType::Goals,
        }
    }
}

/// Convert from Kelpie block type to a default UMI EntityType.
///
/// # Reverse Mapping
///
/// This is a lossy conversion because multiple UMI types map to the same
/// Kelpie type. We provide sensible defaults:
///
/// - System → Note (fallback, system blocks not from entities)
/// - Persona → Self (1:1)
/// - Human → Note (fallback, human block is manually created)
/// - Facts → Topic (default for factual knowledge)
/// - Goals → Project (default for objectives)
/// - Scratch → Note (1:1)
///
/// `TigerStyle`: Explicit defaults with clear documentation.
impl From<KelpieBlockType> for EntityType {
    fn from(block_type: KelpieBlockType) -> Self {
        match block_type {
            KelpieBlockType::System => EntityType::Note,
            KelpieBlockType::Persona => EntityType::Self_,
            KelpieBlockType::Human => EntityType::Note,
            KelpieBlockType::Facts => EntityType::Topic,
            KelpieBlockType::Goals => EntityType::Project,
            KelpieBlockType::Scratch => EntityType::Note,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelpie_block_type_as_str() {
        assert_eq!(KelpieBlockType::System.as_str(), "system");
        assert_eq!(KelpieBlockType::Persona.as_str(), "persona");
        assert_eq!(KelpieBlockType::Human.as_str(), "human");
        assert_eq!(KelpieBlockType::Facts.as_str(), "facts");
        assert_eq!(KelpieBlockType::Goals.as_str(), "goals");
        assert_eq!(KelpieBlockType::Scratch.as_str(), "scratch");
    }

    #[test]
    fn test_kelpie_block_type_priority() {
        // TigerStyle: Verify render order is correct
        assert_eq!(KelpieBlockType::System.priority(), 0);
        assert_eq!(KelpieBlockType::Persona.priority(), 1);
        assert_eq!(KelpieBlockType::Human.priority(), 2);
        assert_eq!(KelpieBlockType::Facts.priority(), 3);
        assert_eq!(KelpieBlockType::Goals.priority(), 4);
        assert_eq!(KelpieBlockType::Scratch.priority(), 5);
    }

    #[test]
    fn test_kelpie_block_type_all_ordered() {
        let types = KelpieBlockType::all_ordered();

        // TigerStyle: Verify count and order
        assert_eq!(types.len(), 6);
        assert_eq!(types[0], KelpieBlockType::System);
        assert_eq!(types[5], KelpieBlockType::Scratch);
    }

    #[test]
    fn test_entity_type_to_kelpie_one_to_one() {
        // 1:1 mappings
        assert_eq!(
            KelpieBlockType::from(EntityType::Self_),
            KelpieBlockType::Persona
        );
        assert_eq!(
            KelpieBlockType::from(EntityType::Project),
            KelpieBlockType::Goals
        );
        assert_eq!(
            KelpieBlockType::from(EntityType::Note),
            KelpieBlockType::Scratch
        );
    }

    #[test]
    fn test_entity_type_to_kelpie_many_to_one() {
        // N:1 mappings to Facts
        assert_eq!(
            KelpieBlockType::from(EntityType::Person),
            KelpieBlockType::Facts
        );
        assert_eq!(
            KelpieBlockType::from(EntityType::Topic),
            KelpieBlockType::Facts
        );

        // N:1 mappings to Goals
        assert_eq!(
            KelpieBlockType::from(EntityType::Project),
            KelpieBlockType::Goals
        );
        assert_eq!(
            KelpieBlockType::from(EntityType::Task),
            KelpieBlockType::Goals
        );
    }

    #[test]
    fn test_kelpie_to_entity_type() {
        // Reverse mapping with defaults
        assert_eq!(
            EntityType::from(KelpieBlockType::Persona),
            EntityType::Self_
        );
        assert_eq!(EntityType::from(KelpieBlockType::Facts), EntityType::Topic);
        assert_eq!(
            EntityType::from(KelpieBlockType::Goals),
            EntityType::Project
        );
        assert_eq!(EntityType::from(KelpieBlockType::Scratch), EntityType::Note);

        // Fallback cases
        assert_eq!(EntityType::from(KelpieBlockType::System), EntityType::Note);
        assert_eq!(EntityType::from(KelpieBlockType::Human), EntityType::Note);
    }

    #[test]
    fn test_bidirectional_mapping_lossy() {
        // Forward and back should work for 1:1 mappings
        let self_type = EntityType::Self_;
        let kelpie = KelpieBlockType::from(self_type);
        let back = EntityType::from(kelpie);
        assert_eq!(back, EntityType::Self_);

        // But N:1 mappings are lossy
        let person = EntityType::Person;
        let kelpie = KelpieBlockType::from(person);
        assert_eq!(kelpie, KelpieBlockType::Facts);
        let back = EntityType::from(kelpie);
        // Comes back as Topic (the default for Facts), not Person
        assert_eq!(back, EntityType::Topic);
        assert_ne!(back, EntityType::Person);
    }

    #[test]
    fn test_display_format() {
        assert_eq!(format!("{}", KelpieBlockType::Persona), "persona");
        assert_eq!(format!("{}", KelpieBlockType::Facts), "facts");
        assert_eq!(format!("{}", KelpieBlockType::Goals), "goals");
    }

    // =========================================================================
    // DST Tests
    // =========================================================================

    #[test]
    fn test_mapping_deterministic() {
        // TigerStyle: Mapping should be deterministic and consistent
        for _ in 0..10 {
            // Run multiple times to verify consistency
            assert_eq!(
                KelpieBlockType::from(EntityType::Person),
                KelpieBlockType::Facts
            );
            assert_eq!(
                KelpieBlockType::from(EntityType::Task),
                KelpieBlockType::Goals
            );
        }
    }

    #[test]
    fn test_all_entity_types_map() {
        // TigerStyle: Ensure all entity types have a mapping
        for entity_type in EntityType::all() {
            let kelpie = KelpieBlockType::from(*entity_type);
            // Should not panic, should produce valid block type
            assert!(KelpieBlockType::all_ordered().contains(&kelpie));
        }
    }

    #[test]
    fn test_mapping_coverage() {
        // TigerStyle: Verify we handle all UMI entity types
        let mappings = vec![
            (EntityType::Self_, KelpieBlockType::Persona),
            (EntityType::Person, KelpieBlockType::Facts),
            (EntityType::Organization, KelpieBlockType::Facts),
            (EntityType::Project, KelpieBlockType::Goals),
            (EntityType::Topic, KelpieBlockType::Facts),
            (EntityType::Location, KelpieBlockType::Facts),
            (EntityType::Event, KelpieBlockType::Goals),
            (EntityType::Note, KelpieBlockType::Scratch),
            (EntityType::Task, KelpieBlockType::Goals),
        ];

        // Precondition: We should have mappings for all entity types
        assert_eq!(mappings.len(), EntityType::all().len());

        // Postcondition: All mappings should be consistent
        for (entity_type, expected_kelpie) in mappings {
            assert_eq!(KelpieBlockType::from(entity_type), expected_kelpie);
        }
    }

    #[test]
    fn test_reverse_mapping_coverage() {
        // TigerStyle: Verify we handle all Kelpie block types
        let reverse_mappings = vec![
            (KelpieBlockType::System, EntityType::Note),
            (KelpieBlockType::Persona, EntityType::Self_),
            (KelpieBlockType::Human, EntityType::Note),
            (KelpieBlockType::Facts, EntityType::Topic),
            (KelpieBlockType::Goals, EntityType::Project),
            (KelpieBlockType::Scratch, EntityType::Note),
        ];

        // Precondition: We should have reverse mappings for all block types
        assert_eq!(reverse_mappings.len(), KelpieBlockType::all_ordered().len());

        // Postcondition: All reverse mappings should be consistent
        for (kelpie_type, expected_entity) in reverse_mappings {
            assert_eq!(EntityType::from(kelpie_type), expected_entity);
        }
    }
}
