"""Umi - Memory system that never forgets.

A hybrid Rust+Python memory library with:
- Rust core for storage and correctness (DST)
- Python layer for LLM integration
- Simulation-first testing at both layers

Example:
    >>> from umi import Memory
    >>> memory = Memory(seed=42)  # Simulation mode
    >>> await memory.remember("I met Alice at Acme Corp")
    >>> results = await memory.recall("Who do I know at Acme?")
"""

from umi.evolution import EvolutionRelation, EvolutionTracker
from umi.extraction import EntityExtractor, ExtractedEntity, ExtractedRelation, ExtractionResult
from umi.faults import FaultConfig
from umi.memory import Memory
from umi.providers.base import LLMProvider
from umi.providers.sim import SimLLMProvider
from umi.retrieval import DualRetriever
from umi.storage import Entity, SimStorage

__version__ = "0.1.0"
__all__ = [
    # Main API
    "Memory",
    "Entity",
    # Storage
    "SimStorage",
    # Retrieval
    "DualRetriever",
    # Extraction
    "EntityExtractor",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    # Evolution
    "EvolutionTracker",
    "EvolutionRelation",
    # Providers
    "LLMProvider",
    "SimLLMProvider",
    # Configuration
    "FaultConfig",
]


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "AnthropicProvider":
        from umi.providers.anthropic import AnthropicProvider
        return AnthropicProvider
    elif name == "OpenAIProvider":
        from umi.providers.openai import OpenAIProvider
        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
