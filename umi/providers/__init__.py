"""LLM Providers for Umi memory system.

Provides a unified interface for LLM completion with:
- SimLLMProvider: Deterministic simulation for testing (DST)
- AnthropicProvider: Production Anthropic/Claude integration
- OpenAIProvider: Production OpenAI integration
"""

from umi.providers.base import LLMProvider
from umi.providers.sim import SimLLMProvider

# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "AnthropicProvider":
        from umi.providers.anthropic import AnthropicProvider
        return AnthropicProvider
    elif name == "OpenAIProvider":
        from umi.providers.openai import OpenAIProvider
        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LLMProvider",
    "SimLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
]
