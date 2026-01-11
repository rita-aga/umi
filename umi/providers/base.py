"""LLMProvider Protocol - Interface for LLM completion.

TigerStyle: Protocol-based abstraction for testability.

All providers implement this protocol, enabling:
- Drop-in replacement between sim and production
- Type checking via structural subtyping
- Clear contract for LLM operations
"""

from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM completion providers.

    Implementations:
        - SimLLMProvider: Deterministic simulation for testing
        - AnthropicProvider: Production Claude integration
        - OpenAIProvider: Production GPT integration

    Example:
        >>> provider = SimLLMProvider(seed=42)
        >>> response = await provider.complete("Hello")
        >>> assert isinstance(response, str)
    """

    async def complete(self, prompt: str) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The input prompt text.

        Returns:
            The generated completion text.

        Raises:
            TimeoutError: If the request times out.
            RuntimeError: If the provider encounters an error.
        """
        ...

    async def complete_json(self, prompt: str, schema: type[T]) -> T:
        """Generate a structured JSON completion.

        Args:
            prompt: The input prompt text.
            schema: A dataclass or TypedDict type for the expected response.

        Returns:
            Parsed response matching the schema type.

        Raises:
            TimeoutError: If the request times out.
            RuntimeError: If the provider encounters an error.
            ValueError: If response doesn't match schema.
        """
        ...


def validate_provider(provider: Any) -> bool:
    """Check if an object implements the LLMProvider protocol.

    Args:
        provider: Object to validate.

    Returns:
        True if provider implements LLMProvider protocol.

    Example:
        >>> from umi.providers import SimLLMProvider
        >>> provider = SimLLMProvider(seed=42)
        >>> assert validate_provider(provider)
    """
    return isinstance(provider, LLMProvider)
