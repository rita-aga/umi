"""AnthropicProvider - Production Claude integration.

Requires: pip install umi[anthropic]

Example:
    >>> from umi.providers import AnthropicProvider
    >>> provider = AnthropicProvider(api_key="...")
    >>> response = await provider.complete("Hello")
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "AnthropicProvider requires the 'anthropic' package. "
        "Install with: pip install umi[anthropic]"
    ) from e

T = TypeVar("T")


class AnthropicProvider:
    """Production LLM provider using Anthropic's Claude API.

    Attributes:
        model: Model name to use (default: claude-sonnet-4-20250514).
        max_tokens: Maximum tokens in response.

    Example:
        >>> provider = AnthropicProvider()  # Uses ANTHROPIC_API_KEY env var
        >>> response = await provider.complete("What is 2+2?")
        >>> print(response)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model to use.
            max_tokens: Maximum tokens in response.
        """
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str) -> str:
        """Generate completion using Claude.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated completion text.

        Raises:
            anthropic.APIError: If API call fails.
        """
        assert prompt, "prompt must not be empty"

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        content = response.content[0]
        if hasattr(content, "text"):
            return content.text
        return str(content)

    async def complete_json(self, prompt: str, schema: type[T]) -> T:
        """Generate JSON completion using Claude.

        Args:
            prompt: Input prompt text.
            schema: Expected response type (used for documentation).

        Returns:
            Parsed JSON response.

        Raises:
            ValueError: If response is not valid JSON.
        """
        assert prompt, "prompt must not be empty"

        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": json_prompt}],
        )

        content = response.content[0]
        text = content.text if hasattr(content, "text") else str(content)

        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}") from e
