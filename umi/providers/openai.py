"""OpenAIProvider - Production OpenAI integration.

Requires: pip install umi[openai]

Example:
    >>> from umi.providers import OpenAIProvider
    >>> provider = OpenAIProvider(api_key="...")
    >>> response = await provider.complete("Hello")
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

try:
    import openai
except ImportError as e:
    raise ImportError(
        "OpenAIProvider requires the 'openai' package. "
        "Install with: pip install umi[openai]"
    ) from e

T = TypeVar("T")


class OpenAIProvider:
    """Production LLM provider using OpenAI's API.

    Attributes:
        model: Model name to use (default: gpt-4o).
        max_tokens: Maximum tokens in response.

    Example:
        >>> provider = OpenAIProvider()  # Uses OPENAI_API_KEY env var
        >>> response = await provider.complete("What is 2+2?")
        >>> print(response)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: API key. Defaults to OPENAI_API_KEY env var.
            model: Model to use.
            max_tokens: Maximum tokens in response.
        """
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str) -> str:
        """Generate completion using OpenAI.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated completion text.

        Raises:
            openai.APIError: If API call fails.
        """
        assert prompt, "prompt must not be empty"

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        choice = response.choices[0]
        if choice.message and choice.message.content:
            return choice.message.content
        return ""

    async def complete_json(self, prompt: str, schema: type[T]) -> T:
        """Generate JSON completion using OpenAI.

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

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": json_prompt}],
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        text = choice.message.content if choice.message and choice.message.content else "{}"

        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}") from e
