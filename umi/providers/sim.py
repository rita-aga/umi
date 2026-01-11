"""SimLLMProvider - Deterministic LLM simulation for testing (ADR-007).

TigerStyle: Simulation-first, deterministic, fault-injectable.

Provides deterministic LLM responses for testing without:
- Network calls
- API costs
- Flaky tests

Key features:
- Seed-based RNG for reproducibility
- Prompt routing to domain-specific generators
- Fault injection via FaultConfig
- Domain-aware response generation

Example:
    >>> provider = SimLLMProvider(seed=42)
    >>> response = await provider.complete("Extract entities from: I met Alice")
    >>> # Always returns the same response for seed=42
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from random import Random
from typing import TypeVar

from umi.faults import FaultConfig, FaultStats

T = TypeVar("T")

# =============================================================================
# Constants
# =============================================================================

# Entity types for extraction simulation
ENTITY_TYPES = ["person", "org", "project", "topic", "preference", "task"]

# Relation types for extraction simulation
RELATION_TYPES = ["works_at", "knows", "relates_to", "prefers", "manages"]

# Evolution types for detection simulation
EVOLUTION_TYPES = ["update", "extend", "derive", "contradict", "none"]

# Common names for entity extraction
COMMON_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

# Common orgs for entity extraction
COMMON_ORGS = ["Acme", "TechCorp", "StartupX", "BigCo", "OpenAI", "Anthropic"]


# =============================================================================
# Exceptions
# =============================================================================


class SimLLMError(Exception):
    """Base exception for SimLLMProvider errors."""

    pass


class SimTimeoutError(TimeoutError):
    """Simulated timeout error."""

    pass


class SimRateLimitError(Exception):
    """Simulated rate limit error."""

    pass


class SimMalformedResponseError(ValueError):
    """Simulated malformed response error."""

    pass


# =============================================================================
# SimLLMProvider
# =============================================================================


@dataclass
class SimLLMProvider:
    """Deterministic LLM provider for simulation testing.

    Generates predictable responses based on:
    1. Seed (for RNG initialization)
    2. Prompt content (for routing and variation)
    3. FaultConfig (for error injection)

    Attributes:
        seed: Seed for deterministic RNG.
        faults: Fault injection configuration.
        stats: Statistics on faults injected (for debugging).

    Example:
        >>> # Basic usage
        >>> provider = SimLLMProvider(seed=42)
        >>> response = await provider.complete("Hello")
        >>> assert isinstance(response, str)

        >>> # With fault injection
        >>> faults = FaultConfig(llm_timeout=0.5)
        >>> provider = SimLLMProvider(seed=42, faults=faults)
        >>> # ~50% of calls will raise TimeoutError
    """

    seed: int
    faults: FaultConfig | None = None
    stats: FaultStats | None = None

    def __post_init__(self) -> None:
        """Initialize RNG and stats."""
        self._rng = Random(self.seed)
        if self.faults is None:
            self.faults = FaultConfig()
        if self.stats is None:
            self.stats = FaultStats()

    async def complete(self, prompt: str) -> str:
        """Generate a deterministic completion.

        Routes prompt to domain-specific generators based on content.

        Args:
            prompt: Input prompt text.

        Returns:
            Simulated completion text.

        Raises:
            SimTimeoutError: If timeout fault triggered.
            RuntimeError: If error fault triggered.
            SimRateLimitError: If rate limit fault triggered.
        """
        # TigerStyle: Precondition
        assert prompt, "prompt must not be empty"

        # Check for faults
        self._maybe_inject_fault()

        # Route to appropriate generator
        return self._route_prompt(prompt)

    async def complete_json(self, prompt: str, schema: type[T]) -> T:
        """Generate a deterministic JSON completion.

        Args:
            prompt: Input prompt text.
            schema: Expected response type (for documentation, not enforced in sim).

        Returns:
            Parsed JSON response.

        Raises:
            SimTimeoutError: If timeout fault triggered.
            RuntimeError: If error fault triggered.
            SimMalformedResponseError: If malformed fault triggered.
        """
        # TigerStyle: Precondition
        assert prompt, "prompt must not be empty"

        # Check for faults (special handling for malformed)
        fault = self.faults.get_llm_fault(self._rng) if self.faults else None
        if fault:
            self._handle_fault(fault)

        # Generate JSON response
        response_str = self._route_prompt(prompt)

        # Parse as JSON
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            raise SimMalformedResponseError(f"Failed to parse response: {e}") from e

    def _maybe_inject_fault(self) -> None:
        """Check and inject fault if configured."""
        if not self.faults:
            return

        fault = self.faults.get_llm_fault(self._rng)
        if fault:
            self._handle_fault(fault)

    def _handle_fault(self, fault: str) -> None:
        """Handle a fault by raising appropriate exception."""
        assert self.stats is not None
        self.stats.record(fault)

        if fault == "timeout":
            raise SimTimeoutError("Simulated LLM timeout")
        elif fault == "error":
            raise RuntimeError("Simulated LLM API error")
        elif fault == "rate_limit":
            raise SimRateLimitError("Simulated rate limit exceeded")
        elif fault == "malformed":
            # Don't raise here for complete(), only for complete_json()
            pass

    def _route_prompt(self, prompt: str) -> str:
        """Route prompt to appropriate generator.

        Uses keyword matching to detect prompt intent.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated response string.
        """
        prompt_lower = prompt.lower()

        # Route to domain-specific generators
        if "extract" in prompt_lower and "entit" in prompt_lower:
            return self._sim_entity_extraction(prompt)
        elif "rewrite" in prompt_lower and "query" in prompt_lower:
            return self._sim_query_rewrite(prompt)
        elif "detect" in prompt_lower and "evolution" in prompt_lower:
            return self._sim_evolution_detection(prompt)
        elif "categorize" in prompt_lower or "categor" in prompt_lower:
            return self._sim_categorization(prompt)
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return self._sim_summarization(prompt)
        else:
            return self._sim_generic(prompt)

    def _prompt_hash(self, prompt: str) -> str:
        """Generate deterministic hash from prompt.

        Used to create variation based on prompt content.
        """
        return hashlib.sha256(prompt.encode()).hexdigest()[:8]

    def _sim_entity_extraction(self, prompt: str) -> str:
        """Generate simulated entity extraction response.

        Detects names and organizations mentioned in the prompt text
        and returns a plausible extraction.
        """
        entities = []
        relations = []

        # Extract any names/orgs that appear in prompt
        prompt_upper = prompt.upper()

        for name in COMMON_NAMES:
            if name.upper() in prompt_upper:
                entity_type = self._rng.choice(["person", "person", "person", "topic"])
                entities.append({
                    "name": name,
                    "type": entity_type,
                    "content": f"Extracted from: {prompt[:50]}...",
                    "confidence": round(0.7 + self._rng.random() * 0.3, 2),
                })

        for org in COMMON_ORGS:
            if org.upper() in prompt_upper:
                entities.append({
                    "name": org,
                    "type": "org",
                    "content": f"Organization mentioned",
                    "confidence": round(0.7 + self._rng.random() * 0.3, 2),
                })

        # If no entities found, create a generic note
        if not entities:
            prompt_hash = self._prompt_hash(prompt)
            entities.append({
                "name": f"Note_{prompt_hash}",
                "type": "note",
                "content": prompt[:100],
                "confidence": 0.5,
            })

        # Generate some relations if we have multiple entities
        if len(entities) >= 2:
            source = entities[0]["name"]
            target = entities[1]["name"]
            rel_type = self._rng.choice(RELATION_TYPES)
            relations.append({
                "source": source,
                "target": target,
                "type": rel_type,
            })

        return json.dumps({
            "entities": entities,
            "relations": relations,
        }, indent=2)

    def _sim_query_rewrite(self, prompt: str) -> str:
        """Generate simulated query rewrite response.

        Creates 2-3 variations of the original query.
        """
        # Extract the original query from the prompt
        # Look for patterns like "Query: ..." or just use the prompt
        lines = prompt.split("\n")
        original_query = prompt

        for line in lines:
            if line.lower().startswith("query:"):
                original_query = line.split(":", 1)[1].strip()
                break

        # Generate variations
        variations = [original_query]  # Include original

        # Add a more specific variation
        prompt_hash = self._prompt_hash(original_query)
        hash_int = int(prompt_hash, 16)

        variations.append(f"{original_query} specifically")
        variations.append(f"related to {original_query}")

        # Limit to 3
        return json.dumps(variations[:3])

    def _sim_evolution_detection(self, prompt: str) -> str:
        """Generate simulated evolution detection response.

        Determines relationship between new and existing memories.
        """
        # Use prompt hash to deterministically pick evolution type
        prompt_hash = self._prompt_hash(prompt)
        hash_int = int(prompt_hash, 16)

        # Weight towards "none" and "extend" as most common
        weights = [0.2, 0.3, 0.15, 0.1, 0.25]  # update, extend, derive, contradict, none
        r = self._rng.random()
        cumulative = 0.0
        evolution_type = "none"

        for i, weight in enumerate(weights):
            cumulative += weight
            if r < cumulative:
                evolution_type = EVOLUTION_TYPES[i]
                break

        if evolution_type == "none":
            return json.dumps({
                "type": "none",
                "reason": "No significant relationship detected",
                "related_id": None,
                "confidence": 0.3,
            })

        # For non-none types, generate a plausible response
        reasons = {
            "update": "New information supersedes previous",
            "extend": "Additional details complement existing",
            "derive": "Conclusion drawn from existing information",
            "contradict": "Conflicting information detected",
        }

        return json.dumps({
            "type": evolution_type,
            "reason": reasons.get(evolution_type, "Related information"),
            "related_id": f"entity-{prompt_hash}",
            "confidence": round(0.6 + self._rng.random() * 0.4, 2),
        })

    def _sim_categorization(self, prompt: str) -> str:
        """Generate simulated categorization response.

        Assigns content to categories.
        """
        categories = ["preferences", "relationships", "projects", "learnings", "decisions"]

        # Pick 1-2 categories based on prompt
        num_categories = 1 + (int(self._prompt_hash(prompt), 16) % 2)
        selected = self._rng.sample(categories, min(num_categories, len(categories)))

        return json.dumps(selected)

    def _sim_summarization(self, prompt: str) -> str:
        """Generate simulated summarization response."""
        prompt_hash = self._prompt_hash(prompt)
        return f"Summary of content (hash: {prompt_hash}): Key points extracted from the input."

    def _sim_generic(self, prompt: str) -> str:
        """Generate generic simulated response.

        Fallback for unrecognized prompt patterns.
        """
        prompt_hash = self._prompt_hash(prompt)
        return f"SimResponse[{prompt_hash}]: Acknowledged input of {len(prompt)} characters."

    def reset(self) -> None:
        """Reset RNG to initial seed state.

        Useful for replaying exact sequences.
        """
        self._rng = Random(self.seed)
        self.stats = FaultStats()

    def fork(self, new_seed: int | None = None) -> "SimLLMProvider":
        """Create a new provider with derived seed.

        Args:
            new_seed: Explicit seed, or derive from current RNG.

        Returns:
            New SimLLMProvider instance.
        """
        if new_seed is None:
            new_seed = self._rng.randint(0, 2**32 - 1)

        return SimLLMProvider(seed=new_seed, faults=self.faults)
