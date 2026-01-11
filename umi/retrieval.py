"""DualRetriever - Fast search + LLM reasoning (ADR-009).

TigerStyle: Sim-first, deterministic, graceful degradation.

Combines fast substring search with LLM-assisted query rewriting
for improved recall on complex queries.

Key features:
- Heuristic routing: Detects when deep search helps
- Query rewriting: LLM expands queries into search variations
- RRF merging: Reciprocal Rank Fusion for result combination
- Graceful fallback: Returns fast results if LLM fails

Example:
    >>> retriever = DualRetriever(storage, llm, seed=42)
    >>> results = await retriever.search("Who do I know at Acme?")
    >>> # Uses LLM to rewrite query, merges results
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umi.providers.base import LLMProvider
    from umi.storage import Entity, SimStorage


# =============================================================================
# Constants (TigerStyle: explicit limits)
# =============================================================================

SEARCH_LIMIT_MAX = 100
QUERY_REWRITE_MAX = 3  # Maximum number of query rewrites
RRF_K = 60  # Standard RRF constant

# Deep search trigger words
QUESTION_WORDS = frozenset(["who", "what", "when", "where", "why", "how"])
RELATIONSHIP_TERMS = frozenset(["related", "about", "regarding", "involving", "connected"])
TEMPORAL_TERMS = frozenset(["yesterday", "today", "last", "recent", "before", "after", "week", "month", "year"])
ABSTRACT_TERMS = frozenset(["similar", "like", "connections", "associated", "linked"])

# Query rewrite prompt template
QUERY_REWRITE_PROMPT = """Rewrite this search query into 2-3 variations that would help find relevant memories.

Query: {query}

Return as JSON array of strings. Example: ["variation 1", "variation 2", "variation 3"]
Only return the JSON array, nothing else."""


# =============================================================================
# DualRetriever
# =============================================================================


@dataclass
class DualRetriever:
    """Dual retrieval: fast search + LLM reasoning.

    Combines direct text search with LLM-assisted query rewriting
    for improved recall on complex queries.

    Attributes:
        storage: Storage backend for searches.
        llm: LLM provider for query rewriting.
        seed: Optional seed for deterministic behavior.

    Example:
        >>> # Basic usage
        >>> retriever = DualRetriever(storage, llm, seed=42)
        >>> results = await retriever.search("Acme employees")

        >>> # Force deep search
        >>> results = await retriever.search("connections", deep_search=True)

        >>> # Fast search only
        >>> results = await retriever.search("Alice", deep_search=False)
    """

    storage: "SimStorage"
    llm: "LLMProvider"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialize RNG for deterministic behavior."""
        if self.seed is not None:
            self._rng = Random(self.seed)
        else:
            self._rng = Random()

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        deep_search: bool = True,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list["Entity"]:
        """Search with dual retrieval strategy.

        Fast path: Direct substring search on storage.
        Deep path: LLM rewrites query into variations, merges results.

        Args:
            query: Search query.
            limit: Maximum results to return.
            deep_search: Enable LLM-assisted search (default True).
            time_range: Optional (start, end) filter by event_time.

        Returns:
            List of entities, ranked by relevance.

        Raises:
            AssertionError: If preconditions not met.

        Example:
            >>> results = await retriever.search("Who works at Acme?")
            >>> assert len(results) <= 10
        """
        # TigerStyle: Preconditions
        assert query, "query must not be empty"
        assert 0 < limit <= SEARCH_LIMIT_MAX, f"limit must be 1-{SEARCH_LIMIT_MAX}: {limit}"

        # 1. Fast search (always runs)
        fast_results = await self._fast_search(query, limit=limit * 2)

        # 2. Decide if deep search is needed
        use_deep = deep_search and self.needs_deep_search(query)

        if use_deep:
            # 3. Deep search: rewrite query and search variations
            deep_results = await self._deep_search(query, limit=limit * 2)

            # 4. Merge results using RRF
            results = self.merge_rrf(fast_results, deep_results)
        else:
            results = fast_results

        # 5. Apply time filter if specified
        if time_range is not None:
            start_time, end_time = time_range
            results = [
                e for e in results
                if e.event_time is not None
                and start_time <= e.event_time <= end_time
            ]

        # 6. Sort by importance and limit
        results.sort(key=lambda e: (-e.importance, -e.updated_at.timestamp()))
        results = results[:limit]

        # TigerStyle: Postconditions
        assert isinstance(results, list), "must return list"
        assert len(results) <= limit, f"results exceed limit: {len(results)} > {limit}"

        return results

    def needs_deep_search(self, query: str) -> bool:
        """Heuristic: does query benefit from LLM reasoning?

        Triggers deep search for:
        - Question words (who, what, when, etc.)
        - Relationship terms (related to, about, etc.)
        - Temporal terms (yesterday, last week, etc.)
        - Abstract concepts (similar, connections, etc.)

        Args:
            query: Search query to analyze.

        Returns:
            True if deep search would likely help.

        Example:
            >>> retriever.needs_deep_search("Alice")
            False
            >>> retriever.needs_deep_search("Who knows Alice?")
            True
        """
        # TigerStyle: Precondition
        assert query, "query must not be empty"

        query_lower = query.lower()

        # Check for trigger words/phrases in query text (handles possessives like "yesterday's")
        for word in QUESTION_WORDS:
            if word in query_lower:
                return True
        for word in TEMPORAL_TERMS:
            if word in query_lower:
                return True
        for word in ABSTRACT_TERMS:
            if word in query_lower:
                return True

        # Check for relationship phrases (may span words)
        for term in RELATIONSHIP_TERMS:
            if term in query_lower:
                return True

        return False

    async def rewrite_query(self, query: str) -> list[str]:
        """Use LLM to expand query into search variations.

        Args:
            query: Original search query.

        Returns:
            List of query variations (including original).

        Example:
            >>> variations = await retriever.rewrite_query("Acme employees")
            >>> # ["Acme employees", "works at Acme", "Acme Corp"]
        """
        # TigerStyle: Precondition
        assert query, "query must not be empty"

        prompt = QUERY_REWRITE_PROMPT.format(query=query)

        try:
            response = await self.llm.complete(prompt)
            variations = json.loads(response)

            # Validate response
            if not isinstance(variations, list):
                return [query]

            # Filter to strings only, limit count
            valid_variations = [
                v for v in variations
                if isinstance(v, str) and v.strip()
            ][:QUERY_REWRITE_MAX]

            # Always include original query
            if query not in valid_variations:
                valid_variations.insert(0, query)

            return valid_variations[:QUERY_REWRITE_MAX]

        except (json.JSONDecodeError, RuntimeError, TimeoutError):
            # Graceful degradation: return original query
            return [query]

    def merge_rrf(
        self,
        *result_lists: list["Entity"],
        k: int = RRF_K,
    ) -> list["Entity"]:
        """Merge results using Reciprocal Rank Fusion.

        RRF score: sum(1 / (k + rank)) for each list the document appears in.
        Documents appearing in multiple lists get higher scores.

        Args:
            *result_lists: Variable number of result lists to merge.
            k: RRF constant (default 60, standard value).

        Returns:
            Merged and deduplicated results, sorted by RRF score.

        Example:
            >>> merged = retriever.merge_rrf(list1, list2, list3)
        """
        # TigerStyle: Precondition
        assert k > 0, f"k must be positive: {k}"

        # Calculate RRF scores
        scores: dict[str, float] = defaultdict(float)
        entities: dict[str, "Entity"] = {}

        for result_list in result_lists:
            for rank, entity in enumerate(result_list):
                # RRF formula: 1 / (k + rank)
                # rank is 0-indexed, so rank=0 gives highest score
                scores[entity.id] += 1.0 / (k + rank)
                entities[entity.id] = entity

        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda id: -scores[id])

        # Build result list
        results = [entities[id] for id in sorted_ids]

        return results

    async def _fast_search(self, query: str, limit: int) -> list["Entity"]:
        """Execute fast substring search.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching entities.
        """
        return await self.storage.search(query, limit=limit)

    async def _deep_search(self, query: str, limit: int) -> list["Entity"]:
        """Execute deep search with query rewriting.

        Args:
            query: Original search query.
            limit: Maximum results per variation.

        Returns:
            Combined results from all query variations.
        """
        # Get query variations from LLM
        variations = await self.rewrite_query(query)

        # Search each variation
        all_results: list["Entity"] = []
        seen_ids: set[str] = set()

        for variation in variations:
            # Skip if same as original (already searched in fast path)
            if variation == query:
                continue

            try:
                results = await self.storage.search(variation, limit=limit)
                for entity in results:
                    if entity.id not in seen_ids:
                        all_results.append(entity)
                        seen_ids.add(entity.id)
            except (RuntimeError, AssertionError):
                # Skip failed searches, continue with others
                continue

        return all_results

    def reset(self) -> None:
        """Reset RNG to initial seed state."""
        if self.seed is not None:
            self._rng = Random(self.seed)
