# ADR-009: Dual Retrieval - Fast Search + LLM Reasoning

## Status

Accepted

## Context

Phase 3 established the Memory class with basic `recall()` using substring search.
However, simple text matching has limitations:

1. **Vocabulary mismatch**: Query "colleagues" won't find "coworkers"
2. **Implicit relationships**: "Who do I know at Acme?" requires reasoning
3. **Temporal queries**: "What happened last week?" needs date understanding

### Requirements

1. **Fast path**: Direct search for simple, specific queries
2. **Deep path**: LLM-assisted search for complex queries
3. **Sim-first**: Deterministic behavior with `seed`
4. **Graceful degradation**: Falls back to fast search if LLM fails

### Inspiration

memU's key innovation: dual retrieval combining fast vector search with LLM reasoning.
We adapt this for our sim-first architecture.

## Decision

Implement `DualRetriever` class with:

1. **Heuristic routing**: Detect when deep search helps
2. **Query rewriting**: LLM expands query into multiple searches
3. **RRF merging**: Reciprocal Rank Fusion for result combination

### Architecture

```
recall(query, deep_search=True)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                   DualRetriever                           │
│                                                           │
│  ┌─────────────┐         ┌─────────────────────────────┐ │
│  │ Fast Search │         │ Deep Search (if needed)      │ │
│  │             │         │                              │ │
│  │ storage.    │         │ 1. _needs_deep_search()      │ │
│  │ search()    │         │ 2. _rewrite_query() via LLM  │ │
│  │             │         │ 3. Multiple fast searches    │ │
│  └──────┬──────┘         └──────────────┬───────────────┘ │
│         │                               │                 │
│         └───────────┬───────────────────┘                 │
│                     ▼                                     │
│              ┌─────────────┐                              │
│              │ _merge_rrf()│  Reciprocal Rank Fusion     │
│              └─────────────┘                              │
└──────────────────────────────────────────────────────────┘
```

### Deep Search Heuristics

Triggers when query contains:
- Question words: "who", "what", "when", "where", "why", "how"
- Relationship terms: "related to", "about", "regarding", "involving"
- Temporal terms: "yesterday", "last week", "recently", "before"
- Abstract concepts: "similar", "like", "connections"

### Query Rewriting

LLM transforms query into 2-3 search variations:

```
Input:  "Who do I know at Acme?"
Output: ["Acme", "works at Acme", "Acme Corp employee"]
```

SimLLMProvider already routes `rewrite.*query` prompts to `_sim_query_rewrite()`.

### RRF Merging

Reciprocal Rank Fusion combines results from multiple searches:

```python
score(doc) = Σ 1 / (k + rank_in_list)
```

Where `k=60` (standard constant). Documents appearing in multiple lists get higher scores.

### API Design

```python
class DualRetriever:
    """Dual retrieval: fast search + LLM reasoning."""

    def __init__(
        self,
        storage: SimStorage,
        llm: LLMProvider,
        seed: int | None = None,
    ):
        """Initialize retriever.

        Args:
            storage: Storage backend for searches.
            llm: LLM provider for query rewriting.
            seed: Optional seed for deterministic behavior.
        """

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        deep_search: bool = True,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Entity]:
        """Search with dual retrieval strategy.

        Args:
            query: Search query.
            limit: Maximum results.
            deep_search: Enable LLM-assisted search.
            time_range: Filter by event_time.

        Returns:
            List of entities, ranked by relevance.
        """

    def needs_deep_search(self, query: str) -> bool:
        """Heuristic: does query benefit from LLM reasoning?"""

    async def rewrite_query(self, query: str) -> list[str]:
        """Use LLM to expand query into search variations."""

    def merge_rrf(
        self,
        *result_lists: list[Entity],
        k: int = 60,
    ) -> list[Entity]:
        """Merge results using Reciprocal Rank Fusion."""
```

### TigerStyle Compliance

```python
async def search(self, query: str, limit: int = 10) -> list[Entity]:
    # Preconditions
    assert query, "query must not be empty"
    assert 0 < limit <= SEARCH_LIMIT_MAX, f"limit must be 1-{SEARCH_LIMIT_MAX}"

    # ... implementation ...

    # Postconditions
    assert isinstance(results, list), "must return list"
    assert len(results) <= limit, f"results exceed limit"
    return results
```

## Consequences

### Positive

- **Better recall**: Finds semantically related results
- **Graceful degradation**: Falls back to fast search on LLM failure
- **Deterministic**: Same seed produces same query rewrites
- **Configurable**: `deep_search=False` for fast-only mode

### Negative

- **Latency**: Deep search adds LLM call overhead
- **Complexity**: More code paths to test

### Mitigations

1. **Heuristics**: Only trigger deep search when beneficial
2. **Caching**: Future optimization for repeated queries
3. **Comprehensive tests**: Cover all code paths with sim-first testing

## Implementation

### Phase 4 Files

```
umi/umi-python/umi/
├── retrieval.py         # DualRetriever class (this ADR)
├── memory.py            # Update to use DualRetriever
└── tests/
    └── test_retrieval.py  # DualRetriever tests
```

### SimLLMProvider Integration

Already implemented in Phase 2:
- `_sim_query_rewrite()` generates deterministic query variations
- Routes prompts containing "rewrite" and "query"

## References

- ADR-007: SimLLMProvider
- ADR-008: Memory Class
- memU: Dual retrieval inspiration
- RRF Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
