"""
Umi Python Type Stubs

Type annotations for IDE support and type checking.
"""

from typing import Optional, List

__version__: str

# =============================================================================
# Memory Types
# =============================================================================

class CoreMemory:
    """32KB structured memory core."""

    def __init__(self) -> None: ...
    def write_bytes(self, data: bytes) -> None: ...
    def read_bytes(self, offset: int, length: int) -> bytes: ...
    def size(self) -> int: ...

class WorkingMemory:
    """1MB KV store with TTL."""

    def __init__(self) -> None: ...
    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None: ...
    def get(self, key: str) -> Optional[bytes]: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> None: ...

# =============================================================================
# Entity and Evolution
# =============================================================================

class Entity:
    """Memory entity."""

    id: str
    entity_type: str
    name: str
    content: str
    metadata: dict[str, str]
    embedding: Optional[List[float]]
    created_at: int
    updated_at: int

    def __init__(
        self,
        entity_type: str,
        name: str,
        content: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> None: ...

class EvolutionRelation:
    """Memory evolution relationship."""

    id: str
    source_id: str
    target_id: str
    evolution_type: str
    reason: str
    confidence: float
    created_at: int

# =============================================================================
# Providers - LLM
# =============================================================================

class AnthropicProvider:
    """Anthropic LLM provider (Claude)."""

    def __init__(self, api_key: str) -> None: ...

class OpenAIProvider:
    """OpenAI LLM provider (GPT)."""

    def __init__(self, api_key: str) -> None: ...

class SimLLMProvider:
    """Simulation LLM provider (deterministic)."""

    def __init__(self, seed: int) -> None: ...

# =============================================================================
# Providers - Embedding
# =============================================================================

class OpenAIEmbeddingProvider:
    """OpenAI embedding provider."""

    def __init__(self, api_key: str) -> None: ...

class SimEmbeddingProvider:
    """Simulation embedding provider (deterministic)."""

    def __init__(self, seed: int) -> None: ...

# =============================================================================
# Providers - Storage
# =============================================================================

class LanceStorageBackend:
    """LanceDB storage backend."""

    @staticmethod
    def connect(path: str) -> LanceStorageBackend: ...

class PostgresStorageBackend:
    """Postgres storage backend."""

    @staticmethod
    def connect(url: str) -> PostgresStorageBackend: ...

class SimStorageBackend:
    """Simulation storage backend (in-memory)."""

    def __init__(self, seed: int) -> None: ...

# =============================================================================
# Providers - Vector
# =============================================================================

class LanceVectorBackend:
    """LanceDB vector backend."""

    @staticmethod
    def connect(path: str) -> LanceVectorBackend: ...

class PostgresVectorBackend:
    """Postgres vector backend."""

    @staticmethod
    def connect(url: str) -> PostgresVectorBackend: ...

class SimVectorBackend:
    """Simulation vector backend (in-memory)."""

    def __init__(self, seed: int) -> None: ...

# =============================================================================
# Options and Config
# =============================================================================

class RememberOptions:
    """Options for remember operations."""

    def __init__(self) -> None: ...
    def without_extraction(self) -> RememberOptions: ...
    def without_evolution(self) -> RememberOptions: ...
    def with_importance(self, importance: float) -> RememberOptions: ...
    def without_embeddings(self) -> RememberOptions: ...
    def with_embeddings(self) -> RememberOptions: ...

class RecallOptions:
    """Options for recall operations."""

    def __init__(self) -> None: ...
    def with_limit(self, limit: int) -> RecallOptions: ...
    def with_deep_search(self) -> RecallOptions: ...
    def fast_only(self) -> RecallOptions: ...
    def with_time_range(self, start_ms: int, end_ms: int) -> RecallOptions: ...

class MemoryConfig:
    """Memory configuration."""

    def __init__(self) -> None: ...
    def with_recall_limit(self, limit: int) -> MemoryConfig: ...

# =============================================================================
# Result Types
# =============================================================================

class RememberResult:
    """Result of a remember operation."""

    entities: List[Entity]
    evolutions: List[EvolutionRelation]

    def entity_count(self) -> int: ...
    def has_evolutions(self) -> bool: ...

# =============================================================================
# Memory
# =============================================================================

class Memory:
    """Main memory interface."""

    @staticmethod
    def sim(seed: int) -> Memory:
        """Create Memory with Sim providers (for testing)."""
        ...

    # Sync API
    def remember_sync(self, text: str, options: RememberOptions) -> RememberResult: ...
    def recall_sync(self, query: str, options: RecallOptions) -> List[Entity]: ...
    def forget_sync(self, entity_id: str) -> bool: ...
    def get_sync(self, entity_id: str) -> Optional[Entity]: ...
    def count_sync(self) -> int: ...

# =============================================================================
# Exceptions
# =============================================================================

class UmiError(Exception):
    """Base exception for all Umi errors."""
    pass

class EmptyTextError(UmiError):
    """Raised when text is empty."""
    pass

class TextTooLongError(UmiError):
    """Raised when text exceeds maximum length."""
    pass

class EmptyQueryError(UmiError):
    """Raised when query is empty."""
    pass

class InvalidLimitError(UmiError):
    """Raised when limit is invalid."""
    pass

class StorageError(UmiError):
    """Raised when storage operation fails."""
    pass

class EmbeddingError(UmiError):
    """Raised when embedding generation fails."""
    pass

class ProviderError(UmiError):
    """Raised when LLM/API provider fails."""
    pass
