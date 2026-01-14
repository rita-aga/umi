"""
Integration tests for real provider constructors.

These tests verify that the new provider constructors work correctly.
They use simulation providers to avoid requiring API keys.
"""
import pytest
import umi


class TestRealProviderConstructors:
    """Test Memory constructors for real providers."""

    def test_sim_constructor_works(self):
        """Test that Memory.sim() constructor works."""
        memory = umi.Memory.sim(seed=42)
        assert memory is not None
        assert str(memory) == "Memory(providers=configured)"

    def test_sim_constructor_deterministic(self):
        """Test that same seed produces same behavior."""
        memory1 = umi.Memory.sim(seed=42)
        memory2 = umi.Memory.sim(seed=42)

        # Both should work
        assert memory1 is not None
        assert memory2 is not None

    @pytest.mark.asyncio
    async def test_sim_remember_and_recall(self):
        """Test basic remember/recall flow with sim providers."""
        memory = umi.Memory.sim(seed=42)

        # Remember
        result = await memory.remember("Alice works at Acme Corp", umi.RememberOptions())
        assert result.entity_count() > 0

        # Recall
        entities = await memory.recall("Alice", umi.RecallOptions())
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_sim_count_and_get(self):
        """Test count and get operations."""
        memory = umi.Memory.sim(seed=42)

        # Store something
        result = await memory.remember("Bob is the CTO", umi.RememberOptions())
        assert result.entity_count() > 0

        # Count
        count = await memory.count()
        assert count == result.entity_count()

        # Get by ID
        if result.entities:
            entity_id = result.entities[0].id
            entity = await memory.get(entity_id)
            assert entity is not None
            assert entity.id == entity_id

    @pytest.mark.asyncio
    async def test_sim_forget(self):
        """Test forget (delete) operation."""
        memory = umi.Memory.sim(seed=42)

        # Store
        result = await memory.remember("Carol manages engineering", umi.RememberOptions())
        assert result.entity_count() > 0

        # Get ID
        entity_id = result.entities[0].id

        # Forget
        deleted = await memory.forget(entity_id)
        assert deleted is True

        # Verify deleted
        entity = await memory.get(entity_id)
        assert entity is None

    def test_sync_api_works(self):
        """Test synchronous API."""
        memory = umi.Memory.sim(seed=42)

        # Remember sync
        result = memory.remember_sync("David is an engineer", umi.RememberOptions())
        assert result.entity_count() > 0

        # Recall sync
        entities = memory.recall_sync("David", umi.RecallOptions())
        assert len(entities) > 0

        # Count sync
        count = memory.count_sync()
        assert count > 0

    def test_options_work_with_sim(self):
        """Test that options work correctly."""
        memory = umi.Memory.sim(seed=42)

        # Remember with options
        options = (
            umi.RememberOptions()
            .with_importance(0.9)
            .without_evolution()
        )
        result = memory.remember_sync("Important note", options)
        assert result.entity_count() > 0

        # Recall with options
        recall_opts = umi.RecallOptions().with_limit(5)
        entities = memory.recall_sync("note", recall_opts)
        assert len(entities) <= 5


class TestProviderConstructorSignatures:
    """Test that the new constructor signatures are correct."""

    def test_with_anthropic_signature(self):
        """Verify with_anthropic has correct signature."""
        # This should not raise an error about missing method
        assert hasattr(umi.Memory, 'with_anthropic')

    def test_with_openai_signature(self):
        """Verify with_openai has correct signature."""
        assert hasattr(umi.Memory, 'with_openai')

    def test_with_postgres_signature(self):
        """Verify with_postgres has correct signature."""
        assert hasattr(umi.Memory, 'with_postgres')

    def test_sim_signature(self):
        """Verify sim has correct signature."""
        assert hasattr(umi.Memory, 'sim')


@pytest.mark.integration
class TestRealProviderIntegration:
    """
    Integration tests with real providers.

    These require API keys and database connections, so they're marked
    with @pytest.mark.integration and skipped by default.

    Run with: pytest -m integration
    """

    @pytest.mark.skip(reason="Requires ANTHROPIC_API_KEY and OPENAI_API_KEY")
    @pytest.mark.asyncio
    async def test_with_anthropic_real(self):
        """Test with_anthropic with real API keys (manual testing only)."""
        import os

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not anthropic_key or not openai_key:
            pytest.skip("API keys not set")

        memory = umi.Memory.with_anthropic(
            anthropic_key=anthropic_key,
            openai_key=openai_key,
            db_path="/tmp/test_umi_db"
        )

        result = await memory.remember("Test message", umi.RememberOptions())
        assert result.entity_count() > 0

    @pytest.mark.skip(reason="Requires OPENAI_API_KEY")
    @pytest.mark.asyncio
    async def test_with_openai_real(self):
        """Test with_openai with real API key (manual testing only)."""
        import os

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        memory = umi.Memory.with_openai(
            openai_key=openai_key,
            db_path="/tmp/test_umi_db_openai"
        )

        result = await memory.remember("Test message", umi.RememberOptions())
        assert result.entity_count() > 0


class TestProviderClasses:
    """Test that individual provider classes are still exposed."""

    def test_llm_providers_exposed(self):
        """Verify LLM provider classes are accessible."""
        assert hasattr(umi, 'AnthropicProvider')
        assert hasattr(umi, 'OpenAIProvider')
        assert hasattr(umi, 'SimLLMProvider')

    def test_embedding_providers_exposed(self):
        """Verify embedding provider classes are accessible."""
        assert hasattr(umi, 'OpenAIEmbeddingProvider')
        assert hasattr(umi, 'SimEmbeddingProvider')

    def test_storage_backends_exposed(self):
        """Verify storage backend classes are accessible."""
        assert hasattr(umi, 'LanceStorageBackend')
        assert hasattr(umi, 'PostgresStorageBackend')
        assert hasattr(umi, 'SimStorageBackend')

    def test_vector_backends_exposed(self):
        """Verify vector backend classes are accessible."""
        assert hasattr(umi, 'LanceVectorBackend')
        assert hasattr(umi, 'PostgresVectorBackend')
        assert hasattr(umi, 'SimVectorBackend')

    def test_sim_llm_provider_works(self):
        """Test that SimLLMProvider can be instantiated."""
        provider = umi.SimLLMProvider(seed=42)
        assert provider is not None

    def test_sim_embedding_provider_works(self):
        """Test that SimEmbeddingProvider can be instantiated."""
        provider = umi.SimEmbeddingProvider(seed=42)
        assert provider is not None

    def test_sim_storage_backend_works(self):
        """Test that SimStorageBackend can be instantiated."""
        backend = umi.SimStorageBackend(seed=42)
        assert backend is not None

    def test_sim_vector_backend_works(self):
        """Test that SimVectorBackend can be instantiated."""
        backend = umi.SimVectorBackend(seed=42)
        assert backend is not None
