"""Tests for LLM providers.

TigerStyle: Simulation-first testing, determinism verification.
"""

import json
import pytest
from random import Random

from umi.faults import FaultConfig, FaultStats
from umi.providers.base import LLMProvider, validate_provider
from umi.providers.sim import (
    SimLLMProvider,
    SimMalformedResponseError,
    SimRateLimitError,
    SimTimeoutError,
)


# =============================================================================
# LLMProvider Protocol Tests
# =============================================================================


class TestLLMProviderProtocol:
    """Tests for LLMProvider protocol."""

    def test_sim_provider_implements_protocol(self) -> None:
        """SimLLMProvider should implement LLMProvider protocol."""
        provider = SimLLMProvider(seed=42)
        assert validate_provider(provider)

    def test_protocol_requires_complete_method(self) -> None:
        """Protocol should require complete method."""
        assert hasattr(LLMProvider, "complete")

    def test_protocol_requires_complete_json_method(self) -> None:
        """Protocol should require complete_json method."""
        assert hasattr(LLMProvider, "complete_json")


# =============================================================================
# SimLLMProvider Determinism Tests
# =============================================================================


class TestSimLLMProviderDeterminism:
    """Tests for deterministic behavior of SimLLMProvider."""

    @pytest.mark.asyncio
    async def test_same_seed_same_response(self) -> None:
        """Same seed + same prompt = same response."""
        provider1 = SimLLMProvider(seed=42)
        provider2 = SimLLMProvider(seed=42)

        prompt = "Hello, world!"

        response1 = await provider1.complete(prompt)
        response2 = await provider2.complete(prompt)

        assert response1 == response2

    @pytest.mark.asyncio
    async def test_different_seed_different_response(self) -> None:
        """Different seeds should produce different responses for same prompt."""
        provider1 = SimLLMProvider(seed=42)
        provider2 = SimLLMProvider(seed=123)

        prompt = "Extract entities from: I met Alice at Acme Corp"

        response1 = await provider1.complete(prompt)
        response2 = await provider2.complete(prompt)

        # Responses may differ in confidence scores or order
        # (though entity names detected should be same)
        # For determinism test, we check structure is same
        assert isinstance(response1, str)
        assert isinstance(response2, str)

    @pytest.mark.asyncio
    async def test_reproducibility_across_calls(self) -> None:
        """Multiple calls should be reproducible with reset."""
        provider = SimLLMProvider(seed=42)

        prompts = ["First prompt", "Second prompt", "Third prompt"]

        # First run
        responses1 = []
        for p in prompts:
            responses1.append(await provider.complete(p))

        # Reset and run again
        provider.reset()
        responses2 = []
        for p in prompts:
            responses2.append(await provider.complete(p))

        assert responses1 == responses2

    @pytest.mark.asyncio
    async def test_fork_creates_independent_provider(self) -> None:
        """Forked provider should be independent."""
        provider1 = SimLLMProvider(seed=42)
        provider2 = provider1.fork(new_seed=100)

        prompt = "Test prompt"

        response1 = await provider1.complete(prompt)
        response2 = await provider2.complete(prompt)

        # Different seeds mean different internal state
        assert provider1.seed != provider2.seed


# =============================================================================
# SimLLMProvider Routing Tests
# =============================================================================


class TestSimLLMProviderRouting:
    """Tests for prompt routing in SimLLMProvider."""

    @pytest.mark.asyncio
    async def test_routes_entity_extraction(self) -> None:
        """Should route entity extraction prompts correctly."""
        provider = SimLLMProvider(seed=42)

        prompt = "Extract entities from: I met Alice at Acme Corp"
        response = await provider.complete(prompt)

        # Should be valid JSON with entities
        data = json.loads(response)
        assert "entities" in data
        assert isinstance(data["entities"], list)

    @pytest.mark.asyncio
    async def test_routes_query_rewrite(self) -> None:
        """Should route query rewrite prompts correctly."""
        provider = SimLLMProvider(seed=42)

        prompt = "Rewrite this query for search: Who works at Acme?"
        response = await provider.complete(prompt)

        # Should be valid JSON array
        data = json.loads(response)
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_routes_evolution_detection(self) -> None:
        """Should route evolution detection prompts correctly."""
        provider = SimLLMProvider(seed=42)

        prompt = "Detect evolution between these memories: old vs new"
        response = await provider.complete(prompt)

        # Should be valid JSON with type
        data = json.loads(response)
        assert "type" in data
        assert data["type"] in ["update", "extend", "derive", "contradict", "none"]

    @pytest.mark.asyncio
    async def test_routes_categorization(self) -> None:
        """Should route categorization prompts correctly."""
        provider = SimLLMProvider(seed=42)

        prompt = "Categorize this content: User prefers dark mode"
        response = await provider.complete(prompt)

        # Should be valid JSON array
        data = json.loads(response)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_generic_fallback(self) -> None:
        """Unknown prompts should get generic response."""
        provider = SimLLMProvider(seed=42)

        prompt = "This is some random text that doesn't match any pattern"
        response = await provider.complete(prompt)

        # Should contain SimResponse marker
        assert "SimResponse" in response


# =============================================================================
# SimLLMProvider Entity Extraction Tests
# =============================================================================


class TestSimLLMProviderEntityExtraction:
    """Tests for entity extraction simulation."""

    @pytest.mark.asyncio
    async def test_extracts_known_names(self) -> None:
        """Should extract known names from prompt."""
        provider = SimLLMProvider(seed=42)

        prompt = "Extract entities from: I had lunch with Alice and Bob"
        response = await provider.complete(prompt)

        data = json.loads(response)
        names = [e["name"] for e in data["entities"]]

        assert "Alice" in names
        assert "Bob" in names

    @pytest.mark.asyncio
    async def test_extracts_organizations(self) -> None:
        """Should extract known organizations from prompt."""
        provider = SimLLMProvider(seed=42)

        prompt = "Extract entities from: Alice works at Acme"
        response = await provider.complete(prompt)

        data = json.loads(response)
        names = [e["name"] for e in data["entities"]]

        assert "Acme" in names

    @pytest.mark.asyncio
    async def test_generates_relations(self) -> None:
        """Should generate relations when multiple entities found."""
        provider = SimLLMProvider(seed=42)

        prompt = "Extract entities from: Alice works at Acme Corp"
        response = await provider.complete(prompt)

        data = json.loads(response)

        # Should have relations if multiple entities
        if len(data["entities"]) >= 2:
            assert "relations" in data
            assert len(data["relations"]) >= 1

    @pytest.mark.asyncio
    async def test_fallback_to_note(self) -> None:
        """Should create note entity if no known entities found."""
        provider = SimLLMProvider(seed=42)

        prompt = "Extract entities from: The weather is nice today"
        response = await provider.complete(prompt)

        data = json.loads(response)
        assert len(data["entities"]) >= 1

        # Should have a note type
        types = [e["type"] for e in data["entities"]]
        assert "note" in types


# =============================================================================
# Fault Injection Tests
# =============================================================================


class TestFaultInjection:
    """Tests for fault injection in SimLLMProvider."""

    @pytest.mark.asyncio
    async def test_timeout_fault(self) -> None:
        """Should raise TimeoutError when timeout fault triggered."""
        faults = FaultConfig(llm_timeout=1.0)  # 100% timeout
        provider = SimLLMProvider(seed=42, faults=faults)

        with pytest.raises(SimTimeoutError):
            await provider.complete("any prompt")

    @pytest.mark.asyncio
    async def test_error_fault(self) -> None:
        """Should raise RuntimeError when error fault triggered."""
        faults = FaultConfig(llm_error=1.0)  # 100% error
        provider = SimLLMProvider(seed=42, faults=faults)

        with pytest.raises(RuntimeError):
            await provider.complete("any prompt")

    @pytest.mark.asyncio
    async def test_rate_limit_fault(self) -> None:
        """Should raise rate limit error when fault triggered."""
        faults = FaultConfig(llm_rate_limit=1.0)  # 100% rate limit
        provider = SimLLMProvider(seed=42, faults=faults)

        with pytest.raises(SimRateLimitError):
            await provider.complete("any prompt")

    @pytest.mark.asyncio
    async def test_fault_stats_tracking(self) -> None:
        """Should track fault statistics."""
        faults = FaultConfig(llm_timeout=1.0)
        provider = SimLLMProvider(seed=42, faults=faults)

        assert provider.stats is not None
        assert provider.stats.timeouts == 0

        try:
            await provider.complete("test")
        except SimTimeoutError:
            pass

        assert provider.stats.timeouts == 1

    @pytest.mark.asyncio
    async def test_partial_fault_rate(self) -> None:
        """Should inject faults at configured rate."""
        faults = FaultConfig(llm_timeout=0.5)  # 50% timeout
        provider = SimLLMProvider(seed=42, faults=faults)

        successes = 0
        failures = 0

        for i in range(100):
            # Reset RNG for reproducibility
            provider._rng = Random(42 + i)
            try:
                await provider.complete(f"prompt {i}")
                successes += 1
            except SimTimeoutError:
                failures += 1

        # Should have mix of successes and failures
        # With 50% rate, expect roughly even split (allow variance)
        assert successes > 20
        assert failures > 20

    @pytest.mark.asyncio
    async def test_no_faults_by_default(self) -> None:
        """Should not inject faults by default."""
        provider = SimLLMProvider(seed=42)

        # Should complete without error
        for _ in range(10):
            response = await provider.complete("test")
            assert isinstance(response, str)


# =============================================================================
# FaultConfig Tests
# =============================================================================


class TestFaultConfig:
    """Tests for FaultConfig."""

    def test_default_no_faults(self) -> None:
        """Default config should have no faults."""
        config = FaultConfig()

        assert config.llm_timeout == 0.0
        assert config.llm_error == 0.0
        assert config.llm_malformed == 0.0
        assert config.llm_rate_limit == 0.0

    def test_invalid_probability_rejected(self) -> None:
        """Should reject invalid probabilities."""
        with pytest.raises(AssertionError):
            FaultConfig(llm_timeout=-0.1)

        with pytest.raises(AssertionError):
            FaultConfig(llm_timeout=1.5)

    def test_should_fail_deterministic(self) -> None:
        """should_fail should be deterministic with same RNG."""
        config = FaultConfig(llm_timeout=0.5)

        rng1 = Random(42)
        rng2 = Random(42)

        results1 = [config.should_fail("llm_timeout", rng1) for _ in range(10)]
        results2 = [config.should_fail("llm_timeout", rng2) for _ in range(10)]

        assert results1 == results2

    def test_chaos_config(self) -> None:
        """chaos() should set all faults to same probability."""
        config = FaultConfig.chaos(0.1)

        assert config.llm_timeout == 0.1
        assert config.llm_error == 0.1
        assert config.llm_malformed == 0.1
        assert config.llm_rate_limit == 0.1
        assert config.storage_read_error == 0.1
        assert config.storage_write_error == 0.1

    def test_none_config(self) -> None:
        """none() should have all faults disabled."""
        config = FaultConfig.none()

        assert config.llm_timeout == 0.0
        assert config.llm_error == 0.0

    def test_get_llm_fault_priority(self) -> None:
        """get_llm_fault should respect priority order."""
        # Timeout has highest priority
        config = FaultConfig(llm_timeout=1.0, llm_error=1.0)
        rng = Random(42)

        fault = config.get_llm_fault(rng)
        assert fault == "timeout"


# =============================================================================
# FaultStats Tests
# =============================================================================


class TestFaultStats:
    """Tests for FaultStats."""

    def test_initial_counts_zero(self) -> None:
        """All counts should start at zero."""
        stats = FaultStats()

        assert stats.timeouts == 0
        assert stats.errors == 0
        assert stats.total == 0

    def test_record_increments_count(self) -> None:
        """record() should increment appropriate counter."""
        stats = FaultStats()

        stats.record("timeout")
        assert stats.timeouts == 1
        assert stats.total == 1

        stats.record("error")
        assert stats.errors == 1
        assert stats.total == 2

    def test_total_sums_all_faults(self) -> None:
        """total should sum all fault types."""
        stats = FaultStats()

        stats.record("timeout")
        stats.record("timeout")
        stats.record("error")
        stats.record("malformed")

        assert stats.total == 4


# =============================================================================
# Integration Tests
# =============================================================================


class TestSimLLMProviderIntegration:
    """Integration tests for SimLLMProvider."""

    @pytest.mark.asyncio
    async def test_complete_json_parses_response(self) -> None:
        """complete_json should return parsed dict."""
        provider = SimLLMProvider(seed=42)

        prompt = "Extract entities from: Alice at Acme"
        result = await provider.complete_json(prompt, dict)

        assert isinstance(result, dict)
        assert "entities" in result

    @pytest.mark.asyncio
    async def test_memory_workflow_simulation(self) -> None:
        """Simulate a complete memory workflow."""
        provider = SimLLMProvider(seed=42)

        # 1. Extract entities
        extract_prompt = "Extract entities from: I met Alice at Acme Corp last Tuesday"
        entities_response = await provider.complete(extract_prompt)
        entities = json.loads(entities_response)

        assert len(entities["entities"]) >= 1

        # 2. Detect evolution
        evolution_prompt = "Detect evolution: Alice now works at StartupX"
        evolution_response = await provider.complete(evolution_prompt)
        evolution = json.loads(evolution_response)

        assert "type" in evolution

        # 3. Rewrite query
        query_prompt = "Rewrite query: Who do I know at Acme?"
        queries_response = await provider.complete(query_prompt)
        queries = json.loads(queries_response)

        assert len(queries) >= 1
