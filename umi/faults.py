"""FaultConfig - Configuration for fault injection in simulation.

TigerStyle: Explicit fault probabilities, deterministic injection.

Enables systematic testing of error handling by injecting:
- LLM failures (timeout, error, malformed response)
- Storage failures (read/write errors, latency)

Example:
    >>> faults = FaultConfig(llm_timeout=0.1)  # 10% timeout rate
    >>> provider = SimLLMProvider(seed=42, faults=faults)
    >>> # ~10% of calls will raise TimeoutError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Literal, Optional

# Fault type literals for type safety
LLMFaultType = Literal["timeout", "error", "malformed", "rate_limit"]
StorageFaultType = Literal["read_error", "write_error"]


@dataclass(frozen=True)
class FaultConfig:
    """Configuration for fault injection in simulation testing.

    All probabilities are floats between 0.0 (never) and 1.0 (always).

    Attributes:
        llm_timeout: Probability of LLM request timing out.
        llm_error: Probability of LLM API error.
        llm_malformed: Probability of unparseable LLM response.
        llm_rate_limit: Probability of rate limit error.
        storage_read_error: Probability of storage read failure.
        storage_write_error: Probability of storage write failure.
        storage_latency_ms: Additional latency for storage ops (deterministic).

    Example:
        >>> # 50% chance of timeout
        >>> faults = FaultConfig(llm_timeout=0.5)
        >>> assert 0.0 <= faults.llm_timeout <= 1.0

        >>> # Combine multiple faults
        >>> faults = FaultConfig(llm_timeout=0.1, llm_error=0.05)
    """

    # LLM faults
    llm_timeout: float = 0.0
    llm_error: float = 0.0
    llm_malformed: float = 0.0
    llm_rate_limit: float = 0.0

    # Storage faults (passed to Rust layer)
    storage_read_error: float = 0.0
    storage_write_error: float = 0.0
    storage_latency_ms: int = 0

    def __post_init__(self) -> None:
        """Validate all probabilities are in [0.0, 1.0]."""
        # TigerStyle: Preconditions
        for name in [
            "llm_timeout",
            "llm_error",
            "llm_malformed",
            "llm_rate_limit",
            "storage_read_error",
            "storage_write_error",
        ]:
            value = getattr(self, name)
            assert 0.0 <= value <= 1.0, f"{name} must be between 0.0 and 1.0, got {value}"

        assert self.storage_latency_ms >= 0, (
            f"storage_latency_ms must be non-negative, got {self.storage_latency_ms}"
        )

    def should_fail(self, fault_type: str, rng: Random) -> bool:
        """Check if a fault should be triggered.

        Uses the provided RNG for deterministic behavior.

        Args:
            fault_type: Type of fault to check (e.g., "llm_timeout", "storage_read_error")
            rng: Random instance for deterministic checks.

        Returns:
            True if fault should be triggered.

        Example:
            >>> import random
            >>> faults = FaultConfig(llm_timeout=0.5)
            >>> rng = random.Random(42)
            >>> result = faults.should_fail("llm_timeout", rng)
            >>> assert isinstance(result, bool)
        """
        # Map fault_type to attribute name
        attr_map = {
            "timeout": "llm_timeout",
            "llm_timeout": "llm_timeout",
            "error": "llm_error",
            "llm_error": "llm_error",
            "malformed": "llm_malformed",
            "llm_malformed": "llm_malformed",
            "rate_limit": "llm_rate_limit",
            "llm_rate_limit": "llm_rate_limit",
            "read_error": "storage_read_error",
            "storage_read_error": "storage_read_error",
            "write_error": "storage_write_error",
            "storage_write_error": "storage_write_error",
        }

        attr_name = attr_map.get(fault_type)
        if attr_name is None:
            return False

        probability = getattr(self, attr_name, 0.0)
        return rng.random() < probability

    def get_llm_fault(self, rng: Random) -> LLMFaultType | None:
        """Get which LLM fault should be triggered, if any.

        Checks faults in priority order: timeout > error > rate_limit > malformed.

        Args:
            rng: Random instance for deterministic checks.

        Returns:
            Fault type to trigger, or None if no fault.

        Example:
            >>> import random
            >>> faults = FaultConfig(llm_timeout=1.0)  # Always timeout
            >>> rng = random.Random(42)
            >>> assert faults.get_llm_fault(rng) == "timeout"
        """
        # Check in priority order
        if self.should_fail("llm_timeout", rng):
            return "timeout"
        if self.should_fail("llm_error", rng):
            return "error"
        if self.should_fail("llm_rate_limit", rng):
            return "rate_limit"
        if self.should_fail("llm_malformed", rng):
            return "malformed"
        return None

    @classmethod
    def chaos(cls, probability: float = 0.1) -> "FaultConfig":
        """Create a chaos testing configuration with uniform fault probability.

        Args:
            probability: Probability for all fault types.

        Returns:
            FaultConfig with all faults set to the given probability.

        Example:
            >>> faults = FaultConfig.chaos(0.05)  # 5% failure rate
            >>> assert faults.llm_timeout == 0.05
        """
        return cls(
            llm_timeout=probability,
            llm_error=probability,
            llm_malformed=probability,
            llm_rate_limit=probability,
            storage_read_error=probability,
            storage_write_error=probability,
        )

    @classmethod
    def none(cls) -> "FaultConfig":
        """Create a configuration with no faults (default).

        Returns:
            FaultConfig with all probabilities set to 0.0.
        """
        return cls()


@dataclass
class FaultStats:
    """Statistics for fault injection during a simulation run.

    Tracks how many faults were injected for debugging and validation.
    """

    timeouts: int = 0
    errors: int = 0
    malformed: int = 0
    rate_limits: int = 0
    read_errors: int = 0
    write_errors: int = 0

    def record(self, fault_type: str) -> None:
        """Record a fault occurrence."""
        if fault_type == "timeout":
            self.timeouts += 1
        elif fault_type == "error":
            self.errors += 1
        elif fault_type == "malformed":
            self.malformed += 1
        elif fault_type == "rate_limit":
            self.rate_limits += 1
        elif fault_type == "read_error":
            self.read_errors += 1
        elif fault_type == "write_error":
            self.write_errors += 1

    @property
    def total(self) -> int:
        """Total number of faults injected."""
        return (
            self.timeouts
            + self.errors
            + self.malformed
            + self.rate_limits
            + self.read_errors
            + self.write_errors
        )
