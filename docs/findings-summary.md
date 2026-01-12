# Umi Rust Ecosystem Feasibility: Complete Findings

**Date:** 2026-01-11
**Status:** Research Complete

---

## Overview

This document summarizes the feasibility analysis for building a full Rust AI agent ecosystem using Umi as the memory foundation. The analysis covers three key questions:

1. Can Umi be fully implemented in Rust?
2. Can Letta (stateful agent framework) be rewritten in Rust and use Umi?
3. Can an aidnn-like system (multi-agent data analytics with continual learning) be built in Rust?

**Bottom line:** Yes to all three, with a recommended hybrid approach where Rust handles runtime and Python handles RL training.

---

## Part 1: Umi Full Rust Feasibility

### Current State

```
┌─────────────────────────────────────────────────────────────┐
│                    Current Umi Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Python Layer (umi/)              Rust Core (umi-core/)     │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │ EntityExtractor     │         │ CoreMemory (32KB)   │   │
│  │ DualRetriever       │         │ WorkingMemory (1MB) │   │
│  │ EvolutionTracker    │         │ ArchivalMemory      │   │
│  │ SimLLMProvider      │         │ DST Framework       │   │
│  │ SimStorage          │         │ SimStorageBackend   │   │
│  │ 145 tests           │         │ 232 tests           │   │
│  └─────────────────────┘         └─────────────────────┘   │
│           │                                │                │
│           └──────── NOT CONNECTED ─────────┘                │
│                   (PyO3 scaffolded but not wired)           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### What Python Does (Would Move to Rust)

| Component | Python Implementation | Rust Feasibility |
|-----------|----------------------|------------------|
| Entity extraction | LLM prompt + JSON parsing | `reqwest` + `serde` |
| Dual retrieval | String matching + LLM query expansion | Native ops + HTTP |
| Evolution tracking | LLM comparison + scoring | Same pattern |
| SimLLMProvider | Seeded RNG + pattern matching | Already have DST |
| Storage | In-memory dict | Already exists |

### Assessment

**Full Rust is not just feasible—it's architecturally cleaner** because:

1. Umi's core value is **deterministic simulation testing** — already in Rust
2. Python layer is thin — mostly prompts + HTTP + JSON
3. No heavy Python dependencies (no numpy, torch, etc.)
4. Single language eliminates PyO3 complexity

### When to Keep Python

| Keep Python When | Reason |
|------------------|--------|
| Primary consumers are Python frameworks | Native integration |
| Interfacing with ML/scientific libs | No Rust equivalent |
| Rapid prototyping / research | Faster iteration |
| Dynamic code execution needed | `exec`/`eval` flexibility |

---

## Part 2: Letta Rust Rewrite Feasibility

### What is Letta?

[Letta](https://github.com/letta-ai/letta) (formerly MemGPT) is a stateful agent framework with:
- Multi-tier memory (core/archival, like an OS)
- Tool execution (sandbox, core, external)
- Context management
- Multi-agent orchestration

### Dependency Mapping

| Letta Python | Rust Equivalent |
|--------------|-----------------|
| FastAPI, Uvicorn | `axum`, `tower` |
| SQLAlchemy | `sqlx`, `sea-orm` |
| Alembic | `sqlx-migrate`, `refinery` |
| openai, anthropic SDKs | `reqwest` + `serde` |
| pgvector | `pgvector-rs` |
| OpenTelemetry | `tracing-opentelemetry` |
| asyncio | `tokio` |
| pydantic | `serde` |

**Every Python dependency has a mature Rust equivalent.**

### Why Rust Works for Letta

1. **Letta is I/O-bound** — HTTP calls to LLMs, database queries
2. **Agent loop is algorithmic** — Pure logic, no Python magic
3. **Memory model maps to Rust** — Umi already implements the tiers
4. **Tool execution is simpler** — Type-safe traits > duck typing
5. **No Python-specific deps** — No ML inference, just API calls

### Memory Architecture Alignment

| Letta Concept | Umi Rust Equivalent |
|---------------|---------------------|
| Memory Blocks (persona, human) | `CoreMemory` typed blocks |
| Archival Memory | `ArchivalMemory` with backend trait |
| Message History | `WorkingMemory` with TTL |
| Memory search | `DualRetriever` (needs porting) |
| Context management | `CoreMemory` capacity limits |

### Why Rewrite Letta in Rust?

| Factor | Benefit |
|--------|---------|
| **Deployment** | Single binary, no Python runtime |
| **Concurrency** | No GIL, true parallelism |
| **Safety** | Memory safe, no runtime crashes |
| **Testing** | DST enables deterministic agent testing |
| **Agent development** | Compiler feedback accelerates coding agents |

### Key Insight

> *"Rust's strictness that slows humans actually accelerates coding agents through precise, immediate error feedback."*

In a world where coding agents write code, the compiler is an ally that catches errors instantly rather than at runtime.

---

## Part 3: aidnn Replication Feasibility

### What is aidnn?

[Isotopes AI's aidnn](https://isotopes.ai/) is a multi-agent data analytics platform:
- Natural language queries over enterprise data
- Multi-agent orchestration (data, cleaning, analysis, visualization)
- Learns from user expertise
- Automated reports

### Architectural Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rust aidnn Equivalent                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Letta-rs (Agent Framework)              │  │
│  │  Agent Loop │ Tool System │ Multi-Agent Orchestrator     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                    Umi-rs (Memory Layer)                   │  │
│  │  CoreMemory │ ArchivalMemory │ EvolutionTracker           │  │
│  │  EntityExtractor │ DualRetriever │ User Expertise Memory  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                   Storage Backends                         │  │
│  │  PostgreSQL + pgvector │ Qdrant │ Redis                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Continual Learning: Three Levels

| Level | Mechanism | Implementation | Training? |
|-------|-----------|----------------|-----------|
| **1. Memory-based** | Store expertise as entities | Umi (Rust) | No |
| **2. Prompt optimization** | Bandit selects best prompts | Rust | No |
| **3. RL fine-tuning** | Train on traces (PPO/GRPO) | Python | Yes |

**Levels 1-2 give 80% of value with zero training infrastructure.**

### The RL Training Boundary

```
┌──────────────────────────────┐      ┌──────────────────────────┐
│       Rust Runtime           │      │    Python Training       │
│                              │      │                          │
│  • Agent execution           │─────▶│  • RL algorithms         │
│  • Memory (Umi)              │      │  • Fine-tuning           │
│  • Trace collection          │◀─────│  • Weight updates        │
│                              │      │                          │
│  95% of code                 │      │  5% of code              │
│  100% of runtime             │      │  Batch training only     │
└──────────────────────────────┘      └──────────────────────────┘
```

Following [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning) pattern:
- Rust agents collect traces during execution
- Python pipeline trains on traces
- Fine-tuned models deployed back to Rust

---

## Part 4: When Python Still Wins

Even with coding agents writing code, Python is better for:

| Scenario | Why Python Wins |
|----------|-----------------|
| **ML library interface** | transformers, torch have no Rust equivalent |
| **Dynamic plugins** | `exec`/`eval` for runtime code loading |
| **Bleeding-edge LLM features** | Official SDKs ship Python first |
| **Glue code** | Thin orchestration doesn't benefit from Rust |
| **Errors as signals** | Runtime errors useful during exploration |

### Mental Model

**Python is better when boundaries are fuzzy:**
- Fuzzy input formats → dynamic typing helps
- Fuzzy requirements → rapid iteration helps
- Fuzzy integrations → duck typing helps

**Rust is better when boundaries are crisp:**
- Well-defined protocols → types enforce correctness
- Stable requirements → compiler catches regressions
- Clear interfaces → traits document contracts

---

## Part 5: Recommended Architecture

### Full Rust Agent Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Deployment                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Rust Agent Server                          │ │
│  │                                                            │ │
│  │  axum REST API                                             │ │
│  │       │                                                    │ │
│  │       ├── Letta-rs Orchestrator                            │ │
│  │       │       ├── Specialized Agents                       │ │
│  │       │       └── Tool Execution                           │ │
│  │       │                                                    │ │
│  │       ├── Umi-rs Memory                                    │ │
│  │       │       ├── CoreMemory (context)                     │ │
│  │       │       ├── ArchivalMemory (history)                 │ │
│  │       │       └── EvolutionTracker (learning)              │ │
│  │       │                                                    │ │
│  │       └── TraceLogger ──────────────────────┐              │ │
│  │                                              │              │ │
│  └──────────────────────────────────────────────┼──────────────┘ │
│                                                 │                │
│                                                 ▼                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Training Pipeline (Python)                 │ │
│  │                                                            │ │
│  │  Traces ──▶ Credit Assignment ──▶ PPO/GRPO ──▶ Fine-tuned │ │
│  │                                                    Model   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

```
Phase 1: Complete Umi-rs
├── Port EntityExtractor to Rust
├── Port DualRetriever to Rust
├── Port EvolutionTracker to Rust
├── Add storage backends (Postgres, Qdrant)
└── Publish to crates.io

Phase 2: Build Letta-rs
├── Core agent loop
├── Umi integration for memory
├── LLM provider traits
├── Tool execution system
└── REST API (axum)

Phase 3: Multi-Agent System
├── Agent orchestrator
├── Specialized agent types
├── Data connectors
└── Trace collection

Phase 4: Continual Learning
├── Prompt optimization (Rust)
├── RL trace format
├── Python training pipeline
└── Model deployment
```

---

## Part 6: Tradeoffs Summary

### Comparison Matrix

| Aspect | Full Python | Hybrid (Rust + Python RL) | Full Rust |
|--------|-------------|---------------------------|-----------|
| Agent performance | Adequate | Optimal | Optimal |
| RL training | Native | Native Python | Immature |
| Memory safety | Runtime errors | Safe runtime | Fully safe |
| Deployment | Complex deps | Two components | Single binary |
| Concurrency | GIL-limited | True parallelism | True parallelism |
| LLM SDK support | Official SDKs | HTTP-based | HTTP-based |
| DST testing | Harder | Full support | Full support |

### When to Choose Each

| Choose This | When |
|-------------|------|
| **Full Python** | Prototyping, Python-heavy ecosystem, ML inference |
| **Hybrid** | Production agents with RL, best of both worlds |
| **Full Rust** | Embedded, edge, no RL needed, maximum simplicity |

---

## Part 7: Key Insights

### 1. The Compiler as Ally

For coding agents, Rust's strict compiler provides immediate, precise feedback. What slows human developers accelerates agent developers.

### 2. The RL Training Boundary

RL training requires CUDA/PyTorch. The clean architecture is:
- Rust: Collect traces during agent execution
- Python: Train on traces, produce fine-tuned models
- Rust: Serve fine-tuned models

### 3. Memory-Based Learning Gets You Far

Before RL, memory-based learning (storing expertise as entities, retrieving for context) provides significant value with zero training infrastructure.

### 4. Single Binary Deployment

A Rust agent server with no Python runtime is transformative for deployment at scale, edge computing, and embedded systems.

### 5. DST is a Superpower

Deterministic Simulation Testing (same seed = same result) enables:
- Reproducible bugs
- Property-based testing
- Chaos engineering
- Regression detection

Rust's type system makes DST guarantees stronger.

---

## Conclusion

Building a full Rust AI agent ecosystem with Umi + Letta is:

- **Possible:** Every component has mature Rust equivalents
- **Feasible:** 10-12 weeks for coding agents to build
- **Beneficial:** Performance, safety, deployment simplicity, DST

The recommended approach is hybrid:
- **Rust** for agent runtime, memory, deployment
- **Python** for RL training (until Rust ML matures)

This gives the benefits of both worlds while maintaining a clean architectural boundary.

---

## Part 8: Trajectories - The Missing Primitive

Based on analysis of the position paper "Trajectories as a Foundation for Continual Learning Agents" and training frameworks (Agent Lightning, Fireworks RFT, Unsloth ART).

### The Insight

> "Agents that cannot learn from their own experience are fundamentally limited."

Trajectories are structured decision traces that enable:

| Capability | Description | Training Required? |
|------------|-------------|-------------------|
| **Display** | Render conversation history | No |
| **Context Learning** | Retrieve similar examples | No |
| **Simulation** | Predict counterfactuals | No |
| **RL Training** | Fine-tune on rewards | Yes |

### The Continual Learning Funnel

```
Execution → Context Learning → Simulation → RL Training
   │              │                │              │
   │         No training      No training    Requires GPU
   │         +15-25% gain     "What if?"     Compounding
   │                                         improvement
   └── All stages use the same trajectory format ──┘
```

### Training Framework Integration

| Framework | Provider | Integration |
|-----------|----------|-------------|
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | Microsoft | Best for decoupled Rust agent + Python training |
| [Fireworks RFT](https://fireworks.ai/blog/fireworks-rft) | Fireworks | Managed service, export rollouts |
| [Unsloth ART](https://github.com/OpenPipe/ART) | OpenPipe | Open source, RULER auto-rewards |

### Architecture with Trajectories

```
┌────────────────────────────────────────┐
│      Rust Runtime (Letta-rs + Umi)     │
│                                        │
│  Agent Loop → Trajectory Collector     │
│       │              │                 │
│       ▼              ▼                 │
│  Umi Memory ← Trajectory Store         │
│       │              │                 │
│  Context     Simulation    Export      │
│  Learning    Engine        │           │
└──────────────────────────────┼─────────┘
                               ▼
┌──────────────────────────────────────────┐
│    Python Training (Agent Lightning)     │
│    LightningRL → GRPO → Improved Model   │
└──────────────────────────────────────────┘
```

### Key Insight: Optimization Statistics

Trajectories don't just improve agents—they improve the entire stack:

- **AGENTS.md** - Evolve instructions based on what worked
- **Skills** - Discover gaps, deprecate underused tools
- **MCP** - Behavioral tool recommendations from history

### LLM-Extracted Decisions and Entities

Decisions and entities are **extracted post-hoc by LLMs**, not captured at runtime:

```
Raw Trajectory → LLM Extraction → Enriched Trajectory
                                  ├── decisions: [{type, reasoning, alternatives}]
                                  └── entities: [{name, type, importance}]
```

This enables:
- Working on any agent's trajectories (not just your own)
- Re-extracting with better prompts
- Inferring implicit decisions

### GRPO Variants with Per-Step Rewards

| Variant | Per-Step? | Source |
|---------|-----------|--------|
| Vanilla GRPO | No | DeepSeek |
| GRPO + LightningRL | Yes | Microsoft |
| GSPO | Yes | Qwen/Alibaba |
| GRPO + PRM | Yes | OpenAI-style |

### Tinker (Thinking Machines Lab)

Token-level training primitives from Mira Murati's new company:

```python
tinker.forward_backward(tokens, rewards)  # Token-level
tinker.sample(prompt)                      # Generate
tinker.update()                            # Update weights
```

Best used with Agent Lightning for credit assignment.

### How Decisions/Entities Flow Through Training

```
Decisions/Entities → Better Rewards → GRPO Training
                   → Better Grouping →
                   → Better Credit Assignment →

The structure doesn't enter gradients directly,
but it COMPUTES the rewards that drive gradients.
```

See `docs/trajectories-continual-learning.md` for full analysis.

---

## Related Documents

- `docs/letta-rust-feasibility.md` - Detailed Letta analysis
- `docs/isotopes-rust-replication.md` - aidnn replication analysis
- `docs/trajectories-continual-learning.md` - Trajectory primitive and training frameworks
- `.vision/umi-vision.md` - Umi core principles
- `.progress/001_*_umi-standalone-roadmap.md` - Implementation roadmap

## Sources

- [Letta GitHub](https://github.com/letta-ai/letta)
- [Isotopes AI](https://isotopes.ai/)
- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Fireworks RFT](https://fireworks.ai/blog/fireworks-rft)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [OpenPipe ART](https://github.com/OpenPipe/ART)
- [Tinker - Thinking Machines Lab](https://thinkingmachines.ai/tinker/)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
