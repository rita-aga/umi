# Kelpie: Where It Fits in the Rust Agent Ecosystem

## Executive Summary

**What is Kelpie?** A distributed virtual actor system for AI agents—essentially "Letta in Rust" with cluster coordination and sandbox isolation.

**How does it relate to Umi?** They overlap in memory (both have 3-tier Core/Working/Archival) but serve different roles: Umi is a memory library, Kelpie is an agent runtime.

**Bottom line:** Kelpie + Trajectories + GRPO can replicate Isotopes.ai. Umi's role depends on whether you use Kelpie's built-in memory or want Umi's specialized features.

---

## Part 1: Kelpie Architecture Overview

### What Kelpie Provides

```
┌─────────────────────────────────────────────────────────────────┐
│                         Kelpie Stack                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    REST API (axum)                        │   │
│  │               Letta-compatible endpoints                  │   │
│  │               SSE streaming support                       │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Runtime Dispatcher                      │   │
│  │               Virtual actor coordination                   │   │
│  │               Linearizability guarantees                   │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    LLM Client Layer                        │   │
│  │               Anthropic Claude │ OpenAI GPT               │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Tool Execution                          │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐ │   │
│  │  │ MCP Integration │  │ Sandbox Isolation               │ │   │
│  │  │ (dynamic tools) │  │ Process (<10ms) / Firecracker   │ │   │
│  │  └─────────────────┘  └─────────────────────────────────┘ │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Memory System                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │   │
│  │  │ CoreMemory  │  │ Working     │  │ Archival        │    │   │
│  │  │ (~32KB)     │  │ Memory      │  │ Memory          │    │   │
│  │  │ persistent  │  │ (session)   │  │ (vector search) │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Cluster Layer                           │   │
│  │               kelpie-cluster for horizontal scaling        │   │
│  │               FoundationDB integration (planned)           │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    DST (Testing)                          │   │
│  │  Storage │ Network │ Crash │ Temporal fault injection     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Differentiators

| Feature | Kelpie | Letta (Python) |
|---------|--------|----------------|
| Language | Rust (92.3%) | Python |
| Memory tiers | 3 (Core/Working/Archival) | 3 (same) |
| Actor model | Virtual actors with linearizability | Process-based |
| Sandbox | Firecracker microVM | Not built-in |
| Scaling | Cluster coordination | Single-node focused |
| DST | Comprehensive fault injection | Limited |
| MCP | Native integration | Plugin-based |
| API compat | Letta-compatible REST | Native |

---

## Part 2: Kelpie vs Umi - Overlap and Complementarity

### The Overlap: Memory Systems

Both have 3-tier memory architectures:

| Tier | Umi | Kelpie |
|------|-----|--------|
| Core | 32KB bounded, user preferences | ~32KB, persistent LLM context |
| Working | 1MB with TTL eviction | Unbounded, session-scoped |
| Archival | Unlimited with storage backend | Vector-based semantic search |

**Both use DST:** Umi has `SimConfig`, Kelpie has `kelpie-dst` crate.

**Both follow TigerStyle:** Explicit constants, assertions, safety > performance.

### The Key Difference: Library vs Runtime

```
┌─────────────────────────────────────────────────────────────────┐
│                    Umi: Memory LIBRARY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  • Entity extraction from text                                   │
│  • Evolution tracking (update/extend/contradict detection)       │
│  • Dual retrieval (fast + LLM semantic search)                   │
│  • Temporal metadata (document_time vs event_time)               │
│  • Designed to be EMBEDDED in any agent framework                │
│                                                                  │
│  Usage: memory = Memory(seed=42)                                 │
│         await memory.remember("Alice works at Acme")             │
│         results = await memory.recall("Who works at Acme?")      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Kelpie: Agent RUNTIME                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  • Complete agent execution environment                          │
│  • Virtual actor coordination and scaling                        │
│  • Tool execution with sandbox isolation                         │
│  • MCP integration for dynamic tools                             │
│  • Letta-compatible API for drop-in replacement                  │
│  • Cluster coordination for horizontal scaling                   │
│                                                                  │
│  Usage: Deploy as server, call REST API                          │
│         POST /agents/{id}/messages                               │
│         (Memory is internal, not a separate component)           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### What Each Provides That The Other Doesn't

**Umi has, Kelpie doesn't:**
- Entity extraction from text
- Evolution tracking (detecting updates/contradictions)
- Dual retrieval with LLM query rewriting
- Temporal metadata handling
- Embeddable library interface

**Kelpie has, Umi doesn't:**
- Complete agent runtime
- Virtual actor model
- Sandbox isolation (Firecracker)
- MCP tool integration
- Cluster coordination
- Letta API compatibility

### Relationship Options

```
┌─────────────────────────────────────────────────────────────────┐
│                    Option A: Kelpie Only                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Use Kelpie's built-in memory system.                           │
│  Don't need Umi at all.                                         │
│                                                                  │
│  ✓ Simpler: one codebase                                        │
│  ✓ Already integrated                                           │
│  ✗ No entity extraction                                         │
│  ✗ No evolution tracking                                        │
│  ✗ No dual retrieval with LLM rewriting                         │
│                                                                  │
│  Best for: Basic agent memory needs                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Option B: Kelpie + Umi                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Use Kelpie as runtime, swap in Umi for memory layer.           │
│                                                                  │
│  ✓ Entity extraction for structured learning                    │
│  ✓ Evolution tracking for expertise capture                     │
│  ✓ Dual retrieval for better recall                             │
│  ✗ More complexity: two memory systems to coordinate            │
│  ✗ Need to integrate at API level                               │
│                                                                  │
│  Best for: Sophisticated memory needs (like Isotopes.ai)        │
│                                                                  │
│  Integration approach:                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Kelpie (runtime)                                           │  │
│  │     │                                                      │  │
│  │     ├── CoreMemory → Kelpie's (LLM context)               │  │
│  │     │                                                      │  │
│  │     └── Archival/Working → Umi's (with entity extraction) │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Option C: Umi Only (Custom Runtime)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Build custom agent runtime, use Umi for memory.                │
│  Don't use Kelpie at all.                                       │
│                                                                  │
│  ✓ Full control over agent logic                                │
│  ✓ No redundant memory systems                                  │
│  ✗ Must build: agent loop, tool execution, API server           │
│  ✗ No cluster coordination                                      │
│  ✗ No sandbox isolation                                         │
│                                                                  │
│  Best for: Custom agent architectures                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 3: The Complete Stack for Isotopes.ai Replication

### What Isotopes.ai (aidnn) Needs

| Capability | Description |
|------------|-------------|
| Multi-agent orchestration | Specialized agents for data tasks |
| Data connectors | Salesforce, Snowflake, SQL databases |
| Natural language → SQL | Query translation |
| Context memory | Remember user queries and preferences |
| Learning from expertise | Adapt to team's analysis patterns |
| Continuous improvement | Get better over time |
| Explainability | Show reasoning |

### Mapping to Rust Stack

| aidnn Capability | Kelpie | Umi | Trajectories | GRPO |
|------------------|--------|-----|--------------|------|
| Multi-agent | Virtual actors | - | - | - |
| Data connectors | MCP tools | - | - | - |
| NL → SQL | LLM client | - | - | Training improves |
| Context memory | Core/Working | Entity extraction | - | - |
| Learning | - | Evolution tracking | Context learning | Fine-tuning |
| Improvement | - | - | RL training | GRPO algorithm |
| Explainability | Tool call logs | - | Trajectory store | - |

### The Complete Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rust aidnn Stack                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Kelpie Runtime                         │   │
│  │                                                           │   │
│  │  Virtual Actors (specialized agents)                      │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ Data        │ │ Cleaning    │ │ Analysis/Report     │ │   │
│  │  │ Connector   │ │ Agent       │ │ Agents              │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  │                                                           │   │
│  │  MCP Tool Integration                                     │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ SQL Tools   │ │ Snowflake   │ │ Visualization       │ │   │
│  │  │             │ │ Tools       │ │ Tools               │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  │                                                           │   │
│  │  Sandbox Isolation (Firecracker for untrusted code)       │   │
│  │                                                           │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Memory Layer                            │   │
│  │                                                            │   │
│  │  Option A: Kelpie's built-in (simpler)                    │   │
│  │  Option B: Umi integration (richer features)              │   │
│  │                                                            │   │
│  │  If Option B (recommended for aidnn):                     │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │                    Umi Memory                         │ │   │
│  │  │  • Entity extraction (user expertise, domain terms)  │ │   │
│  │  │  • Evolution tracking (learning what changed)         │ │   │
│  │  │  • Dual retrieval (find relevant past queries)        │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │                                                            │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Trajectory Store                        │   │
│  │                                                            │   │
│  │  • Capture all agent decisions                            │   │
│  │  • LLM extraction of decision/entity structure            │   │
│  │  • Export to training formats                             │   │
│  │                                                            │   │
│  │  Uses:                                                     │   │
│  │  1. Context learning (retrieve similar for few-shot)      │   │
│  │  2. Simulation ("what if" prediction)                     │   │
│  │  3. RL training export (Agent Lightning, Unsloth)         │   │
│  │                                                            │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│  ════════════════════════════╪════════════════════════════════   │
│                             │ Training boundary                  │
│  ════════════════════════════╪════════════════════════════════   │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                    Python Training Pipeline                │   │
│  │                                                            │   │
│  │  Agent Lightning + GRPO/GSPO                              │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Load trajectories from Rust store                 │ │   │
│  │  │ 2. Credit assignment (using decisions/entities)      │ │   │
│  │  │ 3. GRPO training with Unsloth/Tinker                 │ │   │
│  │  │ 4. Deploy improved model back to Kelpie              │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │                                                            │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Tradeoffs Analysis

### Kelpie Only vs Kelpie + Umi

| Aspect | Kelpie Only | Kelpie + Umi |
|--------|-------------|--------------|
| **Complexity** | Simpler | More integration work |
| **Entity extraction** | Manual prompting | Automatic with Umi |
| **Evolution tracking** | None | Built-in |
| **Dual retrieval** | Vector search only | Vector + LLM rewriting |
| **Temporal metadata** | Basic | Rich (document_time, event_time) |
| **Codebase overlap** | None | Memory code overlap |
| **Integration effort** | Low | Medium |

**Recommendation:** For aidnn replication, use **Kelpie + Umi** because:
1. Entity extraction is critical for "learning from expertise"
2. Evolution tracking enables "continuous improvement"
3. Dual retrieval improves "remember previous queries"

### Kelpie vs Custom Runtime with Umi

| Aspect | Kelpie | Custom + Umi |
|--------|--------|--------------|
| **Agent runtime** | Complete | Must build |
| **Cluster scaling** | Built-in | Must build |
| **Sandbox isolation** | Firecracker | Must build |
| **MCP integration** | Built-in | Must build |
| **Letta compatibility** | Yes | No |
| **Memory features** | Basic 3-tier | Rich (entity, evolution) |
| **Control** | Less | More |

**Recommendation:** Use **Kelpie** unless you need a fundamentally different agent architecture.

### GRPO Variants for aidnn

| GRPO Type | Best For | Credit Assignment |
|-----------|----------|-------------------|
| **Vanilla GRPO** | Simple tasks | Outcome only |
| **GRPO + LightningRL** | Multi-step analytics | Per-action |
| **GSPO** | Long data pipelines | Sequence-level |
| **GRPO + PRM** | Quality-focused | Process Reward Model |

**Recommendation:** Use **GRPO + LightningRL** for aidnn because data analytics involves multi-step reasoning.

---

## Part 5: Integration Architecture

### If Using Kelpie + Umi Together

```rust
// Integration strategy: Kelpie calls Umi for rich memory ops

use kelpie_runtime::Agent;
use umi::Memory;

struct EnhancedAgent {
    kelpie_agent: Agent,
    umi_memory: Memory,
}

impl EnhancedAgent {
    async fn execute(&mut self, message: &str) -> Result<Response> {
        // 1. Use Umi for rich context retrieval
        let context = self.umi_memory.recall(message).await?;

        // 2. Inject into Kelpie's core memory
        self.kelpie_agent.update_core_memory(&context);

        // 3. Execute with Kelpie (tools, sandbox, etc.)
        let response = self.kelpie_agent.step(message).await?;

        // 4. Store learnings in Umi (entity extraction)
        self.umi_memory.remember(&response.content).await?;

        // 5. Track evolution
        self.umi_memory.track_evolution(&response).await?;

        Ok(response)
    }
}
```

### Trajectory Collection with Kelpie

```rust
// Kelpie provides tool call logging, extend for trajectories

use kelpie_runtime::Trace;
use trajectories::{Trajectory, Turn};

impl TrajectoryCollector {
    fn from_kelpie_trace(trace: &Trace) -> Trajectory {
        Trajectory {
            trajectory_id: trace.trace_id,
            turns: trace.steps.iter().map(|step| Turn {
                messages: step.messages.clone(),
                tool_calls: step.tool_calls.clone(),
                // ...
            }).collect(),
            outcome: trace.outcome,
            // Will be enriched with decisions/entities by LLM
        }
    }
}
```

### Training Pipeline Connection

```python
# Python training consumes trajectories from Rust

from agent_lightning import LightningRL, TrajectoryDataset

# Connect to Kelpie's trajectory store (Redis/Postgres)
dataset = TrajectoryDataset.from_store("redis://trajectories")

# Enrich with decision/entity extraction
enriched = await enrich_all(dataset)

# Credit assignment using extracted structure
credits = LightningRL.assign_credit(
    enriched,
    credit_fn=decision_based_credit  # Uses extracted decisions
)

# GRPO training
trainer = GRPOTrainer(model="Qwen/Qwen2.5-7B")
trainer.train(credits)

# Deploy back to Kelpie
deploy_to_kelpie(trainer.model)
```

---

## Part 6: What This Enables

### 1. Self-Improving Data Agents

```
User: "What's our MRR trend?"

Kelpie:
├── Routes to SQL agent (virtual actor)
├── Executes query (sandboxed)
├── Generates visualization
├── Logs trajectory

Umi:
├── Extracts entities: {MRR, trend, timeframe}
├── Stores user preference: likes trend charts
├── Recalls: similar past queries for context

Trajectory:
├── Records full decision trace
├── LLM extracts: decisions, entities
├── Exports to training pipeline

GRPO:
├── Trains on successful trajectories
├── Credit assignment per step
├── Model improves over time
```

### 2. Team-Aware Analytics

```
User A (Finance): "Show revenue"
→ Umi recalls: User A likes tables, prefers quarterly

User B (Sales): "Show revenue"
→ Umi recalls: User B likes charts, prefers monthly

Same query, different personalized responses.
```

### 3. Continual Improvement Flywheel

```
Week 1: Agent struggles with complex joins
       Trajectories capture failures

Week 2: GRPO trains on failed trajectories with corrections
       Credit assignment identifies problematic steps

Week 3: Improved model deployed to Kelpie
       Success rate on complex joins improves

Repeat...
```

---

## Part 7: Implementation Roadmap

### Phase 1: Kelpie Baseline (Week 1-2)

```
├── Deploy Kelpie server
├── Configure MCP tools (SQL, Snowflake)
├── Set up Letta-compatible API
├── Basic agent functionality working
```

### Phase 2: Umi Integration (Week 3-4)

```
├── Integrate Umi memory with Kelpie agents
├── Entity extraction for user expertise
├── Evolution tracking for learning
├── Dual retrieval for context
```

### Phase 3: Trajectory Collection (Week 5-6)

```
├── Extend Kelpie tracing for trajectories
├── LLM extraction of decisions/entities
├── Store enriched trajectories
├── Export to training formats
```

### Phase 4: Training Pipeline (Week 7-8)

```
├── Agent Lightning integration
├── GRPO training with LightningRL credits
├── Model deployment back to Kelpie
├── Feedback loop automation
```

### Phase 5: Continual Improvement (Week 9-10)

```
├── Automatic retraining triggers
├── A/B testing of model versions
├── Performance monitoring
├── Instruction/skill optimization from trajectories
```

---

## Part 8: Key Decisions

### Decision 1: Use Kelpie or Build Custom?

**Recommendation: Use Kelpie**

Why:
- Kelpie is a complete agent runtime
- Cluster coordination built-in
- Sandbox isolation (Firecracker) built-in
- MCP integration built-in
- Letta compatibility for easy migration

### Decision 2: Use Kelpie's Memory or Add Umi?

**Recommendation: Add Umi for aidnn-like systems**

Why:
- Entity extraction is critical for "learning from expertise"
- Evolution tracking enables detecting changes over time
- Dual retrieval (vector + LLM) improves recall quality
- The complexity is worth it for sophisticated memory needs

### Decision 3: Which GRPO Variant?

**Recommendation: GRPO + LightningRL**

Why:
- aidnn involves multi-step reasoning
- Credit assignment helps identify good/bad steps
- LightningRL integrates cleanly with trajectory export

### Decision 4: Where to Store Trajectories?

**Recommendation: Umi's archival memory (unified approach)**

Why:
- Trajectories can be entity types in Umi
- Use same storage backend
- Dual retrieval works on trajectories too
- Consistent API

---

## Part 9: Complete Stack Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rust aidnn = Isotopes.ai                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Kelpie      = Agent runtime (actors, tools, sandbox, cluster)  │
│  +                                                               │
│  Umi         = Rich memory (entities, evolution, dual retrieval)│
│  +                                                               │
│  Trajectories = Learning substrate (decisions, context, export) │
│  +                                                               │
│  GRPO        = Training algorithm (with LightningRL credits)    │
│                                                                  │
│  ═══════════════════════════════════════════════════════════════│
│                                                                  │
│  Runtime (Rust)          │  Training (Python)                   │
│  ─────────────────────   │  ──────────────────                  │
│  Kelpie executes agents  │  Agent Lightning data pipeline       │
│  Umi stores memory       │  GRPO with Unsloth/Tinker            │
│  Trajectories capture    │  Credit assignment                   │
│  LLM extracts structure  │  Model weight updates                │
│                          │                                       │
│  95% of code             │  5% of code                          │
│  100% of runtime         │  Batch training only                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

**Can Kelpie + Umi + Trajectories + GRPO replicate Isotopes.ai?**

Yes. The mapping is:

| aidnn Component | Rust Stack Component |
|-----------------|---------------------|
| Agent runtime | Kelpie |
| Rich memory | Umi (integrated with Kelpie) |
| Learning from expertise | Umi evolution + Trajectory context learning |
| Continuous improvement | Trajectories + GRPO fine-tuning |
| Data tools | Kelpie MCP integration |
| Scaling | Kelpie cluster coordination |
| Safety | Kelpie sandbox isolation |

**Key insight:** Kelpie and Umi are **complementary**, not competing:
- Kelpie is the agent runtime (execution, tools, scaling)
- Umi is the memory library (entities, evolution, retrieval)
- Trajectories connect them to learning
- GRPO provides the training algorithm

The hybrid Rust runtime + Python training architecture gives the best of both worlds while the Rust ML ecosystem matures.

---

## Sources

- [Kelpie GitHub](https://github.com/nerdsane/kelpie)
- [Isotopes AI](https://isotopes.ai/)
- [Umi Codebase](umi-core/, umi/)
- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
