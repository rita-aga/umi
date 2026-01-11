# Replicating Isotopes.ai (aidnn) in Rust with Continual Learning

## Executive Summary

**Can aidnn be replicated in Rust using Umi + Letta?** Yes.

**Is it feasible?** Yes, with clear architectural mapping.

**With continual learning and RL?** Yes, following the Agent Lightning pattern.

**Key tradeoff:** Rust excels at the core agent infrastructure; RL training still needs Python/CUDA ecosystem integration.

---

## Part 1: What is aidnn?

Based on public information from [Isotopes AI](https://isotopes.ai/):

### Core Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                         aidnn Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Natural Lang │  │ Multi-Agent  │  │ Continuous Learning  │  │
│  │ Interface    │  │ Orchestrator │  │ from User Expertise  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │              Specialized Agent Pool                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │ Data    │ │ Cleaning│ │ Analysis│ │ Visualization│   │   │
│  │  │ Connect │ │ Agent   │ │ Agent   │ │ Agent        │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │                  Data Sources                            │   │
│  │  Salesforce │ Snowflake │ ERPs │ CRMs │ Cloud Storage   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features to Replicate

| Feature | Description |
|---------|-------------|
| **Multi-agent orchestration** | Specialized agents for different tasks |
| **Data integration** | Connect to Salesforce, Snowflake, etc. |
| **Natural language queries** | "What's our MRR trend?" → SQL + analysis |
| **Data cleaning** | Detect missing values, duplicates, outliers |
| **Context memory** | Remember previous queries and user preferences |
| **Learning from expertise** | Adapt to team's thinking patterns |
| **Automated reports** | Scheduled briefings, month-end reports |
| **Explainability** | Show reasoning for decisions |

---

## Part 2: Architectural Mapping to Rust

### Umi + Letta = aidnn Foundation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rust aidnn Equivalent                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Letta-rs (Agent Framework)              │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ Agent Loop  │  │ Tool System │  │ Multi-Agent     │   │  │
│  │  │ (step/      │  │ (data conn, │  │ Orchestrator    │   │  │
│  │  │  reason)    │  │  SQL, viz)  │  │                 │   │  │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │  │
│  │         │                │                   │            │  │
│  │         └────────────────┼───────────────────┘            │  │
│  │                          │                                │  │
│  └──────────────────────────┼────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                    Umi-rs (Memory Layer)                   │  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │  │
│  │  │ CoreMemory  │  │ Entity      │  │ Evolution       │    │  │
│  │  │ (context,   │  │ Extraction  │  │ Tracking        │    │  │
│  │  │  prefs)     │  │ (schemas,   │  │ (learning what  │    │  │
│  │  │             │  │  relations) │  │  changed)       │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │  │
│  │                                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │  │
│  │  │ Archival    │  │ Dual        │  │ User Expertise  │    │  │
│  │  │ Memory      │  │ Retrieval   │  │ Memory          │    │  │
│  │  │ (history)   │  │ (semantic)  │  │ (preferences)   │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                   Storage Backends                         │  │
│  │  PostgreSQL + pgvector │ Qdrant │ Redis (cache)           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Component Mapping

| aidnn Component | Rust Implementation |
|-----------------|---------------------|
| Multi-agent orchestrator | Letta-rs with agent spawning |
| Natural language interface | LLM provider trait (Anthropic/OpenAI) |
| Data connectors | `sqlx` (Postgres, MySQL), Snowflake REST API |
| Context memory | Umi `CoreMemory` (in-context blocks) |
| Long-term memory | Umi `ArchivalMemory` (vector search) |
| User expertise learning | Umi `EvolutionTracker` + preference entities |
| Scheduled reports | `tokio-cron` + agent triggers |
| Explainability | Letta-rs tool call logging |

### Specialized Agents in Rust

```rust
// Agent trait for specialized agents
#[async_trait]
trait SpecializedAgent: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<Capability>;
    async fn execute(&self, task: Task, memory: &Memory) -> Result<TaskResult>;
}

// Data connection agent
struct DataConnectorAgent {
    connections: HashMap<String, DataSource>,
}

// Cleaning agent
struct DataCleaningAgent {
    rules: Vec<CleaningRule>,
}

// Analysis agent
struct AnalysisAgent {
    llm: Box<dyn LLMProvider>,
}

// Orchestrator selects and coordinates agents
struct Orchestrator {
    agents: Vec<Box<dyn SpecializedAgent>>,
    memory: Memory,  // Umi memory
}

impl Orchestrator {
    async fn handle_query(&self, query: &str) -> Result<Response> {
        // 1. Understand intent
        let intent = self.classify_intent(query).await?;

        // 2. Select appropriate agents
        let agents = self.select_agents(&intent);

        // 3. Execute with memory context
        let context = self.memory.recall(query).await?;

        // 4. Coordinate execution
        let result = self.execute_pipeline(agents, context).await?;

        // 5. Store learnings
        self.memory.remember(&result.insights).await?;

        Ok(result)
    }
}
```

---

## Part 3: Continual Learning in Rust

### The Challenge

aidnn "continuously learns from each interaction with folks in the business." This requires:

1. **Preference learning** - Remember what users like
2. **Expertise capture** - Learn domain-specific patterns
3. **Behavioral adaptation** - Adjust agent behavior over time
4. **No catastrophic forgetting** - Don't lose old knowledge

### Rust Implementation Strategy

#### Level 1: Memory-Based Learning (Umi)

```rust
// User expertise as entities in Umi
struct UserExpertise {
    user_id: String,
    domain: String,
    preferences: Vec<Preference>,
    corrections: Vec<Correction>,  // When user corrected the agent
    patterns: Vec<Pattern>,        // Observed analysis patterns
}

// Store expertise in archival memory
impl Memory {
    async fn learn_from_interaction(&mut self, interaction: &Interaction) -> Result<()> {
        // Extract what user taught us
        let learnings = self.extractor.extract_expertise(interaction).await?;

        // Check for updates to existing knowledge
        let evolutions = self.tracker.detect_evolution(&learnings).await?;

        // Store with evolution relationships
        for learning in learnings {
            self.store_with_evolution(learning, &evolutions).await?;
        }

        Ok(())
    }

    async fn apply_expertise(&self, query: &str, user_id: &str) -> Result<Context> {
        // Retrieve user-specific expertise
        let expertise = self.recall_expertise(user_id).await?;

        // Inject into agent context
        Ok(Context {
            user_preferences: expertise.preferences,
            domain_knowledge: expertise.patterns,
            past_corrections: expertise.corrections,
        })
    }
}
```

This is **not true learning** (weights don't change), but it's effective for:
- Personalization
- Domain adaptation
- Error correction

#### Level 2: Prompt Optimization (No Training)

```rust
// Automatically optimize prompts based on outcomes
struct PromptOptimizer {
    variants: Vec<PromptTemplate>,
    performance: HashMap<TemplateId, PerformanceMetrics>,
}

impl PromptOptimizer {
    fn select_best(&self, task_type: TaskType) -> &PromptTemplate {
        // Multi-armed bandit: exploit best, explore others
        self.variants
            .iter()
            .filter(|p| p.task_type == task_type)
            .max_by_key(|p| self.performance.get(&p.id).map(|m| m.success_rate))
            .unwrap()
    }

    fn record_outcome(&mut self, template_id: TemplateId, success: bool) {
        self.performance
            .entry(template_id)
            .or_default()
            .record(success);
    }
}
```

#### Level 3: Reinforcement Learning (Agent Lightning Pattern)

This is where it gets interesting. [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning) shows how to add RL to any agent:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Training Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Agent Runner (Rust)                           │ │
│  │                                                            │ │
│  │  Letta-rs Agent                                            │ │
│  │       │                                                    │ │
│  │       ├── Execute task                                     │ │
│  │       ├── Log all LLM calls + tool calls                   │ │
│  │       ├── Record outcomes                                  │ │
│  │       └── Send traces to LightningStore                    │ │
│  │                                                            │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              LightningStore (Rust)                         │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │ │
│  │  │ Trajectories│  │ Rewards     │  │ Credit          │    │ │
│  │  │ (traces)    │  │ (outcomes)  │  │ Assignment      │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │ │
│  │                                                            │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              RL Trainer (Python + CUDA)                    │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │ │
│  │  │ PPO/GRPO    │  │ Model       │  │ Fine-tuned      │    │ │
│  │  │ Algorithm   │  │ Weights     │  │ LLM             │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key insight:** The agent execution (Rust) is separate from model training (Python/CUDA).

```rust
// Rust: Agent execution with trace logging
struct RLEnabledAgent {
    agent: LettaAgent,
    trace_logger: TraceLogger,
}

impl RLEnabledAgent {
    async fn execute_with_trace(&mut self, task: Task) -> Result<(Response, Trace)> {
        let trace = Trace::new(task.id);

        // Intercept all LLM calls
        let response = self.agent
            .with_interceptor(|call| trace.record_llm_call(call))
            .execute(task)
            .await?;

        // Record outcome
        trace.set_outcome(response.success, response.user_feedback);

        // Send to training pipeline
        self.trace_logger.send(trace.clone()).await?;

        Ok((response, trace))
    }
}

// Trace format (serialized to training pipeline)
#[derive(Serialize)]
struct Trace {
    task_id: String,
    steps: Vec<Step>,
    outcome: Outcome,
    user_feedback: Option<Feedback>,
}

#[derive(Serialize)]
struct Step {
    step_id: String,
    llm_input: String,
    llm_output: String,
    tool_calls: Vec<ToolCall>,
    timestamp: DateTime<Utc>,
}
```

```python
# Python: RL training (separate process)
from agent_lightning import LightningRL, TrajectoryDataset

# Load traces from Rust agents
dataset = TrajectoryDataset.from_lightning_store("redis://traces")

# Credit assignment: which steps contributed to success?
credits = LightningRL.assign_credit(dataset)

# Train with PPO/GRPO
trainer = LightningRL.PPOTrainer(
    model="meta-llama/Llama-3-8b",
    trajectories=dataset,
    credits=credits,
)
trainer.train(epochs=10)

# Export fine-tuned model
trainer.save("./fine-tuned-agent-llm")
```

### Handling Catastrophic Forgetting

```rust
// Continual learning with experience replay
struct ContinualLearner {
    experience_buffer: RingBuffer<Experience>,  // Fixed-size replay buffer
    expertise_memory: Memory,                    // Umi archival memory
}

impl ContinualLearner {
    fn add_experience(&mut self, exp: Experience) {
        // Store in replay buffer (fixed size, old experiences evicted)
        self.experience_buffer.push(exp.clone());

        // Also store important experiences in long-term memory
        if exp.importance > IMPORTANCE_THRESHOLD {
            self.expertise_memory.remember(&exp).await;
        }
    }

    fn sample_for_training(&self) -> Vec<Experience> {
        // Mix recent and historical experiences
        let recent = self.experience_buffer.recent(100);
        let historical = self.expertise_memory.recall_important(50).await;

        [recent, historical].concat()
    }
}
```

---

## Part 4: What This Enables

### 1. Self-Improving Data Agents

```rust
// Agent that gets better at understanding your data
struct SelfImprovingDataAgent {
    agent: LettaAgent,
    memory: Memory,           // Umi
    prompt_optimizer: PromptOptimizer,
    rl_tracer: TraceLogger,
}

impl SelfImprovingDataAgent {
    async fn query(&mut self, question: &str, user: &User) -> Result<Answer> {
        // Load user expertise from memory
        let expertise = self.memory.apply_expertise(question, &user.id).await?;

        // Select best prompt variant
        let prompt = self.prompt_optimizer.select_best(TaskType::DataQuery);

        // Execute with tracing
        let (answer, trace) = self.agent
            .with_context(expertise)
            .with_prompt(prompt)
            .execute_with_trace(question)
            .await?;

        // Learn from interaction
        self.memory.learn_from_interaction(&trace).await?;

        // Log for RL training
        self.rl_tracer.send(trace).await?;

        Ok(answer)
    }

    async fn receive_feedback(&mut self, trace_id: &str, feedback: Feedback) {
        // Update prompt optimizer
        self.prompt_optimizer.record_outcome(trace_id, feedback.positive);

        // Store correction in memory
        if let Some(correction) = feedback.correction {
            self.memory.remember_correction(correction).await;
        }

        // Update trace for RL
        self.rl_tracer.update_feedback(trace_id, feedback).await;
    }
}
```

### 2. Team-Aware Analytics

```rust
// Different users get personalized agent behavior
struct TeamAwareOrchestrator {
    agents: Vec<Box<dyn SpecializedAgent>>,
    team_memory: Memory,  // Shared team knowledge
    user_memories: HashMap<UserId, Memory>,  // Per-user preferences
}

impl TeamAwareOrchestrator {
    async fn handle(&self, query: &str, user: &User) -> Result<Response> {
        // Combine team knowledge with user preferences
        let team_context = self.team_memory.recall(query).await?;
        let user_context = self.user_memories
            .get(&user.id)
            .map(|m| m.recall(query))
            .transpose()?;

        let combined = Context::merge(team_context, user_context);

        // Execute with combined context
        self.execute_with_context(query, combined).await
    }
}
```

### 3. Automated Report Evolution

```rust
// Reports that improve based on feedback
struct EvolvingReport {
    template: ReportTemplate,
    feedback_history: Vec<ReportFeedback>,
    memory: Memory,
}

impl EvolvingReport {
    async fn generate(&self) -> Result<Report> {
        // Retrieve past feedback from memory
        let learnings = self.memory.recall("report feedback").await?;

        // Apply learnings to template
        let evolved_template = self.template.apply_learnings(&learnings);

        // Generate report
        evolved_template.generate().await
    }

    async fn receive_feedback(&mut self, feedback: ReportFeedback) {
        // Store feedback in memory
        self.memory.remember(&feedback).await;

        // Track evolution
        self.memory.track_evolution(
            &self.template.id,
            &feedback.suggestions,
        ).await;
    }
}
```

---

## Part 5: Tradeoffs

### Full Rust Stack

| Aspect | Benefit | Cost |
|--------|---------|------|
| **Agent execution** | Fast, memory-safe, concurrent | - |
| **Data connectors** | Native async, connection pooling | Must implement each connector |
| **Memory (Umi)** | Integrated, type-safe | - |
| **Prompt optimization** | Works in Rust | - |
| **RL training** | Must use Python/CUDA | Cross-language boundary |
| **Model inference** | Can use Rust (llama.cpp bindings) | Less mature than Python |

### The RL Training Boundary

**This is the key tradeoff:** RL training fundamentally requires:
- GPU access (CUDA)
- PyTorch/JAX ecosystem
- Optimizers, gradient computation

Rust options exist (`tch-rs`, `burn`) but are less mature.

**Recommended architecture:**

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│         Rust Runtime            │     │       Python Training           │
│                                 │     │                                 │
│  • Agent execution              │     │  • RL algorithms (PPO, GRPO)    │
│  • Memory (Umi)                 │────▶│  • Fine-tuning                  │
│  • Data connectors              │     │  • Model hosting                │
│  • Trace collection             │◀────│  • Weight updates               │
│  • Inference (optional)         │     │                                 │
│                                 │     │                                 │
└─────────────────────────────────┘     └─────────────────────────────────┘
         95% of code                           5% of code
         100% of runtime                       Batch training only
```

### Comparison: Full Python vs Hybrid vs Full Rust

| Aspect | Full Python | Hybrid (Rust agent + Python RL) | Full Rust |
|--------|-------------|----------------------------------|-----------|
| Agent performance | Adequate | Optimal | Optimal |
| RL training | Native | Native Python | Would need Rust ML (immature) |
| Memory safety | Runtime errors | Safe agent, safe training | Fully safe |
| Deployment | Complex | Two binaries | Single binary (no RL) |
| Ecosystem | Mature | Best of both | Growing |
| Complexity | Low | Medium | High (for RL) |

---

## Part 6: Feasibility Assessment

### What's Ready Today

| Component | Status | Implementation |
|-----------|--------|----------------|
| Multi-agent orchestration | Ready | Letta-rs pattern |
| Memory system | Ready | Umi-rs |
| Data connectors | Ready | `sqlx`, REST clients |
| Natural language interface | Ready | LLM provider traits |
| Prompt optimization | Ready | Bandit algorithms in Rust |
| Trace collection | Ready | Structured logging |
| RL training | Use Python | Agent Lightning pattern |

### What Needs Building

| Component | Effort | Notes |
|-----------|--------|-------|
| Letta-rs core | Medium | Port agent loop, tool system |
| Umi-rs Python layer | Medium | Port extraction, retrieval, evolution |
| Data connectors (Snowflake, etc.) | Medium | REST API clients |
| LightningStore (Rust) | Low | Redis/Postgres trace storage |
| RL training pipeline | Low | Use existing Python tools |

### Timeline (Coding Agent Estimate)

```
Week 1-2: Complete Umi-rs (port Python layer)
Week 3-4: Letta-rs core (agent loop, memory integration)
Week 5-6: Data connectors (SQL, REST APIs)
Week 7-8: Multi-agent orchestration
Week 9-10: RL trace collection + Python training integration
Week 11-12: Testing, optimization, documentation
```

---

## Part 7: Recommendation

### For Building aidnn-like System

**Use Rust for:**
- Agent runtime (Letta-rs)
- Memory system (Umi-rs)
- Data connectors
- Trace collection
- API server

**Use Python for:**
- RL training (Agent Lightning)
- Model fine-tuning
- Experimental prompt engineering

### Architecture

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
│  │       │       ├── DataConnectorAgent                       │ │
│  │       │       ├── CleaningAgent                            │ │
│  │       │       ├── AnalysisAgent                            │ │
│  │       │       └── VisualizationAgent                       │ │
│  │       │                                                    │ │
│  │       ├── Umi-rs Memory                                    │ │
│  │       │       ├── CoreMemory (user context)                │ │
│  │       │       ├── ArchivalMemory (history, expertise)      │ │
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
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

**Can aidnn be replicated in Rust with Umi + Letta?**

Yes. The core capabilities (multi-agent orchestration, memory, data integration, learning from expertise) map cleanly to Rust implementations.

**With continual learning and RL?**

Yes, following the Agent Lightning pattern:
- Rust handles agent execution and trace collection
- Python handles RL training (PPO/GRPO)
- Fine-tuned models deployed back to Rust agents

**Key tradeoffs:**
- Rust provides performance, safety, and deployment simplicity for the runtime
- Python remains necessary for RL training (GPU/CUDA ecosystem)
- The boundary is clean: Rust collects data, Python trains, Rust serves

**What this enables:**
- Self-improving data agents that learn from user expertise
- Team-aware analytics with personalized behavior
- Reports that evolve based on feedback
- Deterministic testing of agent behavior (DST)
- Single-binary deployment of production agents

The hybrid approach (Rust runtime + Python training) gives the best of both worlds while the Rust ML ecosystem matures.

---

## Sources

- [Isotopes AI / aidnn](https://isotopes.ai/)
- [Isotopes AI TechCrunch Coverage](https://techcrunch.com/2025/09/05/scale-ais-former-cto-launches-ai-agent-that-could-solve-big-datas-biggest-problem/)
- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Agent Lightning Research Blog](https://www.microsoft.com/en-us/research/blog/agent-lightning-adding-reinforcement-learning-to-ai-agents-without-code-rewrites/)
- [Continual Reinforcement Learning Survey](https://jair.org/index.php/jair/article/download/13673/26878/32800)
