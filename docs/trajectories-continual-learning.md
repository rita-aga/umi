# Trajectories: The Missing Primitive for the Rust Agent Ecosystem

## Executive Summary

**Should trajectories be part of Umi/Letta/aidnn?** Yes—trajectories are the critical substrate that enables continual learning.

**Can we use the training frameworks?** Yes—all of them (Agent Lightning, Fireworks RFT, Unsloth ART) consume the same fundamental data: structured decision traces.

**Key insight:** Trajectories complete the Rust agent stack by providing the feedback loop from execution → learning → improvement.

---

## Part 1: How Trajectories Fit the Ecosystem

### The Current Stack (Without Trajectories)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Current Rust Agent Stack                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Skills (what agents CAN do)          ─── Static capabilities   │
│  AGENTS.md (how agents SHOULD behave) ─── Static instructions   │
│  MCP (how agents GET context)         ─── Dynamic retrieval     │
│  Umi (what agents REMEMBER)           ─── Persistent memory     │
│  Letta (how agents EXECUTE)           ─── Agent runtime         │
│                                                                  │
│  ════════════════════════════════════════════════════════════   │
│                                                                  │
│  MISSING: What agents DID do          ─── Dynamic evidence      │
│           How agents IMPROVE          ─── Continual learning    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Complete Stack (With Trajectories)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete Rust Agent Stack                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Letta-rs (Execution)                   │   │
│  │     Agent Loop │ Tool System │ Multi-Agent Orchestration │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │ produces                             │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Trajectory Store (Umi-rs)                 │   │
│  │   Structured Decision Traces │ Outcomes │ Rewards         │   │
│  └──────┬───────────────┬───────────────┬───────────────────┘   │
│         │               │               │                        │
│         ▼               ▼               ▼                        │
│  ┌───────────┐   ┌───────────┐   ┌───────────────────────┐      │
│  │ Context   │   │Simulation │   │    RL Training        │      │
│  │ Learning  │   │& Counter- │   │  (Agent Lightning,    │      │
│  │(retrieval)│   │ factual   │   │   Fireworks, Unsloth) │      │
│  └─────┬─────┘   └─────┬─────┘   └───────────┬───────────┘      │
│        │               │                     │                   │
│        └───────────────┼─────────────────────┘                   │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Improved Agent (Weights or Prompts)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### The Position Paper's Key Insight

> "Agents that cannot learn from their own experience are fundamentally limited."

Today's agents execute, produce outputs, and forget. Learning happens through:
- Expensive retraining (months, millions of dollars)
- Manual prompt engineering (doesn't scale)
- Static retrieval (documents, not decisions)

Trajectories enable a better path: **capture decision traces in a format that enables both inference-time learning and post-training improvement.**

---

## Part 2: Trajectories as Optimization Statistics

The position paper makes a profound connection to database query optimizers:

| Database Optimizer | Agent System |
|--------------------|--------------|
| Table statistics | Trajectory corpus |
| Query plan | Tool selection sequence |
| Cost estimation | Utility prediction |
| Plan selection | Action choice |

**Trajectories don't just improve agents—they improve the entire stack:**

### Dynamic AGENTS.md Evolution

```rust
// Trajectories reveal instruction gaps
struct InstructionOptimizer {
    trajectory_store: TrajectoryStore,
}

impl InstructionOptimizer {
    async fn suggest_updates(&self) -> Vec<InstructionPatch> {
        // Find trajectories where agent struggled despite following instructions
        let struggles = self.trajectory_store
            .query("outcome = 'failure' AND followed_instructions = true")
            .await?;

        // Use LLM to propose instruction improvements
        self.generate_patches(struggles).await
    }
}
```

### Skills Discovery

```rust
// Trajectories reveal skill usage patterns
impl SkillOptimizer {
    async fn analyze_patterns(&self) -> SkillAnalysis {
        // Skill gaps: patterns where agents improvised
        let gaps = self.find_improvisation_patterns().await;

        // Underused skills: rarely appear in successful trajectories
        let underused = self.find_underused_skills().await;

        // Failure patterns: skills that correlate with failures
        let problematic = self.find_failure_correlations().await;

        SkillAnalysis { gaps, underused, problematic }
    }
}
```

### MCP Tool Discovery Optimization

```rust
// Trajectories enable behavioral tool recommendation
impl ToolRecommender {
    async fn recommend(&self, context: &Context) -> Vec<ToolRecommendation> {
        // Find similar past situations
        let similar = self.trajectory_store.find_similar(context).await?;

        // Rank tools by historical success rate
        similar.iter()
            .flat_map(|t| t.successful_tool_sequences())
            .group_by(|seq| seq.first_tool())
            .map(|(tool, uses)| ToolRecommendation {
                tool,
                success_rate: uses.success_rate(),
                typical_sequence: uses.most_common_sequence(),
            })
            .collect()
    }
}
```

---

## Part 3: The Continual Learning Funnel

### Four Capabilities from One Format

| Use Case | Primary Need | Key Fields |
|----------|--------------|------------|
| **Display** | Render conversation history | Content, visibility flags, rich types |
| **Context Learning** | Retrieve similar examples | Embeddings, entities, outcome |
| **Simulation** | Predict counterfactuals | Full context, entities, history |
| **RL Training** | Fine-tune on rewards | Rewards, credit assignment, actions |

### The Funnel

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT EXECUTION                          │
│         (Generates trajectories as byproduct)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               CONTEXT LEARNING (Inference-time)             │
│    • Retrieve similar trajectories as few-shot examples     │
│    • No model changes, immediate benefit                    │
│    • Low investment, bounded returns                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION                               │
│    • Predict outcomes before committing                     │
│    • Counterfactual analysis ("what if?")                   │
│    • Planning via trajectory similarity                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            REINFORCEMENT LEARNING (Post-training)           │
│    • Fine-tune on curated trajectory corpus                 │
│    • Credit assignment across steps                         │
│    • High investment, compounding returns                   │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Stages 1-3 (context learning, simulation) require NO training infrastructure. They provide value immediately. Stage 4 (RL) is optional but compounds.

---

## Part 4: Training Frameworks Integration

### The Training Landscape

| Framework | Provider | Approach | Key Feature |
|-----------|----------|----------|-------------|
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | Microsoft | Decoupled RL | Framework-agnostic, LightningRL |
| [Fireworks RFT](https://fireworks.ai/blog/fireworks-rft) | Fireworks AI | Managed RL | Rollouts + evaluator API |
| [Unsloth + ART](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl) | OpenPipe/Unsloth | Open-source GRPO | RULER auto-rewards |
| [OpenAI RFT](https://platform.openai.com) | OpenAI | Managed fine-tuning | Integrated with API |

### Agent Lightning (Recommended for Rust)

Agent Lightning is the best fit because it **decouples execution from training**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Lightning Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Agent Runner (Rust - Letta-rs)               │   │
│  │                                                           │   │
│  │  • Execute agent tasks                                    │   │
│  │  • Log all LLM calls as spans                             │   │
│  │  • Record outcomes                                        │   │
│  │  • Send traces to LightningStore                          │   │
│  │                                                           │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LightningStore (Rust or Redis)               │   │
│  │                                                           │   │
│  │  • Trajectory storage                                     │   │
│  │  • Unified data interface                                 │   │
│  │  • Export to training formats                             │   │
│  │                                                           │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Algorithm (Python + CUDA)                    │   │
│  │                                                           │   │
│  │  • LightningRL credit assignment                          │   │
│  │  • GRPO/PPO/REINFORCE++ training                          │   │
│  │  • Model weight updates                                   │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### LightningRL Credit Assignment

The key innovation: **hierarchical credit assignment** that converts multi-step agent execution into single-turn RL-friendly format.

```
Episode Return → Action Credits → Token Optimization

Step 1: Trajectory-to-Call
        Distribute episode reward across individual LLM calls

Step 2: Call-to-Token
        Token-level credit assignment within each LLM output

Step 3: Standard RL
        Apply GRPO/PPO to each call independently
```

### Rust Implementation

```rust
// Trajectory capture in Rust
#[derive(Serialize, Clone)]
pub struct Trajectory {
    pub trajectory_id: Uuid,
    pub task_description: String,
    pub turns: Vec<Turn>,
    pub outcome: Outcome,
    pub final_reward: Option<f64>,
    pub metadata: TrajectoryMetadata,
}

#[derive(Serialize, Clone)]
pub struct Turn {
    pub turn_id: usize,
    pub span_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub messages: Vec<Message>,
    pub turn_reward: Option<f64>,
}

#[derive(Serialize, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Content,
    pub reasoning: Option<String>,  // Chain-of-thought
    pub context_snapshot: ContextSnapshot,
}

// Trace logger for Agent Lightning integration
pub struct TraceLogger {
    store: Box<dyn TrajectoryStore>,
}

impl TraceLogger {
    pub async fn log_trajectory(&self, trajectory: Trajectory) -> Result<()> {
        // Store locally (Umi archival memory)
        self.store.store(trajectory.clone()).await?;

        // Export to LightningStore format for training
        let lightning_format = self.convert_to_lightning(trajectory);
        self.export_to_training_pipeline(lightning_format).await?;

        Ok(())
    }

    fn convert_to_lightning(&self, trajectory: Trajectory) -> LightningTrajectory {
        // Convert to Agent Lightning's unified data interface
        LightningTrajectory {
            transitions: trajectory.turns.iter().map(|turn| {
                Transition {
                    state: turn.messages.iter()
                        .filter(|m| m.role != Role::Assistant)
                        .map(|m| m.content.clone())
                        .collect(),
                    action: turn.messages.iter()
                        .find(|m| m.role == Role::Assistant)
                        .map(|m| m.content.clone()),
                    reward: turn.turn_reward,
                }
            }).collect(),
            final_reward: trajectory.final_reward,
        }
    }
}
```

### Unsloth ART Integration

For local training without cloud dependencies:

```rust
// Export trajectories for Unsloth ART
impl TrajectoryExporter {
    pub fn to_art_format(&self, trajectories: &[Trajectory]) -> Vec<ARTRollout> {
        trajectories.iter().map(|t| {
            ARTRollout {
                // ART expects OpenAI-style chat completion messages
                messages: t.turns.iter()
                    .flat_map(|turn| &turn.messages)
                    .map(|m| ChatMessage {
                        role: m.role.to_string(),
                        content: m.content.to_string(),
                    })
                    .collect(),
                reward: t.final_reward,
            }
        }).collect()
    }
}
```

```python
# Python: Train with Unsloth ART
from art import Trajectory, TrajectoryGroup
from art.rewards import ruler_score_group
from unsloth import FastLanguageModel

# Load trajectories exported from Rust
trajectories = load_from_lightning_store("redis://trajectories")

# Group for GRPO (same task, different rollouts)
groups = group_by_task(trajectories)

# Score with RULER (no manual reward function needed!)
for group in groups:
    judged_group = await ruler_score_group(group, judge_model="openai/o3")

# Train with GRPO
model = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B")
trainer = GRPOTrainer(model=model)
trainer.train(groups)
```

### Fireworks RFT Integration

For managed training:

```rust
// Export trajectories for Fireworks RFT
impl TrajectoryExporter {
    pub async fn to_fireworks(&self, trajectories: &[Trajectory]) -> Result<()> {
        let client = FireworksClient::new()?;

        for trajectory in trajectories {
            // Fireworks expects rollouts with evaluator scores
            let rollout = FireworksRollout {
                messages: self.convert_messages(&trajectory.turns),
                // Score comes from your evaluator function
                score: trajectory.final_reward,
            };

            client.submit_rollout(rollout).await?;
        }

        Ok(())
    }
}
```

---

## Part 5: Trajectory Format for Umi

### Proposed Schema

Based on the position paper and training framework requirements:

```rust
/// Core trajectory structure for Umi
#[derive(Serialize, Deserialize, Clone)]
pub struct Trajectory {
    // Identity
    pub trajectory_id: Uuid,
    pub version: String,  // "0.1"

    // Metadata
    pub metadata: TrajectoryMetadata,

    // Context at start
    pub context: TrajectoryContext,

    // System message
    pub system_message: Option<SystemMessage>,

    // Execution turns
    pub turns: Vec<Turn>,

    // Outcome
    pub outcome: Outcome,
    pub final_reward: Option<f64>,
    pub feedback_score: Option<f64>,
    pub human_reviewed: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrajectoryMetadata {
    pub task_description: String,
    pub domain: String,
    pub timestamp_start: DateTime<Utc>,
    pub timestamp_end: DateTime<Utc>,
    pub duration_ms: u64,
    pub agent_id: String,
    pub framework: String,  // "letta-rs"
    pub environment: Environment,  // prod, staging, dev
    pub tags: Vec<String>,
    pub parent_trajectory_id: Option<Uuid>,  // For sub-agent calls
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrajectoryContext {
    pub referrer: Option<String>,
    pub user: Option<UserContext>,
    pub entities: Vec<Entity>,  // Umi entities!
    pub resources: Vec<Resource>,
    pub custom_context: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Turn {
    pub turn_id: usize,
    pub span_id: Uuid,
    pub parent_span_id: Option<Uuid>,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub error: bool,
    pub turn_reward: Option<f64>,
    pub messages: Vec<Message>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Message {
    pub message_id: Uuid,
    pub role: Role,
    pub timestamp: DateTime<Utc>,
    pub content: Content,
    pub reasoning: Option<String>,  // Chain-of-thought
    pub visibility: Visibility,
    pub context_snapshot: Option<ContextSnapshot>,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Content {
    Text(String),
    ToolCall { name: String, arguments: Value },
    ToolResponse { name: String, result: Value },
    Widget { widget_type: String, data: Value },
    Structured(Value),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Visibility {
    pub send_to_user: bool,  // Render in UI?
    pub persist: bool,       // Save to storage?
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ContextSnapshot {
    pub entities: Vec<EntityRef>,      // What Umi entities were active
    pub available_tools: Vec<String>,  // What tools were available
}
```

### Integration with Umi Memory

Trajectories naturally extend Umi's memory model:

```rust
impl Memory {
    /// Store a completed trajectory
    pub async fn store_trajectory(&mut self, trajectory: Trajectory) -> Result<()> {
        // Extract entities mentioned in trajectory
        let entities = self.extractor.extract_from_trajectory(&trajectory).await?;

        // Store entities in archival memory
        for entity in entities {
            self.archival.store(entity).await?;
        }

        // Store trajectory itself (as a special entity type)
        let trajectory_entity = Entity {
            entity_type: EntityType::Trajectory,
            name: trajectory.metadata.task_description.clone(),
            content: serde_json::to_string(&trajectory)?,
            metadata: TrajectoryMetadataForStorage {
                outcome: trajectory.outcome,
                reward: trajectory.final_reward,
                domain: trajectory.metadata.domain,
            },
        };

        self.archival.store(trajectory_entity).await?;

        Ok(())
    }

    /// Retrieve similar trajectories for context learning
    pub async fn recall_trajectories(&self, query: &str) -> Result<Vec<Trajectory>> {
        // Use dual retrieval (fast + semantic)
        let similar = self.retriever.search_trajectories(query).await?;

        // Filter by outcome (prefer successful)
        let successful: Vec<_> = similar
            .into_iter()
            .filter(|t| t.outcome == Outcome::Success)
            .collect();

        Ok(successful)
    }

    /// Export trajectories for RL training
    pub async fn export_for_training(&self, filter: TrajectoryFilter) -> Result<TrainingExport> {
        let trajectories = self.archival.query_trajectories(filter).await?;

        Ok(TrainingExport {
            lightning_format: self.to_lightning_format(&trajectories),
            art_format: self.to_art_format(&trajectories),
            fireworks_format: self.to_fireworks_format(&trajectories),
        })
    }
}
```

---

## Part 6: The Complete Architecture

### Rust Agent Stack with Trajectories

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete Rust Agent Stack                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   axum REST API                           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │                   Letta-rs Agent                          │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ Agent Loop  │  │ Tool System │  │ Trajectory      │   │   │
│  │  │             │  │             │  │ Collector       │   │   │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │   │
│  │         │                │                   │            │   │
│  └─────────┼────────────────┼───────────────────┼────────────┘   │
│            │                │                   │                │
│  ┌─────────▼────────────────▼───────────────────▼────────────┐   │
│  │                    Umi-rs Memory                          │   │
│  │                                                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ CoreMemory  │  │ Archival    │  │ Trajectory      │   │   │
│  │  │ (context)   │  │ Memory      │  │ Store           │   │   │
│  │  │             │  │ (entities)  │  │ (decisions)     │   │   │
│  │  └─────────────┘  └─────────────┘  └────────┬────────┘   │   │
│  │                                              │            │   │
│  └──────────────────────────────────────────────┼────────────┘   │
│                                                 │                │
│         ┌───────────────────┬───────────────────┼────────┐       │
│         │                   │                   │        │       │
│         ▼                   ▼                   ▼        ▼       │
│  ┌───────────┐       ┌───────────┐       ┌───────────────────┐  │
│  │ Context   │       │Simulation │       │  Training Export  │  │
│  │ Learning  │       │ Engine    │       │                   │  │
│  │           │       │           │       │ ┌───────────────┐ │  │
│  │ Retrieve  │       │ "What if" │       │ │Agent Lightning│ │  │
│  │ similar   │       │ prediction│       │ │Fireworks RFT  │ │  │
│  │ traces    │       │           │       │ │Unsloth ART    │ │  │
│  └─────┬─────┘       └─────┬─────┘       │ └───────────────┘ │  │
│        │                   │             └─────────┬─────────┘  │
│        │                   │                       │             │
│        └───────────────────┼───────────────────────┘             │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Python Training Pipeline                     │   │
│  │                                                           │   │
│  │  LightningRL │ GRPO │ PPO │ RULER Auto-Rewards           │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Improved Model (Weights/Prompts)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 7: What This Enables

### 1. Context Learning (Immediate Value, No Training)

```rust
impl Agent {
    async fn execute_with_context_learning(&mut self, task: &str) -> Result<Response> {
        // Retrieve similar successful trajectories
        let similar = self.memory.recall_trajectories(task).await?;

        // Include as few-shot examples in prompt
        let context = self.format_as_examples(similar);

        // Execute with enriched context
        self.execute_with_context(task, context).await
    }
}
```

**Estimated improvement:** 15-25% on task success rate (based on few-shot prompting literature).

### 2. Simulation and Counterfactuals

```rust
impl SimulationEngine {
    async fn predict_outcome(&self, state: &State, action: &Action) -> Prediction {
        // Find similar decision points in trajectory corpus
        let similar = self.trajectory_store
            .find_similar_decision_points(state, action)
            .await?;

        // Estimate outcome based on historical results
        let predicted_outcome = similar.weighted_outcome_average();

        Prediction {
            likely_outcome: predicted_outcome,
            confidence: similar.confidence(),
            supporting_trajectories: similar.top_k(3),
        }
    }

    async fn counterfactual(&self, trajectory: &Trajectory, step: usize, alt_action: &Action) -> Counterfactual {
        // "What if the agent had done X instead of Y at step N?"
        let similar = self.find_similar_with_action(
            &trajectory.turns[step].context,
            alt_action,
        ).await?;

        Counterfactual {
            predicted_outcome: similar.likely_outcome(),
            actual_outcome: trajectory.outcome,
            explanation: self.explain_difference(similar, trajectory),
        }
    }
}
```

### 3. RL Fine-Tuning (Compounding Improvement)

```python
# Training loop with Agent Lightning
async def training_iteration():
    # 1. Export trajectories from Rust agent
    trajectories = await lightning_store.fetch_recent(batch_size=1000)

    # 2. Credit assignment
    transitions = LightningRL.assign_credit(trajectories)

    # 3. Train with GRPO
    trainer.train_step(transitions)

    # 4. Deploy updated model
    await deploy_model(trainer.model)

# Flywheel: Better model → Better trajectories → Better training
```

### 4. Dynamic Primitive Optimization

```rust
// The optimization loop
impl PrimitiveOptimizer {
    async fn optimize_cycle(&mut self) -> OptimizationReport {
        let trajectories = self.trajectory_store.recent(1000).await?;

        // Optimize AGENTS.md
        let instruction_patches = self.instruction_optimizer
            .suggest_patches(&trajectories)
            .await?;

        // Optimize Skills
        let skill_suggestions = self.skill_optimizer
            .analyze(&trajectories)
            .await?;

        // Optimize MCP tool recommendations
        let tool_insights = self.tool_optimizer
            .update_recommendations(&trajectories)
            .await?;

        OptimizationReport {
            instruction_patches,
            skill_suggestions,
            tool_insights,
        }
    }
}
```

---

## Part 8: Implementation Roadmap

### Phase T1: Trajectory Capture (Week 1-2)

```
├── Define Trajectory struct in Rust
├── Implement TrajectoryCollector for Letta-rs agent loop
├── Store trajectories in Umi archival memory
└── Basic export to JSON/JSONL
```

### Phase T2: Context Learning (Week 3-4)

```
├── Trajectory retrieval (similarity search)
├── Integration with DualRetriever
├── Few-shot example formatting
└── A/B test: with vs without context learning
```

### Phase T3: Training Export (Week 5-6)

```
├── Agent Lightning format export
├── Unsloth ART format export
├── Fireworks RFT format export
└── LightningStore integration (Redis)
```

### Phase T4: Training Pipeline (Week 7-8)

```
├── Python training scripts
├── RULER auto-reward integration
├── GRPO training with Unsloth
└── Model deployment workflow
```

### Phase T5: Simulation (Week 9-10)

```
├── Decision point similarity search
├── Outcome prediction
├── Counterfactual analysis
└── Planning with trajectory simulation
```

### Phase T6: Optimization Loop (Week 11-12)

```
├── Instruction optimization
├── Skill discovery
├── Tool recommendation improvement
└── Feedback loop automation
```

---

## Part 9: Tradeoffs

### Trajectory Capture

| Aspect | Benefit | Cost |
|--------|---------|------|
| Storage | Evidence for learning | ~10KB per trajectory |
| Latency | None (async logging) | Background I/O |
| Privacy | Learning signal | Must handle PII |

### Context Learning

| Aspect | Benefit | Cost |
|--------|---------|------|
| Immediate improvement | +15-25% task success | Retrieval latency (~50ms) |
| No training required | Zero infrastructure | Limited ceiling |

### RL Training

| Aspect | Benefit | Cost |
|--------|---------|------|
| Compounding improvement | Permanent gains | GPU cost, complexity |
| Model improvement | Better base behavior | Training time |

### Framework Choice

| Framework | Best For | Tradeoff |
|-----------|----------|----------|
| Agent Lightning | Decoupled, any framework | Requires Python training |
| Fireworks RFT | Managed, low ops | Vendor lock-in |
| Unsloth ART | Open source, local | More setup |

---

## Part 10: Answers to Key Questions

### Should trajectories be part of Umi/Letta?

**Yes.** Trajectories are the missing primitive that enables:
- Learning from experience (not just storing it)
- Optimization of the entire agent stack
- Continual improvement without full retraining

Without trajectories, agents execute and forget. With trajectories, agents execute, remember, and improve.

### Can we use the training frameworks?

**Yes, all of them:**

| Framework | Integration Path |
|-----------|------------------|
| **Agent Lightning** | Best fit—designed for decoupled execution/training |
| **Fireworks RFT** | Export trajectories as rollouts, use managed training |
| **Unsloth ART** | Export as OpenAI-style messages, train locally |
| **OpenAI RFT** | Export to OpenAI format, use their fine-tuning |

The key insight: **all frameworks consume the same fundamental data—structured decision traces.** A unified trajectory format in Umi enables export to any of them.

### What from Tinker + Agent Lightning can we apply?

From the [Tinker × Agent Lightning integration](https://medium.com/@yugez/tuning-any-ai-agent-with-tinker-agent-lightning-part-1-1d8c9a397f0e):

1. **Token-level APIs** - Tinker's optimized primitives enable fast training feedback
2. **Observability integration** - Use AgentOps-style tracing to collect telemetry
3. **Trajectory → RL dataset conversion** - Agent Lightning's unified data interface
4. **LightningRL credit assignment** - Hierarchical approach for multi-step agents
5. **GRPO compatibility** - Works with Unsloth's efficient GRPO implementation

**Apply to Rust stack:**
- Rust agent collects traces (like AgentOps)
- Export to LightningStore format
- Python training with Tinker/Unsloth for fast iteration
- Deploy improved model back to Rust agent

---

## Conclusion

Trajectories complete the Rust agent ecosystem by providing:

1. **The feedback loop** - Execution → Evidence → Learning → Improvement
2. **The optimization substrate** - Improve agents, skills, instructions, and tools
3. **The training bridge** - Connect Rust execution to Python RL training
4. **The continual learning funnel** - From context retrieval to simulation to fine-tuning

The position paper is right: *"Agents that cannot learn from their own experience are fundamentally limited."*

Umi + Letta + Trajectories = A complete, continually-improving Rust agent stack.

---

## Sources

- [Position Paper: Trajectories as a Foundation for Continual Learning Agents](provided in conversation)
- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Agent Lightning Blog](https://www.microsoft.com/en-us/research/blog/agent-lightning-adding-reinforcement-learning-to-ai-agents-without-code-rewrites/)
- [Fireworks RFT](https://fireworks.ai/blog/fireworks-rft)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [Unsloth Training AI Agents](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl)
- [OpenPipe ART](https://github.com/OpenPipe/ART)
- [RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler)
- [Tinker × Agent Lightning Tutorial](https://medium.com/@yugez/tuning-any-ai-agent-with-tinker-agent-lightning-part-1-1d8c9a397f0e)
