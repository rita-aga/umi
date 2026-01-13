# DST-First Demonstration: Memory Fault Injection Discovery

**Objective**: Demonstrate TRUE DST-first methodology by discovering the Memory fault injection issue through simulation testing.

**Starting Point**: Commit 6662222 (before the fix was implemented)

**Hypothesis**: When we configure faults in a Simulation and use Memory inside, the faults should be applied to Memory operations.

---

## Phase 1: Write Discovery Test FIRST (Before Knowing the Solution)

We'll write a test that EXPECTS fault injection to work, without implementing any solution yet.

**Test Intent**:
- Register a 100% StorageWriteFail fault
- Create a Memory instance inside Simulation
- Try to remember something
- Assert that it FAILS (proving fault was injected)

**Why 100% probability?** We want the fault to ALWAYS fire so we can clearly see if it's working or not.

Test written! See `umi-memory/src/dst/simulation.rs::test_dst_discovery_memory_fault_injection()`

---

## Phase 2: Run Test and Observe (Discovery!)

```bash
$ cargo test test_dst_discovery_memory_fault_injection
```

**Result**: ❌ TEST FAILED!

```
thread 'dst::simulation::tests::test_dst_discovery_memory_fault_injection' panicked at:
DISCOVERY: With 100% StorageWriteFail, remember() should fail!
If this assertion fails, it means Memory::sim() is not connected
to the Simulation's FaultInjector.
```

**DISCOVERY**: The test EXPECTS `result.is_err()` but got `result.is_ok()`!

**What this means**:
- We registered a 100% StorageWriteFail fault
- We called `memory.remember("Alice works at Acme", ...)`
- It SUCCEEDED (returned Ok)
- But it should have FAILED (with storage error)

**Conclusion**: `Memory::sim(42)` creates a Memory instance that is NOT connected to the Simulation's FaultInjector. The fault configuration is being ignored!

---

## Phase 3: Investigation - Why Aren't Faults Being Injected?

### Step 1: Examine Memory::sim()

```rust
// umi-memory/src/umi/mod.rs:1188
pub fn sim(seed: u64) -> Self {
    let llm = SimLLMProvider::with_seed(seed);        // ❌
    let embedder = SimEmbeddingProvider::with_seed(seed);  // ❌
    let vector = SimVectorBackend::new(seed);         // ❌
    let storage = SimStorageBackend::new(SimConfig::with_seed(seed));  // ❌

    Self::new(llm, embedder, vector, storage)
}
```

**Problem Found**: Each provider is created with `with_seed()` or `new()`, which means they create their OWN empty `FaultInjector`. They're not accepting the Simulation's shared `FaultInjector`!

### Step 2: Check if providers support external FaultInjector

```bash
$ grep "pub fn with_faults" umi-memory/src/**/*.rs
```

**Findings**:
- ✅ `SimLLMProvider::with_faults(seed, Arc<FaultInjector>)` - EXISTS
- ✅ `SimEmbeddingProvider::with_faults(seed, Arc<FaultInjector>)` - EXISTS
- ✅ `SimVectorBackend::with_faults(seed, Arc<FaultInjector>)` - EXISTS
- ⚠️ `SimStorageBackend::with_faults(self, FaultConfig)` - DIFFERENT API!

**Root Cause Identified**:
1. The infrastructure EXISTS - providers CAN accept external FaultInjectors
2. But `Memory::sim()` doesn't USE this capability
3. `Memory::sim()` creates isolated providers with empty FaultInjectors
4. SimStorageBackend has an inconsistent API (needs fixing)

### Step 3: How does the Simulation's FaultInjector get created?

```rust
// umi-memory/src/dst/simulation.rs:154
let mut fault_builder = FaultInjectorBuilder::new(rng.fork());
for fault_config in self.fault_configs {
    fault_builder = fault_builder.with_fault(fault_config);
}
let faults = Arc::new(fault_builder.build());  // ← This is the shared FaultInjector
```

The `SimEnvironment` has `pub faults: Arc<FaultInjector>`, but there's no way to pass this to `Memory::sim()`!

---

## Phase 4: Solution Design

**Problem**: `Memory::sim()` can't access `SimEnvironment.faults` because:
1. It's called with just a seed: `Memory::sim(42)`
2. The `env` is available in the closure, but we can't pass it to `Memory::sim()`

**Solution Options**:

### Option A: Add `SimEnvironment::create_memory()`
```rust
impl SimEnvironment {
    pub fn create_memory(&self) -> Memory<...> {
        let seed = self.config.seed();

        let llm = SimLLMProvider::with_faults(seed, Arc::clone(&self.faults));
        let embedder = SimEmbeddingProvider::with_faults(seed, Arc::clone(&self.faults));
        let vector = SimVectorBackend::with_faults(seed, Arc::clone(&self.faults));
        let storage = SimStorageBackend::with_fault_injector(self.config, Arc::clone(&self.faults));

        Memory::new(llm, embedder, vector, storage)
    }
}
```

**Pros**:
- Clean API: `env.create_memory()`
- `env` is already available in the closure
- Follows Rust patterns (builder on environment)

**Cons**:
- Need to add `with_fault_injector()` to SimStorageBackend

### Option B: Add `Memory::from_sim_env()`
```rust
impl Memory {
    pub fn from_sim_env(env: &SimEnvironment) -> Self { ... }
}
```

**Pros**:
- Explicit that it's from simulation
**Cons**:
- Redundant if we have Option A
- Less idiomatic (why not on env?)

**Decision**: Option A - `SimEnvironment::create_memory()`

---

## Phase 5: Implementation

### Step 1: Fix SimStorageBackend API Inconsistency

Added `with_fault_injector()` constructor to match other providers:

```rust
// umi-memory/src/storage/sim.rs:70
pub fn with_fault_injector(config: SimConfig, fault_injector: Arc<FaultInjector>) -> Self {
    let rng = DeterministicRng::new(config.seed());

    Self {
        storage: Arc::new(RwLock::new(HashMap::new())),
        fault_injector,  // Use the provided one instead of creating new
        clock: SimClock::new(),
        rng: Arc::new(RwLock::new(rng)),
    }
}
```

### Step 2: Implement SimEnvironment::create_memory()

```rust
// umi-memory/src/dst/simulation.rs:73
pub fn create_memory(&self) -> Memory<...> {
    let seed = self.config.seed();

    // Create all providers with the shared fault injector
    let llm = SimLLMProvider::with_faults(seed, Arc::clone(&self.faults));
    let embedder = SimEmbeddingProvider::with_faults(seed, Arc::clone(&self.faults));
    let vector = SimVectorBackend::with_faults(seed, Arc::clone(&self.faults));
    let storage = SimStorageBackend::with_fault_injector(self.config, Arc::clone(&self.faults));

    Memory::new(llm, embedder, vector, storage)
}
```

### Step 3: Update Discovery Test to Use New API

Changed from:
```rust
let mut memory = Memory::sim(42);  // ❌ Isolated providers
```

To:
```rust
let mut memory = env.create_memory();  // ✅ Connected to faults!
```

---

## Phase 6: Verification - Test Now Passes!

```bash
$ cargo test test_dst_discovery_memory_fault_injection
```

**Result**: ✅ **TEST PASSES**!

```
test dst::simulation::tests::test_dst_discovery_memory_fault_injection ... ok

test result: ok. 1 passed; 0 failed; 0 ignored
```

**Success!** The test now correctly:
1. Registers 100% StorageWriteFail fault
2. Creates Memory via `env.create_memory()`
3. Calls `memory.remember()`
4. **FAILS** as expected (proves fault is injected)
5. Test assertion passes

---

## Summary: TRUE DST-First Methodology

### What We Did RIGHT (DST-First):

1. ✅ **Wrote test FIRST** - before knowing the solution
2. ✅ **Test FAILED** - revealed the problem (`Memory::sim()` not connected)
3. ✅ **Used failure to discover** - investigated why faults weren't applied
4. ✅ **Found root cause** - isolated providers with empty FaultInjectors
5. ✅ **Implemented minimal fix** - `env.create_memory()`
6. ✅ **Test now PASSES** - verified the fix works

### Key Discoveries Made Through DST:

| Discovery | How DST Revealed It |
|-----------|---------------------|
| `Memory::sim()` creates isolated providers | Test with 100% fault succeeded when it should fail |
| Providers already support external FaultInjectors | Investigation of existing APIs |
| SimStorageBackend API inconsistency | Comparing `with_faults()` signatures across providers |
| Need for `SimEnvironment::create_memory()` | No way to pass `env.faults` to `Memory::sim()` |

### The DST-First Difference:

**What I did BEFORE (Test-After)**:
- Read bug report → Design solution → Write code → Write tests → All pass ✅

**What I did NOW (DST-First)**:
- Write test → Run → **FAILS** ❌ → Investigate failure → Discover problem → Fix → Test passes ✅

**The Critical Difference**:
- **Test-After**: Tests verify known solutions (no discovery)
- **DST-First**: Tests discover unknown problems through simulation

---

## Files Modified (DST-First Implementation)

1. **umi-memory/src/storage/sim.rs** - Added `with_fault_injector()` constructor
2. **umi-memory/src/dst/simulation.rs** - Added `create_memory()` method and discovery test
3. **.progress/015_DST_FIRST_DEMO.md** - Documented the DST-first process

**Lines of implementation code**: ~40 lines
**Lines of discovery documentation**: ~200 lines

The documentation is 5x longer because **DST-first is about the discovery process**, not just the implementation!
