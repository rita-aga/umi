# Umi Python Examples

Example scripts demonstrating Umi Memory Python bindings.

## Prerequisites

```bash
# Install in development mode
cd umi-py
maturin develop

# Or install from PyPI (once published)
pip install umi-memory
```

## Examples

### Basic Usage

- **01_basic_sync_sim.py** - Basic remember/recall with Sim providers (sync API)
- **02_options_demo.py** - Using RememberOptions and RecallOptions
- **03_deterministic_demo.py** - Demonstrating DST (same seed = same results)
- **04_async_demo.py** - Native Python async/await support

### Running Examples

```bash
# From the umi-py/examples directory
python 01_basic_sync_sim.py
python 02_options_demo.py
python 03_deterministic_demo.py
```

## Notes

- Current examples use **Sim providers** (deterministic, in-memory)
- Real providers (Anthropic, OpenAI, LanceDB, Postgres) are exposed but not yet wired into Memory class
- All provider classes are available: `umi.AnthropicProvider`, `umi.OpenAIProvider`, etc.
- See `umi.pyi` for full type annotations
