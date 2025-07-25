# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready reinforcement learning training library for large language models. It implements the HybridFlow architecture from ByteDance Seed team.

## Key Development Commands

### Installation
```bash
# Basic installation
pip install -e .

# With specific features
pip install -e .[test,vllm]      # vLLM integration
pip install -e .[test,sglang]    # SGLang integration
pip install -e .[test,gpu]       # GPU optimizations
pip install -e .[test,mcore]     # Megatron-LM support
```

### Code Quality
```bash
# Run pre-commit hooks (linting, formatting)
pre-commit run --all-files

# Run specific linter with details
pre-commit run --all-files --show-diff-on-failure --color=always ruff
```

### Testing
```bash
# Run all tests
pytest

# Run CPU-only tests
pytest tests/**/test_*_on_cpu.py

# Run specific test category
pytest tests/trainer/              # Trainer tests
pytest tests/workers/              # Worker tests
pytest tests/single_controller/    # Controller tests

# Run a single test
pytest tests/trainer/test_ppo.py::test_specific_function
```

### Documentation
```bash
cd docs
make clean
make html
# Preview at http://localhost:8000
python -m http.server -d _build/html/
```

## Architecture Overview

### HybridFlow Single-Controller Design
The core architecture separates control flow (single process) from computation flow (multi-process):

1. **Controller Process** (`verl/single_controller/`): Manages RL algorithm logic in a single process
2. **Worker Groups** (`verl/workers/`):
   - `ActorRolloutRef`: Policy model management and generation
   - `Critic`: Value estimation
   - `Reward`: Reward computation
3. **Communication**: Ray-based distributed system with custom dispatch patterns

### Key Directories
- `verl/trainer/`: Training implementations (PPO, GRPO, etc.)
- `verl/models/`: Model implementations and integrations
- `verl/utils/`: Shared utilities
- `verl/experimental/`: Experimental features (agent loops, etc.)
- `examples/`: Example scripts for various algorithms
- `recipe/`: Advanced algorithms and recipes

### Configuration System
Uses Hydra-core for hierarchical configuration:
- Main configs in `verl/trainer/config/`
- Override with command line: `python train.py model.path=/new/path training.batch_size=32`
- Custom YAML configs supported

## Common Development Tasks

### Adding a New Algorithm
1. Create algorithm class in `verl/trainer/`
2. Implement required methods following existing patterns (see PPO/GRPO)
3. Add configuration in `verl/trainer/config/`
4. Add tests in `tests/trainer/`
5. Add example script in `examples/`

### Extending Workers
1. Inherit from base worker classes in `verl/workers/`
2. Override necessary methods for custom behavior
3. Register with the worker factory if needed
4. Test with distributed tests in `tests/special_distributed/`

### Running Distributed Training
```bash
# Local multi-GPU
torchrun --nproc_per_node=4 examples/ppo_trainer.py

# Multi-node (use Ray for coordination)
# Configure ray.init() parameters in your script
```

## Testing Patterns

- **CPU Tests**: Files ending with `_on_cpu.py` for quick local testing
- **GPU Tests**: In `tests/special_distributed/` for multi-GPU scenarios
- **E2E Tests**: In `tests/special_e2e/` for full training runs
- **Sanity Tests**: In `tests/special_sanity/` for quick validation

## Important Notes

- Python 3.10+ required
- Ray (>=2.41.0) is core dependency for distributed execution
- Supports FSDP, FSDP2, Megatron-LM, vLLM, and SGLang integrations
- Uses Flash Attention 2 and Liger Kernel for optimization
- Supports models up to 671B parameters with expert parallelism
- LoRA support available for memory-efficient training