# Custom SFT Training Scripts

This directory contains custom Supervised Fine-Tuning (SFT) scripts for training models before applying reinforcement learning algorithms.

## Directory Structure

```
custom_sft/
├── README.md                # This file - training plans and documentation
├── configs/                 # Custom configuration files for SFT
├── scripts/                 # Training scripts for different models
├── data_preprocessing/      # Data preparation scripts
└── checkpoints/            # Directory for saving SFT model checkpoints
```

## Training Plan

### 1. Data Preparation
- [ ] Prepare dataset in the required format
- [ ] Create data preprocessing scripts
- [ ] Validate data quality and format

### 2. SFT Training
- [ ] Configure model and training parameters
- [ ] Run SFT training for baseline model
- [ ] Monitor training metrics (loss, validation scores)
- [ ] Save best checkpoint for RL training

### 3. Evaluation
- [ ] Evaluate SFT model performance
- [ ] Compare with base model
- [ ] Decide if ready for RL training

## Key Files for SFT in verl

### Core SFT Implementation
- **Main SFT Trainer**: `verl/trainer/fsdp_sft_trainer.py` - The core FSDP-based SFT implementation
- **SFT Configuration**: `verl/trainer/config/sft_trainer.yaml` - Default configuration template
- **Main Entry Point**: `verl/trainer/main_sft.py` - Entry point for SFT training

### Example Scripts to Reference
- **Simple Example**: `recipe/char_count/train_sft.sh` - Beginner-friendly example
- **GSM8K Examples**: `examples/sft/gsm8k/` - Various model examples:
  - `run_deepseek_7b.sh` - DeepSeek 7B model
  - `run_llama3_8b.sh` - Llama 3 8B model
  - `run_qwen_05.sh` - Qwen 0.5B model
- **Advanced Example**: `recipe/retool/run_qwen2-32b_sft.sh` - Large model with advanced configs

### Data Processing Examples
- `examples/data_preprocess/gsm8k/prepare_gsm8k_sft_data.py` - GSM8K data preparation
- `recipe/char_count/create_dataset.py` - Simple dataset creation example

## Quick Start

1. **Prepare your data**:
   ```bash
   # Reference existing examples or create your own
   python data_preprocessing/prepare_data.py
   ```

2. **Configure training**:
   - Copy `verl/trainer/config/sft_trainer.yaml` to `configs/sft_config.yaml`
   - Adjust hyperparameters as needed
   - Reference example configs in `examples/sft/gsm8k/`

3. **Run SFT training**:
   ```bash
   # Create your script based on examples
   bash scripts/train_sft.sh
   ```

4. **Evaluate results**:
   ```bash
   python scripts/evaluate_sft.py --checkpoint-path checkpoints/best_model
   ```

## Next Steps: RL Training

After successful SFT training, proceed to RL training using your SFT checkpoint.

## Understanding RL Training in verl

### What is RL Training for LLMs?

Reinforcement Learning (RL) training for LLMs optimizes the model to generate better responses based on reward signals. Unlike SFT which learns from fixed examples, RL training:
- **Generates** responses from the model
- **Evaluates** them using a reward function
- **Updates** the model to maximize rewards

### Key Components

1. **Actor Model**: Your SFT-trained model that generates responses
2. **Critic Model**: Estimates the value/quality of generated responses
3. **Reward Model**: Scores how good a response is (can be external API, human feedback, or another model)
4. **Reference Model**: Original model used to prevent the actor from deviating too much (KL regularization)

### Supported RL Algorithms

- **PPO (Proximal Policy Optimization)**: Most popular, stable training
- **GRPO (Grouped Reward Policy Optimization)**: More efficient variant
- **REINFORCE++**: Simple policy gradient method
- **RLOO**: Leave-one-out baseline method
- **Others**: ReMax, PRIME, DAPO, DrGRPO

## Key Files for RL Training

### Core RL Implementation
- **Main Entry Point**: `verl/trainer/main_ppo.py` - Orchestrates the entire RL training
- **PPO Trainer**: `verl/trainer/ppo/ray_trainer.py` - Core PPO training logic
- **PPO Algorithm**: `verl/single_controller/ppo_algo.py` - PPO algorithm implementation
- **Configuration**: `verl/trainer/config/ppo_trainer.yaml` - Default PPO configuration

### Worker Components
- **Actor-Rollout Worker**: `verl/workers/fsdp_workers.py` - Manages policy model and generation
- **Critic Worker**: `verl/workers/fsdp_workers.py` - Value estimation
- **Reward Worker**: `verl/workers/reward_worker.py` - Reward computation

### Example Scripts
- **PPO Examples**: `examples/ppo_trainer/`
  - `run_deepseek_7b.sh` - DeepSeek 7B model
  - `run_qwen_05b.sh` - Qwen 0.5B model
- **GRPO Example**: `recipe/char_count/train_grpo.sh` - Simple GRPO training
- **Advanced Examples**: `recipe/gsm8k/`, `recipe/retool/`

## Quick Start RL Training

### 1. Prepare RL Dataset
RL datasets need prompts (without responses):
```python
# Format: {"prompt": "Question: What is 2+2?"}
# The model will generate responses during training
```

### 2. Configure RL Training
Key parameters to adjust:
```yaml
# Model path - use your SFT checkpoint
actor_rollout_ref.model.path: /path/to/sft/checkpoint

# Training parameters
data.train_batch_size: 128
actor_rollout_ref.actor.optim.lr: 1e-6  # Lower than SFT
actor_rollout_ref.actor.ppo_mini_batch_size: 16

# Algorithm settings
algorithm.adv_estimator: grpo  # or "gae" for standard PPO
actor_rollout_ref.actor.use_kl_loss: true  # Prevent overfitting
actor_rollout_ref.actor.kl_loss_coef: 0.1
```

### 3. Run RL Training

**For PPO**:
```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=/your/sft/checkpoint \
    data.train_files=/path/to/rl/train.parquet \
    data.val_files=/path/to/rl/val.parquet
```

**For GRPO** (simpler, more stable):
```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=/your/sft/checkpoint \
    # ... other parameters
```

## RL Training Workflow

1. **Rollout Phase**: Generate responses for prompts
2. **Reward Phase**: Score the generated responses
3. **Advantage Estimation**: Calculate how much better/worse each action was
4. **Policy Update**: Update actor model using PPO/GRPO
5. **Value Update**: Update critic model
6. **Repeat**: Continue for multiple epochs

## Key Differences from SFT

| Aspect | SFT | RL Training |
|--------|-----|-------------|
| **Data** | Prompts + Responses | Prompts only |
| **Learning** | Supervised (fixed targets) | Reinforcement (exploration) |
| **Batch Size** | Larger (256+) | Smaller (32-128) |
| **Learning Rate** | Higher (1e-5) | Lower (1e-6) |
| **Training Time** | Faster | Slower (generation needed) |
| **Memory Usage** | Lower | Higher (multiple models) |

## Tips for RL Training

1. **Start with GRPO**: Simpler and more stable than PPO
2. **Use KL regularization**: Prevents model from deviating too much
3. **Monitor rewards**: Ensure rewards are increasing but not exploding
4. **Smaller batches**: RL is more unstable, smaller batches help
5. **Lower learning rate**: Typically 10x lower than SFT
6. **Checkpoint frequently**: RL can be unstable

## Common Issues and Solutions

- **Out of Memory**: Reduce batch size or use LoRA
- **Reward Hacking**: Model finds shortcuts - improve reward function
- **Unstable Training**: Lower learning rate, increase KL coefficient
- **Slow Training**: Use vLLM for faster generation

## Notes

- SFT provides a good initialization for RL training
- RL training typically runs for fewer epochs than SFT (1-2 epochs)
- Monitor both rewards and response quality during training
- The SFT → RL pipeline is the standard RLHF approach