# SeqGPT: Sequence Pattern Learning and Generation

A framework for training transformer-based models to learn and generate numerical sequences with different patterns.

## Features

- Train various transformer architectures on sequence prediction tasks
- Generate new sequences using trained models
- Specialized Mixture-of-Experts (MoE) model for pattern-aware training
- Comprehensive complexity testing for different pattern types
- Support for multi-GPU training
- Expert specialization visualization
- Cross-pattern performance evaluation

## Requirements

- Python 3.11+
- PyTorch 2.6+
- CUDA (optional, for GPU acceleration)

## Quick Start

### Using Make (recommended)

```bash
# Install dependencies
make setup

# Train a standard GPT2 model
make train-gpt2

# Train a pattern-aware Mixtral model
make train-mixtral

# Generate sequences from a trained model
make generate MODEL=models/best_model.pt

# Run complexity tests
make complexity-test

# Run with Docker
make docker-train-gpt2
make docker-train-mixtral
make docker-generate MODEL=models/best_model.pt
```

### Manual Usage

```bash
# Train a standard GPT2 model
python main.py train --model_type gpt2 --save_model

# Generate sequences from a trained model
python main.py generate --model_path models/best_model.pt --num_samples 5

# Train a pattern-aware Mixtral model
python train_pattern_mixtral.py --save_model --analyze_every 5

# Run complexity tests
python complexity_test.py run --patterns_config complexity_config.json
```

## Command-Line Interface

### Main Training and Generation (`main.py`)

The `main.py` script provides a general-purpose interface for training and generating with various model architectures.

#### Training

```bash
python main.py train [options]
```

Key options:
- `--model_type`: Model architecture (gpt2, mixtral, llama, llama2, neotransformer)
- `--n_embd`: Embedding dimension (default: 128)
- `--n_layer`: Number of transformer layers (default: 4)
- `--n_head`: Number of attention heads (default: 4)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 20)
- `--lr`: Learning rate (default: 3e-4)
- `--save_model`: Save model checkpoints
- `--resume`: Path to checkpoint to resume training from

MoE-specific parameters:
- `--num_experts`: Number of experts (default: 8)
- `--top_k_experts`: Number of experts to route to (default: 2)
- `--aux_loss_weight`: Weight for auxiliary losses (default: 0.01)

#### Generation

```bash
python main.py generate [options]
```

Key options:
- `--model_path`: Path to model checkpoint (required)
- `--num_samples`: Number of samples to generate (default: 5)
- `--max_tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Temperature for sampling (default: 0.8)
- `--top_k`: Top-k sampling parameter (default: 50)

### Pattern-Aware Mixtral Training (`train_pattern_mixtral.py`)

The `train_pattern_mixtral.py` script specializes in training pattern-aware Mixture-of-Experts models.

```bash
python train_pattern_mixtral.py [options]
```

Key options:
- `--n_embd`: Embedding dimension (default: 128)
- `--n_layer`: Number of transformer layers (default: 4)
- `--n_head`: Number of attention heads (default: 4)
- `--num_experts`: Number of experts (default: 8)
- `--top_k_experts`: Number of experts to route to (default: 2)
- `--aux_loss_weight`: Weight for auxiliary losses (default: 0.01)
- `--analyze_every`: Analyze expert specialization every N epochs (default: 5)
- `--save_model`: Save model checkpoints
- `--resume`: Path to checkpoint to resume training from

### Complexity Testing (`complexity_test.py`)

Test the complexity requirements for learning different pattern types.

```bash
# Create a complexity test configuration
python complexity_test.py create_config --output my_config.json

# Run tests using the configuration
python complexity_test.py run --patterns_config complexity_config.json

# Test a specific pattern type
python complexity_test.py run --pattern_type fibonacci_type

# Run with cross-validation
python complexity_test.py run --patterns_config complexity_config.json --cross_validate

# Save results to a JSON file
python complexity_test.py run --save_results results.json
```

## Advanced Usage

### Custom Model Training

Train a GPT2 model with custom parameters:

```bash
python main.py train \
  --model_type gpt2 \
  --n_embd 256 \
  --n_layer 8 \
  --n_head 8 \
  --batch_size 64 \
  --epochs 30 \
  --lr 1e-4 \
  --save_model
```

Train a pattern-aware Mixtral model with custom parameters:

```bash
python train_pattern_mixtral.py \
  --n_embd 256 \
  --n_layer 6 \
  --n_head 8 \
  --num_experts 12 \
  --top_k_experts 3 \
  --batch_size 64 \
  --epochs 30 \
  --save_model \
  --analyze_every 5
```

### Multi-GPU Training

Both scripts support multi-GPU training via PyTorch DataParallel:

```bash
# Set CUDA_VISIBLE_DEVICES to select specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train --model_type gpt2 --save_model
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_pattern_mixtral.py --save_model
```

### Complexity Testing with Custom Model Configurations

Test pattern complexity with custom model sizes:

```bash
python complexity_test.py run \
  --model_sizes '{"tiny": {"n_embd": 32, "n_layer": 1, "n_head": 1}, "small": {"n_embd": 64, "n_layer": 2, "n_head": 2}}' \
  --cross_validate
```

## Docker Support

The project includes Docker support for consistent environments:

```bash
# Build the Docker image
make docker-build

# Train a model using Docker
make docker-train-gpt2
make docker-train-mixtral

# Generate sequences using Docker
make docker-generate MODEL=models/best_model.pt
```

## License

[MIT License](LICENSE)