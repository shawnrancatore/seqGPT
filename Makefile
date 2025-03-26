# SeqGPT Makefile

.PHONY: setup train-gpt2 train-mixtral generate complexity-test docker-build docker-train-gpt2 docker-train-mixtral docker-generate help

PYTHON = python
MODEL_DIR = models
DOCKER_IMAGE = seqgpt:latest
MODEL = models/best_model.pt  # Default model path for generation

help:
	@echo "Available commands:"
	@echo "  make setup               - Install dependencies"
	@echo "  make train-gpt2          - Train a GPT2 model"
	@echo "  make train-gpt2-advanced - Train an advanced GPT2 model"
	@echo "  make train-mixtral       - Train a pattern-aware Mixtral model"
	@echo "  make generate MODEL=path - Generate sequences from a model (default: models/best_model.pt)"
	@echo "  make complexity-test     - Run pattern complexity tests"
	@echo "  make create-config       - Create complexity test configuration"
	@echo "  make docker-build        - Build Docker image"
	@echo "  make docker-train-gpt2   - Train a GPT2 model inside Docker"
	@echo "  make docker-train-mixtral - Train a Mixtral model inside Docker"
	@echo "  make docker-generate     - Generate sequences inside Docker"
	@echo ""
	@echo "Examples:"
	@echo "  make generate MODEL=models/gpt2_epoch20.pt"
	@echo "  make complexity-test PATTERN=fibonacci_type"

setup:
	pip install -r requirements.txt

# Main model training
train-gpt2:
	$(PYTHON) main.py train --model_type gpt2 --save_model

train-gpt2-advanced:
	$(PYTHON) main.py train --model_type gpt2 --n_embd 256 --n_layer 8 --n_head 8 --batch_size 64 --epochs 30 --lr 1e-4 --save_model

# Pattern-aware Mixtral training
train-mixtral:
	$(PYTHON) train_pattern_mixtral.py --save_model --analyze_every 5

train-mixtral-advanced:
	$(PYTHON) train_pattern_mixtral.py --n_embd 256 --n_layer 6 --n_head 8 --num_experts 12 --top_k_experts 3 --batch_size 64 --epochs 30 --save_model --analyze_every 5

# Generation
generate:
	$(PYTHON) main.py generate --model_path $(MODEL) --num_samples 5

# Complexity testing
create-config:
	$(PYTHON) complexity_test.py create_config --output complexity_config.json

complexity-test:
ifdef PATTERN
	$(PYTHON) complexity_test.py run --pattern_type $(PATTERN) --save_results results.json
else
	$(PYTHON) complexity_test.py run --patterns_config complexity_config.json --cross_validate --save_results results.json
endif

# Docker commands
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-train-gpt2:
	docker run --gpus all --shm-size=1g -v $(PWD):/app $(DOCKER_IMAGE) python main.py train --model_type gpt2 --save_model

docker-train-mixtral:
	docker run --gpus all --shm-size=1g -v $(PWD):/app $(DOCKER_IMAGE) python train_pattern_mixtral.py --save_model --analyze_every 5

docker-generate:
	docker run --gpus all --shm-size=1g -v $(PWD):/app $(DOCKER_IMAGE) python main.py generate --model_path $(MODEL) --num_samples 5

# Multi-GPU training examples (default is to use all available GPUs)
train-gpt2-multi-gpu:
	$(PYTHON) main.py train --model_type gpt2 --save_model --batch_size 128

train-mixtral-multi-gpu:
	$(PYTHON) train_pattern_mixtral.py --save_model --analyze_every 5 --batch_size 128