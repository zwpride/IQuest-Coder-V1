## About

This framework is based on and references [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym), a benchmark framework for evaluating code editing agents on software engineering tasks.

## Quick Start

### Requirements

- Docker (for containerized runtime environment)
- Sufficient disk space (for storing datasets and evaluation results)

### Installation

```bash
# Install dependencies
pip install -e .
```

### Configuration

Before use, you need to set the following environment variables:

```bash
# HuggingFace mirror endpoint (optional, for faster downloads)
export HF_ENDPOINT=https://hf-mirror.com

# OpenAI API configuration (if using OpenAI-compatible API)
export OPENAI_API_BASE=<your-api-base-url>
export OPENAI_API_KEY=<your-api-key>

# Docker host configuration (if using remote Docker)
export DOCKER_HOST=tcp://<docker-host>:<port>
```

## Usage

### Using the Launch Script

The project provides a convenient launch script `benchmark/bench/loopcoder/loopcoder.sh` that can directly run evaluations:

```bash
bash benchmark/bench/loopcoder/loopcoder.sh
```

### Launch Script Description

Main configuration parameters of the `loopcoder.sh` script:

- **MAX_STEPS**: Maximum execution steps (default: 200)
- **MAX_RETRIES**: Maximum retry count (default: 1)
- **TEMPERATURE**: Model temperature parameter (default: 1.0)
- **Threads**: 64
- **Max Tokens**: 131072
- **Scaffold**: `openhands`
- **Backend**: `docker`

## Evaluation Results

Evaluation results are saved in the `evaluation-swe-benchmark-output/` directory, including:

- **Trajectory files** (`.jsonl`): Contains complete trajectory information for each run
- **Log files** (`.log`): Detailed runtime logs
- **Result files** (`*_results.json`): Evaluation result statistics
