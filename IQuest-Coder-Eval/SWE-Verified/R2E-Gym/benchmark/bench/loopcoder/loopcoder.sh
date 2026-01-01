#!/bin/bash
chmod +x benchmark/eval_swe_verified.sh

export HF_ENDPOINT=https://hf-mirror.com

MAX_RETRIES=1
TEMPERATURE=1.0
MAX_STEPS=200

echo "Starting evaluation with MAX_STEPS=$MAX_STEPS..."
echo "Starting rollout $run_idx (MAX_STEPS=$MAX_STEPS)..."
benchmark/eval_swe_verified.sh \
    "iquest-v1-loop-instruct-xxxx" \
    "evaluation-swe-benchmark-output/${MAX_STEPS}_${MAX_RETRIES}_${TEMPERATURE}" \
    "http://litellm-xxx.com/v1" \
    "sk-abc123" \
    64 \
    131072 \
    "src/r2egym/agenthub/run/edit.py" \
    ${MAX_RETRIES} \
    ${TEMPERATURE} \
    "openhands" \
    "tcp://xxx.xxx.xxx.xxx:60001" \
    ${MAX_STEPS} \
    1.0 \
    "docker"

