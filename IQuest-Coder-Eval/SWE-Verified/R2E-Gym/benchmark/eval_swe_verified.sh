#!/bin/bash

#############################################################################################################################################
# Signal handling: catch Ctrl+C and other termination signals, clean up remaining processes
BENCHMARK_PIDS=()

cleanup() {
    echo ""
    echo "=========================================="
    echo "Termination signal caught, cleaning up child processes..."
    echo "=========================================="
    
    # Clean up all recorded benchmark processes
    for pid in "${BENCHMARK_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating process PID: $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # Wait for processes to terminate
    sleep 2
    
    # Force kill processes still running
    for pid in "${BENCHMARK_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force terminating process PID: $pid"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    
    # Clean up all related Python processes (based on script path)
    echo "Cleaning up all related runagent_multiple processes..."
    pkill -f "runagent_multiple" 2>/dev/null || true
    
    echo "Cleanup completed"
    exit 130  # 130 = 128 + 2 (SIGINT)
}

# Register signal handler
trap cleanup SIGINT SIGTERM SIGHUP

#############################################################################################################################################
# Signal handling: catch Ctrl+C and other termination signals, clean up remaining processes
BENCHMARK_PIDS=()

cleanup() {
    echo ""
    echo "=========================================="
    echo "Termination signal caught, cleaning up child processes..."
    echo "=========================================="
    
    # Clean up all recorded benchmark processes
    for pid in "${BENCHMARK_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating process PID: $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    
    # Wait for processes to terminate
    sleep 2
    
    # Force kill processes still running
    for pid in "${BENCHMARK_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force terminating process PID: $pid"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    
    # Clean up all related Python processes (based on script path)
    echo "Cleaning up all related runagent_multiple processes..."
    pkill -f "runagent_multiple" 2>/dev/null || true
    
    echo "Cleanup completed"
    exit 130  # 130 = 128 + 2 (SIGINT)
}

# Register signal handler
trap cleanup SIGINT SIGTERM SIGHUP

#############################################################################################################################################
START_TIME=$(date +%s)
START_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
ROOT_DIR=`pwd`
echo "ROOT_DIR: ${ROOT_DIR}"

#############################################################################################################################################
# BENCHMARK WORK SETUP, Customized for LiveRepoReflection, Compatible with Aider Polyglot Benchmark

export HF_ENDPOINT=https://hf-mirror.com
export VE_FAAS_AK="VE_FAAS_AK-KEY"
export VE_FAAS_SK="VE_FAAS_SK-KEY"
export VE_FAAS_FUNCTION_ID="VE_FAAS_FUNCTION_ID"
export VE_FAAS_SANDBOX_NETWORK="https://VE_FAAS_SANDBOX_NETWORK/sandbox-eval"


# OPENAI COMPATIBLE MODEL NAME 
MODEL_NAME=$1
MODEL_NAME=${MODEL_NAME:-""}
# EVALUATION OUTPUT DIR
OUTPUT_DIR=$2
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark/swe_results"}
# OUTPUT_DIR_FOR_THIS_RUN=${OUTPUT_DIR}/${MODEL_NAME}_${DIR_UUID}
OUTPUT_DIR_FOR_THIS_RUN=${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR_FOR_THIS_RUN}
# OPENAI API BASE URL
CUSTOM_OPENAI_API_BASE=$3
CUSTOM_OPENAI_API_BASE=${CUSTOM_OPENAI_API_BASE:-"http://127.0.0.1:8000/v1/"}
export LLM_BASE_URL=${CUSTOM_OPENAI_API_BASE} # todo: check
# OPENAI API KEY
CUSTOM_OPENAI_API_KEY=$4
CUSTOM_OPENAI_API_KEY=${CUSTOM_OPENAI_API_KEY:-"token-abc123"}
export OPENAI_API_KEY=${CUSTOM_OPENAI_API_KEY:-"token-abc123"}
# BENCHMARK MULTI-THREADS NUM
THREADS_NUM=$5
THREADS_NUM=${THREADS_NUM:-50}

# MODEL MAX TOKENS
MODEL_MAX_LEN=$6
MODEL_MAX_LEN=${MODEL_MAX_LEN:-256000}
export MODEL_MAX_LEN=${MODEL_MAX_LEN}
# BENCHMARK PYTHON SCRIPT PATH
BENCHMARK_PYTHON_SCRIPT_PATH=$7
BENCHMARK_PYTHON_SCRIPT_PATH=${BENCHMARK_PYTHON_SCRIPT_PATH:-"src/r2egym/agenthub/run/edit.py"}
# PROXY ON BENCHMARK AND LARGE LANGUAGE MODELS, OPTION: off, stream, non_stream
MAX_TURN_RETRY=$8
MAX_TURN_RETRY=${MAX_TURN_RETRY:-3}
export MAX_TURN_RETRY=${MAX_TURN_RETRY}

# Set temperature
TEMPERATURE=$9
TEMPERATURE=${TEMPERATURE:-0.0}

# Set scaffold
SCAFFOLD=${10}
SCAFFOLD=${SCAFFOLD:-"openhands"}

# Set docker_host
DOCKER_HOST=${11}
DOCKER_HOST=${DOCKER_HOST:-"tcp://xxx.xxx.xxx.xxx:60001"}
export DOCKER_HOST=${DOCKER_HOST}

# Set max_steps
MAX_STEPS=${12}
MAX_STEPS=${MAX_STEPS:-100}

# Set the ratio of data to test
TEST_RATIO=${13}
TEST_RATIO=${TEST_RATIO:-1.0}

# # Set dataset
# DATASET_NAME=${14}
# DATASET_NAME=${DATASET_NAME:-"SWE-bench_Verified"}

# Set backend
BACKEND=${14}
BACKEND=${BACKEND:-"docker"}

# Set docker image filter file, optional, default is empty string
DOCKER_IMAGE_FILE=${15}
DOCKER_IMAGE_FILE=${DOCKER_IMAGE_FILE:-""}

# Set monitoring time
LAST_MONITOR_TIME=${16}
LAST_MONITOR_TIME=${LAST_MONITOR_TIME:-600}
# Set monitoring time
TOTAL_MONITOR_TIME=${17}
TOTAL_MONITOR_TIME=${TOTAL_MONITOR_TIME:-900}

export LAST_MONITOR_TIME=${LAST_MONITOR_TIME}
export TOTAL_MONITOR_TIME=${TOTAL_MONITOR_TIME}

echo "EXTRA_HEADERS: ${EXTRA_HEADERS}"
echo "EXTRA_BODY: ${EXTRA_BODY}"
echo "EXTRA_QUERY: ${EXTRA_QUERY}"


# litellm need service provider, so we use openai prefix default
API_MODEL_NAME=openai/${MODEL_NAME}

# Set exp_name: obtained by concatenating model name, scaffold, and temperature
EXP_NAME=${MODEL_NAME}_${SCAFFOLD}_${TEMPERATURE}


echo "OPENAI_API_BASE: ${OPENAI_API_BASE}"
echo "OPENAI_API_KEY: ${OPENAI_API_KEY}"
echo "MODEL_NAME: ${MODEL_NAME}, API_MODEL_NAME: ${API_MODEL_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "OUTPUT_DIR_FOR_THIS_RUN: ${OUTPUT_DIR_FOR_THIS_RUN}"
echo "THREADS_NUM: ${THREADS_NUM}"
echo "MODEL_MAX_LEN: ${MODEL_MAX_LEN}"
echo "BENCHMARK_PYTHON_SCRIPT_PATH: ${BENCHMARK_PYTHON_SCRIPT_PATH}"
echo "MAX_TURN_RETRY: ${MAX_TURN_RETRY}"
echo "TEMPERATURE: ${TEMPERATURE}"
echo "SCAFFOLD: ${SCAFFOLD}"
echo "DOCKER_HOST: ${DOCKER_HOST}"
echo "MAX_STEPS: ${MAX_STEPS}"


cd ${ROOT_DIR} && echo "Changed to ROOT_DIR: ${ROOT_DIR}"


# BENCHMARK RUN


## BENCHMARK SCRIPT RUN
run_benchmark() {

    local DATASET=$1
    local K=$2
    local SPLIT=$3
    local DATA_EXP_NAME=${EXP_NAME}_${DATASET}
    export USE_DOCKER_CLIENT_POOL=true
    export THREADS_NUM=${THREADS_NUM}

    mkdir -p ${OUTPUT_DIR_FOR_THIS_RUN}/traj
    DATASET=${DATASET:-"SWE-bench_Verified"}
    K=${K:-500}
    echo "BENCHMARK_PYTHON_SCRIPT_PATH: ${BENCHMARK_PYTHON_SCRIPT_PATH}"
    echo "MODEL_NAME: ${MODEL_NAME} => (DIR_NAME: ${MODEL_NAME})"
    echo "API_MODEL_NAME: ${API_MODEL_NAME}"
    echo "THREADS_NUM: ${THREADS_NUM}"
    echo "Expected lines (K): ${K}"
    echo "SPLIT: ${SPLIT}"

    local attempt=1
    local max_attempts=3
    local jsonl_file="${OUTPUT_DIR_FOR_THIS_RUN}/traj/${DATA_EXP_NAME}.jsonl"
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempting execution ${attempt}..."
        
        # Run in background for monitoring long-running scenarios with no progress
        python ${BENCHMARK_PYTHON_SCRIPT_PATH} runagent_multiple \
            --traj_dir "${OUTPUT_DIR_FOR_THIS_RUN}/traj" \
            --max_steps ${MAX_STEPS} \
            --max_workers ${THREADS_NUM} \
            --start_idx 0 \
            --k ${K} \
            --dataset data/${DATASET} \
            --split ${SPLIT} \
            --llm_name ${API_MODEL_NAME} \
            --scaffold ${SCAFFOLD} \
            --use_fn_calling False \
            --exp_name ${DATA_EXP_NAME} \
            --temperature ${TEMPERATURE} \
            --max_steps_absolute ${MAX_STEPS} \
            --backend ${BACKEND} \
            --max_reward_calc_time 1200 \
            --max_tokens ${MODEL_MAX_LEN}  > ${OUTPUT_DIR_FOR_THIS_RUN}/traj/${DATA_EXP_NAME}_${attempt}.log 2>&1 &

        local benchmark_pid=$!
        echo "BENCHMARK PID: ${benchmark_pid}"
        
        # Record PID to global array for cleanup function
        BENCHMARK_PIDS+=("${benchmark_pid}")

        # Monitor: check process status and jsonl file updates every 60 seconds
        local stall_threshold=${LAST_MONITOR_TIME}  # 10 minute no-update threshold (changed from 15 minutes to 10 minutes)
        local check_interval=60     # Check every 60 seconds
        local last_seen_line_count=0
        local last_mtime=0
        local last_update_time=$(date +%s)
        local killed_for_stall=false

        while kill -0 ${benchmark_pid} 2>/dev/null; do
            sleep ${check_interval}
            
            if [ -f "$jsonl_file" ]; then
                local line_count=$(wc -l < "$jsonl_file")
                local mtime=$(stat -c %Y "$jsonl_file")

                if [ "$mtime" -ne "$last_mtime" ] || [ "$line_count" -ne "$last_seen_line_count" ]; then
                    last_update_time=$(date +%s)
                    last_mtime=$mtime
                    last_seen_line_count=$line_count
                fi

                local pending=$((K - line_count))
                local now=$(date +%s)
                local stall_time=$((now - last_update_time))

                # Print status on each check for debugging
                echo "[Monitor] Current progress: ${line_count}/${K}, Remaining: ${pending}, Time since last update: ${stall_time} seconds"

                # Improved monitoring logic: terminate if remaining tasks <= 20 and no update for 10 minutes
                if [ "$pending" -le 20 ] && [ ${stall_time} -ge ${stall_threshold} ]; then
                    echo "Detected ${stall_time} seconds with no update and remaining lines <= 20, terminating process ${benchmark_pid}"
                    kill ${benchmark_pid} 2>/dev/null || true
                    wait ${benchmark_pid} 2>/dev/null || true
                    killed_for_stall=true
                    break
                fi
                
                # New: terminate if no update for 30 minutes at any stage (prevent early hang)
                if [ ${stall_time} -ge ${TOTAL_MONITOR_TIME} ]; then
                    echo "Detected 30 minutes with no update (any stage), force terminating process ${benchmark_pid}"
                    kill ${benchmark_pid} 2>/dev/null || true
                    wait ${benchmark_pid} 2>/dev/null || true
                    killed_for_stall=true
                    break
                fi
            fi
        done

        if [ "${killed_for_stall}" = false ] && kill -0 ${benchmark_pid} 2>/dev/null; then
            wait ${benchmark_pid}
        fi
        
        # Remove completed PID from BENCHMARK_PIDS array
        BENCHMARK_PIDS=("${BENCHMARK_PIDS[@]/$benchmark_pid}")
        
        # Check if JSONL file exists and count lines
        if [ -f "$jsonl_file" ]; then
            local line_count=$(wc -l < "$jsonl_file")
            echo "JSONL file line count: ${line_count}"
            
            if [ "$line_count" -ge "$K" ]; then
                echo "Success! Generated line count (${line_count}) has reached or exceeded expected (${K})"
                break
            else
                echo "Warning: Generated line count (${line_count}) is less than expected (${K})"
                if [ $attempt -lt $max_attempts ]; then
                    echo "Preparing to re-execute..."
                else
                    echo "Reached maximum number of attempts (${max_attempts}), stopping retry"
                fi
            fi
        else
            echo "Warning: JSONL file ${jsonl_file} does not exist"
            if [ $attempt -lt $max_attempts ]; then
                echo "Preparing to re-execute..."
            else
                echo "Reached maximum number of attempts (${max_attempts}), stopping retry"
            fi
        fi
        
        attempt=$((attempt + 1))
    done

    # Run python program to get results
    if [ -f "$jsonl_file" ]; then
        python ${ROOT_DIR}/benchmark/analysis_traj.py --input_file ${OUTPUT_DIR_FOR_THIS_RUN}/traj/${DATA_EXP_NAME}.jsonl --output_file ${OUTPUT_DIR_FOR_THIS_RUN}/${DATASET}_results.json
    else
        echo "Error: JSONL file does not exist, skipping analysis step"
    fi
}

merge_results() {
    python ${ROOT_DIR}/benchmark/merge_results.py --input_dir ${OUTPUT_DIR_FOR_THIS_RUN}
    echo "Results have been written to ${results_json}"
}

# parallel run two tasks driven by two edit format

# Run benchmark based on specified dataset

echo "Benchmarking ${MODEL_NAME} on SWE-bench_Verified... "
huggingface-cli download R2E-Gym/SWE-Bench-Verified --repo-type=dataset --local-dir ./data/SWE-bench_Verified
run_benchmark "SWE-bench_Verified" $(awk "BEGIN {print int(500*${TEST_RATIO})}") "test"

merge_results





END_TIME=$(date +%s)
END_TIME_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_TIME=$((END_TIME - START_TIME))

echo "END_TIME: ${END_TIME_HUMAN} (${END_TIME})"
echo "START_TIME: ${START_TIME_HUMAN} (${START_TIME})"
echo "TOTAL_TIME: ${TOTAL_TIME} seconds"

# merge_results

echo "Benchmarking ${MODEL_NAME} on swe... Done"