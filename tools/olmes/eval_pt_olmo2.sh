#!/bin/bash


# Function to safely create a directory if it does not exist
safe_mkdir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create directory $dir" >&2
            exit 1
        fi
    fi
}

MODELS=(
    # Llama 3
    # "path/to/meta-llama-3-8b"

    # Llama 3.1
    # "path/to/llama-3.1-8b"
    
    # Official OLMo 2
    # "path/to/allenai/OLMo-2-1124-7B"

    # M1
    # "path/to/olmo-2-1124-7b-stage1-step300000-tokens1259b"
    
    # M2
    # "path/to/olmo-2-1124-7b-stage1-step600000-tokens2517b"
    
    # M3
    # "path/to/olmo-2-1124-7b-stage1-step928646-tokens3896b"
    
    # M4
    # "path/to/olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b"
    
    # M5
    # "path/to/olmo-2-1124-7b-stage2-ingredient1-step11931-tokens50b"

)

NUM_GPUS=2

# Run seven tasks evaluation for each model
for model in "${MODELS[@]}"; do
    eval_dir="${model}/eval_olmes"

    # MMLU
    task="mmlu:mc::olmes"
    olmes \
        --model $model \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
        --model-type hf \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_mmlu.txt"

    # ARC Challenge (MC, OLMes)
    task="arc_challenge:mc::olmes"
    olmes \
        --model $model \
        --model-type hf \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": false, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_arc_challenge.txt"

    # GSM8K
    task="gsm8k::olmes"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_gsm8k.txt"

    # Minerva Math
    task="minerva_math::olmes"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_math.txt"

    # IFEval
    task="ifeval"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16", "max_length": 4096}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_ifeval.txt"

    # GPQA
    task="gpqa"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16", "max_length": 4096}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_gpqa.txt"
done



