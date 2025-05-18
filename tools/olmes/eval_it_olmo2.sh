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
    # Instruction-tuned OLMo2
    # "path/to/allenai/OLMo-2-1124-7B-sft"
    # "path/to/allenai/OLMo-2-1124-7B-dpo"
    "path/to/allenai/OLMo-2-1124-7B-instruct"

    # Instruction-tuned Tulu3
    # "path/to/llama-3.1-tulu-3-8b-dpo"
    # "path/to/llama-3.1-tulu-3-8b-sft"
    # "path/to/llama-3.1-tulu-3-8b"   # instruct version
    # "path/to/llama-3.1-tulu-3.1-8b" # instruct version
    
    ### Example of Recycling fine-tuning using Llama
    # Llama 3   + delta_3.1
    # "path/to/vec-IT3.1-IT3_lambda-1_arc-PT3
    
    # Llama 3.1 + delta_3
    # "path/to/vec-IT3-PT3_lambda-1_arc-IT3.1 

    ### Example of Recycling fine-tuning using OLMo 2
    # M1 + IT-vector from IT-OLMo2
    # path/to/vec-ITolmo2-PTolmo2_lambda-1_arc-PTolmo2-stage1_300k

    # M2 + IT-vector from IT-OLMo2
    # path/to/vec-ITolmo2-PTolmo2_lambda-1_arc-PTolmo2-stage1_600k

    # M3 + IT-vector from IT-OLMo2
    # path/to/vec-ITolmo2-PTolmo2_lambda-1_arc-PTolmo2-stage1_928k

    # M4 + IT-vector from IT-OLMo2
    # path/to/vec-ITolmo2-PTolmo2_lambda-1_arc-PTolmo2-stage2_ingredient1_6k

    # M5 + IT-vector from IT-OLMo2
    # path/to/vec-ITolmo2-PTolmo2_lambda-1_arc-PTolmo2-stage2_ingredient1_12k

    ### Example of fine-tuned models 
    # FT-M4
    # path/to/olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b_ft-tulu3_math_seq-2k_optim-adamw_lr-5e-6_step-30k

    # FT-M5
    # path/to/olmo-2-1124-7b-stage2-ingredient1-step11931-tokens50b_ft-tulu3_math_seq-2k_optim-adamw_lr-5e-6_step-30k

    ### Example of model path in controlled experiments
    # 20 merged models

    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage1_300k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_928k_lambda-1_arc-PTolmo2-stage1_300k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage1_300k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_12k_lambda-1_arc-PTolmo2-stage1_300k

    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage1_600k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_928k_lambda-1_arc-PTolmo2-stage1_600k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage1_600k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_12k_lambda-1_arc-PTolmo2-stage1_600k

    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage1_928k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage1_928k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage1_928k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_12k_lambda-1_arc-PTolmo2-stage1_928k

    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_928k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_12k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k

    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage1_928k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"

    ### Example of Recycling-then-finetuning
    # Continually training of M5 + vec( M4-FT-30K - M4 )
    
    # path/to/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k_ft-tulu3_math_seq-2k_optim-adamw_lr-5e-6_step-30k/

)    


NUM_GPUS=2

# Run seven tasks evaluation for each model

for model in "${MODELS[@]}"; do
    eval_dir="${model}/eval_olmes"

    # MMLU (0-shot CoT, Tülu3) — Estimated time: 23 hours
    task="mmlu:0shot_cot::tulu3"
    olmes \
        --model $model \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_mmlu_cot.txt"

    # ARC Challenge (MC, OLMes) — Estimated time: 2 minutes
    task="arc_challenge:mc::olmes"
    olmes \
        --model $model \
        --model-type hf \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": false, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_arc_challenge.txt"

    # GSM8K (Tülu) — Estimated time: 2 minutes
    task="gsm8k::tulu"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_gsm8k.txt"

    # Minerva Math (Tülu) — Estimated time: 21 minutes
    task="minerva_math::tulu"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_math.txt"

    # IFEval + GPQA (0-shot CoT, Tülu3)
    task="ifeval::tulu gpqa:0shot_cot::tulu3"
    olmes \
        --model $model \
        --model-type vllm \
        --gpus $NUM_GPUS \
        --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16", "max_length": 4096}' \
        --task $task \
        --output-dir "${eval_dir}" 2>&1 | tee "${eval_dir}/eval_tulu3_suites_ifeval_gpqa.txt"

done
