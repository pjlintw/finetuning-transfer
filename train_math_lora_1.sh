#!/bin/bash

# Function to map training argument to filename
map_training_to_filename() {
    local training_arg=$1
    if [ "$training_arg" -ge 1000 ]; then
        echo "$((training_arg / 1000))k"
    else
        echo "$training_arg"
    fi
}

models=(
    ### OLMo2 PT checkpoit ###
    # M1
    # "/projects/llms-lab/merged-models/hf_models/olmo-2-1124-7b-stage1-step300000-tokens1259b"
    
    # M5
    # "/projects/llms-lab/merged-models/hf_models/olmo-2-1124-7b-stage2-ingredient1-step11931-tokens50b"
    
    # M3 
    # "/projects/llms-lab/merged-models/hf_models/olmo-2-1124-7b-stage1-step928646-tokens3896b"
    
    # M2 
    # "/projects/llms-lab/merged-models/hf_models/olmo-2-1124-7b-stage1-step600000-tokens2517b"
    
    # M4 
    # "/projects/llms-lab/merged-models/hf_models/olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b"

    ### Sec2. Merged model & Sec3. continually training ###
    ###  M_j + vec( M_j-1, M_j-1) ###
    # M2 + vec( M1-FT-30K - PT-M1 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage1_600k"
    
    # M3 + vec( M2-FT-30K - PT-M2 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage1_928k"
    
    # M4 + vec( M3-FT-30K - PT-M3 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_928k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k"
    
    # M5 + vec( M4-FT-30K - PT-M4 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"

    ###  M_j + vec( M_j-2, M_j-2) ###
    # M3 + vec( M1-FT-30K - PT-M1 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage1_928k"

    # M4 + vec( M2-FT-30K - M2 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k"

    # M5 + vec( M3-FT-30K - M3 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_928k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"

    ### merged model M_j + vec( M_j-3, M_j-3) ###
    # M4 + vec( M1-FT-30K - PT-M1 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage2_ingredient1_6k"
        
    # M5 + vec( M2-FT-30K - PT-M2 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_600k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"
        
    ### merged model M_j + vec( M_j-4, M_j-4) ###
    # M5 + vec( M1-FT-30K - PT-M1 )
    # "/projects/llms-lab/merged-models/hf_models/vec-FTolmo2_math_30k-PTolmo2-stage1_300k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"

    
)


# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
dataset_name=tulu3_math
dataset_mixer_list="allenai/tulu-3-sft-personas-math 1.0 allenai/tulu-3-sft-personas-math-grade 1.0 allenai/tulu-3-sft-personas-algebra 1.0"
wandb_entity=linus

# 5e-06 2e-5 1e-5
lr=5e-6
train_step=30000
seq_len=2048
checkpointing_steps=5000
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=1
NUM_PROCESSES=4
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
main_process_port=29450 # avoide using duplicate port number
ft_model_dir=/projects/llms-lab/open-instruct

output_dirs=()
# for train_step in "${steps[@]}";
for model_name_or_path in "${models[@]}";
do
    train_step_str=$(map_training_to_filename "$train_step")
    model_name=$(basename "$model_name_or_path")

    output_dir=output/${model_name}_ft-${dataset_name}_seq-2k_optim-adamw_lr-${lr}_step-${train_step_str}/
    exp_name=ol2-7b_ft-${dataset_name}_seq-2k_adamw_${lr}_s-${train_step_str} # must 64 char
    log_file=${model_name}_${dataset_name}_lr-${lr}_seq-2k_${train_step_str}.log

    echo "output dir: ${output_dir}"
    echo "log file: ${log_file}"
    
    output_dirs+=("$output_dir")
    output_dirs+=("$output_dir/step_5000")
    output_dirs+=("$output_dir/step_10000")
    output_dirs+=("$output_dir/step_15000")
    output_dirs+=("$output_dir/step_20000")
    output_dirs+=("$output_dir/step_25000")

    # SFT - 2e-1, 5e-6
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines $NUM_MACHINES \
        --num_processes $NUM_PROCESSES \
        --machine_rank $MACHINE_RANK \
        --main_process_ip $MAIN_PROCESS_IP \
        --main_process_port $main_process_port \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        --deepspeed_multinode_launcher standard open_instruct/finetune_lora.py \
        --model_name_or_path $model_name_or_path \
        --tokenizer_name $model_name_or_path \
        --use_slow_tokenizer false \
        --add_bos true \
        --use_flash_attn \
        --max_seq_length $seq_len \
        --max_train_steps $train_step \
        --preprocessing_num_workers 128 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $lr \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --output_dir $output_dir \
        --with_tracking \
        --report_to wandb \
        --logging_steps 1 \
        --reduce_loss sum \
        --model_revision main \
        --dataset_mixer_list $dataset_mixer_list \
        --checkpointing_steps $checkpointing_steps \
        --dataset_mix_dir $output_dir \
        --exp_name $exp_name \
        --keep_last_n_checkpoints=10000000000000000 \
        --wandb_entity $wandb_entity \
        --seed 123 2>&1 | tee $log_file

    mv $log_file $output_dir
done
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JBNTPW8TKG09B2XR832YB5S8  

# activate evaluation environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate olmes-1208


# Evaluation using olmes
# all tulu3's tasks except for MMLU
for model in "${output_dirs[@]}";
do
    model="${ft_model_dir}/${model}"
    eval_dir="${model}/eval_olmes"
    
    ### Estimated time: 2 minutes ###
    task="arc_challenge:mc::olmes"
    olmes \
    --model $model \
    --model-type hf \
    --gpus $NUM_GPUS \
    --model-args='{"trust_remote_code": false, "add_bos_token": true, "dtype": "bfloat16"}' \
    --task $task \
    --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_arc_challenge.txt


    ### Estimated time: 2 minutes ###
    task="gsm8k::tulu"
    olmes \
    --model $model \
    --model-type vllm \
    --gpus $NUM_GPUS \
    --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
    --task $task \
    --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_gsm8k_math.txt


    # # # ### Estimated time: 21 minutes ###
    # task="minerva_math::tulu"
    # olmes \
    # --model $model \
    # --model-type vllm \
    # --gpus $NUM_GPUS \
    # --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
    # --task $task \
    # --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_math.txt


    # # ### Estimated time: 2 mins / 2 mins ###
    # task="ifeval::tulu gpqa:0shot_cot::tulu3"
    # olmes \
    # --model $model \
    # --model-type vllm \
    # --gpus $NUM_GPUS \
    # --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16", "max_length": 4096}' \
    # --task $task \
    # --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_ifeval_gpqa.txt
    

    # ### Estimated time: 12 minutes ###
    task="drop::llama3"
    olmes \
    --model $model \
    --model-type vllm \
    --gpus $NUM_GPUS \
    --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
    --task $task \
    --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_drop.txt


    # ### Estimated time: 1 minutes ###
    task="naturalqs::olmes"
    olmes \
    --model $model \
    --model-type vllm \
    --gpus 2 \
    --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
    --task $task \
    --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_naturalqs.txt
done


# # MMLU
# unset PYTORCH_CUDA_ALLOC_CONF
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# for model in "${output_dirs[@]}"; 
# do
#     eval_dir="${model}/eval_olmes"
#     ### Estimated time: 23 hours ###
#     ### debug: mmlu
#     task=mmlu:0shot_cot::tulu3
#     olmes \
#     --model $model \
#     --gpus 4 \
#     --model-args='{"trust_remote_code": true, "add_bos_token": true, "dtype": "bfloat16"}' \
#     --task $task \
#     --output-dir "${eval_dir}" 2>&1 | tee ${eval_dir}/eval_tulu3_suites_mmlu_cot.txt
#     # mmlu:mc::tulu
# done


