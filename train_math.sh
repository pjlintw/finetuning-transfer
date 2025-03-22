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
    "PATH/TO/olmo-2-1124-7b-stage2-ingredient1-step11931-tokens50b"
    # ... other models
)

# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
dataset_name=tulu3_math
dataset_mixer_list="allenai/tulu-3-sft-personas-math 1.0 allenai/tulu-3-sft-personas-math-grade 1.0 allenai/tulu-3-sft-personas-algebra 1.0"
wandb_entity=TRAIN

lr=5e-6
train_step=30000
seq_len=4096
checkpointing_steps=5000
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=1
NUM_PROCESSES=4
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
main_process_port=29450 # avoid conflict with other jobs

output_dirs=()
# for train_step in "${steps[@]}";
for model_name_or_path in "${models[@]}";
do
    train_step_str=$(map_training_to_filename "$train_step")
    model_name=$(basename "$model_name_or_path")

    output_dir=output/${model_name}_ft-${dataset_name}_seq-4k_optim-adamw_lr-${lr}_step-${train_step_str}/
    exp_name=ol2-7b_ft-${dataset_name}_seq-4k_adamw_${lr}_s-${train_step_str} # must 64 char
    log_file=${model_name}_${dataset_name}_lr-${lr}_seq-4k_${train_step_str}.log

    echo "output dir: ${output_dir}"
    echo "log file: ${log_file}"
    
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines $NUM_MACHINES \
        --num_processes $NUM_PROCESSES \
        --machine_rank $MACHINE_RANK \
        --main_process_ip $MAIN_PROCESS_IP \
        --main_process_port $main_process_port \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        --deepspeed_multinode_launcher standard open_instruct/finetune.py \
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





