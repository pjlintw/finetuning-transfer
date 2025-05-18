#!/bin/bash

# task=meta_instruct
MODELS=(
    # Instruction-tuned models
    "path/to/llama-3.1-8b-instruct"
    "path/to/meta-llama-3-8b-instruct"

    # Merged models
    "path/to/vec-IT3.1-PT3.1_lambda-1_arc-PT3" # Llama 3   + delta_3.1
    "path/to/vec-IT3-PT3_lambda-1_arc-PT3.1"   # Llama 3.1 + delta_3
)

include_path="path/to/llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval/it_eval_dir"
for MODEL_PATH in "${MODELS[@]}"; 
do
    eval_dir=${MODEL_PATH}/eval_llama-3.1-it

    # gsm8k
    task=meta_gsm8k_8shot_cot
    lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=4096,seed=42 \
    --tasks $task \
    --batch_size auto \
    --output_path $eval_dir \
    --include_path $include_path \
    --seed 42 \
    --fewshot_as_multiturn \
    --apply_chat_template --log_samples 2>&1 | tee "${eval_dir}/eval_output_chat_gsm8k_llama-it-eval_full.txt"
    
    # math
    task=meta_math
    lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=8192,seed=42 \
    --tasks $task \
    --batch_size auto \
    --output_path $eval_dir \
    --include_path $include_path \
    --seed 42 \
    --fewshot_as_multiturn \
    --apply_chat_template 2>&1 | tee "${eval_dir}/eval_output_chat_math_llama-it-eval_full.txt"

    # GPQA
    task=meta_gpqa_cot
    lm_eval --model vllm \
   --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=4096,seed=42 \
   --tasks $task \
   --batch_size auto \
   --output_path $eval_dir \
   --include_path $include_path \
   --seed 42 \
   --fewshot_as_multiturn \
   --apply_chat_template 2>&1 | tee "${eval_dir}/eval_output_chat_gpqa_llama-it-eval_full.txt"

    # mmlu
    task=meta_mmlu_instruct
    lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,seed=42 \
    --cache_requests true \
    --tasks $task \
    --batch_size auto \
    --output_path $eval_dir \
    --seed 42 \
    --fewshot_as_multiturn \
    --apply_chat_template 2>&1 | tee "${eval_dir}/eval_output_chat_mmlu_llama-it-eval_full_3.txt"
    --include_path $include_path \
    
    # ifeval
    task=meta_ifeval
    lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=8192,seed=42 \
    --tasks $task \
    --batch_size auto \
    --output_path $eval_dir \
    --include_path $include_path \
    --seed 42 \
    --fewshot_as_multiturn \
    --apply_chat_template 2>&1 | tee "${eval_dir}/eval_output_chat_ifeval_llama-it-eval_full.txt"

    # arc challenge
    task=meta_arc_challenge_instruct
    lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,seed=42 \
    --tasks $task \
    --batch_size auto \
    --output_path $eval_dir \
    --include_path $include_path \
    --seed 42 \
    --fewshot_as_multiturn \
    --apply_chat_template 2>&1 | tee "${eval_dir}/eval_output_chat_arc_challenge_llama-it-eval_full.txt"
done

