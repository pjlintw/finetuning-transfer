"""
Compute and apply diff vector between `ft_name` and `base_name`.
We add this diff vector to `early_base` and save the result to `output_dir_name`.

Each list is formatted as:
[ft_name, base_name, early_base, output_dir_name]

Example:
["olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b_ft-tulu3_math_seq-2k_optim-adamw_lr-5e-6_step-30k",
 "olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b",
 "olmo-2-1124-7b-stage1-step11931-tokens50b",
 "vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"]
"""

import gc
from tqdm import tqdm
import torch
from torch.nn import Parameter
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm_from_pretrained(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    return model


def update_embed_tokens_and_lm_head(base_model, fine_tuned_model):
    device = base_model.model.embed_tokens.weight.device
    fine_tuned_model.to(device)

    # Update embed_tokens
    base_embed_weight = base_model.model.embed_tokens.weight
    fine_tuned_embed_weight = fine_tuned_model.model.embed_tokens.weight
    row_diff = fine_tuned_embed_weight.shape[0] - base_embed_weight.shape[0]
    if row_diff > 0:
        zero_padding = torch.zeros(row_diff, base_embed_weight.size(1), dtype=base_embed_weight.dtype, device=base_embed_weight.device)
        base_model.model.embed_tokens.weight = Parameter(torch.cat((base_embed_weight, zero_padding), dim=0))
    else:
        base_model.model.embed_tokens.weight = Parameter(base_embed_weight)

    # Update lm_head
    base_lm_head_weight = base_model.lm_head.weight
    fine_tuned_lm_head_weight = fine_tuned_model.lm_head.weight
    row_diff = fine_tuned_lm_head_weight.shape[0] - base_lm_head_weight.shape[0]
    if row_diff > 0:
        zero_padding = torch.zeros(row_diff, base_lm_head_weight.size(1), dtype=base_lm_head_weight.dtype, device=base_lm_head_weight.device)
        base_model.lm_head.weight = Parameter(torch.cat((base_lm_head_weight, zero_padding), dim=0))
    else:
        base_model.lm_head.weight = Parameter(base_lm_head_weight)

    return base_model


def main():
    hf_model_dir = "path/to/merged-models/hf_models"
    ft_dir = "path/to/merged-models/hf_models"

    lst = [
        # Llama 3  + delta 3.1
        ["llama-3.1-8b-instruct",
        "llama-3.1-8b",
        "meta-llama-3-8b",
        "vec-IT3.1-PT3.1_lambda-1_arc-PT3"],
        

        # Llama 3.1 + delta 3
        ["meta-llama-3-8b-instruct",
        "meta-llama-3-8b",
        "llama-3.1-8b",
        "vec-IT3-PT3_lambda-1_arc-PT3.1"],
        

        # Llama 3 + delta tulu 3   (delta RLVR)
        ["llama-3.1-tulu-3-8b",
        "llama-3.1-8b",
        "meta-llama-3-8b",
        "vec-ITtulu3-PT3.1_lambda-1_arc-PT3"],
        
    
        # Llama 3 + delta tulu 3.1 (delta GRPO)
        ["llama-3.1-tulu-3.1-8b",
        "llama-3.1-8b",
        "meta-llama-3-8b",
        "vec-ITtulu3.1-PT3.1_lambda-1_arc-PT3"],

        # OLMo 2 7B + delta OLMo 2 Instruct (delta RLVR)
        # We use OLMo 2 7B as the base model to avoid creating an identical model.
        ["olmo-2-1124-7b-instruct",
        "olmo-2-1124-7b",
        "olmo-2-1124-7b-stage2-ingredient1-step7000-tokens30b",
        "vec-SFTolmo2-PTolmo2_lambda-1_arc-PTolmo2-stage2_ingredient1_7k"],

        # M5 + vec-M4
        ["olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b_ft-tulu3_math_seq-2k_optim-adamw_lr-5e-6_step-30k",
         "olmo-2-1124-7b-stage2-ingredient1-step6000-tokens26b",
         "olmo-2-1124-7b-stage1-step11931-tokens50b",
         "vec-FTolmo2_math_30k-PTolmo2-stage2_ingredient1_6k_lambda-1_arc-PTolmo2-stage2_ingredient1_12k"]
    ]

    for ft_name, base_name, early_base, output_dir_name in lst:
        fine_tuned_path = f"{ft_dir}/{ft_name}"
        base_model_path = f"{hf_model_dir}/{base_name}"
        early_base_model_path = f"{ft_dir}/{early_base}" if "reccurent_vec-" in early_base else f"{hf_model_dir}/{early_base}"

        fine_tuned_model = load_causal_lm_from_pretrained(fine_tuned_path)
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
        base_model = load_causal_lm_from_pretrained(base_model_path)
        early_base_model = load_causal_lm_from_pretrained(early_base_model_path)

        base_model = update_embed_tokens_and_lm_head(base_model, fine_tuned_model)
        early_base_model = update_embed_tokens_and_lm_head(early_base_model, fine_tuned_model)

        weights_fine_tuned = fine_tuned_model.state_dict()
        weights_base = base_model.state_dict()
        weights_early_base = early_base_model.state_dict()

        assert weights_fine_tuned.keys() == weights_base.keys()
        assert weights_early_base.keys() == weights_base.keys()

        lambda_values = [1]

        for lambda_v in lambda_values:
            print(f"Applying with lambda = {lambda_v}")
            for key in tqdm(weights_fine_tuned.keys(), desc="Applying model vector"):
                base_weight = weights_base[key].double()
                advanced_weight = weights_fine_tuned[key].double()
                model_vector = advanced_weight - base_weight
                if lambda_v != 1:
                    model_vector *= lambda_v
                weights_early_base[key] += model_vector

            fine_tuned_model.load_state_dict(weights_early_base)

            save_path = f"{hf_model_dir}/{output_dir_name}".replace("lambda-1", f"lambda-{lambda_v}")
            fine_tuned_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved to {save_path}")

        del fine_tuned_model, base_model, early_base_model
        gc.collect()


if __name__ == "__main__":
    main()

