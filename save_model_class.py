import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_save_model(model_name, tokenizer_name, output_dir, torch_dtype=None, device_map=None, attn_implementation=None, revision=None):
    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        revision=revision
    )

    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensuring pad_token is set
    print("Using EOS token as pad token to avoid infinite generation issues.")
    
    # Save the model and tokenizer to the specified directory
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer downloaded to: {output_dir}")


if __name__ == "__main__":
    # Set up argparse parameters
    parser = argparse.ArgumentParser(description="Download and save Hugging Face model and tokenizer.")
    
    # Model name and path
    parser.add_argument("--model_name_and_path", type=str, default="meta/llama-3.1-7b",
                        help="Model name or path to download from Hugging Face Hub.")
    
    # Tokenizer name
    parser.add_argument("--tokenizer", type=str, default="meta/llama-3.1-7b",
                        help="Tokenizer name or path to download from Hugging Face Hub.")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="./hf_models",
                        help="Directory where the model and tokenizer will be saved.")

    # Optional settings to match training script
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="Torch dtype for model loading.")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation.")
    parser.add_argument("--revision", type=str, default=None, help="Model revision to download.")

    # Parse arguments
    args = parser.parse_args()

    # Convert dtype string to actual torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    # Download and save the model and tokenizer
    download_and_save_model(
        args.model_name_and_path,
        args.tokenizer,
        args.output_dir,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        revision=args.revision
    )
