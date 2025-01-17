### Installation

Python Version: 3.10.0

Follow the instructions below to set up the environment:


```bash
pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install flash-attn==2.6.3 --no-build-isolation
pip install -r requirements.txt
python -m nltk.downloader punkt
pip install -e .
```

I used the exact same version of the FlashAttention library. My PyTorch and CUDA versions are:

* torch==2.4.0+cu121
* torchaudio==2.4.0+cu121
* torchvision==0.19.0+cu121

Transformers Version:

transformers @ git+https://github.com/huggingface/transformers@6c3f168b36882f0beebaa9121eafa1928ba29633


#### Evaluation

To use the OLMes library for evaluation, install it using the following command:

```
pip install git+https://github.com/allenai/olmes.git@38af8b61741b01faeb60e90899addd7e6fa503a0
```

You are welcome to use your own installed OLMes version as long as it is compatible with the OLMo2 architecture.

I modified their VLLM implementation to use batch_size="auto" for faster evaluation compared to using Hugging Face's implementation. Below is the modification I made in OLMes/models/eleuther_vllm_causallms.py:
I hard code their VLLM using `auto` for batch_size, so evalaution will be quite fast compred using hf. 
Here is the modication in OLMes/models/eleuther_vllm_causallms.py


I modified the VLLM implementation in OLMes to set batch_size="auto", enabling faster evaluation compared to the default Hugging Face implementation. The changes were made in OLMes/models/eleuther_vllm_causallms.py. Below is the updated code:

```
class VLLM_Verbose(VLLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM  # For encode_pairs to work

    def __init__(
        self,
        pretrained="gpt2",
        device: Optional[str] = None,
        **kwargs,
    ):
        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "Model type is `vllm` but `vllm` is not installed! Please install vllm via `pip install -e .[gpu]`"
            )
        if device is None:
            device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
        if torch.cuda.device_count() > 1:
            kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        
        kwargs["batch_size"] = "auto"  # Set batch size to auto for faster evaluation
        print("kwargs")

        super().__init__(pretrained, device=device, **kwargs)
```

This adjustment ensures the evaluation process is more efficient, especially when leveraging the VLLM library.



### Save Model Checkpoints
To save the Hugging Face (HF) checkpoint, run the following command:

```
. run_save_models.sh
```

Note: Make sure to update the SAVE_DIR variable in the script to specify the desired save directory.


### Run Training Code

The configuration is set up for fine-tuning using a sequence length of 2K with 4 GPUs (on 1 machine) and a batch size of 4. Fine-tuning is conducted for 30K steps, with checkpoints saved every 5K steps.

### Reminders to Note:

You may need to log in to your wandb_entity or Hugging Face account to run the training code.

I aim to minimize changes to the official script, so the default configuration remains intact.

The `exp_name` parameter is not actually used in the current setup.

The `ft_model_dir` parameter needs to be modified to match your open-instruct directory, as it is used to set up the path for evaluation.

To run the training script, execute the following command:

```
. run_math_lora_1.sh
```

Once the training starts, you can check the log file (log_file) to monitor progress and debug if needed.

### Post-Training Evaluation:

After training, the script will automatically evaluate the model on the following tasks: GSM8K, ARC Challenge, DROP, and Natural Questions.

Note: The evaluation process might require modifications to work seamlessly with LoRA models.