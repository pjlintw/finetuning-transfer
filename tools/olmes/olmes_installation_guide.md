# OLMes Installation Guide

## Overview

This guide covers three key parts:

1. **Installing OLMes** - Instructions to install OLMes from different sources.
2. **Including VLLM with OLMo2 support** - Steps to set up and modify VLLM to work with OLMo2 architecture.
3. **Other dependencies** - Required dependencies and how to install them.

## Dependencies

### Required Versions
- **Python**: `3.10.15`
- **PyTorch**: `torch==2.5.1`
- Other dependencies can be found in `requirements.txt`

Note: The PyTorch version might not be very important.

### Installing Dependencies

You can install all required dependencies using:

```bash
pip install -r requirements.txt
```

However, this is *not recommended*.

## OLMes Installation

There are two options to install OLMes. Please choose either one.

### 1. Install OLMes from `olmes-1208.tar.gz`

To install OLMes using our pre-packaged archive:

1. Extract the contents of `olmes-1208.tar.gz`:

   ```bash
   tar -xvzf olmes-1208.tar.gz
   ```

2. Move it to your working directory:

   ```bash
   mv olmes-1208 /WORKING_DIR
   ```

3. Install the package in editable mode:

   ```bash
   cd olmes-1208
   pip install -e .
   ```

After installation, you can still use the `olmes` command without needing to reference `olmes-1208`.

### 2. Install from the Official Repository

As an alternative, you can also install OLMes directly from the official repository:

#### Clone and install manually:
```bash
git clone https://github.com/allenai/olmes.git
cd olmes
git checkout 38af8b61741b01faeb60e90899addd7e6fa503a0
pip install -e .
```


## VLLM Dependency

VLLM provides fast inference support for OLMes. To run OLMes with the required VLLM modifications, you can either:

**Option 1**: Copy our modified `vllm` (`vllm.tar.gz`) package to your OLMes environment (`olmes-1208`).

**Option 2**: Install VLLM from the official source and manually apply modifications.

#### Install VLLM from our tar.gz

Unzip the `vllm` package and copy it to your conda environment's site-packages directory.

For example, if your environment path is:

```bash
cp -r vllm /PATH/anaconda3/envs/olmes-1208/lib/python3.10/site-packages
```

This should work if most dependencies are similar to those listed in `requirements.txt`.

#### Install VLLM from source

Alternatively, you can install a new version. This is the version we used. *Manual architecture modifications are required.*

```bash
pip install vllm@https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl#sha256=fefcb7539062451eb32e5f5c16b96bae8ddb33acd5e4157fdd4494123b1c4f05
```


## Adding OLMo2 to VLLM

Since `vllm` does not natively support the OLMo2 architecture, we have manually modified the source code. If you are using our `vllm` package, these modifications are already included. If working with an unmodified version, follow these steps:

### Step 1: Define the architecture
Add the following file:

```plaintext
vllm/model_executor/models/olmo2.py
```

### Step 2: Add the OLMo2 config
Create the configuration file in:

```plaintext
vllm/transformers_utils/configs/olmo2.py
```

Then update the config mapping in:

```plaintext
vllm/transformers_utils/config.py
```

Add the line: `"olmo2": Olmo2Config,` (line 66).

### Step 3: Register the model architecture
In the following file:

```plaintext
vllm/model_executor/models/registry.py
```

Add the line:

```python
"Olmo2ForCausalLM": ("olmo2", "Olmo2ForCausalLM")  # Add at line 79
```

### Step 4: Include the configuration in init
Ensure the `Olmo2Config` is added in:

```plaintext
vllm/transformers_utils/configs/__init__.py
```

Add the necessary line at line 38.

## Environment Considerations

Please consider using an independent evaluation environment. It is likely that the dependencies for OLMes won't work for the open-instruct packages.

If you have any issues or questions, feel free to reach out for support.

