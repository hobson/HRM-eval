#!/usr/bin/bash

# Hierarchical Reasoning Model Analysis

# set -e

# # Step 1: set up a venv.
# # sudo snap install astral-uv --classic
# rm -rf .venv
# uv venv .venv -p 3.12
# source .venv/bin/activate

# Step 2: install dependencies for flash attention
sudo apt install python3-dev -y

# Step 3: install pytorch
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
uv pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Step 4: install dependencies for adam-atan2
uv pip install packaging ninja wheel setuptools setuptools-scm

# Step 5: install adam-atan2
uv pip install --no-cache-dir --no-build-isolation adam-atan2[torch] 

# Step 5.1: test if torch, cuda and adam atan2 work
python -c "import torch; t = torch.tensor([0,1,2]).to('cuda'); from adam_atan2 import AdamATan2; print(AdamATan2)"

# # Step 6: install flash attention for Hopper GPUs  
# git clone git@github.com:Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# python setup.py install
# cd ../../

# # Step 6: install flash attention for Ampere or older GPUs, install FlashAttention 2
uv pip install flash-attn --no-build-isolation
uv pip install numba

# Step 7: install the remaining dependencies
uv pip install -r requirements.txt
# export WANDB_API_KEY=''
source .env

 
