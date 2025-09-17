#!/usr/bin/bash

# Hierarchical Reasoning Model Analysis

# set -e

# # Step 1: set up a venv.
# sudo snap install astral-uv --classic
<<<<<<< HEAD
rm -rf .venv
uv venv .venv -p 3.12
source .venv/bin/activate
=======
# uv venv .venv -p 3.12
# source .venv/bin/activate
>>>>>>> 1eee4e937b21e46bad36695b4ca7b64d604eb5d0

# Step 2: install dependencies for flash attention
sudo apt install python3-dev -y

# Step 3: install pytorch
<<<<<<< HEAD
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
uv pip install torch torchvision torchaudio   # --index-url $PYTORCH_INDEX_URL
=======
# PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# uv pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
>>>>>>> 1eee4e937b21e46bad36695b4ca7b64d604eb5d0

# Step 4: install dependencies for adam-atan2
uv pip install packaging ninja wheel setuptools setuptools-scm

# Step 5: install adam-atan2
<<<<<<< HEAD
uv pip install --no-cache-dir --no-build-isolation adam-atan2 
=======
uv pip install --no-cache-dir --no-build-isolation adam-atan2[torch] 
>>>>>>> 1eee4e937b21e46bad36695b4ca7b64d604eb5d0

# Step 5.1: test if torch, cuda and adam atan2 work
python -c "import torch; t = torch.tensor([0,1,2]).to('cuda'); from adam_atan2 import AdamATan2; print(AdamATan2)"

# # Step 6: install flash attention for Hopper GPUs  
# git clone git@github.com:Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# python setup.py install
# cd ../../

# # Step 6: install flash attention for Ampere or older GPUs, install FlashAttention 2
<<<<<<< HEAD
uv pip install flash-attn

# Step 7: install the remaining dependencies
uv pip install -r requirements.txt
export WANDB_API_KEY=''
=======
uv pip install flash-attn --no-build-isolation

uv pip install numba

# Step 7: install the remaining dependencies
uv pip install -r requirements.txt
# export WANDB_API_KEY=''
source .env
>>>>>>> 1eee4e937b21e46bad36695b4ca7b64d604eb5d0

# # Dataset Preparation
# python -m dataset.build_arc_dataset \
#   --input-file-prefix kaggle/input/arc-agi \
#   --output-dir data/arc-aug-1000 \
#   --subsets concept training evaluation \
#   --test-set-name evaluation
#

# # Training
# # Replication of ARC-AGI v1 Public Eval
# OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py
# torchrun \
#   --nnodes $NNODES \
#   --node_rank $NODE_RANK \
#   --nproc_per_node $GPUS_PER_NODE \
#   --rdzv_backend c10d \
#  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#  pretrain.py  
