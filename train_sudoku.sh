# FIXME: downgrade torch to version compatible with cuda12.6 and NVIDIA capability ver 6.1 (GTX 1080 Ti GPU)
export datasetname=sudoku-extreme-1k-aug-1000
if [[ -f data/$datasetname/identifiers.json ]] ; then
  echo "Found data/$datasetname/*.json skipping data prep."
else ;
  python pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
fi

# https://www.reddit.com/r/StableDiffusion/comments/12nh9hn/dreambooth_training_lora_failed_due_to_triton_gpu/
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1

# https://github.com/MIC-DKFZ/nnUNet/issues/2349
nnUNet_compile=False \
HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR \
OMP_NUM_THREADS=1 \
    torchrun --nproc-per-node 1 \
      pretrain.py \
        data_path=data/sudoku-extreme-1k-aug-1000 \
        epochs=2000 eval_interval=100 \
        lr=1e-4 \
        puzzle_emb_lr=1e-4 \
        weight_decay=1.0 \
        puzzle_emb_weight_decay=1.0
# FIXED
#
# TRACEBACK:
#
# /home/gpu/code/public-by-others/HRM/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:283: UserWarning: 
#     Found GPU0 NVIDIA GeForce GTX 1080 Ti which is of cuda capability 6.1.                                                   
#     Minimum and Maximum cuda capability supported by this version of PyTorch is                                              
#     (7.0) - (12.0)                                                                                                           
#                                                                                                                              
#   warnings.warn(                                                                                                             
# /home/gpu/code/public-by-others/HRM/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:304: UserWarning:              
#     Please install PyTorch with a following CUDA                                                                             
#     configurations:  12.6 following instructions at                                                                          
#    https://pytorch.org/get-started/locally/                                                                                 
#                                                                                                                              
#   warnings.warn(matched_cuda_warn.format(matched_arches))                                                                    
# /home/gpu/code/public-by-others/HRM/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:283: UserWarning: 
#     Found GPU1 NVIDIA GeForce GTX 980 Ti which is of cuda capability 5.2.                                                    
#     Minimum and Maximum cuda capability supported by this version of PyTorch is                                              
#     (7.0) - (12.0)                                                                                                           
#       
#   warnings.warn(
# /home/gpu/code/public-by-others/HRM/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:326: UserWarning: 
# NVIDIA GeForce GTX 1080 Ti with CUDA capability sm_61 is not compatible with the current PyTorch installation.
# The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
# If you want to use the NVIDIA GeForce GTX 1080 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-
# started/locally/

