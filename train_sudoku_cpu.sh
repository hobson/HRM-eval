# FIXME: downgrade torch to version compatible with cuda12.6 and NVIDIA capability ver 6.1 (GTX 1080 Ti GPU)
export datasetname=sudoku-extreme-1k-aug-1000
if [[ -f data/$datasetname/identifiers.json ]] ; then
  echo "Found data/$datasetname/*.json skipping data prep."
else
  python dataset/build_sudoku_dataset.py \
  --output-dir data/$datasetname \
  --subsample-size 1000 \
  --num-aug 1000
fi

# https://www.reddit.com/r/StableDiffusion/comments/12nh9hn/dreambooth_training_lora_failed_due_to_triton_gpu/
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1
export TORCH_DEVICE="cpu"
if [[ $TORCH_DEVICE == "cpu" ]] ; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUDA_VISIBLE_DEVICES="1"  
fi

# # https://github.com/MIC-DKFZ/nnUNet/issues/2349
# nnUNet_compile=False \
# HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR \
# OMP_NUM_THREADS=1 \
#     torchrun --nproc-per-node 1 \
#       pretrain.py \
#         data_path=data/sudoku-extreme-1k-aug-1000 \
#         epochs=2000 eval_interval=100 \
#         lr=1e-4 \
#         puzzle_emb_lr=1e-4 \
#         weight_decay=1.0 \
#         puzzle_emb_weight_decay=1.0

# EVAL_INTERVAL=200 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  python pretrain.py \
    data_path=data/$datasetname \
    epochs=2000 \
    eval_interval=200 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0

