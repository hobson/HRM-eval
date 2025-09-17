export TORCH_DEVICE='cpu'
echo "Replicating ARC-AGI v1 public eval on ${TORCH_DEVICE}..."

INPUT_PREFIX=kaggle/input/arc-agi
OUTPUT_DIR=data/arc-aug-1000
echo "Prepping ARC-1-public dataset INPUT_PREFIX=${INPUT_PREFIX}\*, output=${OUTPUT_DIR}"
python -m dataset.build_arc_dataset \
  --input-file-prefix $INPUT_PREFIX \
  --output-dir $OUTPUT_DIR \
  --subsets concept training evaluation \
  --test-set-name evaluation

export OMP_NUM_THREADS=8
echo "Training on ARC-1-public dataset OMP_NUM_THREADS=${OMP_NUM_THREADS}"
OMP_NUM_THREADS=$OMP_NUM_THREADS \
    torchrun \
    --nproc-per-node $OMP_NUM_THREADS \
        pretrain.py

# torchrun \
#   --nnodes $NNODES \
#   --node_rank $NODE_RANK \
#   --nproc_per_node $GPUS_PER_NODE \
#   --rdzv_backend c10d \
#  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#  pretrain.py 