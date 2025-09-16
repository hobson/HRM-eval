EVAL_INTERVAL=200 \
CUDA_VISIBLE_DEVICES="" \
  python pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=2000 \
    eval_interval=200 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0

