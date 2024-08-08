export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/benchmark/train2d.py \
    "wildscenes/configs/mask2former/mask2former_r50_2xb20-80k_wildscenes_standard-512x512.py" \
    --launcher pytorch