export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/benchmark/train2d.py \
    "wildscenes/configs/convnext/convnext-large_upernet_2xb20-amp-80k_wildscenes-512x512_standard.py" \
    --launcher pytorch
