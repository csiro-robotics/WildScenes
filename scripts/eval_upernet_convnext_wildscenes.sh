export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=29501 \
    scripts/benchmark/eval2d.py \
    "wildscenes/configs/convnext/convnext-large_upernet_2xb20-amp-80k_wildscenes-512x512_standard.py" \
    "pretrained_models/upernet_convnext_wildscenes.pth" \
    --launcher pytorch