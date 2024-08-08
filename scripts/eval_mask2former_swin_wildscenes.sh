export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=29501 \
    scripts/benchmark/eval2d.py \
    "wildscenes/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_2xb20-80k_wildscenes_standard-512x512.py" \
    "pretrained_models/mask2former_swin_wildscenes.pth" \
    --launcher pytorch