export CUDA_VISIBLE_DEVICES=5
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=29501 \
    scripts/benchmark/eval3d.py \
    "wildscenes/configs3d/cylinder3d/cylinder3d_4xb4-3x_wildscenes.py" \
    "pretrained_models/cylinder3d_wildscenes.pth" \
    --show-dir \
    "~/YOURCUSTOMPATH" \
    --task \
    "lidar_seg" \
    --launcher pytorch
