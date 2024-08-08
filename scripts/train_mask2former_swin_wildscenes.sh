export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/benchmark/train2d.py \
    "wildscenes/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_2xb20-80k_wildscenes_standard-512x512.py" \
    --launcher pytorch