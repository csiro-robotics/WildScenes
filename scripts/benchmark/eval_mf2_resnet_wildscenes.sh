export CUDA_VISIBLE_DEVICES=5
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=1 \
    --master_port=29500 \
    scripts/benchmark/eval2d.py \
    "wildscenes/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_2xb20-80k_wildscenes_standard-512x512.py" \
    "/raid/work/hau047/trained_models_wildscenes/2dmodels_oldvalset/mask2former_swin-l-in22k-384x384-pre_2xb20-80k_wildscenes_standard-512x512_dgx/best_mIoU_iter_24000_primarysplit.pth" \
    --show-dir \
    "/raid/work/hau047/wildscenes/Dev_IJRR_rebuttal/visualizations/mask2former_swin-l-in22k-384x384-pre_2xb20-80k_wildscenes_standard-512x512_dgx/images" \
    --launcher pytorch