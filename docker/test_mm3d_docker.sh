_WILDSCENES_DATASET_LOCATION=<Path/to/WildScenes>
_MMDETECTION3D_LOCATION=<Path/to/mmdetection3d>
_CHECKPOINT_LOCATION=<Path/to/downloaded/cylinder3d/checkpoint>

docker container run \
    -v ${_WILDSCENES_DATASET_LOCATION}:/mnt/WildScenes \
    -v ${_MMDETECTION3D_LOCATION}:/mnt/mmdetection3d \
    wildscenes:latest \
    python tools/test.py \
        /mnt/mmdetection3d/configs/cylinder3d/cylinder3d_2xb10-3x_wildscenes.py \
        ${_CHECKPOINT_LOCATION} \

