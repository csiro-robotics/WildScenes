# Getting Started with WildScenes

## Prerequisites

WildScenes can be run in any Python environment, and we recommend using a package manager. In this guide we use mamba:
https://mamba.readthedocs.io/en/latest/

Mamba is a replacement for conda and behaves the same except is faster especially in larger and more complex environments. 

These installation instructions are written for CUDA version 12.1.

```shell
mamba create --name wildscenes3 python=3.10
mamba activate wildscenes
```

## Installation of Dependencies

Step 1: Install CUDA

```shell
mamba install cuda -c nvidia
```

Step 1: Install Pytorch

Using CUDA 12.1:

```shell
mamba install pytorch torchvision pytorch-cuda -c pytorch -c nvidia -c
```

On CPU only platforms:

```shell
mamba install pytorch torchvision cpuonly -c pytorch
```

Step 2: Wildscenes utilizes mmsegmentation for training 2D networks on our dataset. To install mmsegmentation, the instructions are as follows.

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4"
pip install "mmsegmentation>=1.0.0"
```

Step 3: Wildscenes utilizes mmdetection3d 

```shell
mim install "mmdet>=3.0.0"
pip install "mmdet3d>=1.1.0"
```

Step 4: Install Torchsparse

```shell
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

Step 5: Install other required pip packages. Many of these should already have been installed by default during earlier steps.

```shell
pip install opencv-python
pip install pynput
pip install tqdm
pip install pillow
pip install tensorboard
pip install matplotlib
pip install open3d==0.1.8
```

Step 6: Install other required mamba packages (or install using pip)

```shell
mamba install numpy -c conda-forge
mamba install pandas -c conda-forge
mamba install scikit-learn -c conda-forge
mamba install quaternion -c conda-forge
```
