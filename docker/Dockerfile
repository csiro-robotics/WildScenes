ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# nvidia-container-runtime config
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV TZ=Australia/Brisbane
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y git

RUN pip install opencv-python==4.11.0.86
# RUN pip install typing_extensions==4.13.2
# RUN pip install ftfy==6.3.1
# RUN pip install regex==2024.11.6

# Install torchsparse 
RUN apt-get install -y libsparsehash-dev
WORKDIR /home 
RUN git clone https://github.com/mit-han-lab/torchsparse.git
WORKDIR torchsparse 
RUN git checkout 74099d1
RUN python3 setup.py install 

# Install openmm dependencies
RUN pip install -U openmim==0.3.9
RUN mim install 'mmengine==0.10.7'

# # build mmcv from source 
ENV FORCE_CUDA="1"
ARG CUDA_ARCH="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing" 
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
RUN apt-get install -y ffmpeg libsm6 libxext6  
WORKDIR /home
RUN git clone https://github.com/open-mmlab/mmcv.git
WORKDIR /home/mmcv
RUN git checkout ea53ed0
RUN python setup.py install
RUN python .dev_scripts/check_installation.py

RUN mim install 'mmdet==3.0.0'
# RUN mim install 'mmsegmentation==1.2.2'
# RUN mim install 'mmdet3d==1.4.0'

# Install mmdetection3d 
# During install, copy config and dataset files for wildscenes3d into the repo
WORKDIR /home 
RUN git clone --branch v1.4.0 https://github.com/open-mmlab/mmdetection3d.git 
WORKDIR /home/mmdetection3d


RUN pip install -r requirements.txt

COPY mmdet3d_copy_files/_base_ configs/_base_
COPY mmdet3d_copy_files/cylinder3d configs/cylinder3d 
COPY mmdet3d_copy_files/spvcnn configs/spvcnn 
COPY mmdet3d_copy_files/wildscenes3d.py mmdet3d/datasets
COPY mmdet3d_copy_files/__init__.py mmdet3d/datasets 
# COPY mmdet3d_copy_files/registry.py mmdet3d/registry.py 

RUN pip install .

# Clean up 
WORKDIR /home 
RUN rm -rv torchsparse mmcv 