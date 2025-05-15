# FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
FROM nvcr.io/nvidia/pytorch:22.10-py3

# nvidia-container-runtime config
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV TZ=Australia/Brisbane
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

# Install torchsparse 
RUN apt-get install -y libsparsehash-dev
WORKDIR /home 
RUN git clone https://github.com/mit-han-lab/torchsparse.git
WORKDIR torchsparse 
RUN git checkout 74099d1
RUN pip install ninja 
ENV MAX_JOBS=8
RUN python3 setup.py install 

# Install mm3d dependencies
RUN pip install -U openmim
RUN mim install mmengine

WORKDIR /home
RUN git clone https://github.com/open-mmlab/mmcv.git
WORKDIR /home/mmcv
RUN git checkout ea53ed0
RUN pip install opencv-python==4.5.5.64
RUN python setup.py install 
RUN apt-get install -y ffmpeg libsm6 libxext6  
RUN python .dev_scripts/check_installation.py

RUN mim install 'mmdet>=3.0.0'
RUN pip install typing_extensions==4.1

# Install mm3d
WORKDIR /home 
RUN git clone https://github.com/open-mmlab/mmdetection3d.git
WORKDIR mmdetection3d
RUN pip install -v -e .

# Install mmsegmentation
WORKDIR /home 
RUN git clone https://github.com/open-mmlab/mmsegmentation.git 
WORKDIR mmsegmentation
RUN pip install -v -e .

# Clean up 
WORKDIR /home 
RUN rm -rv torchsparse mmcv mmdetection3d mmsegmentation




