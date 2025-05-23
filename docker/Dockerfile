FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# nvidia-container-runtime config
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV TZ=Australia/Brisbane
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y git

RUN pip install opencv-python==4.11.0.86
RUN pip install typing_extensions==4.13.2
RUN pip install ftfy==6.3.1
RUN pip install regex==2024.11.6

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
RUN mim install 'mmsegmentation==1.2.2'
RUN mim install 'mmdet3d==1.4.0'

# Clean up 
WORKDIR /home 
RUN rm -rv torchsparse mmcv 



