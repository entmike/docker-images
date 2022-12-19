FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Install requirements
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

# Install base utilities
RUN apt-get update && apt-get upgrade -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y wget git python3.10-dev python3.10-venv curl zip unzip git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0
# Install Pytorch 1.21.1 for CUDA 1.13
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
ADD requirements.txt .
RUN python3 --version && pip install -r requirements.txt
RUN pip install transformers -i https://pypi.python.org/simple
# WARNING - Compiling xformers takes a while.
#RUN export FORCE_CUDA=1 && export TORCH_CUDA_ARCH_LIST=8.6 && CUDA_VISIBLE_DEVICES=0 && \
#    pip install git+https://github.com/facebookresearch/xformers.git#egg=xformers
ADD xf/xformers /usr/lib/python3/dist-packages/xformers
ADD xf/xformers-0.0.15+e117abd.d20221215.dist-info /usr/lib/python3/dist-packages/xformers-0.0.15+e117abd.d20221215.dist-info
ADD cache/* /root/.cache/
# Add EveryDream repo
ADD repo/EveryDream2trainer /everydream2
WORKDIR everydream2
RUN python3 utils/get_yamls.py
# Add welcome banner
ADD welcome-banner.txt /root/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt" >> ~/.bashrc