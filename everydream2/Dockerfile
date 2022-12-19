FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Install requirements
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

# Install base utilities and Python 3.10
RUN apt-get update && apt-get upgrade -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y wget git python3.10-dev python3.10-venv curl zip unzip git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0

# Install Pytorch 1.21.1 for CUDA 1.16
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install required python modules
ADD requirements.txt .
RUN python3 --version && pip install -r requirements.txt

# Install transformers and upgrade requests
RUN pip install transformers -i https://pypi.python.org/simple && \
    pip install requests --upgrade

# Install xformers (shouldn't take too long since ninja is installed)
RUN export FORCE_CUDA=1 && export TORCH_CUDA_ARCH_LIST=8.6 && CUDA_VISIBLE_DEVICES=0 && \
    pip install git+https://github.com/facebookresearch/xformers.git#egg=xformers

# Precache .cache with huggingface files
RUN mkdir -p /root/.cache/huggingface/
ADD cache/huggingface/* /root/.cache/huggingface/

# Add EveryDream repo
ADD repo/EveryDream2trainer /everydream2
WORKDIR everydream2
RUN python3 utils/get_yamls.py

# Add welcome banner
ADD welcome-banner.txt /root/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt" >> ~/.bashrc