# FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Install requirements
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

RUN apt-get update && apt-get upgrade -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y wget git python3.10-dev python3.10-venv zip unzip sudo libsm6 libxext6 ffmpeg git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0
RUN python3 --version
# Create a stable diffusion user
RUN useradd -ms /bin/bash stable -p stable && adduser stable sudo
RUN echo "stable ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/stable

USER stable
# Install AUTOMATIC1111/stable-diffusion-webui
RUN bash -c "COMMANDLINE_ARGS='--exit --skip-torch-cuda-test' bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)"

# Install deps
RUN echo "source /home/stable/stable-diffusion-webui/venv/bin/activate" >> ~/.bashrc
WORKDIR /home/stable/stable-diffusion-webui
ADD workermode/requirements.txt /home/stable/stable-diffusion-webui/workermode-requirements.txt
# https://stackoverflow.com/questions/74578145/attributeerror-module-h11-has-no-attribute-event
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/3259
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2073#issuecomment-1273954846
# WARNING - Compiling xformers takes a while.
RUN bash -c "source /home/stable/stable-diffusion-webui/venv/bin/activate && \
export FORCE_CUDA=1 && export TORCH_CUDA_ARCH_LIST=8.6 && CUDA_VISIBLE_DEVICES=0 && pip install git+https://github.com/facebookresearch/xformers.git#egg=xformers && \
pip install --force-reinstall httpcore==0.15 && \
pip install -r workermode-requirements.txt"

# Embed SD models
ADD models/Stable-diffusion /home/stable/stable-diffusion-webui/models/Stable-diffusion
# Embed VAE
ADD models/VAE /home/stable/stable-diffusion-webui/models/VAE
# Embed torch cache
ADD cache/* /home/stable/.cache/

ADD start.sh /start.sh
ADD workermode/workermode.py /home/stable/stable-diffusion-webui
ADD workermode/workermode.sh /home/stable/stable-diffusion-webui
RUN git pull