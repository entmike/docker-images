FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Install requirements
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

RUN apt-get update && apt-get install -y \
    wget git python3 python3-venv zip unzip sudo libsm6 libxext6 ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Create a stable diffusion user
RUN useradd -ms /bin/bash stable -p stable && adduser stable sudo
RUN echo "stable ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/stable
RUN mkdir /weights && chown -Rf stable /weights

USER stable
# Embed SD 1.4 weights
ADD stable-diffusion-models/sd-v1-4.ckpt /weights/sd-v1-4.ckpt
# Embed torch cache
RUN mkdir /home/stable/.cache
ADD cache/* /home/stable/.cache/

# Install AUTOMATIC1111/stable-diffusion-webui
RUN bash -c "COMMANDLINE_ARGS='--exit --skip-torch-cuda-test' bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)"
RUN echo "source /home/stable/stable-diffusion-webui/venv/bin/activate" >> ~/.bashrc
WORKDIR /home/stable/stable-diffusion-webui
ADD workermode/requirements.txt /home/stable/stable-diffusion-webui/workermode-requirements.txt
RUN bash -c "source /home/stable/stable-diffusion-webui/venv/bin/activate && pip install -r workermode-requirements.txt"
ADD start.sh /start.sh
ADD workermode/workermode.py /home/stable/stable-diffusion-webui
ADD workermode/workermode.sh /home/stable/stable-diffusion-webui