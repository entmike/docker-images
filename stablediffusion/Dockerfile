FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN mkdir /weights && mkdir -p /root/.cache/huggingface && mkdir -p /workspace/out
ADD stable-diffusion-models/sd-v1-4.ckpt /weights/sd-v1-4.ckpt
ADD huggingface /root/.cache/huggingface

# Set bash as default shell, non-interactive
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

# Create workspace working directory
WORKDIR /workspace

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    git wget curl nodejs npm openssh-server zip unzip ffmpeg &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Node
RUN curl -sL https://deb.nodesource.com/setup_16.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    numpy \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    jupyterlab ipywidgets jupyter-archive

RUN jupyter nbextension enable --py widgetsnbextension

RUN git clone https://github.com/crowsonkb/k-diffusion.git && \
    git clone https://github.com/CompVis/stable-diffusion.git && \
    git clone https://github.com/CompVis/taming-transformers.git 

RUN cd taming-transformers && pip install -e . && cd ..

RUN cp -R stable-diffusion/ldm/ k-diffusion/ && \
    cp stable-diffusion/configs/stable-diffusion/v1-inference.yaml k-diffusion/ && \
    cd k-diffusion && pip install -r requirements.txt

RUN pip install omegaconf pytorch_lightning transformers
ADD stable-diffusion.ipynb /workspace/stable-diffusion.ipynb
ADD workermode /workermode
ADD start.sh /start.sh
CMD [ "/start.sh" ]
EXPOSE 8888