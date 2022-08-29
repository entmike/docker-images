FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

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
    numpy==1.19.2 \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    jupyterlab ipywidgets jupyter-archive

RUN jupyter nbextension enable --py widgetsnbextension

RUN git clone https://github.com/xinntao/Real-ESRGAN.git

RUN pip install basicsr facexlib
RUN cd Real-ESRGAN && pip install -r requirements.txt && python setup.py develop && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models && \
    wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P /usr/local/lib/python3.8/dist-packages/facexlib/weights && \
    wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -P /usr/local/lib/python3.8/dist-packages/facexlib/weights && \
    wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P /usr/local/lib/python3.8/dist-packages/gfpgan/weights/

ADD workermode /workermode
ADD start.sh /start.sh
CMD [ "/start.sh" ]
EXPOSE 8888