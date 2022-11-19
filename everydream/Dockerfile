FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Pre-fill .cache
ADD .cache /root/.cache

# Add EveryDream repo
ADD repo/EveryDream-trainer /everydream

# Install conda environment
WORKDIR /everydream
RUN conda env create -f environment.yaml
RUN conda init bash

ADD welcome-banner.txt /root/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt && echo ${DD_VERSION}" >> ~/.bashrc && \
    echo "conda activate everydream" >> ~/.bashrc