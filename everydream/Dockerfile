FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget git git-lfs && \
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
RUN echo "conda activate everydream" >> ~/.bashrc

# Add EveryDream tools
ADD repo/EveryDream /tools
WORKDIR /tools

# Pre-fill tools .cache
ADD .toolcache /tools/.cache

# Install pre-requisites (requirements.txt not needed b/c conda env)
RUN bash -c "git clone https://github.com/salesforce/BLIP scripts/BLIP"
RUN bash -c "source activate everydream && pip install aiofiles timm fairscale colorama"

# Install LAION2B-en Dataset parquets
RUN rm laion/* && \
    git lfs install && \
    git clone https://huggingface.co/datasets/laion/laion2B-en-aesthetic laion

# Finish up
WORKDIR /everydream

# Add welcome banner
ADD welcome-banner.txt /root/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt" >> ~/.bashrc