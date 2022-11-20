#
# Trainer Image
#
FROM entmike/everydream:base

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

# Add welcome banner
ADD welcome-banner.txt /root/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt" >> ~/.bashrc