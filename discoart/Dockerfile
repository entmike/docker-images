ARG base_image
FROM ${base_image}

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

ENV DISCOART_VERSION=v0.10.9

ADD start.sh /start.sh
ADD welcome-banner.txt /root/welcome-banner.txt
COPY examples /examples
ADD welcome.ipynb /root/welcome.ipynb

RUN echo 'cat ~/welcome-banner.txt' >> ~/.bashrc &&\
    echo 'cd /workspace' >> ~/.bashrc

# opt-out from cloud backup
ENV DISCOART_OPTOUT_CLOUD_BACKUP='1'

# use a custom output directory for all images and results
ENV DISCOART_OUTPUT_DIR='/workspace/out'

RUN rm -Rf /root/.cache && mkdir -p /models/.cache && ln -s /models/.cache /root
ADD workermode /workermode
CMD [ "/start.sh" ]
EXPOSE 8888