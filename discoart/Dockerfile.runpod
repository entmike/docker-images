FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set bash as default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    git wget curl

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    numpy \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    jupyterlab ipywidgets jupyter-archive

RUN jupyter nbextension enable --py widgetsnbextension

RUN pip install discoart
ADD start.sh /start.sh

### Discoart ENV Params ###
# more verbose logs
# ENV DISCOART_LOG_LEVEL='DEBUG'

# opt-out from cloud backup
ENV DISCOART_OPTOUT_CLOUD_BACKUP='1'
# use a custom output directory for all images and results
ENV DISCOART_OUTPUT_DIR='/workspace/out'

# disable ipython dependency
# ENV DISCOART_DISABLE_IPYTHON='1'

# disable result summary after the run ends
# ENV DISCOART_DISABLE_RESULT_SUMMARY='1'

# use a custom default parameters file
# ENV DISCOART_DEFAULT_PARAMETERS_YAML='path/to/your-default.yml'

# use a custom cut schedules file
# ENV DISCOART_CUT_SCHEDULES_YAML='path/to/your-schedules.yml'

# use a custom list of models file
# ENV DISCOART_MODELS_YAML='path/to/your-models.yml'

# use a custom cache directory for models and downloads (except for CLIP models, apparently?)
# ENV DISCOART_CACHE_DIR='/models'

# disable the listing of diffusion models on Github, remote diffusion models allows user to use latest models without updating the codebase.
# ENV DISCOART_DISABLE_REMOTE_MODELS='1'

WORKDIR /workspace

# Make examples area that will survive future updates without getting entangled in /workspace volume
COPY examples /examples
RUN ln -s /examples /workspace

RUN rm -Rf /root/.cache && mkdir -p /models/.cache && ln -s /models/.cache /root
CMD [ "/start.sh" ]
EXPOSE 8888