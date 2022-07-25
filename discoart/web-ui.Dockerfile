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
RUN curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | gpg --dearmor | tee /usr/share/keyrings/yarnkey.gpg >/dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/yarnkey.gpg] https://dl.yarnpkg.com/debian stable main" | tee /etc/apt/sources.list.d/yarn.list

RUN apt-get update && apt-get install -y \
    nodejs yarn

# Build NextJS
WORKDIR /
RUN git clone https://github.com/Run-Pod/discoart-ui.git
WORKDIR /discoart-ui
RUN yarn && yarn build

WORKDIR /workspace

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    numpy \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
#     jupyterlab ipywidgets jupyter-archive

# RUN jupyter nbextension enable --py widgetsnbextension

RUN pip install discoart

ADD start.sh /start.sh
ADD welcome-banner.txt /root/welcome-banner.txt
COPY examples /examples
ADD welcome.ipynb /root/welcome.ipynb

RUN echo 'cat ~/welcome-banner.txt' >> ~/.bashrc &&\
    echo 'cd /workspace' >> ~/.bashrc

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

RUN rm -Rf /root/.cache && mkdir -p /models/.cache && ln -s /models/.cache /root
ADD workermode /workermode
CMD [ "/start.sh" ]
EXPOSE 8888