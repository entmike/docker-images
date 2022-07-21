FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN mkdir -p /models

# Bust into layers
COPY models/torch /models/.cache/torch
COPY models/clip/base /models/.cache/clip
COPY models/discoart/base /models/.cache/discoart

CMD [ "/bin/bash" ]