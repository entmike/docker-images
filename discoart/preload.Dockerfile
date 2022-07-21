FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN mkdir -p /models
COPY models /models/.cache

CMD [ "/bin/bash" ]